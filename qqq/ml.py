import numpy as np
import numpy.random as npr
import theano as th
import theano.tensor.nnet as thn
import theano.tensor as T
from tqdm import tqdm

from typing import Dict, List, Set # noqa

from .util import ensure_list, check_all_same_length


def gen_mlc_weights(Y, a=1, dist_weight=1, n_iter=25000):
    '''
    Generates multilabel classification sample weights.

    Solve for probability over Y, p, such that
      a*H(p)/log(N) - sum([H(y_i)/(log(2)*len(labels)) for i in labels])
    is maximized.

    This process can be quite slow, but produces a good distribution. It's
    worth it, given it only needs to be run once per data set.

    Arguments:
        dist_weight: relative weight of the distribution entropy loss
        n_iter: number of iterations for which to train
    '''

    N = Y.shape[0]

    t_Hp_fac = th.shared(dist_weight / np.log(N))
    t_Hb_fac = th.shared(1 / (Y.shape[1] * np.log(2)))

    t_softits = th.shared(npr.normal(0, 1, Y.shape[0]))
    t_Y = th.shared(Y)

    def softmax(arr):
        e = T.exp(arr)
        return e / T.sum(e)

    def bentropy(marg):
        return -T.sum(marg * T.log(marg) + (1 - marg) * T.log(1 - marg))

    def entropy(softits):
        p = softmax(softits)
        return -T.dot(p, T.log(p))

    dist_gain = t_Hp_fac * entropy(t_softits)
    marg = T.dot(thn.softmax(t_softits), t_Y)
    marg_gain = t_Hb_fac * bentropy(marg)
    t_gain = dist_gain + marg_gain

    t_a = th.shared(a)
    # ghetto nesterov momentum
    t_b = th.shared(1 - 0.9999)
    t_d = th.shared(np.zeros(Y.shape[0]))
    g = th.grad(t_gain, t_softits)

    f = th.function(
        [],
        [dist_gain, marg_gain],
        updates=[
            (t_softits, t_softits + t_a * t_d),
            (t_d, t_b * t_d + g),
        ],
    )

    pbar = tqdm(range(n_iter), desc='optimizing sampling distribution')
    for i in pbar:
        x = f()
        if not i % 100:
            pbar.set_description(
                f'normed sampling distribution entropy {float(x[0]):4.3f}, '
                f'normed marginal entropy {float(x[1]):4.3f}'
            )

    return th.function([], [softmax(t_softits)])()[0]


def undersample_mlc(y):
    '''
    Returns the undersampling indices.

    An approximate undersampling by iteratively choosing samples from the
    least total-represented class.

    The maximum remaining imbalance, largest to smallest, is n_labels : 1.
    '''

    ns_per_class = y.sum(axis=0)
    target_per_class = np.min(ns_per_class)

    ixes = []  # type: List[int]
    for next_class_ix in np.argsort(ns_per_class):
        candidates_pos = np.where(y[:, next_class_ix] >= 0.5)[0]
        candidates_neg = np.where(y[:, next_class_ix] <= 0.5)[0]

        n_have_pos = y[ixes].sum(axis=0)[next_class_ix]
        n_have_neg = len(ixes) - n_have_pos

        n_needed_pos = target_per_class - n_have_pos
        n_needed_neg = target_per_class - n_have_neg

        if n_needed_pos > 0:
            ixes += list(npr.choice(candidates_pos, size=n_needed_pos))
        if n_needed_neg > 0:
            ixes += list(npr.choice(candidates_neg, size=n_needed_neg))

    return ixes


def mk_bin_from_mlc(y_mlc):

    return [
        np.stack([y_mlc[:, i], 1 - y_mlc[:, i]], axis=1)
        for i in range(y_mlc.shape[1])
    ]


def gen_random_labels(X, n_classes, pvec=None):
    '''
    Returns a random labelling of dataset X.

    Labels are one-hot encoded, with `n_classes` dimensions.

    Args:
        X: number of labels to generate, integer or array-like.
            If array-like, `X.shape[0]` labels will be generated.
        n_classes: number of output classes
        pvec: array-like, gives probabilities of each class. If `None`, all
            classes are equiprobable.

    '''

    try:
        num = X.shape[0]
    except AttributeError:
        num = X

    pvec = np.ones((n_classes,)) / n_classes

    return npr.multinomial(1, pvec, size=num)


def generate_batches(x, xn: str, y=None, yn=None, *, bs=128,  # noqa
                     balance=False, sample_weights=None, sequential=False):
    '''
    Generate batches of data from a larger array.

    Arguments:
        x: input array
        y: label array
        bs: batch size
        balance: if y is one-hot: if True, return batches with balanced
            classes, on average.
    '''
    have_y = y is not None
    if have_y and yn is None:
        raise ValueError('need to give names to y values if they are given')

    x, y = ensure_list(x), ensure_list(y)
    xn, yn = ensure_list(xn), ensure_list(yn)

    check_all_same_length(*xn, *yn)

    if len(xn) != len(x) or len(yn) != len(y):
        raise ValueError(
            f'wrong number of names! {len(xn), len(x)}, {len(yn), len(y)}')

    # explicit weights overrde balancing
    have_sw = sample_weights is not None
    if have_sw:
        probs = sample_weights.astype(np.float64) / sample_weights.sum()
        balance = False

    elif balance and have_y:
        if len(y) != 1:
            raise TypeError(
                'balancing is undefined unless y '
                'is a single array of 1-hot vectors'
            )
        n_per_class = y[0].sum(axis=0, dtype=np.uint64)
        n_classes = y[0].shape[1]
        p_ixes = y[0].argmax(axis=1)
        probs = (
            np.ones(len(y[0]), dtype=np.float64) /
            (n_classes * n_per_class[p_ixes])
        )
    else:
        probs = np.ones(x[0].shape[0]) / x[0].shape[0]

    ix_arange = npr.permutation(len(x[0]))

    seq = 0
    while True:
        if sequential:
            ixes = np.arange(seq, seq + bs) % x[0].shape[0]
            seq = (seq + bs) % x[0].shape[0]
        else:
            ixes = npr.choice(ix_arange, size=bs, p=probs)

        Xb = [sub_x[ixes] for sub_x in x]
        X_out = {n: x for n, x in zip(xn, Xb)}

        if have_y:
            Yb = [sub_y[ixes] for sub_y in y]
            Y_out = {n: y for n, y in zip(yn, Yb)}

            yield X_out, Y_out
        else:
            yield X_out


def batch_transformer(gen, f, *, inplace: bool,  # noqa
                      joint=False, get_shape=None,
                      in_key='x0', out_key='x0'):
    '''
    Transforms a function over individual samples into a chainable interator
    over batches.

    Arguments:
        gen: the generator whose data stream to transform. Must output
            dicts or tuples of dicts.
        f: the function to apply to individual samples
        inplace: True indicates the function is in-place and will overwrite
            the batch array
        get_shape: calculate the output shape from the input shape for
            individual samples. Has no meaning if `inplace` is True. If None,
            output shape is assumed unchanged by the function.
    '''

    get_shape = get_shape or (lambda shape: shape)

    if in_key is None:
        in_key = 'x0'
    if out_key is None:
        out_key = 'x0'

    while True:
        xy = next(gen)

        if isinstance(xy, tuple):
            xbd_in, ybd_in = xy
        else:
            xbd_in, ybd_in = xy, None

        if not joint:
            out_in_y = False
            if in_key in xbd_in.keys():
                zb_in = xbd_in[in_key]
            elif in_key in ybd_in.keys():
                out_in_y = True
                zb_in = ybd_in[in_key]
            else:
                raise ValueError(
                    'the value of inplace must correspond to a data key')

            xbd_out, ybd_out = xbd_in, ybd_in

            if inplace:
                for z_in in zb_in:
                    f(z_in)

            else:
                zb_out = np.zeros(
                    zb_in.shape[0:1] + get_shape(zb_in.shape[1:])
                )
                for z_out, z_in in zip(zb_out, zb_in):
                    z_out[:] = f(z_in)

                if out_in_y:
                    ybd_out[out_key] = zb_out
                else:
                    xbd_out[out_key] = zb_out

        else:
            raise NotImplementedError

        yield xbd_out, ybd_out


def get_k_of_each(y, k):
    '''
    Returns the indices of a subarray containing  min(#members, k)
    members of each class of y.

    Args:
        y: binary-encoded labels

    Returns:
        ixes: indices of the selected elements

    Examples:
        >>> y_sub = y[get_k_of_each(10, y)]
    '''
    if len(y.shape) != 2:
        raise ValueError('This function expects a 2D array.')

    ixes = []
    ymax = np.argmax(y, axis=1)

    for i in range(y.shape[1]):
        ixes_i = np.where(ymax == i)[0]
        ixes.append(npr.choice(ixes_i, min(len(ixes_i), k), replace=False))

    return np.concatenate(ixes)


def complement_ixes(ixes, y):
    '''
    Generates the complement of an array of indices of a given array.

    Arguments:
        ixes: the list of indices to complement
        y: the array, or length of array, with respect to which to complement
            the indices
    '''
    try:
        y = len(y)
    except Exception:
        pass

    all_ixes = np.ones(y, dtype=np.uint8)
    all_ixes[ixes] = 0

    return np.where(all_ixes == 1)[0]


def bagged_kfold(self, load_fun, k=3):
    for kx in range(k):
        train = set()
        val = set()  # type: Set[str]
        for fn, val in self._map:
            if kx / k < val / self.range <= (kx + 1) / k:
                val.add(fn)
            else:
                train.add(fn)


def round_to_pow2(n):
    return 2 ** np.ceil(np.log2(n))
