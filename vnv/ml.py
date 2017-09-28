from typing import Dict, List, Set # noqa
from warnings import warn

import numpy as np
import numpy.random as npr

from .util import as_list, check_all_same_length
from .log import get_logger


LOG = get_logger(__name__)


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


def get_dict_data_len(x_dict):
    '''
    Get the number of samples of data packed into a keras_style dictionary.
    '''
    return check_all_same_length(*x_dict.values())


def dict_tv_split(*ds, f_train=0.85, shuffle=True, seed=None):
    vals = []
    for d in ds:
        for v in d.values():
            vals.append(v)

    n_samples = check_all_same_length(*vals)
    n_train = int(f_train * n_samples)

    if shuffle:
        if seed is not None:
            npr.seed(seed)
        ixes = npr.permutation(n_samples)
    else:
        ixes = np.arange(n_samples)

    t_ixes = ixes[:n_train]
    v_ixes = ixes[n_train:]

    train_dicts = tuple({k: v[t_ixes] for k, v in d.items()} for d in ds)
    val_dicts = tuple({k: v[v_ixes] for k, v in d.items()} for d in ds)

    return train_dicts, val_dicts


def get_train_test_gens(x_dict, y_dict, val_split, seed=None, **kwargs):

    if kwargs.pop('bound_ixes', None) is not None:
        LOG.warn(f'ignoring passed bound_ixes in favour of random split')

    npr.seed(seed)
    random_ixes = npr.permutation(len([*x_dict.values()][0]))
    n_train = int(val_split * len(random_ixes))

    gen_t = generate_batches(
        x_dict, y=y_dict, bound_ixes=random_ixes[:n_train], **kwargs,
    )
    gen_v = generate_batches(
        x_dict, y=y_dict, bound_ixes=random_ixes[n_train:], **kwargs,
    )

    return gen_t, gen_v


def generate_batches(
        x_dict, *, y=None,
        bs=128, balance=False, sequential=False,
        sample_weights=None, bound_ixes=None,
        ):
    '''
    Generate batches of data from a larger array.

    Arguments:
        x_dict: input dictionary of input names to arrays
        y_dict: label dictionary of label names to arrays
        bs: size of batch to generate
        balance: if y is one-hot: if True, return batches with balanced
            classes, on average.
        sample_weights: possibly unnormalized probabilities with which to
            draw each sample, independently. Supersedes `balance`.
        sequential: if True, will iterate over the input arrays in sequence,
            yielding contiguous batches. Still yields forever, looping to the
            start when input arrays are exhausted
        bound_ixes: sequence of indicies into the x/y arrays. Elements at those
            indices will be treated as though they comprise the whole array.
            In sequential mode, elements will be traversed in the order given
            by this sequence.
    Yields:
        Xb: dictionary of x batches
        Yb: dictionary of y batches, or None
    '''
    y_dict = y or {}
    have_y = bool(y_dict)

    raw_n_samples = check_all_same_length(*x_dict.values(), *y_dict.values())
    if bound_ixes is None:
        ix_arange = np.arange(raw_n_samples, dtype=np.uint64)
    else:
        ix_arange = np.array(bound_ixes)
    n_samples = len(ix_arange)

    # explicit weights overrde balancing
    have_sw = sample_weights is not None
    if have_sw:
        sample_weights = np.array(sample_weights)[ix_arange].astype(np.uint64)
        probs = sample_weights / sample_weights.sum()
        balance = False

    elif balance and have_y:
        if len(y_dict) != 1:
            raise ValueError(
                'balancing is undefined when multiple label sets are present')

        y_arr = list(y_dict.values())[0][bound_ixes]
        if np.max(y_arr.sum(axis=1) > 1.):
            warn('given label array does not appear to be one-hot encoded',
                 RuntimeWarning)

        n_per_class = y_arr.sum(axis=0, dtype=np.uint64)
        n_classes = y_arr.shape[1]
        p_ixes = y_arr.argmax(axis=1)
        probs = (
            np.ones(n_samples, dtype=np.float64) /
            (n_classes * n_per_class[p_ixes])
        )
    else:
        probs = np.ones(n_samples) / n_samples

    seq = 0

    while True:
        if sequential:
            ixes = ix_arange[np.arange(seq, seq + bs) % n_samples]
            seq = (seq + bs) % n_samples
        else:
            ixes = npr.choice(ix_arange, size=bs, p=probs)

        xbd = {k: x[ixes] for k, x in x_dict.items()}

        if have_y:
            ybd = {k: y[ixes] for k, y in y_dict.items()}
            yield xbd, ybd
        else:
            yield xbd


def apply_bts(gen, bts, *, train):
    for bt in bts:
        gen = bt(gen, train=train)
    return gen


def batch_transformer( # noqa
    f, *, inplace: bool, mode='each', get_shape=None,
    in_keys=None, out_keys=None, out_in_ys=None, pop_in_keys=True,
    train_only=False):
    '''
    Transforms functions over individual samples into functions taking
    batch generators and returning batch generators of transformed data.

    Arguments:
        f: the function to apply to individual samples
        inplace: whether the function operates in-place
        mode: how to apply the transformations. 'each' means the function
            will be applied to each input in in_keys and output on the
            same key. 'mix' means the function will be called with the values
            corresponding to each of the in_keys as an argument, and is
            expected to produce a value for each of the output keys.
        get_shape: calculate the output shape from the input shape for
            individual samples. Has no meaning if `inplace` is True. If None,
            output shape is assumed unchanged by the function.  If mode is
            'mix', assumed to the shapes corresponding to the input keys as
            arguments and to return a shape for each of the output keys, in
            their respective order.
        in_keys: the input keys over which to operate. Can be in either the
            x or the y dict
        out_keys: the output keys to which to move the input. Defaults to
            `in_keys`.
        pop_in_keys: if True, input keys will be popped from the dictionary.
            Naturally, if some output keys share the names of some input
            keys, they will be set after the originals are deleted and will
            be present in the output.
        out_in_ys: list of bools, of same length as out_keys. A True entry
            indicates the corresponding output should be placed in the y
            dictionary. Defaults to using the dictionary of the corresponding
            input key. This default will break if out_keys is of different
            length than in_keys, and should be used carefully if mode is 'mix'.
        train_only: if True, the transformations will only be applied if the
            returned generator is instantiated with a `train=True` kwarg.

    Returns:
        a function taking a generator and a train flag and returning the
        transformed generator
    '''

    get_shape = get_shape or (lambda shape: shape)

    in_keys = as_list(in_keys)
    if out_keys and inplace:
        raise ValueError('cannot set explicit out_keys if inplace is True')

    in_in_ys = None  # type: ignore

    def bt(gen, train=True):
        '''
        Arguments:
            gen: the generator whose data stream to transform. Must output
                dicts or tuples of dicts.
            train: if False, and the batch_transformer defining this function
                was called with `train_only=True`, the data will be returned
                unchanged
        '''
        nonlocal in_keys
        nonlocal out_keys
        nonlocal in_in_ys
        nonlocal out_in_ys

        while True:
            xy = next(gen)
            have_y = isinstance(xy, tuple)

            if have_y:
                xbd, ybd = xy
            else:
                xbd, ybd = xy, {}

            bs = check_all_same_length(
                *xbd.values(), *ybd.values(),
                msg='differently sized batch elements')

            if train_only and not train:
                if have_y:
                    yield xbd, ybd
                else:
                    yield xbd
                continue

            if in_keys is None:
                # by default, apply transformations to all inputs and no labels
                in_keys = [*xbd.keys()]
            out_keys = out_keys or in_keys
            if in_in_ys is None:
                in_in_ys = []
                for in_key in in_keys:
                    if in_key in xbd.keys():
                        in_in_y = False
                    elif in_key in ybd.keys():
                        in_in_y = True
                    else:
                        raise KeyError(
                            f'the key {in_key} was not found. Found keys: '
                            f'{set(xbd.keys())}, {set(ybd.keys())}'
                        )

                    in_in_ys.append(in_in_y)

                out_in_ys = out_in_ys or in_in_ys
                check_all_same_length(
                    out_in_ys, out_keys,
                    msg='output dict specfication must have the same '
                        'length as the number output keys')

            if mode == 'each':
                for ik, ok, iiy, oiy in\
                        zip(in_keys, out_keys, in_in_ys, out_in_ys):

                    zbd_in = ybd if iiy else xbd
                    zb_in = zbd_in[ik]
                    zbd_out = ybd if oiy else xbd

                    if inplace:
                        for z_in in zb_in:
                            f(z_in)
                    else:
                        zb_out = np.zeros((bs,) + get_shape(zb_in.shape[1:]))

                        for bx in range(bs):
                            zb_out[bx] = f(zb_in[bx])

                        if pop_in_keys:
                            del zbd_in[ik]

                        zbd_out[ok] = zb_out

            elif mode == 'mix':
                zb_ins = [
                    (ybd if iiy else xbd)[ik]
                    for ik, iiy in zip(in_keys, in_in_ys)
                ]
                shapes = get_shape(*[zb_in.shape[1:] for zb_in in zb_ins])
                zb_outs = [np.zeros((bs,) + shape) for shape in shapes]

                for bx, z_ins in enumerate(zip(*zb_ins)):
                    rs = f(*z_ins)
                    for ox, r in enumerate(rs):
                        zb_outs[ox][bx] = r

                for ik, ok, iiy, oiy, zb_out in\
                        zip(in_keys, out_keys,  # type: ignore
                            in_in_ys, out_in_ys, zb_outs):

                    if pop_in_keys:
                        del (ybd if iiy else xbd)[ik]

                    (ybd if oiy else xbd)[ok] = zb_out

            else:
                raise ValueError(f'invalid mode {mode}')

            if have_y:
                yield xbd, ybd
            else:
                yield xbd

    return bt


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
