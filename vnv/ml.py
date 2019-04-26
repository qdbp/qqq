"""
Utilities for manipulating and preprocessing data flows.
"""
from copy import copy
from typing import (Any, Callable, Collection, Dict, Generator, List, Sequence,
                    Set, Tuple, Type, Union)
from warnings import warn

import numpy as np
import numpy.random as npr
from sklearn.model_selection import ShuffleSplit, train_test_split

from .log import get_logger
from .util import alleq, as_list, check_all_same_length, die, flatten

LOG = get_logger(__name__)

Shape_t = Tuple[int, ...]
DataDict = Dict[str, np.ndarray]


def undersample_mlc(y):
    """
    Returns the undersampling indices.

    An approximate undersampling by iteratively choosing samples from the
    least total-represented class.

    The maximum remaining imbalance, largest to smallest, is n_labels : 1.
    """

    ns_per_class = y.sum(axis=0)
    target_per_class = np.min(ns_per_class)

    ixes: List[int] = []
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


def gen_random_labels(
    X: Union[np.ndarray, int], n_classes: int, pvec=None
) -> np.ndarray:
    """
    Returns a random labelling of dataset X.

    Labels are one-hot encoded, with `n_classes` dimensions.

    Accepts an optional vector of probabilities with which to generate each
    class.

    Args:
        X: number of labels to generate, integer or array-like.
            If array-like, `X.shape[0]` labels will be generated.
        n_classes: number of output classes
        pvec: array-like, gives probabilities of each class. If `None`, all
            classes are equiprobable.

    Returns:
        ndarray[N, n_classes], the random, one-hot encoded labels.
    """

    if isinstance(X, int):
        num = X
    else:
        num = X.shape[0]

    pvec = np.ones((n_classes,)) / n_classes

    return npr.multinomial(1, pvec, size=num)


def get_dict_data_len(x_dict: Dict[Any, Collection]):
    """
    Get the number of samples of data packed into a dictionary of arrays.

    Raises ValueError if the arrays in the dictionary are not all the same
    length.
    """
    return check_all_same_length(*x_dict.values())


def dict_tt_split(*data_dicts, **tts_kwargs):
    """
    Split in tandem data arrays held as keys in one or more data dictionaries.

    Uses `sklearn.model_selection.train_test_split` internally, and accepts
    that method's options.

    Arguments:
        data_dicts: the dicts to split. Should contain equal-length arrays as
            values
        tts_kwargs:
            options to pass to `train_test_split`
    Returns:
        Two tuples each with the same number of dictionaries as was passed,
        each with the same keys as the original. The dicts in the first tuple
        hold the train data, those in the second the test data.
    """

    rs = tts_kwargs.pop("random_state", None) or npr.RandomState()

    arrs = flatten([list(d.values()) for d in data_dicts])
    check_all_same_length(*arrs)

    train_dict = [{} for d in data_dicts]  # type: ignore
    test_dict = [{} for d in data_dicts]  # type: ignore
    for dx, data_dict in enumerate(data_dicts):
        for k, arr in data_dict.items():
            train, test = train_test_split(
                arr, random_state=copy(rs), **tts_kwargs, shuffle=False
            )
            train_dict[dx][k] = train
            test_dict[dx][k] = test

    return tuple(train_dict), tuple(test_dict)


def dict_arr_concat(
    *dicts: Dict[Any, np.ndarray], check_keys=False, check_lens=False, axis=0
):
    """
    Concatenate numpy arrays held in dictionaries.

    Arguments:
        dicts: dictionaries of numpy arrays
        check_keys: if `True`, will raise a `ValueError` if the sets of keys of
            each element `dicts` are not all equal.
        check_lens: if `True`, will raise a `ValueError` if, within each
            dictionary, the length of all value arrays are not exactly equal.
            For greater clarity, the lengths of value arrays across different
            dicts are not compared.
    Returns:
        A dictionary. Its keys are the union of the keys in `dicts`. The value
        mapped by each key `k` is the concatenation of all arrays contained in
        the input dicts under `k`, in the order they are found in `dicts`.
    """

    if check_lens and not check_keys:
        LOG.warn("Enabling check_keys because check_lens is enabled")
        check_keys = True

    if check_keys:
        if not alleq([d.keys() for d in dicts]):
            raise ValueError(
                "The given dictionaries do not all have the same keys"
            )

    all_keys: Set[Any] = set()
    for d in dicts:
        all_keys |= d.keys()

    shapes: Dict[Any, Tuple[Shape_t, Shape_t]] = {}
    dtypes: Dict[Any, Type] = {}

    for k in all_keys:
        if not alleq(
            [
                d[k].shape[:axis] + d[k].shape[axis + 1 :]
                for d in dicts
                if k in d
            ]
        ):
            raise ValueError(f"Incompatible shapes on key '{k}'!")
        if not alleq([d[k].dtype for d in dicts if k in d]):
            raise ValueError(f"Incompatible array types for key '{k}'!")

        for d in dicts:
            if k in d:
                shapes[k] = (d[k].shape[:axis], d[k].shape[axis + 1 :])
                dtypes[k] = d[k].dtype
                break
        else:
            assert 0, "BUG! key in all_ke"

    if check_lens:
        for dx, d in enumerate(dicts):
            if not alleq([arr.shape[axis] for arr in d.values()]):
                raise ValueError(
                    f"Dict number {dx} has arrays of differing lengths, "
                    "and check_lens is enabled."
                )

    lens = {
        k: sum([d[k].shape[axis] for d in dicts if k in d]) for k in all_keys
    }
    out = {
        k: np.zeros(shapes[k][0] + (lens[k],) + shapes[k][1], dtype=dtypes[k])
        for k in all_keys
    }

    ixes = {k: 0 for k in all_keys}
    for d in dicts:
        for k, v in d.items():
            dl = v.shape[axis]
            sl = (
                (slice(None),) * len(shapes[k][0])
                + (slice(ixes[k], ixes[k] + dl),)
                + (slice(None),) * len(shapes[k][1])
            )
            out[k][sl] = v[...]
            ixes[k] += dl

    assert (
        ixes == lens
    ), "BUG! did not align arrays when filling `out` correctly"
    return out


def dict_arr_eq(d1: DataDict, d2: DataDict):
    """
    Correctly compares dictionaries of numpy arrays for equality correctly.
    """
    return (
        d1.keys() == d2.keys()
        and all(d1[k].shape == d2[k].shape for k in d1.keys())
        and all(np.allclose(d1[k], d2[k]) for k in d1.keys())
    )


def get_train_test_gens(
    *data_dicts: DataDict, splitter=ShuffleSplit, splitter_opts, **genbatchopts
):
    """
    Creates training and validation generators from a list of data
    dictionaries.

    First, a sequence of index arrays is created by the splitter object passed
    to this function. Then, each collection of inputs is partitioned according
    to these indices. Finally, partition is then passed to `generate_batches`.

    Arguments:
        data_dicts: the data dictionaries to create training and validation
            generators from.
        splitter: an object, or class of such an object, which has a `split`
            method returning a sequence of index arrays, one for each split.
            The default is the `sklearn.model_selection.ShuffleSplit` class
        splitter_opts:
            If `splitter` is passed as a class, it will be instantiated with
            `splitter_opts` as constructor arguments. If `splitter` is an
            object, these do nothing.
        genbatchopts:
            Options to pass to `generate_batches`.

    Returns:
        N generators, where N is the number of index arrays returned by the
        `splitter`. Each generator will independently yield samples from its
        partition according to the `genbatchopts` options passed.
    """

    if isinstance(splitter, type):
        splitter = splitter(**splitter_opts)
    else:
        if splitter_opts:
            warn(
                "Passed splitter_opts, but splitter is an existing object."
                "They will be ignored."
            )

    index_arrays = splitter.split()

    out = []
    for ixes in index_arrays:
        out.append(
            generate_batches(*data_dicts, bound_ixes=ixes, **genbatchopts)
        )

    return tuple(out)


def generate_batches(
    *data_dicts: DataDict,
    bs=128,
    balance_key: str = None,
    sequential: bool = False,
    sample_weights: np.ndarray = None,
    bound_ixes: np.ndarray = None,
):
    """
    Generate batches of data from a sequence of data dictionaries, i.e.
    dictionaries with numpy arrays as values.

    All values arrays across all dictionaries are assumed to hold arrays with
    the same number of samples N.

    Arguments:
        data_dicts: dictionaries holding data arrays
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
        a tuple of dictionaries, each a batch-sampled version of an input
        dictionary, in the same position. This is an infinite generator.
    """

    raw_n_samples = check_all_same_length(
        *flatten([list(d.values()) for d in data_dicts])
    )

    if bound_ixes is None:
        ix_arange = np.arange(raw_n_samples, dtype=np.uint64)
    else:
        ix_arange = np.asarray(bound_ixes)
    n_samples = len(ix_arange)

    # explicit weights overrde balancing
    have_sw = sample_weights is not None
    if have_sw:
        sample_weights = np.array(sample_weights)[ix_arange].astype(np.uint64)
        balance_key = None

        probs = sample_weights / sample_weights.sum()

    elif balance_key is not None:
        balance_values = [
            d[balance_key] for d in data_dicts if balance_key in d
        ]
        if not balance_values:
            raise ValueError(
                f"No data found for balance key '{balance_key}' in any dict."
            )
        elif len(balance_values) > 1:
            raise ValueError(
                f"More than one data array found for '{balance_key}'"
            )

        # list of dicts with the key -> first elem -> actual array in dict
        balance_arr = balance_values.pop()
        if len(balance_arr.shape) != 2:
            raise ValueError(
                f"Can't balance based on data with shape {balance_arr.shape}!"
            )

        if not np.allclose(balance_arr.sum(axis=1), np.ones(n_samples)):
            warn("Label array does not appear to be one-hot encoded")

        n_per_class = balance_arr.sum(axis=0, dtype=np.uint64)
        n_classes = balance_arr.shape[1]
        p_ixes = balance_arr.argmax(axis=1)

        probs = np.ones(n_samples, dtype=np.float64) / (
            n_classes * n_per_class[p_ixes]
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

        out = tuple({k: arr[ixes] for k, arr in d.items()} for d in data_dicts)
        if len(out) == 1:
            yield out[0]
        else:
            yield out


def apply_bts(gen, bts, *, train):
    for bt in bts:
        gen = bt(gen, train=train)
    return gen


def batch_transformer(
    f: Callable[..., np.ndarray],
    *,
    inplace: bool,
    mode: str = "each",
    batchwise_apply: bool = False,
    in_keys: Sequence[str] = None,
    out_keys: Sequence[Union[str, Tuple[int, str]]] = None,
    pop_in_keys: bool = True,
    train_only: bool = False,
):
    """
    Transforms functions over individual arrays into functions taking batch
    generators and returning batch generators of transformed data.

    Arguments:
        f: a batchwise function
        inplace: whether the function operates in-place
        mode: how to apply the transformations. The supported modes are:
            - 'each': the function will be applied to each input in in_keys,
              and its output will be under the same key.
            - 'mix': the function will be called with the values corresponding
              to each of the in_keys as an argument, in the same order. It is
              expected to produce one output for each of the output keys, in
              the given order.
        in_keys: which keys to apply the function to. All other keys will be
            passed through.
        out_keys: the output keys to which to move the input. Defaults to
            `in_keys`. If this is a list of keys. Each element must be either a
            string or a (int, str) tuple. If the latter, the int gives the
            index of the output dictionary. The length of this sequence must
            equal to that of `in_keys` in mode 'each', and to the the number of
            return values of `f`.  in mode 'mix'.
        pop_in_keys: if True, input keys will be popped from the dictionary.
            Naturally, if some output keys share the names of some input
            keys, they will be set after the originals are deleted and will
            be present in the output.
        train_only: if True, the transformations will only be applied if the
            returned generator is instantiated with a `train=True` kwarg.
        batchwise_apply: if True, the transformer function is assumed to be
            able to take the entire batch array as an argument. Otherwise, it
            will be applied to the individual elements of the batch in a loop.

    Returns:
        a function taking a generator and a train flag and returning the
        transformed generator
    """

    in_keys = as_list(in_keys)
    if out_keys and inplace:
        raise ValueError("cannot set explicit out_keys if inplace is True")

    batch_gen_t = Generator[Union[DataDict, Tuple[DataDict, ...]], None, None]

    def bt(gen: batch_gen_t, train=True):
        """
        Arguments:
            gen: the generator whose data stream to transform. Must output
                dicts or tuples of dicts.
            train: if False, and the batch_transformer defining this function
                was called with `train_only=True`, the data will be returned
                unchanged
        """
        nonlocal in_keys
        nonlocal out_keys
        nonlocal inplace

        def split_flatbatch(flatbatch, key_to_dix):
            out: List[DataDict] = [
                {} for i in range(max(key_to_dix.values()) + 1)
            ]
            for key, dix in key_to_dix.items():
                out[dix][key] = flatbatch[key]

            return tuple(out)

        if mode not in ["mix", "each"]:
            raise ValueError("Unrecognized mode f{mode}")

        if mode == "mix" and inplace:
            warn("Using inplace with 'mix' mode, inplace will be ignored")
            inplace = False

        # should loop forever
        for batch in gen:
            if not isinstance(batch, tuple):
                batch = (batch,)

            keys: Set = set()
            for d in batch:
                for key in d.keys():
                    if key in keys:
                        raise ValueError(
                            "Duplicate keys in data dicts not supported!"
                        )
                    keys.add(key)
            del keys

            bs = check_all_same_length(
                *flatten([list(d.values()) for d in batch])
            )

            if train_only and not train:
                yield batch

            flatbatch: DataDict = {
                k: arr for d in batch for k, arr in d.items()
            }
            key_to_dix: Dict[str, int] = {
                k: ix for ix, d in enumerate(batch) for k in d.keys()
            }

            # by default, apply transformations to all inputs and no labels
            if not in_keys:
                in_keys = list(flatbatch.keys())

            if out_keys and mode == "each" and inplace and not pop_in_keys:
                warn(
                    "Using explicit output keys with an inplace function and"
                    "without popping output keys - data will be duplicated!"
                )

            # if out_keys is not passed, assume they're the same as the in keys
            out_keys = out_keys or in_keys

            # canonicalize to explicit output dict index
            # novel keys without an explicit dict index will go into output 0
            clean_out_keys = []
            for ok in out_keys:
                if isinstance(ok, tuple):
                    clean_out_keys.append(ok)
                elif ok in key_to_dix:
                    clean_out_keys.append((key_to_dix[ok], ok))
                else:
                    if max(key_to_dix.values()) > 1:
                        warn(
                            f"""
                            Output key {ok} is neither an input key nor has an
                            output index. It will be put in the first output
                            dict."""
                        )
                    clean_out_keys.append((0, ok))

            out_keys = clean_out_keys

            if mode == "each" and len(out_keys) != len(in_keys):
                raise ValueError(
                    'In mode "each", the number of out_keys '
                    "must equal the number of in_keys"
                )

            if batchwise_apply:
                if mode == "each":
                    if inplace:
                        for key in in_keys:
                            f(flatbatch[key])
                        out = [flatbatch[key] for key in in_keys]
                    else:
                        out = [f(flatbatch[key]) for key in in_keys]
                elif mode == "mix":
                    out = f(*[flatbatch[key] for key in in_keys])
                else:
                    die("unreachable")
            else:
                if mode == "each" and inplace:
                    for key in in_keys:
                        varr = flatbatch[key]
                        for bx in range(bs):
                            f(varr[bx])
                    out = [flatbatch[key] for key in in_keys]
                else:
                    if mode == "each":
                        first_elems = [f(flatbatch[key][0]) for key in in_keys]
                    elif mode == "mix":
                        first_elems = f(*[flatbatch[key][0] for key in in_keys])
                        if len(first_elems) != len(out_keys):
                            raise ValueError(
                                f"Transformer function {f} returned "
                                f"{len(first_elems)} values, but "
                                f"{len(out_keys)} was expected based on the "
                                f"output keys {out_keys}."
                            )
                    else:
                        die("unreachable")

                    out = [
                        np.zeros((bs,) + elem.shape, elem.dtype)
                        for elem in first_elems
                    ]
                    for outarr, elem in zip(out, first_elems):
                        outarr[0, ...] = elem

                    if mode == "each":
                        for kx, key in enumerate(in_keys):
                            varr = flatbatch[key]
                            for bx in range(1, bs):
                                out[kx][bx, ...] = f(varr[bx, ...])
                    # this might be slow
                    elif mode == "mix":
                        for bx in range(1, bs):
                            res = f(*[flatbatch[key][bx] for key in in_keys])
                            for outarr, r in zip(out, res):
                                outarr[bx, ...] = r
                    else:
                        die("unreachable")

            in_keys_with_ixes = set((ix, k) for k, ix in key_to_dix.items())

            for (oix, okey), res in zip(out_keys, out):
                flatbatch[okey] = res
                key_to_dix[okey] = oix

            if pop_in_keys:
                would_del_output = set(out_keys) & in_keys_with_ixes
                to_pop = set(in_keys) - {x[1] for x in would_del_output}
                for ikey in to_pop:
                    del flatbatch[ikey]
                    del key_to_dix[ikey]

            yield split_flatbatch(flatbatch, key_to_dix)

    return bt


def get_k_of_each(y, k):
    """
    Returns the indices of a subarray containing  min(  # members, k)
    members of each class of y.

    Args:
        y: binary-encoded labels

    Returns:
        ixes: indices of the selected elements

    Examples:
        >> > y_sub=y[get_k_of_each(10, y)]
    """
    if len(y.shape) != 2:
        raise ValueError("This function expects a 2D array.")

    ixes = []
    ymax = np.argmax(y, axis=1)

    for i in range(y.shape[1]):
        ixes_i = np.where(ymax == i)[0]
        ixes.append(npr.choice(ixes_i, min(len(ixes_i), k), replace=False))

    return np.concatenate(ixes)


def complement_ixes(ixes, y):
    """
    Generates the complement of an array of indices of a given array.

    Arguments:
        ixes: the list of indices to complement
        y: the array, or length of array, with respect to which to complement
            the indices
    """
    try:
        y = len(y)
    except Exception:
        pass

    all_ixes = np.ones(y, dtype=np.uint8)
    all_ixes[ixes] = 0

    return np.where(all_ixes == 1)[0]
