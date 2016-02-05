import numpy.random as npr
from numpy.lib.stride_tricks import as_strided


def sl_window(a, w, s, axis=0):
    """
    Generates staggered windows of an array.

    Given an array a of dimension N, stride size s, and window size w,
    returns an array of dimension N + 1 of w-sized windows, each offset
    by s from the previous. The "sliding" happens along the given axis.

    Args:
        a: array over which to generate windows
        w: window size
        s: stride size
        axis: axis of a along which to slide the window

    Return:
        out: array of windows; shape nwindows on zeroth axis,
             w on axis corresponding to 'axis' argument, other
             dimensions unchanged
    """

    nw = 1 + (a.shape[axis] - w)//s

    nas = list(a.shape)
    nas[axis] = w
    ns = (nw,) + tuple(nas)

    ss = (s*a.strides[axis],) + a.strides
    out = as_strided(a, ns, ss)

    return out


def ttsplit(arrs, f=0.2, align=0, shf_train=True, shf_test=False):
    """ Does a test-train split for multiple arrays.

        Given any number of arrays of the same length,
        splits off from each a fraction given by f,
        and returns the split parts the train and the test.

        Args:
            arrs: the arrays to split
            f: the fraction to split off
            shf_train: whether to shuffle the train arrays
            shf_test: whether to shuffle the test arrays

        Return:
            L: the train arrays, length (1-f)*original
            V: the test arrays, length f*original
        """

    assert len(set([len(arr) for arr in arrs])) == 1

    l = len(arrs[0])
    s = int(l*(1-f))

    L, V = [], []

    rs = npr.get_state()
    for arr in arrs:
        if shf_train and shf_test:
            npr.shuffle(arr)
            npr.set_state(rs)
        L.append(arr[:s])
        V.append(arr[s:])

    if shf_train and not shf_test:
        shuffle_arrs(L, rs)
    if shf_test and not shf_train:
        shuffle_arrs(V, rs)

    return L, V


def shuffle_arrs(arrs, rs=None):
    """ Shuffles a collection of arrays in tandem.

        Given a sequence of arrays, shuffles
        each to the same permutation, so that
        elements at a given index in each array
        stay aligned after shuffling.

        Args:
            arrs: collection of arrays to shuffle
            rs: random state to shuffle by """

    for arr in arrs:
        npr.shuffle(arr)
        if rs is not None:
            npr.set_state(rs)
