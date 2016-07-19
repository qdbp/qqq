import numpy.random as npr
from numpy.lib.stride_tricks import as_strided


def decay(k, x, x0):
    return k*x + (1-k)*x0



def sl_window(a, w, s, axis=0, sl_axis=0):
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
        sl_axis: axis along which windows lie

    Return:
        out: array of windows; shape nwindows on zeroth axis,
             w on axis corresponding to 'axis' argument, other
             dimensions unchanged
    """

    nw = 1 + (a.shape[axis] - w)//s

    nas = list(a.shape)
    nas[axis] = w
    ns = tuple(nas[:sl_axis] + [nw] + nas[sl_axis:])

    nss = list(a.strides)
    ss = tuple(nss[:sl_axis] + [s*a.strides[axis]] + nss[sl_axis:])
    out = as_strided(a, ns, ss)

    return out


def unsl_window(a):
    """
    Undoes the action of sl_window to the extent possible.
    """

    return a.base.base


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

if __name__ == "__main__":
    import numpy as np

    test_arr = np.asarray(range(200))
    test_arr = np.reshape(test_arr, (20, 10))

    print(test_arr.shape)
    print(test_arr)

    
    print('slide w{} s{} axis{}'.format(5, 3, 0))
    t1 = sl_window(test_arr, 5, 3, axis=0)
    print(t1.shape)
    print(t1)

    print('slide w{} s{} axis{}'.format(3, 3, 1))
    t2 = sl_window(test_arr, 3, 3, axis=1)
    print(t2.shape)
    print(t2)

    print('slide w{} s{} axis{} sl_axis{}'.format(5, 3, 0, 1))
    t3 = sl_window(test_arr, 5, 3, axis=0, sl_axis=1)
    print(t1.shape)
    print(t1)

    print('unsl t1')
    print(unsl_window(t1))
    print('unsl t2')
    print(unsl_window(t2))
    print('unsl t2')
    print(unsl_window(t3))
