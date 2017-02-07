from threading import Lock

import numpy as np
import numpy.random as npr
from numpy.lib.stride_tricks import as_strided


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.minimum(x, 0)


def decay(k, x, x0):
    return k * x + (1 - k) * x0


def sl_window(arr: np.ndarray, window: int, stride: int, axis=0, sl_axis=0):
    '''
    Generates staggered windows of an array.

    Given an array a of dimension N, stride size `stride`, and window size
    `window`, returns an array of dimension N + 1 of `window`-sized windows,
    each offset by `stride` from the previous. The 'sliding' happens along
    `axis` and the windows lie along `sl_axis` of the output array.

    Args:
        arr: array over which to generate windows
        window: window size
        stride: stride size
        axis: axis of `arr` along which to slide the window
        sl_axis: axis of output array along which windows will lie

    Returns:
        out: array of windows; shape nwindows on zeroth axis,
             w on axis corresponding to 'axis' argument, other
             dimensions unchanged
    '''

    num_windows = 1 + (arr.shape[axis] - window) // stride
    win_stride = stride * arr.strides[axis]

    new_shape = arr.shape[:axis] + (window,) + arr.shape[axis + 1:]
    new_shape = new_shape[:sl_axis] + (num_windows,) + new_shape[sl_axis:]

    new_strides = arr.strides[:sl_axis] + (win_stride,) + arr.strides[sl_axis:]

    return as_strided(arr, new_shape, new_strides)


def unsl_window(a):
    '''
    Undoes the action of sl_window to the extent possible.
    '''

    return a.base.base


def _tandem(mode, *args, rs=None):
    if not args:
        return ()

    if len({len(a) for a in args}) > 1:
        raise ValueError('arrays must be of the same length for tandem ops')

    args = [np.asarray(a) for a in args]
    l = len(args[0])

    with Lock():
        if rs:
            npr.set_state(rs)

        if mode == 'shuffle':
            ixes = npr.permutation(l)
        elif mode == 'resample':
            ixes = npr.choice(np.arange(l, dtype=np.int64),
                              size=l, replace=True)
        else:
            raise ValueError('invalid mode in _tandem')

    return tuple(arr[ixes] for arr in args)


def tandem_shuffle(*args, rs=None):
    ''' Shuffles a collection of arrays in tandem.

        Given a sequence of arrays, shuffles
        each to the same permutation, so that
        elements at a given index in each array
        stay aligned after shuffling.

        Args:
            *args: array-likes of the same length
            rs: random state to shuffle by. If None, uses the current.
    '''

    return _tandem('shuffle', *args, rs=rs)


def tandem_resample(*args, rs=None):
    ''' Chooses the same random subset from a collection of arrays.

        Given a sequence of arrays, resamples each with replacement.
        Equivalent to `zip(*resample(zip(*args)))`

        Args:
            *args: array-likes of the same length
            rs: random state to shuffle by. If None, uses the current.
    '''

    return _tandem('resample', *args, rs=rs)


if __name__ == '__main__':
    import numpy as np
