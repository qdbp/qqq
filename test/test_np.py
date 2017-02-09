import pytest

from numpy import array
import numpy as np


def test_slarr():
    from qqq.np import sl_window, unsl_window

    x1 = np.linspace(0, 5, num=6)
    x2 = np.linspace(0, 50, num=6)
    arr = x1[:, np.newaxis] + x2

    print(arr)

    t1 = sl_window(arr, 2, 1)
    print(t1.shape)
    print(repr(t1))
    assert np.allclose(
        t1,
        array([[[0.,  10.,  20.,  30.,  40.,  50.],
                [1.,  11.,  21.,  31.,  41.,  51.]],

               [[1.,  11.,  21.,  31.,  41.,  51.],
                [2.,  12.,  22.,  32.,  42.,  52.]],

               [[2.,  12.,  22.,  32.,  42.,  52.],
                [3.,  13.,  23.,  33.,  43.,  53.]],

               [[3.,  13.,  23.,  33.,  43.,  53.],
                [4.,  14.,  24.,  34.,  44.,  54.]],

               [[4.,  14.,  24.,  34.,  44.,  54.],
                [5.,  15.,  25.,  35.,  45.,  55.]]])
    )

    t2 = sl_window(arr, 2, 2, axis=1)
    print(arr)
    print(repr(t2))
    assert np.allclose(
        t2,
        array([[[0.,  10.],
                [1.,  11.],
                [2.,  12.],
                [3.,  13.],
                [4.,  14.],
                [5.,  15.]],

               [[20.,  30.],
                [21.,  31.],
                [22.,  32.],
                [23.,  33.],
                [24.,  34.],
                [25.,  35.]],

               [[40.,  50.],
                [41.,  51.],
                [42.,  52.],
                [43.,  53.],
                [44.,  54.],
                [45.,  55.]]])
    )

    t3 = sl_window(arr, 4, 2, axis=1, sl_axis=2)
    print(arr)
    print(repr(t3))
    assert np.allclose(
        t3,
        array([[[0.,  20.],
                [10.,  30.],
                [20.,  40.],
                [30.,  50.]],

               [[1.,  21.],
                [11.,  31.],
                [21.,  41.],
                [31.,  51.]],

               [[2.,  22.],
                [12.,  32.],
                [22.,  42.],
                [32.,  52.]],

               [[3.,  23.],
                [13.,  33.],
                [23.,  43.],
                [33.,  53.]],

               [[4.,  24.],
                [14.,  34.],
                [24.,  44.],
                [34.,  54.]],

               [[5.,  25.],
                [15.,  35.],
                [25.,  45.],
                [35.,  55.]]])
    )

    for t in [t1, t2, t3]:
        assert np.allclose(unsl_window(t), arr)


def test_tandems():
    from qqq.np import tandem_shuffle, tandem_resample

    a = list(range(10))
    b = tuple(range(10))

    sa, sb = tandem_shuffle(a, b)
    qa, qb = tandem_resample(a, b)

    assert isinstance(sb, np.ndarray)
    assert np.allclose(sa, sb)
    assert np.allclose(qa, qb)

    c = list(range(9))

    with pytest.raises(ValueError):
        sa, sc = tandem_shuffle(a, c)