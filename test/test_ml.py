from pytest import raises
import numpy as np
import numpy.random as npr


def test_generate_batches():
    from qqq.ml import generate_batches

    x1 = np.ones((5, 2, 2))
    x2 = np.ones((5, 5)) * 2
    y1 = np.ones((5, 3)) * 3

    x_bad = np.ones((6, 2, 2))

    # basic test
    gen = generate_batches({'x1': x1})
    next(gen)

    # shape tests
    gen = generate_batches({'x1': x1, 'x2': x2}, y={'y1': y1}, bs=3)

    x, y = next(gen)

    assert {*x.keys()} == {'x1', 'x2'}
    assert {*y.keys()} == {'y1'}

    assert x['x1'].shape == (3, 2, 2)
    assert y['y1'].shape == (3, 3)

    with raises(ValueError):
        gen = generate_batches({'xb': x_bad}, y={'y': y1})
        next(gen)

    x = {'x': np.arange(500)}
    y = {'y': np.arange(500)}

    npr.seed(1337)
    g = generate_batches(x, y, bs=100)
    xb, yb = next(g)
    assert np.allclose(xb['x'], yb['y'])
    assert np.any(xb['x'][:-1] > xb['x'][1:])

    g = generate_batches(x, y, sequential=True, bs=200)
    xb, yb = next(g)

    assert np.allclose(xb['x'], yb['y'])
    assert np.allclose(xb['x'][100:200], x['x'][100:200])

    sw = np.ones(500)
    sw[::2] = 0

    g = generate_batches(x, sample_weights=sw)
    xb = next(g)

    assert np.allclose(xb['x'] % 2, 1.)


def test_batch_transformer():
    from qqq.ml import batch_transformer

    bs = 50

    def source_1():
        while True:
            yield {
                'x0': np.ones((bs, 10, 20)),
                'x1': 2 * np.ones((bs, 2, 3, 4)),
                'x2': 3 * np.ones((bs, 70))
            }, {
                'y0': np.ones((bs,))
            }

    def source_2():
        while True:
            yield {
                'x0': np.ones((50, 10, 20)),
            }, {
                'y0': np.ones((bs, 10))
            }

    def f1(x):
        x[:] *= 2

    def f2(x):
        return x.mean(axis=-1)

    def f2_shape(shape):
        return shape[:-1]

    def f3(x1, x2):
        return x1.mean(axis=-1) + np.sum(x2), x2.mean(axis=-1)

    def f3_shape(shape1, shape2):
        return shape1[:-1], shape2[:-1]

    flow1 = batch_transformer(f1, inplace=True)(source_1())
    n, _ = next(flow1)

    assert np.allclose(n['x0'], 2.)
    assert np.allclose(n['x1'], 4.)

    flow2 = batch_transformer(
        f2, inplace=False,
        get_shape=f2_shape, in_keys=['x1', 'x2'], out_keys=['x1', 'xz'],
    )(source_1())

    nx, ny = next(flow2)
    assert ny['y0'].shape == (bs,)
    assert nx['x0'].shape == (bs, 10, 20)
    assert nx['x1'].shape == (bs, 2, 3)
    assert nx['xz'].shape == (bs,)
    assert 'x2' not in nx

    flow25 = batch_transformer(
        f2, inplace=False, get_shape=f2_shape, in_keys=['x1']
    )(source_1())

    nx, ny = next(flow25)
    assert ny['y0'].shape == (bs,)
    assert nx['x0'].shape == (bs, 10, 20)
    assert nx['x1'].shape == (bs, 2, 3)

    flow3 = batch_transformer(
        f3, inplace=False, mode='mix', get_shape=f3_shape,
        in_keys=['x0', 'y0'], out_keys=['yb', 'x1'], pop_in_keys=True,
        out_in_ys=[True, False])(source_2())

    nx, ny = next(flow3)
    assert 'x0' not in nx
    assert 'x1' in nx
    assert 'y0' not in ny
    assert 'yb' in ny


def test_dict_tv_split():
    from qqq.ml import dict_tv_split

    bs = 50

    x = {'xa': np.ones((bs, 5, 5)),
         'xb': np.arange(bs)}

    y = {'ya': np.arange(bs)}
    y_bad = {'ya': np.arange(bs + 1) * 3}

    (xt, yt), (xv, yv) = dict_tv_split(x, y, f_train=0.5)

    assert np.allclose(xt['xb'], yt['ya'])
    assert len(xt['xb']) == 25

    with raises(ValueError):
        dict_tv_split(x, y_bad)
