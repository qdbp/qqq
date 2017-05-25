from pytest import raises
import numpy as np
import numpy.random as npr

from qqq.ml import batch_transformer, generate_batches


def test_generate_batches():

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

    bt1 = batch_transformer(source_1(), f1, inplace=True)
    n, _ = next(bt1)

    assert np.allclose(n['x0'], 2.)
    assert np.allclose(n['x1'], 2.)

    bt2 = batch_transformer(
        source_1(), f2, inplace=False,
        get_shape=f2_shape, in_keys=['x1', 'x2'], out_keys=['x1', 'xz'],
    )

    nx, ny = next(bt2)
    assert ny['y0'].shape == (bs,)
    assert nx['x0'].shape == (bs, 10, 20)
    assert nx['x1'].shape == (bs, 2, 3)
    assert nx['xz'].shape == (bs,)
    assert 'x2' not in nx

    bt3 = batch_transformer(
        source_2(), f3, inplace=False, mode='mix', get_shape=f3_shape,
        in_keys=['x0', 'y0'], out_keys=['yb', 'x1'], pop_in_keys=True,
        out_in_ys=[True, False])

    nx, ny = next(bt3)
    assert 'x0' not in nx
    assert 'x1' in nx
    assert 'y0' not in ny
    assert 'yb' in ny

