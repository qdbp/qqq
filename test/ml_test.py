import numpy as np
import numpy.random as npr
from pytest import raises, warns


def test_generate_batches():
    from vnv.ml import generate_batches

    x1 = np.ones((5, 2, 2))
    x2 = np.ones((5, 5)) * 2
    y1 = np.ones((5, 3)) * 3
    z1 = np.ones((5, 3, 3, 3))

    x_too_many_samples = np.ones((6, 2, 2))

    # basic test
    gen = generate_batches({"x1": x1})
    next(gen)

    # shape tests
    gen = generate_batches({"x1": x1, "x2": x2}, {"y1": y1}, {'z': z1}, bs=3)

    x, y, z = next(gen)

    assert {*x.keys()} == {"x1", "x2"}
    assert {*y.keys()} == {"y1"}
    assert {*z.keys()} == {"z"}
    assert np.allclose(z['z'], np.ones((3, 3, 3, 3)))

    assert x["x1"].shape == (3, 2, 2)
    assert y["y1"].shape == (3, 3)

    with raises(ValueError):
        gen = generate_batches({"xb": x_too_many_samples}, {"y": y1})
        next(gen)

    x = {"x": np.arange(500)}
    y = {"y": np.arange(500)}

    npr.seed(1337)
    g = generate_batches(x, y, bs=100)
    xb, yb = next(g)
    assert np.allclose(xb["x"], yb["y"])
    assert np.any(xb["x"][:-1] > xb["x"][1:])

    # SEQUENTIAL
    g = generate_batches(x, y, sequential=True, bs=200)
    xb, yb = next(g)

    # tandmeness
    assert np.allclose(xb["x"], yb["y"])
    # sequentiality
    assert np.allclose(xb["x"][100:200], x["x"][100:200])

    # WEIGHTS
    sw = np.ones(500)
    sw[::2] = 0

    g = generate_batches(x, sample_weights=sw)
    xb = next(g)

    assert np.allclose(xb["x"] % 2, 1.)

    # BOUND IXES
    x = np.arange(500)[:, None]
    g = generate_batches({"x": x}, bound_ixes=np.arange(0, 500, 3))
    xb = next(g)
    assert np.allclose(next(g)["x"] % 3, 0.)
    assert not np.allclose(xb["x"], next(g)["x"])


def test_batch_transformer():
    from vnv.ml import batch_transformer

    bs = 50

    def source_1():
        while True:
            yield {
                "x0": np.ones((bs, 1, 2)),
                "x1": 2 * np.ones((bs, 2, 3, 4)),
                "x2": 3 * np.ones((bs, 7)),
            }, {"y0": np.ones((bs,))}

    def f1(x):
        x[:] *= 2

    def f2(x):
        return x.mean(axis=-1)

    flow1 = batch_transformer(
        f1, inplace=True, batchwise_apply=True)(source_1())
    n, _ = next(flow1)

    assert np.allclose(n["x0"], 2.)
    assert np.allclose(n["x1"], 4.)

    flow2 = batch_transformer(f2, inplace=False,
                              in_keys=["x1", "x2"], out_keys=["x1", "xz"],
                              )(source_1())

    nx, ny = next(flow2)
    # untouched
    assert ny["y0"].shape == (bs,)
    assert nx["x0"].shape == (bs, 1, 2)
    # dimreduced by f2
    assert nx["x1"].shape == (bs, 2, 3)
    assert nx["xz"].shape == (bs,)
    assert "x2" not in nx

    flow25 = batch_transformer(f2, inplace=False, in_keys=["x1"])(source_1())

    nx, ny = next(flow25)
    assert ny["y0"].shape == (bs,)
    assert nx["x0"].shape == (bs, 1, 2)
    assert nx["x1"].shape == (bs, 2, 3)

    def source_2():
        while True:
            yield {"x0": np.ones((bs, 2, 1))},\
                {"y0": np.ones((bs, 1))},\
                {"z0": np.ones((bs, 4, 4, 4))}

    def f3(x1, x2):
        return x1.mean(axis=-1) + np.sum(x2), x2.mean(axis=-1)

    flow3 = batch_transformer(
        f3,
        inplace=False,
        mode="mix",
        in_keys=["x0", "z0"],
        out_keys=[(1, "m1"), (3, "m3")],
        pop_in_keys=True,
    )(source_2())

    e1, d1, e2, d3 = next(flow3)
    assert e1 == e2 == {}
    assert set(d3.keys()) == {'m3'}
    assert set(d1.keys()) == {'y0', 'm1'}

    # f3(source2[x0], source2[z0])
    # np.mean(np.ones(2, 1), axis=-1) + np.sum(np.ones((4, 4, 4)))
    assert np.allclose(d1['m1'], 65 * np.ones((bs, 2)))
    # np.mean(np.ones((4, 4, 4), axis = -1))
    assert np.allclose(d3['m3'], np.ones((bs, 4, 4,)))


def test_dict_arr_eq():
    from vnv.ml import dict_arr_eq

    x1 = {"a": np.ones(5), "b": np.ones(5)}
    x2 = {"a": np.ones(5), "b": np.ones(5)}
    x3 = {"b": np.ones(5)}
    x4 = {"a": np.ones(6), "b": np.ones(5)}

    assert dict_arr_eq(x1, x2)
    assert not dict_arr_eq(x1, x3)
    assert not dict_arr_eq(x1, x4)


def test_dict_tt_split():
    from vnv.ml import dict_tt_split

    bs = 50

    x = {"xa": np.ones((bs, 5, 5)), "xb": np.arange(bs)}
    y = {"ya": np.arange(bs)}

    (xt, yt), (xv, yv) = dict_tt_split(x, y, test_size=0.5)

    assert np.allclose(xt["xb"], yt["ya"])
    assert len(xt["xb"]) == 25

    y_bad = {"ya": np.arange(bs + 1) * 3}

    with raises(ValueError):
        dict_tt_split(x, y_bad)


def test_dict_arr_concat_basics():

    from vnv.ml import dict_arr_concat, dict_arr_eq

    x1 = {"a": np.ones(1), "b": np.ones(1)}
    x2 = {"a": np.ones(2), "b": np.ones(2)}
    x9 = {"a": np.ones(2), "b": np.ones(9)}

    xz = {"c": np.arange(3), "b": np.ones(5)}

    assert dict_arr_eq(
        dict_arr_concat(x1, x2, check_keys=True, check_lens=True),
        {"a": np.ones(3), "b": np.ones(3)},
    )

    assert dict_arr_eq(
        dict_arr_concat(x1, x9, check_keys=True, check_lens=False),
        {"a": np.ones(3), "b": np.ones(10)},
    )

    with raises(ValueError):
        dict_arr_concat(x1, x9, check_keys=True, check_lens=True)

    assert dict_arr_eq(
        dict_arr_concat(x1, xz, check_keys=False, check_lens=False),
        {"a": np.ones(1), "b": np.ones(6), "c": np.arange(3)},
    )
    with raises(ValueError):
        dict_arr_concat(x1, xz, check_keys=False, check_lens=True)
    with raises(ValueError):
        dict_arr_concat(x1, xz, check_keys=True, check_lens=False)


def test_dict_arr_concat_axes():

    from vnv.ml import dict_arr_concat, dict_arr_eq

    x1 = {"a": np.ones((1, 2)), "b": np.ones((1, 2))}
    x2 = {"a": np.ones((2, 2)), "b": np.ones((2, 2))}

    assert dict_arr_eq(
        dict_arr_concat(x1, x2, axis=0), {
            "a": np.ones((3, 2)), "b": np.ones((3, 2))}
    )

    with raises(ValueError):
        dict_arr_concat(x1, x2, axis=1)

    y1 = {"a": np.ones((1, 1, 5)), "b": np.ones((1, 1, 5))}
    y2 = {"a": np.ones((1, 4, 5)), "b": np.ones((1, 4, 5))}
    y3 = {"a": np.ones((1, 5, 5)), "b": np.ones((1, 5, 5))}

    assert dict_arr_eq(
        dict_arr_concat(y1, y2, y3, axis=1),
        {"a": np.ones((1, 10, 5)), "b": np.ones((1, 10, 5))},
    )

    with raises(ValueError):
        assert dict_arr_concat(y1, y2, axis=2)
