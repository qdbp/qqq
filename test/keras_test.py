from glob import glob
from os import environ, remove

import keras.activations as ka
import keras.callbacks as kc
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pytest
from filelock import FileLock
from sklearn.metrics import accuracy_score

from vnv.keras import apply_layers, compile_model, data_fnist

TFLOCK = FileLock("/tmp/vnv_test_tflock")

(Xt, yt), (Xv, yv) = data_fnist()

GIT_PREPUSH = environ.get("VNV_GIT_PREPUSH", False)


def test_compile_model():
    from vnv.keras import compile_model

    i0 = km.Input(shape=(16,))
    i1 = km.Input(shape=(16,))

    h0 = kl.Dense(16)(i0)
    h1 = kl.Dense(16)(i1)

    y0 = kl.Dense(2)(h0)
    y01 = kl.Dense(2)(h0)
    ym = kl.Dense(2)(kl.Concatenate()([h0, h1]))
    y1 = kl.Dense(2)(h1)

    compile_model(i0, y0)
    compile_model([i0, i1], [y0, y1], loss="mse")
    compile_model([i0, i1], [y0, y1], loss=["mse", "cxe"])
    compile_model(i0, [y0, y01], loss="cxe")
    compile_model([i0, i1], ym, loss=["cxe"])

    # TODO aux losses, metrics


# @pytest.mark.skipif(GIT_PREPUSH, reason="git prepush")
@pytest.mark.skip
def test_LayerFactory():

    from vnv.keras import LayerFactory, Stacked

    cfac = LayerFactory("base_conv", kl.Conv2D, 64, (3, 3), activation="relu")
    fpac = LayerFactory("base_pool", kl.MaxPooling2D, (2, 2))

    i = km.Input(Xt.shape[1:])
    y = Stacked(
        (cfac + fpac) * 2,
        kl.Flatten(),
        kl.Dense(128, activation="relu"),
        kl.Dense(yv.shape[-1], activation="softmax"),
    )(i)

    with TFLOCK:
        m = compile_model(i, y, loss="cxe", metrics="acc")
        m.summary()
        m.fit(Xt, yt, epochs=1)
        yp = np.argmax(m.predict(Xv), axis=-1)

    acc = accuracy_score(np.argmax(yv, axis=-1), yp)
    assert acc > 0.8


# @pytest.mark.skipif(GIT_PREPUSH, reason="git prepush")
@pytest.mark.skip
def test_Gated():

    from vnv.keras import Gated, LayerFactory as LF, Stacked

    f_s1 = LF("base_conv", kl.Conv2D, 128, (3, 3), strides=1)
    f_s2 = LF("base_conv", kl.Conv2D, 128, (3, 3), strides=2)
    # f_pool = LF('base_pool', kl.MaxPooling2D, (2, 2))

    t_gate = Gated(f_s1) + Gated(f_s2)
    # t_gwp = Stacked(t_gate, f_pool)
    # stk = Stacked(t_gwp, t_gwp, t_gwp)

    i = km.Input(Xt.shape[1:])

    y = Stacked(
        f_s1,
        kl.Activation("relu"),
        t_gate * 2,
        kl.Flatten(),
        kl.Dense(128, activation="relu"),
        kl.Dropout(0.5),
        kl.Dense(128, activation="relu"),
        kl.Dropout(0.5),
        kl.Dense(128, activation="relu"),
        kl.Dropout(0.5),
        kl.Dense(yt.shape[-1], activation="softmax"),
    )(i)

    with TFLOCK:
        m = compile_model(i, y, loss="cxe", metrics="acc")
        m.summary()
        m.fit(Xt, yt, epochs=5, validation_data=(Xv, yv))
        yp = np.argmax(m.predict(Xv), axis=-1)

    acc = accuracy_score(np.argmax(yv, axis=-1), yp)
    assert acc > 0.7


# @pytest.mark.skipif(GIT_PREPUSH, reason="git prepush")
@pytest.mark.skip
def test_Residual():

    from vnv.keras import LayerFactory as LF, Residual, Stacked, Gated

    f_conv = LF("base_conv", kl.Conv2D, 128, (3, 3), padding="same")
    f_pool = LF("res_pool", kl.MaxPooling2D, (2, 2))

    t_res = Residual(Gated(f_conv) * 2, activation="relu") + f_pool
    t_stk = t_res * 3

    i = km.Input(Xt.shape[1:])
    y = Stacked(
        t_stk,
        kl.Flatten(),
        kl.Dense(128, activation="relu"),
        kl.Dropout(0.5),
        kl.Dense(128, activation="relu"),
        kl.Dense(yt.shape[-1], activation="softmax"),
    )(i)

    with TFLOCK:
        m = compile_model(i, y, loss="cxe", metrics="acc")
        m.summary()
        m.fit(Xt, yt, epochs=5)
        yp = np.argmax(m.predict(Xv), axis=-1)

    acc = accuracy_score(np.argmax(yv, axis=-1), yp)
    assert acc > 0.8


if __name__ == "__main__":
    # test_LayerFactory()
    test_Gated()
    # test_Residual()
