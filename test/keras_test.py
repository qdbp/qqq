from os import remove, environ
from glob import glob

import keras.optimizers as ko
import keras.activations as ka
import keras.callbacks as kc
import keras.layers as kl
import keras.models as km
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pytest

from sklearn.metrics import accuracy_score
from vnv.keras import data_fnist


(Xt, yt), (Xv, yv) = data_fnist()

GIT_PREPUSH = environ.get('VNV_GIT_PREPUSH', False)


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
    compile_model([i0, i1], [y0, y1], loss='mse')
    compile_model([i0, i1], [y0, y1], loss=['mse', 'cxe'])
    compile_model(i0, [y0, y01], loss='cxe')
    compile_model([i0, i1], ym, loss=['cxe'])

    # TODO aux losses, metrics


@pytest.mark.skipif(GIT_PREPUSH, reason='git prepush')
def test_SelfGated():

    from vnv.keras import SelfGated, compile_model

    i = km.Input(Xt.shape[1:])
    h = SelfGated(kl.Conv2D(64, (3, 3)))(i)
    f = kl.Flatten()(h)
    d = kl.Dense(128, activation='relu')(f)
    y = kl.Dense(yt.shape[-1], activation='softmax')(d)

    m = compile_model(i, y, loss='cxe', metrics='acc')
    m.fit(Xt, yt, epochs=1)

    yp = np.argmax(m.predict(Xv), axis=-1)
    acc = accuracy_score(np.argmax(yv, axis=-1), yp)
    assert acc > 0.8


@pytest.mark.skipif(GIT_PREPUSH, reason='git prepush')
def test_Residual():

    from vnv.keras import Residual, compile_model

    c0 = kl.Conv2D(64, (3, 3), activation='relu', padding='same')
    c1 = kl.Conv2D(64, (3, 3), activation='linear', padding='same')

    i = km.Input(Xt.shape[1:])
    c = kl.Conv2D(64, (3, 3), activation='relu', padding='same')(i)
    h = Residual(c0, c1, activation='relu')(c)
    f = kl.Flatten()(h)
    d = kl.Dense(128, activation='relu')(f)
    y = kl.Dense(yt.shape[-1], activation='softmax')(d)

    m = compile_model(i, y, loss='cxe', metrics='acc')
    m.fit(Xt, yt, epochs=1)

    yp = np.argmax(m.predict(Xv), axis=-1)
    acc = accuracy_score(np.argmax(yv, axis=-1), yp)
    assert acc > 0.8


if __name__ == '__main__':
    test_Residual()
