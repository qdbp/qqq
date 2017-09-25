from keras.engine.topology import Layer
from keras.layers import GRU
from keras import backend as K

import numpy as np


# LAYERS

# http://arxiv.org/pdf/1409.7495.pdf
class NegGrad(Layer):

    def __init__(self, *, lbd, **kwargs):
        super().__init__(**kwargs)
        self._lbd = lbd

    def build(self, input_shape):
        self.lbd = self.add_weight(
                'lbd', (1,), trainable=False,
                initializer=lambda x: self._lbd,
            )

    def call(self, x, mask=None):
        return (1 + self.lbd) * K.stop_gradient(x) - self.lbd * x

    def get_output_shape_for(self, input_shape):
        return input_shape

    def set_lbd(self, lbd):
        self.lbd.set_value(lbd)

    def get_config(self):
        config = {
            'lbd': float(self.lbd.get_value()),
        }
        config.update(super().get_config())
        return config


# WRAPPERS

class RNNUnroll:
    '''
    Applies an RNN to a fixed input vector and its own output to produce
    an output sequence.
    '''

    def __init__(self, layer, output_length, *, consume_output=):
        self.layer = layer
        self.output_length = output_length


# ACTIVATIONS
def negabs(x):
    return -K.abs(x)
