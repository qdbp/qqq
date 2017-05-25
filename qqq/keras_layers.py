from keras.engine.topology import Layer
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K

import numpy as np
import theano


# http://arxiv.org/pdf/1409.7495.pdf
# https://github.com/fchollet/keras/issues/3119#issuecomment-230289301
class NegGradOp(theano.Op):
    '''
    Reverses gradient flow.
    '''

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super().__init__()
        self.hp_lambda = hp_lambda
        self.lbd = theano.shared(np.asarray(hp_lambda,
                                 dtype=theano.config.floatX))

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.lbd * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes


class NegGrad(Layer):

    def __init__(self, hp_lambda, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.gr_op = NegGradOp(self.hp_lambda)
        self.lbd = self.gr_op.lbd

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return self.gr_op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "lambda": self.hp_lambda}
        base_config = super().get_config()
        base_config.update(config)
        print(base_config)

        return base_config

    def set_lbd(self, lbd):
        self.lbd.set_value(lbd)


# ACTIVATIONS
def negabs(x):
    return -K.abs(x)
