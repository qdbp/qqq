import argparse as arg
import cmd
import os
import os.path as osp
import sys
from datetime import datetime as dtt
from datetime import timedelta
from glob import glob
from threading import Thread
from typing import Optional

import numpy as np
import numpy.random as npr

from keras import backend as K
from keras.callbacks import Callback
from keras.layers.noise import GaussianNoise
from qqq.qlog import get_logger

log = get_logger(__file__)

HDF5_EXT = 'hdf5'
JSON_EXT = 'json'
PNG_EXT = 'png'


def stack_layers(inp, *layers):
    for l in layers:
        inp = l(inp)

    return inp


def shuffle_weights(weights):
    return [npr.permutation(w.flat).reshape(w.shape) for w in weights]


class ModelHandler:
    '''
    Adjutant object to a keras model for weight handling and loading.
    '''

    @classmethod
    def attach(cls, model, name, **kwargs):
        '''
        Attaches handler to the model `model`.
        '''
        if not hasattr(model, 'handler'):
            model.handler = cls(model, name, **kwargs)
        return model, model.handler

    def __init__(self, model, name, min_lr_factor=1e-3, max_sigma=1,
                 weights_dir='./weights/'):
        '''
        Args:
            name: name to use for the model and associated files
            min_lr_factor: learning rate will never be adjusted lower than
                the current learning rate times this factor
            weights_dir: directory to store weights files
        '''
        self.model = model
        self.init_weights = model.get_weights()
        self.init_lr = self.get_lr()
        self.name = name
        self.root = os.environ.get('QQQ_WEIGHTSIDR', weights_dir)

        self.bag = None
        self.fold = None

        self.max_lr = self.get_lr()
        # XXX: hacky
        self.min_lr = self.max_lr * min_lr_factor

        self.max_sigma = max_sigma
        self.noise_layers = [l for l in model.layers
                             if isinstance(l, GaussianNoise)]

        self.init_sigma = [n.sigma for n in self.noise_layers]

        self._w_re = r'weights_{}_([0-9\.]+)\.hdf5'

    def reinit_model(self):
        self.model.set_weights(shuffle_weights(self.init_weights))
        self.set_lr(self.init_lr)
        for ix, s in enumerate(self.init_sigma):
            self.set_noise(s, ix)
        log.info('reinitialized model')

    # TODO
    def iter_weights(self, *, do_folds=True):
        '''
        Returnes an iterator over the model's component weight files.

        Used to load a set of related (bagged and/or folded) weights
        in sequence for aggregate predictions.
        '''
        raise NotImplementedError

    @property
    def weights_dir(self) -> str:
        path = [self.root, self.name,
                'no_bag/' if self.bag is None else f'bag_{self.bag}',
                'no_fold/' if self.fold is None else f'fold_{self.fold}']

        out_path = osp.join(*path)
        os.makedirs(out_path, exist_ok=True)

        return out_path

    @property
    def _w_temp(self):
        return osp.join(self.weights_dir + 'weights_{}_{:.4f}.hdf5')

    @property
    def _w_glob(self):
        return osp.join(self.weights_dir + 'weights_{}_*.hdf5')

    def set_bag(self, bag: Optional[int]) -> None:
        self.bag = bag

    def set_fold(self, fold: Optional[int]) -> None:
        self.fold = fold

    def set_name(self, name: str) -> None:
        self.name = name

    def save_weights(self, typ: str, param: float) -> None:
        old_fns = sorted(glob(self._w_glob.format(typ)))
        for old_fn in old_fns:
            os.remove(old_fn)
        new_fn = self._w_temp.format(typ, param)
        self.model.save_weights(new_fn)
        log.debug(f'saved new weights to {new_fn}')

    def load_weights(self, typ):
        fns = sorted(glob(self._w_glob.format(typ)))
        if fns:
            fn = fns[0]
            log.info(f'loaded weights from {fn}')  # type: ignore
            self.model.load_weights(fn)

    def get_lr(self):
        return K.get_value(self.model.optimizer.lr)

    def set_lr(self, lr):
        lr = np.asarray(lr).astype(np.float32)
        K.set_value(self.model.optimizer.lr, lr)
        log.debug(f'set learning rate to {lr!s}')

    def adj_lr(self, dlr):
        '''
        Adjusts the learning given the training and validation losses.

        A learning rate controller (`lr_ctl`) must be set beforehand using
        `attach_lr_ctl`.

        Args:
            dlr: learning rate adjustment, in units of current learning rate
        '''

        if dlr is not None:
            lr = self.get_lr()
            true_dlr = dlr * lr
            self.set_lr(np.maximum(lr + true_dlr, self.min_lr))

    def adj_noise(self, dsigma, ix):
        s = self.get_noise(ix)
        new_sigma = np.minimum(np.maximum(s + s * dsigma, 0), self.max_sigma)
        self.set_noise(new_sigma, ix)

    def set_noise(self, new_sigma, ix):
        self.noise_layers[ix].sigma = new_sigma

    def get_noise(self, ix):
        return self.noise_layers[ix].sigma

    def __getattr__(self, name):
        return getattr(self.model, name)


class TrainerUI:

    class TrainCMD(cmd.Cmd):
        intro = '~~~ QQQ Trainer UI ~~~'
        prompt = ':: '

        def do_train(self):
            pass

        def set_lr(self):
            pass

    def __init__(self):
        self.model = None
        self.h = None

        self._setup_parser()
        self._run()

    def _setup_parser(self):
        self._parser = arg.ArgumentParser('TrainerUI')

    def add_model(self, model, name):
        self.model, self.h = ModelHandler.attach(model, name)


class HandlerCallback(Callback):

    def set_model(self, model):
        if not hasattr(model, 'handler'):
            raise ValueError(
                'This callback requires a model with a ModelHandler'
            )
        super().set_model(model)


class WeightSaver(HandlerCallback):

    def __init__(self, *args, min_improvement=0.95, load=True, **kwargs):
        self.min_improvement = 0.95
        self.load = load
        super().__init__(*args, **kwargs)

    def on_train_begin(self, logs=None):
        self.best_loss = np.inf
        if self.load:
            self.model.handler.load_weights('best')

    def on_epoch_end(self, epoch, logs):
        vl = logs['val_loss']
        if vl < self.best_loss * self.min_improvement:
            self.model.handler.save_weights('best', vl)


class NoiseControl(HandlerCallback):
    '''
    Adjusts the noise in a layer.

    Determines the change based on the difference between the training
    and validation loss.
    '''

    def __init__(self, index, k=0.2):
        self.index = index
        self.k = k

    def on_epoch_end(self, epoch, logs):
        tl, vl = logs['loss'], logs['val_loss']
        sigma = self.k * 2 * (vl - tl) / (vl + tl)
        self.model.handler.adj_noise(sigma, self.index)

        new_sigma = self.model.handler.get_noise(self.index)
        s = f'noise level on index {self.index} now at {new_sigma:.2f}'
        if (epoch % 9):
            log.verbose(s)
        else:
            log.info(s)


class ProgLogger(HandlerCallback):

    def on_train_begin(self, logs=None):
        h = self.model.handler
        name, bag, fold = h.name, h.bag, h.fold
        log.info(
            f'beginning training for model "{name}", bag {bag}, fold {fold}')

        self._ulr = float(self.model.handler.get_lr()) * 1e6
        self._tst = dtt.now()

    def on_epoch_begin(self, epoch, logs=None):
        self._est = dtt.now()
        # check for lr changes
        _cur_ulr = float(self.model.handler.get_lr()) * 1e6
        if not np.isclose(self._ulr, _cur_ulr):
            log.info(f'learning rate changed from {self._ulr:3.0} x 10^-6 to '
                     f'{_cur_ulr:3.0} x 10^-6')
        self._ulr = _cur_ulr

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        tl, vl = logs['loss'], logs.get('val_loss', '-----')
        now = dtt.now()
        tim = timedelta(seconds=int((now - self._est).total_seconds()))
        ttim = timedelta(seconds=int((now - self._tst).total_seconds()))

        s = (f'E{epoch:04d} finished in {tim} ({ttim} total)'
             f' | TL {tl:5.3} VL {vl:5.3}')

        if epoch % 5:
            log.verbose(s)
        else:
            log.info(s)

    def on_train_end(self, epoch, logs=None):
        log.info(f'Training finished in {dtt.now() - self._tst}')
