import glob
import hashlib as hsh
import json
import os
import os.path as osp
import sys
import time

import keras.backend as K
import keras.callbacks as kcb
import keras.models as krm
# import keras.utils.visualize_util as kuv
import numpy as np
import sklearn.metrics as skm

from qqq.np import decay as decay

HDF5_EXT = 'hdf5'
JSON_EXT = 'json'
PNG_EXT = 'png'


def mk_identifier(m):
    # sort_keys is very important
    reset = False
    try:
        lr = K.get_value(m.optimizer.lr)
        m.optimizer.lr.set_value(0)
        reset = True
    except AttributeError:
        pass
    j = m.to_json(sort_keys=True)
    h = hsh.sha512(j.encode('utf-8')).hexdigest()[:8].upper()
    if reset:
        K.set_value(m.optimizer.lr, lr)
    return j, h


class ModelHandler:

    def __init__(self, model_dir, model_root):
        self.MODEL_ROOT = osp.join(model_dir, model_root)

    def get_model_path(self, m):
        _, h = mk_identifier(m)
        p = osp.join(self.MODEL_ROOT, h)
        os.makedirs(p, exist_ok=True)
        return p

    def get_in_model_path(self, m, fn):
        return osp.join(self.get_model_path(m), fn)

    def mk_model_name(self, m, mn=None, get_jh=False,
                      ext=JSON_EXT):
        j, h = mk_identifier(m)
        fn = ('{}'.format(mn) if mn else '') + '.{}'.format(ext)
        fn = osp.join(self.get_model_path(m), fn)
        if get_jh:
            return fn, j, h
        else:
            return fn

    def save_model(self, m, mn=None):
        fn, j, _ = self.mk_model_name(m, mn=mn, get_jh=True)
        pn = self.mk_model_name(m, mn=mn, ext=PNG_EXT)
        with open(fn, 'w') as f:
            f.write(j)
        
        # kuv.plot(m, to_file=pn)

    def load_model(self, m, mn=None):
        fn = self.mk_model_name(m, mn=mn)
        m.from_json(fn)

    def save_weights(self, m, mn=None, **kwargs):
        fn = self.mk_model_name(m, mn=mn, ext=HDF5_EXT)
        m.save_weights(fn, **kwargs)

    def load_weights(self, m, mn=None, in_name='best'):
        # TODO: slash is intentional, find better way of setting mn
        match = glob.glob(self.get_model_path(m) + '/' + mn +
                          '*{}*'.format(in_name) + HDF5_EXT)
        if len(match) > 1:
            print('found more than one matching model:\n{}\nloading first'
                  .format('\n\t'.join(match)))
        m.load_weights(match[0])

    def load_model_json(self, hsh, mn, reset_lr=1e-3):
        fn = osp.join(self.MODEL_ROOT, hsh, mn+'.'+JSON_EXT)
        with open(fn, 'r') as f:
            m = krm.model_from_json(f.read())
        if reset_lr:
            K.set_value(m.optimizer.lr, reset_lr)
        return m

class PIDLearner:
    def __init__(self, base_lr, patience=1e4,
                 p=0.6, i=0.2, d=0.1, k=1/5, scale=1e-3):

        self.pat = patience
        self.b_lr = base_lr
        self.p = p
        self.i = i
        self.d = d
        self.k = k

        self._c_lr = 1
        self._std = 1
        self._mu = 0
        self._p_val = 0
        self._i_val = 0
        self._d_val = 0

    def step(self, v_loss):
        if self._mu is None:
            self._mu = v_loss

        self._std = decay(self.k, v_loss - self.mu, self._std)

        # time for magic fudge; should learn this
        self._i_val += v_loss - self._mu
        self._p_val = v_loss - self._mu

        self._mu = decay(self.k, v_loss, self._mu)
        # the "maybe, sort of derivative"(tm) from Naumov MathWorks(R)
        self._d_val = (v_loss - self._mu)/self._mu

        lr_adj = -(self._i_val * self.i +
                   self._p_val * self.p +
                   self._d_val * self.d)
        kill = self._i_val > self.patience
        return lr_adj, self._i_val, kill

class TrainingMonitor(kcb.Callback):
    """
    The systemd of keras callbacks. A monolithic and
    well-integrated monster which handles, among other
    things, data validation, learning rater adjustment,
    and saving weights.

    Planned features include and input data stream control.
    """
    def __init__(self, vgen, vsamples, mhd, key='y',
                 stagmax=2, mname='unnamed_model',
                 aeons=5, aeon_lr_factor=1/3,
                 aeon_stag_factor=2, init_lr=1e-3,
                 save_best_train=False):
        self.vgen = vgen
        self.vsamples = vsamples
        self.mhd = mhd
        self.key = key
        self.mname = mname

        self.aeons = aeons
        self.aeon_lr_factor = aeon_lr_factor
        self.aeon_stag_factor = aeon_stag_factor
        self.stagmax = stagmax
        self.stagcnt = 0

        self.init_lr = init_lr

        self.best_loss = np.inf
        self.best_tloss = np.inf
        self.current_best_weights = None
        self.save_best_train = save_best_train

        self.aeon = 0
        self.epoch = 0
        self.batch = 0

        self.cur_loss = None
        self.k = 0.1

        self.start_mark = 0
        self.epoch_mark = 0

        self.begin_str =\
            """{}\nTRAINING MODEL {} FOR {} AEONS"""\

        self.train_str = 'A{:02d} E{:03d} B{:04d}×{} L{:0.3f} '
        self.val_str = ('| VL{:0.3f} [stagcnt {}/{}] '
                        '[{:6.0f} epoch secs, {:6.0f} total]')

        super().__init__()

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.init_lr)
        self.start_mark = time.time()
        print(self.begin_str
              .format('='*80,
                      self.mhd.mk_model_name(self.model,
                                             self.mname,
                                             get_jh=True)[2],
                      self.aeons
                      )
              )

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch

    def on_batch_end(self, batch, logs=None):
        l = float(logs['loss'])
        if self.cur_loss is None:
            self.cur_loss = l
        else:
            self.cur_loss = self.k * l + (1 - self.k) * self.cur_loss
        sys.stdout.write('\r' + self.train_str.format(self.aeon,
                                                      self.epoch,
                                                      self.batch+1,
                                                      logs['size'],
                                                      self.cur_loss))

        if self.save_best_train and self.cur_loss < self.best_tloss:
            self.mhd.save_weights(self.model, mn=self.mname+'_best_train',
                                  overwrite=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_mark = time.time()
        self.epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['val_loss']
<<<<<<< HEAD
=======
        tloss = logs['loss']
>>>>>>> 3ca0ca0ee497cfab926025333d69abe107d49da6

        if loss >= self.best_loss:
            self.stagcnt += 1
        else:
            self.stagcnt = 0
            self.best_loss = loss
            self.mhd.save_weights(self.model, mn=self.mname+'_backup',
                                  overwrite=True)
            self.current_best_weights = self.model.get_weights()

        now = time.time()
        sys.stdout.write(self.val_str.format(loss,
                                             self.stagcnt, self.stagmax,
                                             now - self.epoch_mark,
                                             now - self.start_mark)+'\n')
        sys.stdout.flush()

        if self.stagcnt >= self.stagmax:
            sys.stdout.write('AEON {} COMPLETE'.format(self.aeon))
            if self.aeon >= self.aeons - 1:
                self.save_best_weights()
                print(': TRAINING FINISHED')
                self.model.stop_training = True
            else:
                self.aeon += 1
                self.epoch = 0
                self.stagcnt = 0
                self.stagmax *= self.aeon_stag_factor
                lr = (K.get_value(self.model.optimizer.lr) *
                      self.aeon_lr_factor)
                print(': LEARNING RATE SET TO {:.3E}, NEW STAGMAX {}\n'
                      .format(float(lr), self.stagmax))
                K.set_value(self.model.optimizer.lr, lr)

    def save_best_weights(self):
        self.model.set_weights(self.current_best_weights)
        self.mhd.save_weights(self.model,
                              mn=self.mname+'_best_l:{:.3f}'
                              .format(self.best_loss), overwrite=True)
