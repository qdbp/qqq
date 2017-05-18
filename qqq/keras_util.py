import argparse as arg
import cmd
import concurrent.futures as cfu
# from functools import partial
import os
import os.path as osp
import sys
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime as dtt
from datetime import timedelta
from glob import glob
from threading import Lock, Thread
from typing import Optional

import click as clk
import keras.backend as K
import keras.callbacks as kcb
import keras.layers as kr
import matplotlib.collections as mplc
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy.random as npr
from colored import attr, fg
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.engine.topology import Layer
from keras.layers.noise import GaussianNoise
from keras.utils import to_categorical
from tqdm import tqdm

from qqq.qlog import get_logger

LOG = get_logger(__file__)

HDF5_EXT = 'hdf5'
JSON_EXT = 'json'
PNG_EXT = 'png'


def apply_layers(inp, *stacks, td=False):

    y = inp
    for stack in stacks:
        for layer in stack:
            if td:
                y = kr.TimeDistributed(layer)(y)
            else:
                y = layer(y)
    return y


def shuffle_weights(weights):
    return [npr.permutation(w.flat).reshape(w.shape) for w in weights]


def get_callbacks(name, *, pat_stop=9, pat_lr=4):
    return [
        kcb.EarlyStopping(patience=pat_stop),
        kcb.ReduceLROnPlateau(patience=pat_lr, factor=0.2),
        kcb.ModelCheckpoint(f'./weights/{name}.hdf5', save_best_only=True),
        ValLossPP(),
        KQLogger(name),
        LiveLossPlot(name),
    ]


# DATA
def standard_mnist(flatten=True):
    (Xt, yt), (Xv, yv) = mnist.load_data()

    Xt = Xt.reshape(Xt.shape[0],
                    -1 if flatten else Xt.shape[1:]).astype(np.float32)
    Xv = Xv.reshape(Xv.shape[0],
                    -1 if flatten else Xv.shape[1:]).astype(np.float32)

    m = np.mean(Xt, axis=0)
    s = np.std(Xt)

    Xt = (Xt - m) / s
    Xv = (Xv - m) / s

    yt = to_categorical(yt)
    yv = to_categorical(yv)

    return (Xt, yt), (Xv, yv)


# TODO: probably
class KProject:

    def __init__(self):
        self.archs = {}

    def add_architecture(self, name: str, arch: "KArch"):
        self.archs[name] = arch

    def add_writer(self, writer):
        self.writer = writer

    @clk.command
    @clk.argument('arch_name')
    @clk.option('--test')
    def execute(self, arch_name, * test):
        arch = self.archs[arch_name]
        arch.build()

        if test:
            arch.test()
        else:
            arch.train()


class KArch:

    def __init__(
            self,
            name: str,
            f_build_model, f_evaluate_model, f_predict_model):
        self.name = name
        self.model = None

    @abstractmethod
    def build(self, **hyp):
        '''
        Create and compile the keras model.
        '''

    @abstractmethod
    def train(self, X, y=None, sample_weight=None, class_weight=None):
        '''
        Trains the model.
        '''

    @abstractmethod
    def evaluate(self, X):
        '''
        Evaluates the model.
        '''

    def hyp_search(self, hyps, log_fn=None):
        if log_fn is None:
            log_fn = f'./hyp_search_{self.name}.csv'
        pass


class KQLogger(Callback):

    def __init__(self, name):
        self.name = name
        self.best_val_loss = np.inf

    def on_epoch_end(self, logs):
        self.best_val_loss =\
            min(logs.get('val_loss', np.inf), self.best_val_loss)

    def on_train_begin(self, logs):
        LOG.info(f'begin training model {self.name}'.upper())

    def on_train_end(self, logs):
        if self.best_val_loss < np.inf:
            LOG.info(f'best val loss {self.best_val_loss:.4f}')
        LOG.info(f'end training model {self.name}'.upper())

class ValLossPP(Callback):

    compare_more = {'categorical_accuracy', 'binary_accuracy'}

    def __init__(self):
        self.best_val_loss = {}

    def on_epoch_begin(self, epoch, logs):
        try:
            total = self.params['steps']
        except KeyError:
            total =\
                self.params.get('samples') // self.params.get('batch_size') + 1
        self.counter = tqdm(
            total=total,
            desc=f'Epoch {epoch}',
            leave=False,
        )

    def on_batch_end(self, batch, logs):
        self.counter.update()

    def on_epoch_end(self, epoch, logs):  # noqa
        self.counter.close()
        # out = f'Epoch {epoch}\n'
        print(f'Epoch {epoch}')

        greens = set()
        losses = {}
        val_losses = {}

        for key, val in logs.items():

            if key not in self.params['metrics']:
                continue

            if key.startswith('val_'):
                val_losses[key] = val
                if key not in self.best_val_loss:
                    self.best_val_loss[key] = val
                    greens.add(key)

                elif ((key[4:] in self.compare_more and
                        val > self.best_val_loss[key]) or (
                        key[4:] not in self.compare_more and
                        val < self.best_val_loss[key])):
                    greens.add(key)
                    self.best_val_loss[key] = val
            else:
                losses[key] = val

        for key in val_losses.keys():
            vl = val_losses[key]
            tl = losses[key[4:]]

            vls = '{:.3f}'.format(float(vl))
            tls = '{:.3f}'.format(float(tl))

            if key in greens:
                vls = fg('green') + attr('bold') + vls + attr('reset')

            # out += f'{key[4:]:-<40.40s}: train {tls} - {vls} val\n'
            print(f'{key[4:]:-<40.40s}: train {tls} - {vls} val')


class LiveLossPlot(Callback):

    def __init__(self, name, plot_style=None,
                 batches_per_point=10, bpp_inflation=1.1):

        self.name = name
        self.batches_per_point = batches_per_point
        self.bpp_mult = 1.0
        self.bpp_inflation = bpp_inflation

        self.plot_style = plot_style or {}

        self.batch_data = defaultdict(list)
        self.proxy_lines = {}
        self.line_data = {}

        self.monotonic_batch = 0
        self.exe = cfu.ThreadPoolExecutor(max_workers=1)
        self._setup_axes()

    def _setup_axes(self):
        self.fig = plt.figure()
        self.fig.set_size_inches(15, 8)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.set_title(f'{self.name} training plot')
        self.ax.set_ylabel('loss')
        self.ax.set_xlabel('batch')

        self.ax.set_xscale('log')

        self.ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(n=5))
        self.ax.yaxis.set_minor_formatter(mtick.NullFormatter())
        self.ax.grid(b=True, which='major', color='#999999', lw=0.3)
        self.ax.grid(b=True, which='minor', color='#aaaaaa', lw=0.3, ls=':')

        os.makedirs('./plots/', exist_ok=True)

    def _update_plot(self, data):
        batch = data.pop('batch')

        to_plot = []
        colors = []
        new_lines = False
        for lab, loss in data.items():
            if lab not in self.proxy_lines:
                self.proxy_lines[lab] = self.ax.plot([], [], label=lab,)[0]
                self.line_data[lab] = [(batch, loss)]
                new_lines = True
            else:
                self.line_data[lab].append((batch, loss))
                to_plot.append(self.line_data[lab][-2:])
                colors.append(self.proxy_lines[lab].get_color())

        if new_lines:
            self.ax.legend(loc=3)

        if to_plot:
            lc = mplc.LineCollection(to_plot, colors=colors, linewidths=0.75)
            self.ax.add_collection(lc)
            self.ax.autoscale()
            self.ax.set_ylim(bottom=0)
            self.ax.set_xlim(left=1)
            self.ax.margins(0.1)

            self.fig.canvas.draw()
            self.fig.savefig(f'./plots/{self.name}.svg')

            self.bpp_mult *= self.bpp_inflation

    def on_batch_end(self, batch, logs):

        add_point =\
            not bool(
                (self.monotonic_batch - 1) %
                int(self.batches_per_point * self.bpp_mult)
            )
        if add_point:
            to_append = dict(batch=self.monotonic_batch)

        loss_log =\
            {k: v for k, v in logs.items() if k in self.params['metrics']}

        for lab, loss in loss_log.items():
            if add_point:
                mval = np.mean(self.batch_data[lab])
                to_append.update({lab: mval})
                del self.batch_data[lab][:]
            else:
                self.batch_data[lab].append(loss)

        if add_point:
            self.exe.submit(self._update_plot, to_append)

        self.monotonic_batch += 1

    def on_epoch_end(self, epoch, logs):
        val_logs = {k: v for k, v in logs.items() if k[:4] == 'val_'}
        val_logs['batch'] = self.monotonic_batch
        self._update_plot(val_logs)

    def on_train_end(self, logs):
        self.fig.canvas.draw()
        self.fig.savefig(f'./plots/{self.name}.svg')
