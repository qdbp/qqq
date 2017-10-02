import concurrent.futures as cfu
import os
from abc import abstractmethod
from collections import defaultdict

import click as clk
import keras.backend as K
import keras.callbacks as kc
import keras.layers as kl
import matplotlib.collections as mplc
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import numpy.random as npr
from colored import attr, fg
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.engine.topology import Layer
from keras.models import Model
from keras.utils import to_categorical
from tqdm import tqdm

from .log import get_logger
from .util import as_list

LOG = get_logger(__file__)

HDF5_EXT = 'hdf5'
JSON_EXT = 'json'
PNG_EXT = 'png'


# MODEL CONSTRUCTION

def apply_layers(inp, *stacks, td=False):
    '''
    Applies the Keras layers in the list of stacks sequentially to the input.

    If `td` is True, applies one layer of TimeDistributed to each layer.
    If `td` is an integer, applies `td` levels of TimeDistributed to each
    layer.
    '''

    td = int(td)
    y = inp
    for stack in stacks:
        for layer in stack:
            for i in range(td):
                layer = kl.TimeDistributed(layer)
            y = layer(y)
    return y


def compile_model(
        i, y, *,
        loss='categorical_crossentropy',
        losses=None,
        optimizer='nadam',
        metrics=None):
    '''
    Compiles a keras model with sensible defaults.

    Arguments:
        i:
            The input tensor(s).
        y:
            The output tensor(s).
        loss:
            The primary loss to use. Of the same form as accepted by
            `Model.compile`. Understands the following shorthand:
                cxe -> categorical_crossentropy
                bxe -> binary_crossentropy
        losses:
            Auxiliary loss tensors to add to the model with `add_loss`.
        optimizer:
            The optimizer to use. Of the same form as accepted by
            `Model.compile`.
        metrics:
            The metrics to use. Of the same form as accepted by
            `Model.compile`. Understands the following shorthand:
                acc -> categorical_accuracy
                bacc -> binary_accuracy
    '''

    def _rectify(inp, tr_dict):
        if not isinstance(inp, dict):
            return [tr_dict.get(x, x) for x in as_list(inp)]
        else:
            return {k: tr_dict.get(v, v) for k, v in inp.items()}

    loss_map = {
        'cxe': 'categorical_crossentropy',
        'bxe': 'categorical_crossentropy',
    }
    loss = _rectify(loss, loss_map)
    losses = as_list(losses)

    metric_map = {
        'acc': 'categorical_accuracy',
        'bacc': 'binary_accuracy',
    }
    metrics = _rectify(metrics, metric_map)

    m = Model(inputs=as_list(i), outputs=as_list(y))
    for aux_loss in losses:
        m.add_loss(aux_loss)
    m.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return m


def get_callbacks(
        name, *, pat_stop=9, pat_lr=3, plot=False, val=True,
        epochs=50, base_lr=0.001):
    '''
    Returns some sensible default callbacks for Keras model training.
    '''
    monitor = 'val_loss' if val else 'loss'

    os.makedirs('./weights/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)

    out = [
        kc.ModelCheckpoint(
            f'./weights/{name}.hdf5', save_best_only=True, monitor=monitor),
        ValLossPP(),
        KQLogger(name),
        kc.CSVLogger(f'./logs/{name}.csv'),
        # kc.ReduceLROnPlateau(
        #     patience=pat_lr, factor=1/np.e, monitor=monitor),
        # linear lr schedule
        kc.LearningRateScheduler(
            lambda epoch:  base_lr * (1 - epoch / epochs)),
        kc.EarlyStopping(patience=pat_stop, monitor=monitor),
    ]

    if plot:
        out.append(LiveLossPlot(name))

    return out


def shuffle_weights(weights):
    return [npr.permutation(w.flat).reshape(w.shape) for w in weights]


# TODO
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


# TODO
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


# DATA

def standard_mnist(flatten=True, featurewise_norm=True):
    (Xt, yt), (Xv, yv) = mnist.load_data()

    Xt = Xt.reshape(Xt.shape[0],
                    -1 if flatten else Xt.shape[1:]).astype(np.float32)
    Xv = Xv.reshape(Xv.shape[0],
                    -1 if flatten else Xv.shape[1:]).astype(np.float32)

    if featurewise_norm:
        m = np.mean(Xt, axis=0)
    else:
        m = np.mean(Xt)
    s = np.std(Xt)

    Xt = (Xt - m) / s
    Xv = (Xv - m) / s

    yt = to_categorical(yt)
    yv = to_categorical(yv)

    return (Xt, yt), (Xv, yv)


# CALLBACKS

class HypergradientScheduler(Callback):
    '''
    A learning rate scheduler that plays nice with other schedulers by
    mixing in a multiplier rather than overwriting the lr wholesale.
    '''
    # FIXME: can only be feasibly implemented as an optimizer


class KQLogger(Callback):

    def __init__(self, name):
        self.name = name
        self.best_val_loss = np.inf
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs):
        self.best_val_loss =\
            min(logs.get('val_loss', np.inf), self.best_val_loss)
        self.last_epoch += 1

    def on_train_begin(self, logs):
        LOG.info(f'begin training model {self.name}'.upper())

    def on_train_end(self, logs):
        if self.best_val_loss < np.inf:
            LOG.info(f'best val loss {self.best_val_loss:.4f}')
        LOG.info(
            f'end training model {self.name},'
            f' {self.last_epoch} epochs'.upper()
        )


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

        for key in losses.keys():
            tls = '{:.3f}'.format(losses[key])
            vl = val_losses.get('val_' + key)
            if vl is None:
                vls = '---'
            else:
                vls = '{:.3f}'.format(vl)

            if 'val_' + key in greens:
                vls = fg('green') + attr('bold') + vls + attr('reset')

            # out += f'{key[4:]:-<40.40s}: train {tls} - {vls} val\n'
            print(f'{key:-<40.40s}: train {tls} - {vls} val')


class LiveLossPlot(Callback):

    def __init__(self, name, plot_style=None, bpp=10, bpp_inflation=1.1):

        self.name = name
        self.bpp = bpp
        self.bpp_mult = 1.0
        self.bpp_inflation = bpp_inflation

        self.plot_style = plot_style or {}

        self.batch_data = defaultdict(list)
        self.proxy_lines = {}
        self.line_data = {}

        self.monotonic_batch = 0
        self.next_plot_batch = bpp
        self.exe = cfu.ThreadPoolExecutor(max_workers=1)
        self._setup_axes()

    def _get_lab_axis(self, lab):
        return self.ax_acc if self._is_lab_acc(lab) else self.ax

    def _is_lab_acc(self, lab):
        return 'acc' in lab

    def _recenter_axes(self):
        self.ax.autoscale()

        self.ax.set_ylim(bottom=0)
        self.ax_acc.set_ylim((0., 1))

        self.ax.set_xlim(left=1)

        self.ax.margins(0.1)
        self.ax_acc.margins(0.1)

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
        self.ax.grid(b=True, which='major', color='#7799bb', lw=0.3)
        self.ax.grid(b=True, which='minor', color='#88aacc', lw=0.3, ls=':')

        self.ax_acc = self.ax.twinx()
        self.ax_acc.grid(
            b=True, which='major', color='#bb9977', lw=0.3)
        self.ax_acc.grid(
            b=True, which='minor', color='#ccaa88', lw=0.3, ls=':')

        self.ax_acc.set_ylabel('accuracy')
        self._recenter_axes()

        os.makedirs('./plots/', exist_ok=True)

    def _update_plot(self, data):
        batch = data.pop('batch')

        to_plot = []
        to_plot_acc = []

        colors = []
        colors_acc = []

        new_lines = False
        for lab, loss in data.items():
            if lab not in self.proxy_lines:
                self.proxy_lines[lab] =\
                    self._get_lab_axis(lab).plot([], [], label=lab,)[0]
                self.line_data[lab] = [(batch, loss)]
                new_lines = True
            else:
                plot_arr = to_plot_acc if self._is_lab_acc(lab) else to_plot
                c_arr = colors_acc if self._is_lab_acc(lab) else colors

                self.line_data[lab].append((batch, loss))
                plot_arr.append(self.line_data[lab][-2:])
                c_arr.append(self.proxy_lines[lab].get_color())

        if new_lines:
            self.ax.legend(loc=3)
            self.ax_acc.legend(loc=6)

        if to_plot:
            lc = mplc.LineCollection(to_plot, colors=colors, linewidths=0.75)
            lc_acc = mplc.LineCollection(
                to_plot_acc, colors=colors_acc, linewidths=0.75)
            self.ax.add_collection(lc)
            self.ax_acc.add_collection(lc_acc)

            self._recenter_axes()

            self.fig.canvas.draw()
            self.fig.savefig(f'./plots/{self.name}.svg')

    def on_batch_end(self, batch, logs):

        add_point = all((
            bool(self.monotonic_batch),
            self.monotonic_batch == self.next_plot_batch,
        ))

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
            self.next_plot_batch += int(self.bpp_mult * self.bpp)
            self.bpp_mult *= self.bpp_inflation

        self.monotonic_batch += 1

    def on_epoch_end(self, epoch, logs):
        val_logs = {k: v for k, v in logs.items() if k[:4] == 'val_'}
        val_logs['batch'] = self.monotonic_batch
        self._update_plot(val_logs)

    def on_train_end(self, logs):
        self.fig.canvas.draw()
        self.fig.savefig(f'./plots/{self.name}.svg')


# LAYERS

class NegGrad(Layer):

    def __init__(self, lbd, **kwargs):
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


# ACTIVATIONS

def negabs(x):
    return -K.abs(x)


# LOSSES AND METRICS
def f2_crossentropy(yt, yp):
    '''
    Biased binary crossentropy, favouring recall.
    '''
    yp = K.clip(yp, K.epsilon(), 1 - K.epsilon())
    return K.mean(-(2 * yt * K.log(yp) + (1 - yt) * K.log(1 - yp)))
