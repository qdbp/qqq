import concurrent.futures as cfu
import os
import warnings
from abc import abstractmethod
from collections import Counter, defaultdict, deque
from functools import lru_cache

import keras.backend as K
import keras.callbacks as kc
import keras.layers as kl
import matplotlib.collections as mplc
import matplotlib.ticker as mtick
import numpy as np
import numpy.random as npr
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras.models import Model
from keras.utils import to_categorical

from .log import get_logger
from .util import as_list

try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

try:
    from colored import attr, fg
except ImportError:
    warnings.warn("Install the 'colored' package for color output support")

    def attr(s: str) -> str:
        return s

    fg = attr


LOG = get_logger(__file__)

HDF5_EXT = "hdf5"
JSON_EXT = "json"
PNG_EXT = "png"


# MODEL CONSTRUCTION


def apply_layers(inp, *stacks, td=False):
    """
    Applies the Keras layers in the list of stacks sequentially to the input.

    If `td` is True, applies one layer of TimeDistributed to each layer.
    If `td` is an integer, applies `td` levels of TimeDistributed to each
    layer.
    """

    td = int(td)
    y = inp
    for stack in stacks:
        for layer in stack:
            for _ in range(td):
                layer = kl.TimeDistributed(layer)
            y = layer(y)
    return y


def compile_model(
    x, y, *, loss="cxe", aux_losses=None, optimizer="nadam", metrics=None
):
    """
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
    """

    outputs = as_list(y)
    inputs = as_list(x)

    def _rectify(arg, tr_dict):

        if arg is None:
            return None

        if not isinstance(arg, dict):
            arg = as_list(arg)

            if len(arg) == 1:
                arg = arg * len(outputs)

        if len(arg) != len(outputs):
            raise ValueError(
                "Number of losses doesn't match the number of outputs"
            )

        if not isinstance(arg, dict):
            return [tr_dict.get(x, x) for x in arg]
        else:
            return {k: tr_dict.get(v, v) for k, v in arg.items()}

    loss_map = {
        "cxe": "categorical_crossentropy",
        "bxe": "categorical_crossentropy",
    }
    loss = _rectify(loss, loss_map)
    losses = as_list(aux_losses)

    metric_map = {"acc": "categorical_accuracy", "bacc": "binary_accuracy"}
    metrics = _rectify(metrics, metric_map)

    m = Model(inputs=inputs, outputs=outputs)
    for aux_loss in losses:
        m.add_loss(aux_loss)
    m.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return m


def get_callbacks(
    name,
    *,
    pat_stop=15,
    pat_lr=3,
    plot=False,
    val=True,
    epochs=50,
    base_lr=0.001,
):
    """
    Returns some sensible default callbacks for Keras model training.
    """
    monitor = "val_loss" if val else "loss"

    os.makedirs("./weights/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    out = [
        kc.ModelCheckpoint(
            f"./weights/{name}.hdf5", save_best_only=True, monitor=monitor
        ),
        ValLossPP(),
        KQLogger(name),
        kc.CSVLogger(f"./logs/{name}.csv"),
        # kc.ReduceLROnPlateau(
        #     patience=pat_lr, factor=1/np.e, monitor=monitor),
        # linear lr schedule
        kc.LearningRateScheduler(lambda epoch: base_lr * (1 - epoch / epochs)),
        kc.EarlyStopping(patience=pat_stop, monitor=monitor),
    ]

    if plot:
        out.append(LiveLossPlot(name))

    return out


def shuffle_weights(weights):
    return [npr.permutation(w.flat).reshape(w.shape) for w in weights]


# DATA


def data_mnist(**kwargs):
    from keras.datasets import mnist

    return normalize_data(mnist.load_data, **kwargs)


def data_fnist(**kwargs):
    from keras.datasets import fashion_mnist

    return normalize_data(fashion_mnist.load_data, **kwargs)


def normalize_data(load_func, flatten=False, featurewise_norm=True):
    (Xt, yt), (Xv, yv) = load_func()

    # we squish to (batch, rest) for non-conv
    if flatten:
        Xt = Xt.reshape(Xt.shape[0], -1)
        Xv = Xv.reshape(Xv.shape[0], -1)
    # if not, we add a dummy colour channel for conv
    else:
        Xt = Xt[..., None]
        Xv = Xv[..., None]

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
    """
    A learning rate scheduler that plays nice with other schedulers by
    mixing in a multiplier rather than overwriting the lr wholesale.
    """

    # FIXME: can only be feasibly implemented as an optimizer


class KQLogger(Callback):
    def __init__(self, name):
        self.name = name
        self.best_val_loss = np.inf
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs):
        self.best_val_loss = min(
            logs.get("val_loss", np.inf), self.best_val_loss
        )
        self.last_epoch += 1

    def on_train_begin(self, logs):
        LOG.info(f"begin training model {self.name}".upper())

    def on_train_end(self, logs):
        if self.best_val_loss < np.inf:
            LOG.info(f"best val loss {self.best_val_loss:.4f}")
        LOG.info(
            f"end training model {self.name},"
            f" {self.last_epoch} epochs".upper()
        )


class ValLossPP(Callback):

    compare_more = {"accuracy"}

    @classmethod
    @lru_cache(maxsize=None)
    def is_compare_more(cls, key):
        """
        Returns True for keys where more is better.
        """
        for cm in cls.compare_more:
            if cm in key:
                return True
        return False

    def __init__(self):
        self.best_val_loss = {}

    def on_epoch_begin(self, epoch, logs):
        try:
            total = self.params["steps"]
        except KeyError:
            total = (
                self.params.get("samples") // self.params.get("batch_size") + 1
            )
        self.counter = tqdm(total=total, desc=f"Epoch {epoch}", leave=False)

    def on_batch_end(self, batch, logs):
        self.counter.update()

    def on_epoch_end(self, epoch, logs):  # noqa
        self.counter.close()
        print(f"Epoch {epoch}")

        greens = set()
        losses = {}
        val_losses = {}

        for key, val in logs.items():

            if key not in self.params["metrics"]:
                continue

            if key.startswith("val_"):
                val_losses[key] = val
                if key not in self.best_val_loss:
                    self.best_val_loss[key] = val
                    greens.add(key)

                elif (
                    self.is_compare_more(key) and val > self.best_val_loss[key]
                ) or (
                    not self.is_compare_more(key)
                    and val < self.best_val_loss[key]
                ):
                    greens.add(key)
                    self.best_val_loss[key] = val
            else:
                losses[key] = val

        for key in losses.keys():
            tls = "{:.3f}".format(losses[key])
            vl = val_losses.get("val_" + key)
            if vl is None:
                vls = "---"
            else:
                vls = "{:.3f}".format(vl)

            if "val_" + key in greens:
                vls = fg("green") + attr("bold") + vls + attr("reset")

            # out += f'{key[4:]:-<40.40s}: train {tls} - {vls} val\n'
            print(f"{key:-<40.40s}: train {tls} - {vls} val")


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
        return "acc" in lab

    def _recenter_axes(self):
        self.ax.autoscale()

        self.ax.set_ylim(bottom=0)
        self.ax_acc.set_ylim((0.0, 1))

        self.ax.set_xlim(left=1)

        self.ax.margins(0.1)
        self.ax_acc.margins(0.1)

    def _setup_axes(self):
        self.fig = plt.figure()
        self.fig.set_size_inches(15, 8)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.set_title(f"{self.name} training plot")
        self.ax.set_ylabel("loss")
        self.ax.set_xlabel("batch")
        self.ax.set_xscale("log")
        self.ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(n=5))
        self.ax.yaxis.set_minor_formatter(mtick.NullFormatter())
        self.ax.grid(b=True, which="major", color="#7799bb", lw=0.3)
        self.ax.grid(b=True, which="minor", color="#88aacc", lw=0.3, ls=":")

        self.ax_acc = self.ax.twinx()
        self.ax_acc.grid(b=True, which="major", color="#bb9977", lw=0.3)
        self.ax_acc.grid(b=True, which="minor", color="#ccaa88", lw=0.3, ls=":")

        self.ax_acc.set_ylabel("accuracy")
        self._recenter_axes()

        os.makedirs("./plots/", exist_ok=True)

    def _update_plot(self, data):
        batch = data.pop("batch")

        to_plot = []
        to_plot_acc = []

        colors = []
        colors_acc = []

        new_lines = False
        for lab, loss in data.items():
            if lab not in self.proxy_lines:
                self.proxy_lines[lab] = self._get_lab_axis(lab).plot(
                    [], [], label=lab
                )[0]
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
                to_plot_acc, colors=colors_acc, linewidths=0.75
            )
            self.ax.add_collection(lc)
            self.ax_acc.add_collection(lc_acc)

            self._recenter_axes()

            self.fig.canvas.draw()
            self.fig.savefig(f"./plots/{self.name}.svg")

    def on_batch_end(self, batch, logs):

        add_point = all(
            (
                bool(self.monotonic_batch),
                self.monotonic_batch == self.next_plot_batch,
            )
        )

        if add_point:
            to_append = dict(batch=self.monotonic_batch)

        loss_log = {
            k: v for k, v in logs.items() if k in self.params["metrics"]
        }

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
        val_logs = {k: v for k, v in logs.items() if k[:4] == "val_"}
        val_logs["batch"] = self.monotonic_batch
        self._update_plot(val_logs)

    def on_train_end(self, logs):
        self.fig.canvas.draw()
        self.fig.savefig(f"./plots/{self.name}.svg")


# LAYERS


class NegGrad(Layer):
    """
    Inverts gradient flow, scalding by an adjustable parameter.
    """

    def __init__(self, lbd, **kwargs):
        super().__init__(**kwargs)
        if lbd < 0:
            raise ValueError("lbd must be nonnegative.")
        self._lbd = lbd

    def build(self, input_shape):
        self.lbd = self.add_weight(
            "lbd",
            (1,),
            trainable=False,
            initializer=lambda x: K.variable(self._lbd),
        )
        self.built = True

    def call(self, x, mask=None):
        return (1 + self.lbd) * K.stop_gradient(x) - self.lbd * x

    def get_output_shape_for(self, input_shape):
        return input_shape

    def set_lbd(self, lbd):
        self.lbd.set_value(lbd)

    def get_config(self):
        config = {"lbd": float(K.get_value(self.lbd))}
        config.update(super().get_config())
        return config


class CReLU(Layer):
    """
    Concatenated relu activation.
    """

    def build(self):
        self.built = True

    def calculate_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return NotImplemented

        return input_shape[:-1] + (2 * input_shape[-1],)

    def call(self, x):

        p = K.relu(x)
        n = K.relu(-x)

        return K.concatenate([n, p], axis=-1)


# ACTIVATIONS
def negabs(x):
    return -K.abs(x)


# LOSSES AND METRICS
def f2_crossentropy(yt, yp):
    """
    Biased binary crossentropy, favouring recall.
    """
    yp = K.clip(yp, K.epsilon(), 1 - K.epsilon())
    return K.mean(-(2 * yt * K.log(yp) + (1 - yt) * K.log(1 - yp)))


def r2(yt, yp):
    m = K.mean(yt)

    ss = K.sum((yt - m) ** 2)
    sr = K.sum((yt - yp) ** 2)

    return K.mean(1 - (sr + K.epsilon()) / (ss + K.epsilon()))


# TEMPLATES


class Template:
    """
    A higher level model building block, based on the formulaic combination
    of multiple layers.
    """

    name_resolver = Counter()
    name_stack = deque([])

    def __init__(self, name=None, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    # TODO
    # def clone(self, new_name, *args, **kwargs):
    #     new_kwargs = self.kwargs.copy(self.kwargs)update(kwargs)
    #     if args:
    #         new_args = args
    #     else:
    #         new_args = self.args

    #     return self.__class__(*new_args, name=new_name, **new_kwargs)

    def __call__(self, *xs):
        if self.name is not None:
            self.__class__.name_stack.append(self.name)
            try:
                return self.call(*xs)
            finally:
                self.__class__.name_stack.pop()
        else:
            return self.call(*xs)

    @abstractmethod
    def call(self, *xs):
        """
        Applies this template to the keras tensor x.
        """

    def __mul__(self, n):
        if isinstance(n, int):
            return Stacked(*([self] * n))
        else:
            return NotImplemented

    # FIXME use >> << instead
    def __add__(self, t):
        if isinstance(t, (Template, kl.Layer)):
            return Stacked(self, t)
        else:
            return NotImplemented

    def __rmul__(self, i):
        # XXX: this will break if __mul__ implements type overloading
        return self.__mul__(i)

    def __radd__(self, i):
        if isinstance(i, (Template, kl.Layer)):
            return Stacked(i, self)
        else:
            return NotImplemented

    def __or__(self, t):
        if isinstance(t, (Template, kl.Layer)):
            return Parallel(self, t)
        else:
            return NotImplemented

    def __and__(self, t):
        if isinstance(t, (Template, kl.Layer)):
            return Reduce(kl.Concatenate, self, t)
        else:
            return NotImplemented

    def get_name(self):
        out = ".".join(self.__class__.name_stack)
        index = self.name_resolver[out]
        self.name_resolver[out] += 1
        out += "_" + str(index)
        return out

    def suffix_name(self, name_suffix):
        class _:
            def __enter__(inner_self):
                self.__class__.name_stack.append(name_suffix)

            def __exit__(inner_self, *_):  # noqa
                self.__class__.name_stack.pop()

        return _()


class Die(Template):
    """
    The lowest-level template: produces a new instance of a single layer
    with the parameters given at instantiation of this class.
    """

    def __init__(self, name: str, lcls: type, *args, **kwargs) -> None:
        super().__init__(name)
        self.lcls = lcls
        self.args = args
        self.kwargs = kwargs

        # get rid of an erroneous layer name; layers take the template name
        self.kwargs.pop("name", None)

        self._instances = {}  # type: ignore

    def clone(self, name, **kwargs):
        new_kwargs = self.kwargs.copy()
        new_kwargs.update(kwargs)
        return self.__class__(name, self.lcls, *self.args, **new_kwargs)

    def call(self, xs):
        layer = self.lcls(*self.args, name=self.get_name(), **self.kwargs)

        self._instances[layer.name] = layer

        return layer(xs)


class Parallel(Template):
    def __init__(self, *objs, name=None):
        super().__init__(name or "parallel")
        self.objs = objs

    def call(self, x):
        return [obj(x) for obj in self.objs]


class Reduce(Template):
    def __init__(self, objs, name=None) -> None:
        super().__init__(name or "reduce")
        self.objs = objs

    def call(self, xs):
        true_ys = []  # type: ignore
        for obj in self.objs:
            y = obj(xs)
            if isinstance(y, (list, tuple)):
                true_ys.extend(y)
            else:
                true_ys.append(y)

        return self.merge_cls(name=self.get_name())(true_ys)


class Stacked(Template):
    """
    It stacks.

    It asks no questions about what it stacks.
    """

    def __init__(self, *objs, name=None):
        super().__init__(name=name)
        self.objs = objs

    def call(self, x):
        y = x
        for obj in self.objs:
            y = obj(y)
        return y

    def __getitem__(self, ix):
        return self.objs[ix]


class Gated(Template):
    """
    Must be instantiated with a factory.

    On production, instantiates a second product with sigmoid activations.
    This product is called on the same inputs as the principal, and the
    principal's output is multiplied by that of the gater.
    """

    def __init__(self, factory: Die) -> None:
        super().__init__(name="gated")
        self.fac = factory.clone(name=factory.name, activation=None)
        self.gfac = factory.clone(
            name=factory.name + "_gate", activation="hard_sigmoid"
        )

    def call(self, x):
        g = self.gfac(x)
        y = self.fac(x)

        with self.suffix_name("mul_gate"):
            return kl.Multiply(name=self.get_name())([g, y])


class Residual(Template):
    """
    Sums the template's output with its own input, and opionally
    applies an activation to the result.
    """

    def __init__(self, *objs, activation=None) -> None:
        super().__init__(name="res")
        self.objs = objs
        self.activation = activation

    def call(self, x):
        y = x

        if self.activation is not None:
            with self.suffix_name("res_activation"):
                y = kl.Activation(self.activation, name=self.get_name())(y)

        for obj in self.objs:
            y = obj(y)

        y = Reduce(kl.Add, name="add_input")(x, y)

        return y


class TCTX:
    """
    Template execution context and parameter generator.
    """

    def new_frame(self, template: Template):
        class _frame:
            def __enter__(inner_self):
                self._enter(template)

            def __exit__(inner_self, *_):  # noqa
                self._exit(template)

        return _frame()

    @abstractmethod
    def _enter(self, template):
        pass

    @abstractmethod
    def _exit(self, template):
        pass
