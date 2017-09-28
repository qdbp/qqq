from functools import namedtuple
from collections.abc import Mapping, Sequence, Iterable
from functools import wraps
from inspect import signature, Parameter
import os.path as osp
import pickle
import time
from threading import Lock
from typing import Generic, TypeVar


PrioTup = namedtuple('PrioTup', ('prio', 'value'))


def ensure_type(obj, t, *args, **kwargs):
    if obj is None:
        return t(*args, **kwargs)
    elif not isinstance(obj, t):
        raise ValueError(f'need a {t.__name__} object, got {obj}')
    else:
        return obj


def as_list(obj):
    '''
    Ensures output is a list of objects.

    Iterables are read into a list, non-iterables are wrapped in a singleton
    list.
    '''

    if obj is None:
        return []

    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


def sift_kwargs(f):
    '''
    Lets the wrapped function silently ignore invalid kwargs.
    '''
    @wraps(f)
    def _f(*args, **kwargs):
        return f(*args, **kwsift(kwargs, f))
    return _f


def kwsift(kw, f):
    '''
    Sifts a keyoword argument dictionary with respect to a function.

    Returns a dictionary with those entries that the given function
    accepts as keyword arguments.

    If the function is found to accept a variadic keyword dictionary
    (**kwargs), the first argument is returned unchanged, since any keyword
    argument is therefore legal.
    '''

    sig = signature(f)
    kw_kinds = {Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
    out = {}
    # go backward to catch **kwargs on the first pass
    for name, p in list(sig.parameters.items())[::-1]:
        if p.kind == p.VAR_KEYWORD:
            return kw
        elif p.kind in kw_kinds and name in kw.keys():
            out[name] = kw[name]

    return out


def check_all_same_length(*args, allow_none=False, msg=None):
    '''
    Raises ValueError if arguments' lengths differ.

    Returns arguments' shared length.
    '''
    if not args:
        return 0

    s = {
        len(arg) for arg in args
        if not allow_none or arg is not None
    }

    if len(s) > 1:
        raise ValueError(
            f'arguments have different lengths! {s}\n' + (msg or ''))

    return len(args[0])


def pickled(fn, func, *args, **kwargs):
    if osp.isfile(fn):
        with open(fn, 'rb') as f:
            out = pickle.load(f)
    else:
        out = func(*args, **kwargs)
        with open(fn, 'wb') as f:
            pickle.dump(out, f)
    return out


class UQLError(Exception): pass  # noqa


def uql(d, k, default=None):
    '''
    Micro-Query Language.

    Dissect nested dicts. Dismember JSON without mercy.

    Valid keys for a given subcontainer which are not found
    in that subcontainer trigger the default return. On the other hand,
    invalid keys or attempting to index non-containers raises a
    UQLError.

    >>> d = {'a': {'1': 'foo', '2': ['bar', 'baz']}}
    >>> uql(d, 'a.2.1')
    "bar"
    >>> uql(d, 'a.1.b')
    InvalidPathError
    >>> uql(d, 'a.2.3', 'qux')
    "qux"
    '''

    key, _, rest = k.partition('.')

    if isinstance(d, Mapping):
        if key in d:
            if not rest:
                return d[key]
            else:
                return uql(d[key], rest, default=default)
        else:
            try:
                if int(key) in d:
                    if not rest:
                        return d[key]
                    else:
                        uql(d[key], rest, default=default)
            except ValueError:
                return default

    elif isinstance(d, Sequence):
        try:
            ix = int(key)
        except ValueError:
            raise UQLError(
                f'bad subkey {key} for subcontainer {d}'
            )
        if not rest:
            try:
                return d[ix]
            except IndexError:
                return default
        else:
            return uql(d[ix], rest, default=default)

    else:
        raise UQLError(
            f'attempt to get key {key} from terminal value {d}'
        )


class LockedRef:

    def __init__(self, obj, lock):
        self.obj = obj
        self.lock = lock

    def __del__(self):
        self.lock.release()

    def __getattr__(self, key):
        return getattr(self.obj, key)


class Sync:
    '''
    Class to transparently syncrhonize access to an object.

    Uses the descriptor protocol to return an exclusive reference wrapper
    which blocks attribute lookup until it is destroyed.
    '''

    def __init__(self, obj):
        self.obj = obj
        self._lock = Lock()

    def __get__(self, obj, cls):
        self._lock.acquire()

        return LockedRef(self.obj, self._lock)

    def __set__(self, obj, val):
        with self._lock():
            self.obj = val


class Timer():
    def __init__(self):
        self.mark = None

    def set(self):
        self.mark = time.time()

    @property
    def now(self):
        return time.time()

    @property
    def elapsed(self):
        if self.mark is None:
            raise ValueError("mark not set")
        return time.time() - self.mark

    @property
    def lap(self):
        if self.mark is None:
            raise ValueError("mark not set")
        t = time.time()
        d = t - self.mark
        self.mark = t
        return d

    def clear(self):
        self.mark = None

    def sleep(self, t):
        time.sleep(t)

    def wait(self, until):
        time.sleep(until - time.time())
