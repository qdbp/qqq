import os.path as osp
import pickle
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import namedtuple, wraps
from inspect import Parameter, signature
from time import sleep, time
from typing import Dict, Generic, Optional, Counter as CounterT

PrioTup = namedtuple('PrioTup', ('prio', 'value'))


def ensure_type(obj, t, *args, **kwargs):
    '''
    Validate that `obj` is an object of type `t`.

    If `obj` is None, the constructor `t` is called with *args, *kwargs.
    '''

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


def save_pickle(obj, name):
    with open(f'{name}.p', 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(name):
    with open(f'{name}.p', 'rb') as f:
        return pickle.load(f)


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


class Stopwatch:
    '''
    Convenience class for timing operations.

    Can track a number of timers, identified by strings keys, in parallel.
    '''
    @property
    @classmethod
    def now(cls):
        '''
        Returns the current time.

        Convenience method to avoid explicit imports of `time`.
        '''
        return time()

    def __init__(self) -> None:
        self.marks: Dict[Optional[str], float] = {}
        self.dts: Dict[Optional[str], float] = {}

    def is_set(self, key: str=None):
        return self.marks[key] is not None

    def set(self, key: str=None) -> None:
        '''
        (Re)sets the timer identified by `key`.
        '''
        self.marks[key] = time()

    def elapsed(self, key=None) -> float:
        '''
        Returns the time elapsed since the key was set.
        '''
        if self.marks[key] is None:
            raise ValueError(f"Mark for key {key} was not set")
        return time() - self.marks[key]

    def lap(self, key: str=None) -> float:
        '''
        Returns the time elapsed since the last time the key was set,
        and sets the key.
        '''
        elapsed = self.elapsed(key)
        self.set(key)
        return elapsed

    def clear(self, key=None):
        del self.marks[key]

    def sleep(self, t):
        '''
        Convenience wrapper around time.sleep()
        '''
        sleep(t)

    def wait(self, until):
        dt = until - time()
        if dt <= 0.:
            return
        sleep(dt)


# class RateCounter:
#     '''
#     Class for keeping temporal statistics for events.
#     '''
# 
#     def __init__(self):
#         self._counter: CounterT[str] = Counter()
#         self._stopwatch: Stopwatch()
# 
#     def register(self, key):
#         self._counter[key] = 0
# 
# 
#     def fire(self, key: str) -> None:
#         self._counter[str] += 1
# 
