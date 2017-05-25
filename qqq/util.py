from inspect import signature, Parameter
import os.path as osp
import pickle
import time
from threading import Lock


def ensure_type(obj, t, *args, **kwargs):
    if obj is None:
        return t(*args, **kwargs)
    elif not isinstance(obj, t):
        raise ValueError(f'need a {t.__name__} object, got {obj}')
    else:
        return obj


def ensure_list(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


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


class Sync:
    '''
    class to transparently syncrhonize access to an object

    any `method` call an object wrapped in this class is equivalent to

    >>> with obj_lock:
    >>>     obj.method()

    this does not work for __dunder__ methods
    '''
    def __init__(self, obj):
        self.obj = obj
        self._lock = Lock()
        self._memo_table = {}

    def __getattr__(self, name):
        m = getattr(self.obj, name)
        if hasattr(m, '__call__'):
            try:
                return self._memo_table[name]
            except KeyError:
                def wrapped_call(*args, **kwargs):
                    with self._lock:
                        return m.__call__(*args, **kwargs)
                self._memo_table[name] = wrapped_call
                return wrapped_call
        else:
            return m


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
