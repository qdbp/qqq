import time
from threading import Lock


def fx(f, x):
    """ function takes a callable f and argument to said callable x
        and returns f(x)
    """
    return f(x)


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
