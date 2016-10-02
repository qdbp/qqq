import time


def fx(f, x):
    """ function takes a callable f and argument to said callable x
        and returns f(x)
    """
    return f(x)


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
