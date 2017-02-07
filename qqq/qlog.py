import gc
import inspect
import logging as lgg
import string
import sys
import time
from collections import deque
from functools import lru_cache
from io import IOBase

import numpy as np

# TODO
COLOR_CRITICAL = '#bf3965'
COLOR_ERROR = '#bf6956'
COLOR_WARNING = '#dc9656'

VERBOSE = 15


class QBox:
    '''
    Class to draw a simplified curses-like box.
    '''

    upline = '\033[F'

    def __init__(self, width, height, border='', blank=' '):
        self.width = width
        self.height = height
        self.border = border
        self.blank = blank

        self._state = np.empty((height, width), dtype='U')
        self.clear()

    def __setitem__(self, k, v: str) -> None:
        if isinstance(k, tuple):
            x, y = k
        else:
            x, y = k, 0

        if not 0 <= x < self.height:
            raise ValueError(f'row position {x} out of bounds')
        elif not 0 <= y < self.width:
            raise ValueError(f'column position {y} out of bounds')

        x = self.height - x - 1
        x = self.height - min(self.height, x)

        l = min(len(v), self.width - y)
        self._state[self.height - x, y:y + l] =\
            np.asarray(list(v))[:l]

    def __delitem__(self, k):
        if isinstance(k, tuple):
            if len(k) == 3:
                x, y, lim = k
                self._state[x, y:lim] = ' '
            elif len(k) == 2:
                x, y = k
                self._state[x, y:] = ' '
            elif len(k) == 1:
                self.__delitem__(k[0])
            elif len(k) == 0:
                self._state[:, :] = ' '
        else:
            self._state[x, :] = ' '

    def _state_to_str(self, state):
        out = (
            self.border * (self.width + 2) + '\n' +
            '\n'.join(
                [self.border + ''.join(row) + self.border for row in state]
            ) + '\n' +
            self.border * (self.width + 2) +
            self.upline * (self.height + 1)
        )
        return out

    def clear(self):
        self._state.fill(self.blank)

    def draw(self, stream):
        stream.write(self._state_to_str(self._state))


class SplitStream(IOBase):
    '''
    Writes logs to stdout in multiple columns.
    '''

    def __init__(self, width, height, channels=2, chan_split='║',
                 stream=sys.stdout, border='.'):
        self.width, self.height = width, height
        self.chan_split = chan_split
        self.stream = stream

        self._sw = (width - channels * len(chan_split)) // channels
        self._sbs = [deque([], maxlen=height) for ix in range(channels)]

        self.qbox = QBox(width, height, border=border)

    def write(self, msg, c):

        msg = str(msg)
        if len(msg) > self._sw:
            msg = msg[:self._sw - 3] + '...'

        self._sbs[c].appendleft(msg)
        self._redraw()

    def _redraw(self):
        self.qbox.clear()
        for dx, dq in enumerate(self._sbs):
            for mx, msg in enumerate(dq):
                self.qbox[mx, dx * self._sw + dx] = msg
            for mx in range(self.height):
                self.qbox[mx, self._sw * (dx + 1)] = self.chan_split
        self.qbox.draw(self.stream)


class QFormatter(lgg.Formatter):

    def __init__(self, *args, do_color=False, **kwargs):
        self.do_color = do_color
        super().__init__(*args, **kwargs)

    def formatTime(self, record, datefmt=None):
        t = self.converter(record.created)
        prefix = time.strftime('%Y-%j-', t)
        prefix += '{:05d}'.format(3600 * t.tm_hour + 60 * t.tm_min + t.tm_sec)
        return prefix


class QLogger(lgg.Logger):

    def verbose(self, *args, **kwargs):
        if self.isEnabledFor(VERBOSE):
            self._log(VERBOSE, *args, **kwargs)


lgg.setLoggerClass(QLogger)


@lru_cache(maxsize=None)
def fun_from_frame(frame):
    for o in gc.get_objects():
        if inspect.isfunction(o) and o.__code__ is frame.f_code:
            return o


@lru_cache(maxsize=None)
def compress_snake_case(name):
    return '_'.join(('' if not p else p[0]) for p in name.split('_'))


@lru_cache(maxsize=None)
def compress_upcase(name):
    return ''.join(c for c in name if c in string.ascii_uppercase)


@lru_cache(maxsize=None)
def compress_qualname(qname, module):
    names = qname.split('.')
    s = ''

    if module is not None:
        s += '.'.join(compress_snake_case(m) for m in module.split('.')) + ':'

    for n in names[:-1]:
        if n.startswith('<'):
            s += f'<n[1]>'
        else:
            upper = compress_upcase(n)
            if upper:
                s += upper
            else:
                s += compress_snake_case(n)
        s += '.'

    return s + names[-1]


def with_stackinfo(logger):
    old_log = logger._log

    def new_log(self, *args, **kwargs):
        # _log -> info, etc. -> true caller
        frame = inspect.currentframe().f_back.f_back
        fun = fun_from_frame(frame)
        qname = compress_qualname(fun.__qualname__, fun.__module__)

        kwargs['extra'] = kwargs.get('extra', {})
        kwargs['extra']['qname'] = qname

        return old_log(self, *args, **kwargs)

    logger._log = new_log
    return logger


def setup_logger(logger, *, fn, do_stdout=False, mode='a'):
    logger.setLevel('INFO')

    formatter = QFormatter(
        '[{levelname:.1s}] {asctime} {qname}: {message}',
        style='{',
    )

    fn_fmt = QFormatter(
        '[{levelname:.1s}] {asctime} {qname}: {message}',
        style='{',
        do_color=False,
    )
    fn_h = lgg.FileHandler(fn, mode=mode)
    fn_h.setFormatter(fn_fmt)
    logger.addHandler(fn_h)

    if not hasattr(logger.__class__, 'verbose'):

        logger.__class__.verbose = verbose

    handler = lgg.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)



def get_logger(name, **kwargs):
    logger = with_stackinfo(lgg.getLogger(name))
    setup_logger(logger, fn='./log.txt', **kwargs)
    return logger
