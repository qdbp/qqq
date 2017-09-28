import gc
import inspect
import logging as lgg
import os.path as osp
import re
import string
import time
from functools import lru_cache

# TODO
COLOR_CRITICAL = '#bf3965'
COLOR_ERROR = '#bf6956'
COLOR_WARNING = '#dc9656'

VERBOSE = 15
lgg.addLevelName(VERBOSE, 'Verbose')


class VNVFormatter(lgg.Formatter):

    UNCOLOR_RE = re.compile(r'\x1b[^m]*m')

    def __init__(self, *args, do_color=False, strip_color=False, **kwargs):
        self.do_color = do_color
        self.strip_color = strip_color
        super().__init__(*args, **kwargs)

    def format(self, record):
        if self.strip_color:
            record.msg = self.UNCOLOR_RE.sub('', record.msg)
        return super().format(record)

    def formatTime(self, record, datefmt=None):
        t = self.converter(record.created)
        prefix = time.strftime('%Y-%j-', t)
        return prefix + f'{3600*t.tm_hour + 60*t.tm_min + t.tm_sec:05d}'


@lru_cache(maxsize=None)
def fun_from_frame(frame):
    for o in gc.get_objects():
        if inspect.isfunction(o) and o.__code__ is frame.f_code:
            return o


class VNVLogger(lgg.Logger):

    def verbose(self, msg, *args, **kwargs):
        if self.isEnabledFor(VERBOSE):
            self._log(VERBOSE, msg, args, **kwargs)

    def _log(self, level, msg, args, **kwargs):
        # _log -> info, etc. -> true caller
        try:
            frame = inspect.currentframe().f_back.f_back  # type: ignore
            fun = fun_from_frame(frame)
        except AttributeError:
            fun = None

        if fun is not None:
            qname = fun.__qualname__
            module = fun.__module__
        else:
            qname = '-'
            module = '-'

        qname = compress_qualname(qname, module)

        kwargs['extra'] = kwargs.get('extra', {})
        kwargs['extra']['qname'] = qname

        return super()._log(level, msg, args, **kwargs)  # type: ignore


lgg.setLoggerClass(VNVLogger)


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
            s += f'<{n[1]}>'
        else:
            upper = compress_upcase(n)
            if upper:
                s += upper
            else:
                s += compress_snake_case(n)
        s += '.'

    return s + names[-1]


def setup_logger(logger, *, log_fn, log_level=lgg.INFO,
                 log_to_stdout=True, log_to_file=False, mode='a'):
    '''
    Sets up the logger. Should not be called directly.

    Arguments:
        log_to_stdout (bool): whether to log to stdout
        log_to_file (bool): whether to log to a file
        log_level (int): the log level to set the logger to.
        mode (str): the file open mode for the log file. "w" to overwrite any
            previous file with the given log name.
    '''

    logger.setLevel(lgg.INFO)

    if log_to_stdout:
        formatter = VNVFormatter(
            '{levelname:.1s} {asctime} {qname}: {message}',
            style='{',
        )

        handler = lgg.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_to_file:
        fn_fmt = VNVFormatter(
            '{levelname:.1s} {asctime} {qname}: {message}',
            style='{',
            do_color=False,
            strip_color=True,
        )

        fn_h = lgg.FileHandler(log_fn, mode=mode)
        fn_h.setFormatter(fn_fmt)
        logger.addHandler(fn_h)


def get_logger(name, *, log_fn=None, **kwargs):
    '''
    Retrieves a qqq-style logger with the given name.

    Arguments:
        log_fn (bool): if logging to a file, the file name to log to. If
            this is not given, a sensible default name in the current directory
            is chosen.
    '''
    log_fn = log_fn or f'./qlog_{osp.basename(name)}.txt'
    logger = lgg.getLogger(name)
    setup_logger(logger, log_fn=log_fn, **kwargs)
    return logger
