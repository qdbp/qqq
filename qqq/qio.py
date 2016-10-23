import concurrent.futures as cfu
import functools as fun
import os.path as osp
import pickle as pkl
from queue import Queue, Empty
import time
from threading import Lock, RLock, Event, Thread
import traceback as trc
import sched
import sys

import lxml


def wr(*args):
    '''
    Convenience function to print something in the style of
    `sys.stdout.write(something+'\r')
    '''
    print(*args, end='\r', flush=True)


class SplitQueue:
    '''
    class splitting multiplt `queue.Queue` objects into several
    '''
    def __init__(self, input_q, n_branches, cond_func,
                 halt=None, _debug=False):
        '''
        Arguments:
            n_branches:
                number of output queues to split into. these queues are stored
                in a dict `output_qs` with integer keys in [0, `n_branches`)
            cont_func:
                function taking an object found in the input queue `input_q`
                and returning an integer [0, `n_branches`). this integer gives
                the key of output queue into which the object should be sorted.
            halt:
                a `threading.Event` object. when set, the loop will shut down.
                if `None`, a new internal `Event` will be instantiated. the
                splitqueue can still be shut down using the `halt` method.

                warning: if using a shared event object `e`, this will
                propagate the halt signal to all other object using `e`
        '''
        self.input_q = input_q
        self.n_branches = n_branches
        self.cond_func = cond_func

        self.output_qs = {i: Queue() for i in range(n_branches)}

        if halt is None:
            self._halt = Event()
        else:
            assert isinstance(halt, Event)
            self._halt = halt

        self._debug = _debug

        Thread(target=self._loop, daemon=True, name='split_loop').start()

    def halt(self):
        '''
        shut down the split queue
        '''
        self._halt.set()

    def _loop(self):
        while not self._halt.is_set():
            try:
                obj = self.input_q.get(timeout=0.2)
                k = self.cond_func(obj)
                self.output_qs[k].put(obj)
            except Empty:
                print('empty', self._halt.is_set())
                continue

    def __getitem__(self, key):
        '''
        to support eas(ier) named assignment, e.g.:

        >>> first, second = SplitQueue(in, 2, cond)[:]
        '''
        t = tuple(self.output_qs[i] for i in range(self.n_branches))
        return t.__getitem__(key)


class MergeQueue:
    '''
    class merging multiple `queue.Queue` objects into one

    relative order of objects within any input queue is preserved in the output
    queue
    '''
    def __init__(self, input_qs, halt=None):
        '''
        Arguments:
            input_qs:
                `queue.Queue` objects which to aggregate
            halt:
                a `threading.Event` object. when set, the loop will shut down.
                if `None`, a new internal `Event` will be instantiated. the
                splitqueue can still be shut down using the `halt` method.
        '''
        assert all([isinstance(iq, Queue) for iq in input_qs])
        self.input_qs = input_qs
        self.output_q = Queue()

        if halt is None:
            self._halt = Event()
        else:
            assert isinstance(halt, Event)
            self._halt = halt
            
        self._poll_lag = 0.05

        Thread(target=self._loop, daemon=True).start()

    def halt(self):
        self._halt.set()

    def _loop(self):
        while not self._halt.is_set():
            for iq in self.input_qs:
                while True:
                    try:
                        self.output_q.put(iq.get(timeout=self._poll_lag))
                    except Empty:
                        break


class Scraper:
    '''
    class implementing a basic multithreaded processing pipeline
    '''
    def __init__(self, input_q, work_func, output_q=None, collect_output=True,
                 ignore_none=True, halt_on_empty=True,
                 workers=16, use_processes=False, halt=None,
                 allow_fail=None, verbose=False):
        '''
        a number of workers are spawned, and for each `arg` in `args`,
        `work_func` is executed asynchronously on one of these workers.

        the control loops are run in a separate thread, and all workers are
        daemonic. this means that invocation of `run` return does not block,
        and work will be aborted if the main threads returns.

        no verification of what work has actually completed is implemented.
        this might be changed in the future if compelling cases where such
        logic cannot be feasibly implemented in work_func itself (for instance
        where there is some nontrivial cost to invoking work_func on a given
        argument more than once, even across different program invocations).

        Arguments:
            input_q:
                a `queue.Queue` instance, or an iterable. if iterable, all of
                its elemented will be added to a new `queue.Queue` instance
                accessible as `scraper.input_q`.
                this is the way the scraper instance receives work to do.
                `input_q` will be polled, and its contents gotten and fed into
                `worker_func`
            work_func:
                function to be called once for each argument passed to the
                object through `add_arg` or `add_args`.
            workers:
                maximum number of instances of `work_func` to be run together.
            allow_fail:
                list of exception classes which when raised by `work_func`
                will be suppressed rather than terminate the operation.
            halt:
                a `threading.Event` object. when set, the loop will shut down.
                if `None`, a new internal `Event` will be instantiated. the
                splitqueue can still be shut down using the `halt` method.
            verbose:
                whether to print allowed exceptions when they arise.
            use_processes:
                if `True`, `work_func` workers run in separate processes. else,
                they run in separate threads. may not work with all types of
                `work_func` if `True`.
            halt_on_empty:
                if `True`, the scraper will shut down when all arguments have
                been exhausted. once shutdown is triggered, adding more
                arguments to the input queue will not restart the scraper
            collect_output:
                if `True`, the return values of `work_func` will be made
                available, in order of completion. should be set to False
                if `work_func` operates through side effects exclusively.
                if `True`, the return values of `work_func` can be accessed
                through `scraper.outs_q`
            ignore_none:
                if `True`, `None` return values will not be put in the
                output queue, and will be tacitly dropped instead.
        '''

        self.f = work_func
        self.workers = workers

        if allow_fail is None:
            allow_fail = []
        else:
            assert all([issubclass(e, Exception) for e in allow_fail])
        # simplifies the processing loop
        allow_fail.append(cfu.CancelledError)
        self.allow_fail = allow_fail

        self.verbose = verbose

        if use_processes:
            _work_exe = cfu.ProcessPoolExecutor
        else:
            _work_exe = cfu.ThreadPoolExecutor

        if isinstance(input_q, Queue):
            self.input_q = input_q
        else:
            self.input_q = Queue()
            for arg in input_q:
                self.input_q.put(arg)

        # not Queue so we can use cfu.wait then set difference with 'done'
        self._wfut_s = set()
        self._wfut_lock = RLock()

        if output_q is not None:
            assert isinstance(output_q, Queue)
            self.output_q = output_q
        elif collect_output:
            self.output_q = Queue()
        else:
            self.output_q = None
        self.collect_output = collect_output
        self.ignore_none = ignore_none

        self._wx = _work_exe(max_workers=self.workers)

        if halt is None:
            self._halt = Event()
        else:
            assert isinstance(halt, Event)
            self._halt = halt

        self._sleep_in = 0.2
        self._sleep_out = 0.2

        self._in_t = Thread(target=self._in_loop, daemon=True,
                            name='scraper_in')
        self._out_t = Thread(target=self._out_loop, daemon=True,
                             name='scraper_out')

    def run(self):
        self._in_t.start()
        self._out_t.start()
        return self

    def halt(self):
        self._halt.set()
        if self.verbose:
            print('shutting down scraper')
        with self._wfut_lock:
            for f in self._wfut_s:
                f.cancel()
        self._wx.shutdown(True)

    def _in_loop(self):
        while not self._halt.is_set():
            try:
                arg = self.input_q.get_nowait()
                wf = self._wx.submit(self.f, arg)
                with self._wfut_lock:
                    self._wfut_s.add(wf)
            except Empty:
                time.sleep(self._sleep_in)
            except Exception:
                print('warning: unexpected exception in _in_loop',
                      file=sys.stderr)
        else:
            self.halt()

    def _out_loop(self):
        while not self._halt.is_set():
            with self._wfut_lock:
                done, pend = cfu.wait(self._wfut_s, timeout=0)
                self._wfut_s -= done
            for wf in done:
                try:
                    res = wf.result()
                    if self.collect_output and not (self.ignore_none and res is None):
                        self.output_q.put(res)
                except Exception as e:
                    if all([not isinstance(e, af) for af in self.allow_fail]):
                        print('fatal: disallowed exception')
                        trc.print_exc()
                        self.halt()
                        break
                    elif self.verbose:
                        print('allowed exception raised:')
                        trc.print_exc()
            time.sleep(self._sleep_out)
        else:
            self.halt()


class PoolThrottle:
    '''
    Function call throttle, "X calls in Y seconds, sliding window" style.

    Constrains call flow to adhere to a "max X requests in Y seconds" schema.
    Calls made in excess of this rate will block until there is room in the
    pool.

    Functions to be throttled should be decorated directly by an instance of
    this class. Multiple functions decorated by the same instance will share
    the same pool.
    '''
    def __init__(self, X, Y, res=1e-1):
        '''
        Args:
            X:
                max requests allowed in Y seconds
            Y:
                span of time in which X requests are allowed
            res:
                time resolution with which call attempts are made if the pool
                is full
        '''
        self.X = X
        self.Y = Y
        self.res = res

        self._sched = sched.scheduler()
        # current number of calls "in the pool"
        self._ifl = 0
        self._ifl_lock = RLock()

    def _dec_ifl(self):
        with self._ifl_lock:
            self._ifl -= 1

    def __call__(self, f):
        @fun.wraps(f)
        def wrapped(*args, **kwargs):
            while True:
                # check for decrements
                self._sched.run()
                with self._ifl_lock:
                    do_call = self._ifl < self.X
                    if do_call:
                        self._ifl += 1
                if do_call:
                    out = f(*args, **kwargs)
                    self._sched.enter(self.Y, 1, self._dec_ifl)
                    return out
                else:
                    time.sleep(self.res)
        return wrapped


class FunctionPrinter:
    def __init__(self, tab_depth=4):
        self.depth = 0
        self.tab_depth = 4
        self.cur_fn = None
        self.fn_cache = {}

        self.fresh_line = True

    def decorate(self, f):
        @fun.wraps(f)
        def wrapped(*args, **kwargs):
            self.depth += 1
            sys.stdout = self
            self.fn_cache[self.depth] = f.__name__
            try:
                return f(*args, **kwargs)
            finally:
                self.depth -= 1
                if self.depth == 0:
                    sys.stdout = sys.__stdout__
        return wrapped

    def write(self, s):
        if self.fresh_line:
            sys.__stdout__.write('{}{}: {}'.format(' '*(self.depth-1)*self.tab_depth,
                                                   self.fn_cache[self.depth],
                                                   s)
                                )
        else:
            sys.__stdout__.write(s)

        if '\n' in s:
            self.fresh_line = True
        else:
            self.fresh_line = False

    def __getattr__(self, attr):
        return getattr(sys.__stdout__, attr)
