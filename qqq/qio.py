from collections import deque
import concurrent.futures as cfu
import functools as fun
import os.path as osp
import pickle as pkl
from queue import Queue, Empty
import time
from threading import Lock, RLock, Event, Thread
import traceback as trc
import sys

import lxml


def wr(*args):
    '''
    Convenience function to print something in the style of
    `sys.stdout.write(something+'\r')
    '''
    print(*args, end='\r', flush=True)


def p(fn, f, args=(), kwargs=None, d='.', ow=False, owa=(), owk=None):
    """ pickle cache manager. will cache the results of
        a particular computation to a file with the given
        name if it doesn't exist, else return the contents
        of the file with that name.

        No attempt to match computation to file is
        made in any way, that is for the user to track. """
    if kwargs is None:
        kwargs = {}
    if owk is None:
        owk = {}

    if callable(ow):
        do_ow = ow(*owa, **owk)
    elif isinstance(ow, bool):
        do_ow = ow

    assert isinstance(do_ow, bool),\
        'overwrite condition {} untenable'.format(do_ow)

    fp = osp.join(d, fn)
    if osp.isfile(fp) and not do_ow:
        with open(fp, 'rb') as f:
            return pkl.load(f)
    else:
        res = f(*args, **kwargs)
        with open(fp, 'wb') as f:
            pkl.dump(res, f)
        return res


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
    class implemented a basic asynchronous processing pipeline
    '''
    def __init__(self, input_q, work_func, output_q=None, collect_output=True,
                 ignore_none=True, halt_on_empty=True,
                 workers=16, use_processes=False, halt=None,
                 collect_output=True, allow_fail=None, verbose=False):
        '''
        a number of workers are spawned, and for each `arg` in `args`,
        `get_func` is executed asynchronously on one of these workers.

        in a separate worker pool, `process_func` is called once on each
        out-of-order output of the `get_func` swarm.

        the control loop itself is run in a separate thread, and this function
        returns instantly.

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
                    res = wf.result(timeout=0)
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


        

def scrape(get_func, args, process_func, get_workers=16,
           process_workers=4, sleep=0.05,
           allow_fail=[], verbose=False, get_mode='thread',
           process_mode='thread'):
    '''
    function implementing a basic asynchronous processing pipeline

    a number of workers are spawned, and for each `arg` in `args`, `get_func` is
    executed asynchronously on one of these workers.
    
    in a separate worker pool, `process_func` is called once on each
    out-of-order output of the `get_func` swarm.

    the control loop itself is run in a separate thread, and this function
    returns instantly.
    Arguments:
        get_func:
            function taking a single positional argument, returning
            an object `process_func` can accept.
        args:
            list of arguments to `get_func`. A single instance will be
            spawned for each arg in `args`.
        process_func:
            function which takes the output of `get_func` and does
            something useful with it, like storing it in a database.
        get_workers:
            maximum number of instances of `get_func` to run concurrently
        process_workers:
            maximum number of instances of `process_func` to run concurrently
        allow_fail:
            list of exception classes which when raised by `process_func`
            will be suppressed rather than terminate the operation.
        verbose:
            whether to print allowed exceptions when they arise.
        get_mode:
            "thread" or "process" - whether to put jobs in separate threads,
            or separate processes. The latter may not be possible in all
            circumstances, depending on get_function.
        process_mode:
            same as `get_mode`, but for the `process_func` pool
        sleep:
            time to sleep each time no data from `get_func`s is available
            for process_func.
    Returns:
        halt:
            `Event` handle. setting it to `True` with `halt.set()` will order
            the processing loop to cancel all pending instances of `get_func`
            and to shut down. instances of `get_func` which are running when
            `halt` is set will finish, but will **not** be processed!

            any pending `process_func` processes, however, will be allowed to
            run to completion. this is in line with the philosophy that
            `process_func` handles the caller's bookkeeping and should be run
            to maintain consistent books.
        futs:
            a set containing all the `concurrent.futures.Future`-wrapped
            instances of `get_func`, one per `arg` in `args`
    '''

    assert all([issubclass(e, Exception) for e in allow_fail])
    # simplifies the processing loop
    allow_fail.append(cfu.CancelledError)

    if get_mode == 'thread':
        get_exe = cfu.ThreadPoolExecutor
    elif get_mode == 'process':
        get_exe = cfu.ProcessPoolExecutor
    else:
        raise ValueError('invalid mode {}'.format(mode))

    if process_mode == 'thread':
        px_exe = cfu.ThreadPoolExecutor
    elif process_mode == 'process':
        px_exe = cfu.ProcessPoolExecutor
    else:
        raise ValueError('invalid mode {}'.format(mode))

    q = que.Queue()
    futs = set()

    halt = Event()
    re_init = Event()
    re_init.set()

    pfuts = deque([])
    
    gx = get_exe(max_workers=get_workers)
    px = px_exe(max_workers=process_workers)
    lx = cfu.ThreadPoolExecutor(max_workers=1)
    # executor for the processing loop, always threaded

    def _shutdown():
        if verbose:
            print('shutting down scraper')
        for f in futs:
            f.cancel()
        # we do not cancel processing futures, however
        gx.shutdown(False)
        px.shutdown(False)
        lx.shutdown(False)
    
    def queuer(arg):
        q.put(get_func(arg))

    def process_loop():
        nonlocal futs
        try:
            while not halt.is_set():

                # (re)initialize the worker tast set
                # run once when scrape is called
                # can be called again
                if re_init.is_set():
                    futs.clear()
                    for arg in args:
                        futs.add(gx.submit(queuer, arg))

                while not q.empty():
                    pfuts.appendleft(px.submit(process_func, q.get()))

                # if we have nothing to process, and the getter is also done
                # we're done
                if len(pfuts) == 0 and all([f.done() for f in futs]):
                    # not break to keep halt in a consistent state
                    halt.set()
                    continue

                # barrier
                pfuts.appendleft(None)
                while True:
                    pfut = pfuts.pop()
                    if pfut is None:
                        break
                    try:
                        _ = pfut.result(timeout=0)
                    # pending or running, keep it in the queue
                    except cfu.TimeoutError:
                        pfuts.appendleft(pfut)
                    # cancelled or failed as allowed, implicitly drop it
                    except Exception as e:
                        # nothing to see here
                        if type(e) in allow_fail:
                            continue
                        # uhoh
                        print('fatal: disallowed exception')
                        trc.print_exc()
                        halt.set()
                        break
                    except KeyboardInterrupt:
                        print('user shutdown')
                        halt.set()
                        break

                time.sleep(sleep)
        finally:
            _shutdown()

    for arg in args:
        futs.add(gx.submit(queuer, arg))

    lx.submit(process_loop)

    return halt, futs


class PoolThrottle:
    '''
    function call throttle, "X calls in Y seconds, sliding window" style

    constrains call flow to adhere to a "max X requests in Y seconds" schema.
    calls made in excess of this rate will block until there is room in the
    pool.

    the canonical use case is throttling network requests
    (i.e. `f` is `requests.get` or similar), with the time spent in GIL free IO
    dominating the time in each call.
    '''
    def __init__(self, f, X, Y, res=5e-2):
        '''
        Args:
            f:
                callable which will be throttled
            X:
                max requests allowed in Y seconds
            Y:
                span of time in which X requests are allowed
            res:
                time resolution with which the pool is refreshed
        '''
        self.f = f
        self.X = X
        self.Y = Y

        self._res = res
        self._pool = deque([])
        self._ifl = 0

        self._call_lock = Lock()
        self._ifl
        self._clean_lock = Lock()

    def _clean(self):
        # return instantly if we can't get the lock
        clean = self._clean_lock.acquire(False)
        if not clean:
            return False
        # if lock is not held by any thread we clean...
        if clean:
            now = time.time()
            while True:
                try:
                    if self._pool[-1] < now:
                        self._pool.pop()
                    else:
                        break
                except IndexError:
                    break
                finally:
                    # ... then sleep with the lock, guaranteeing
                    # other threads' calls to _clean will return
                    # for the duration of the sleep
                    time.sleep(self._res)
                    self._clean_lock.release()
                    return True


    def __call__(self, *args, **kwargs):
        while True:
            # thwart race conditions without locking call to f
            with self._call_lock:
                do_call = len(self._pool) < self.X
                if do_call:
                    self._pool.appendleft(time.time() + self.Y)
            if do_call:
                return self.f(*args, **kwargs)
            else:
                # if we're not sleeping with the clean lock
                # we ... just sleep. all threads should sleep once.
                if not self._clean():
                    time.sleep(self._res)


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
