from collections import deque
import concurrent.futures as cfu
import functools as fun
import os.path as osp
import pickle as pkl
import queue as que
import time
from threading import Lock, Event
import traceback as trc
import sys


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
    '''
    # TODO: async process_func
    '''
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
        try:
            while not halt.is_set():
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
        self._next_clean = 0

        self._call_lock = Lock()
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

if __name__ == '__main__':
    fp = FunctionPrinter()

    @fp.decorate
    def func():
        sys.stdout.write('printing func!\n')

    @fp.decorate
    def gunc():
        sys.stdout.write('printing gunc 1!\n')
        func()
        sys.stdout.write('printing gunc 2 without newline...')
        sys.stdout.write('... still new stuff ...')
        sys.stdout.write('... and done!\n') 
        print('using print in gunc!')

    @fp.decorate
    def hunc():
        sys.stdout.write('printing hunc 1!\n')
        gunc()
        junc()
        func()
        raise ValueError()
        gunc()
        sys.stdout.write('printing hunc 2!\n')

    def junc():
        sys.stdout.write('unwrapped function junc!\n')

    def kunc():
        sys.stdout.write('unwrapped function kunc 1, will call hunc, junc!\n')
        try:
            hunc()
        except Exception as e:
            print('got exception {}'.format(e))
        junc()
        gunc()
        sys.stdout.write('unwrapped function kunc 2\n')

    kunc()
