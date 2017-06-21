# import pyximport; pyximport.install()  # noqa
# from .qiox import flif_to_rgba_arr  # noqa

import sched
import time
from abc import abstractmethod
from functools import wraps
from inspect import isgeneratorfunction as isgenerator
from itertools import cycle
from queue import Empty, Full, PriorityQueue, Queue
from random import choice, randrange
from threading import Event, Lock, Thread
from typing import Any, Iterable, List, TypeVar

from .qlog import get_logger
from .util import ensure_type

log = get_logger(__file__)

T = TypeVar('T')


def iterable_to_q(iterable: Iterable[T]):
    q = Queue()  # type: Queue[T]
    for item in iterable:
        q.put(item)
    return q


class HaltMixin:
    '''
    Adds a _halt handle.
    '''

    def __init__(self, *, halt=None, **kwargs):
        '''
        Args:
            halt (None | threading.Event):
                When set, the loop will shut down.  If `None`, a new internal
                `Event` will be instantiated.
        '''

        self._halt = ensure_type(halt, Event)
        # XXX mypy doesn't like mixins? or am I doing something dumb?
        super().__init__(**kwargs)  # type: ignore

    def halt(self):
        '''
        Halts the processor.
        '''
        self._halt.set()


class ControlThreadMixin:
    '''
    Adds a run method and control thread.

    Requires _run_target method.
    '''

    def __init__(self, **kwargs):

        self._control_thread = Thread(target=self._control_loop, daemon=True)

        super().__init__(**kwargs)  # type: ignore

    def run(self):
        '''
        Starts the control thread.
        '''
        self._control_thread.start()

    def join(self, timeout=None):
        self._control_thread.join(timeout=timeout)

    @abstractmethod
    def _control_loop(self):
        '''
        Function to be exectued in control thread.
        '''


class ManyToManyQueueMux(HaltMixin, ControlThreadMixin):
    '''
    Moves output from mulitple queues to multiple queues by given function.
    '''

    def __init__(self, *, input_qs, output_qs, cond_func,
                 greedy=True, delay=0.1, timeout=0.03, **kwargs):
        '''
        Arguments:
            input_qs (Queue | Iterable[Queue] | Dict[IKey, Queue]):
                An input queue or collection of input Queues.  If an iterable,
                the keys will be the integer indices of the iterable.  If
                single queue will be treated like a singleton iterable.
            output_qs (Queue | Iterable[Queue] | Dict[OKey, Queue]):
                An output queue or collection of output Queues.  If an
                iterable, the keys will be the integer indices of the iterable.
                If single queue will be treated like a singleton iterable.
            cond_func (Callable[Tuple[IKey, Value],
                                (None | Tuple[OKey, Value])]):
                A function which takes an input key and a value from the queue
                and produces an output key and a list of new value. Should not
                be an expensive computation, as it will be run in the single
                work thread. If `None` is returned as the key, the output
                will be placed in the output queue with the fewest elements.
            greedy (bool):
                If `True`, will use a greedy strategy, emptying each input
                queue fully before moving on to the next. Otherwise will
                alternate strictly between input Queues, up to timeout.
            timeout (float):
                Timetout used for internal queue operations. Smaller values
                increase responsiveness at the expense of busywait CPU usage.
            delay (float):
                If an output queue is full, will attempt reinsertion after
                `delay` seconds.
        '''
        super().__init__(**kwargs)

        self.input_qs, self.input_q_map = self.__rectify_q_arg(input_qs)
        self.output_qs, self.output_q_map = self.__rectify_q_arg(output_qs)

        self.cond_func = cond_func

        self.greedy = greedy

        self.delay = delay
        self.timeout = timeout

        self._in_cycle = iter(cycle(self.input_q_map.items()))
        self._prio = PriorityQueue(
            maxsize=max(len(self.input_qs), len(self.output_qs))
        )

    def _oq_from_key(self, out_key):
        if out_key is not None:
            return self.output_q_map[out_key]
        else:
            candidates = sorted(
                [(oq.qsize(), oq)
                 for oq in self.output_q_map.values()
                 if not oq.full()]
            )
            if candidates:
                return candidates[0][1]
            else:
                return choice(self.output_qs)

    def __rectify_q_arg(self, q_arg):
        if isinstance(q_arg, dict):
            out = list(q_arg.values())
            out_map = q_arg
        elif isinstance(q_arg, Queue):
            out = [q_arg]
            out_map = {0: q_arg}
        elif isinstance(q_arg, (list, tuple)):
            out = list(q_arg)
            out_map = {ix: q for ix, q in enumerate(q_arg)}

        return out, out_map

    def _control_loop(self):
        while not self._halt.is_set():
            # read phase
            while True:
                if self._prio.full():
                    break

                key, in_q = next(self._in_cycle)
                try:
                    obj = in_q.get(timeout=self.timeout)
                except Empty:
                    break

                out_key, out = self.cond_func(key, obj)
                self._prio.put_nowait((time.time(), (out_key, out)))
            # flush phase
            for i in range(self._prio.qsize()):
                try:
                    due, (out_key, val) = self._prio.get_nowait()
                    self._oq_from_key(out_key).put(val, timeout=self.timeout)
                except Empty:
                    break
                except Full:
                    self._prio.put_nowait((due + self.delay, (out_key, val)))


class QueueExpander(ManyToManyQueueMux):
    '''
    Splits a queue evenly.
    '''

    def __init__(self, *, input_q, n_outputs,
                 balance='min', limit=0, **kwargs):
        '''
        Arguments:
            limit (int): maxsize for the output_qs.
            balance ('rr' | 'min' | 'rand'): If 'rr', distributes intput
                to outputs by a round robin scheme. If 'rand', assigns
                outputs independently uniformly at random. If 'min', assigns
                inputs to a queue with the minimum number of items.
        '''

        output_qs = [
            Queue(maxsize=limit) for x in range(n_outputs)
        ]  # type: List[Queue[Any]]
        self._rr_ix = 0

        def expander(key, obj):
            # exploit mtmqm behaviour
            if balance == 'min':
                return None, obj
            elif balance == 'rand':
                return randrange(n_outputs), obj
            elif balance == 'rr':
                out = self._rr_ix, obj
                self._rr_ix = (self._rr_ix + 1) % n_outputs
                return out
            else:
                raise ValueError(f'unknown balancing scheme {balance}')

        super().__init__(input_qs=[input_q],
                         output_qs=output_qs,
                         cond_func=expander,
                         **kwargs)


class QueueCondenser(ManyToManyQueueMux):
    '''
    Condenses many queues to one.
    '''

    def __init__(self, *, input_qs, output_q=None, limit=0, **kwargs):

        output_q = ensure_type(output_q, Queue, maxsize=limit)

        def condenser(key, obj):
            return 0, obj

        super().__init__(input_qs=input_qs,
                         output_qs=[output_q],
                         cond_func=condenser,
                         **kwargs)


class WorkPipe(HaltMixin, ControlThreadMixin):
    '''
    Turns a function into a work pipe between two queues.

    Detects and drains generators automatically.
    '''

    def __init__(self, *, input_q, work_func, output_q=None,
                 unpack=False, discard=False, discard_empty=True,
                 limit=0, **kwargs):
        '''
        Arguments:
            input_q: input Queue from which arguments are read
            work_func: function to call on the arguments
            output_q: output Queue in which to place results. If None, a new
                Queue is created.
            halt: Event object which when set will stop execution
            unpack: if True, the function will be called as `f(*args)`, where
                args is the value received on input_q. Otherwise, will be
                called as `f(args)`
            discard: if `True`, function outputs will not be collected at all.
            discard_empty: if `True`, function outputs will not be collected
                if they evaluate to `False`.
            limit: the function will block if there are this many or more
                outputs in the output queue already. Same semantics as for
                `Queue(maxsize=)`
        '''

        super().__init__(**kwargs)

        self.unpack = unpack
        self.work_func = work_func
        self.discard = discard
        self.discard_empty = discard_empty

        self.input_q = ensure_type(input_q, Queue)
        self.output_q = ensure_type(output_q, Queue, maxsize=limit)

        self._is_gen = isgenerator(work_func)

    def _control_loop(self):
        while not self._halt.is_set():
            try:
                args = self.input_q.get()
                if self.unpack:
                    out = self.work_func(*args)
                else:
                    out = self.work_func(args)

                if not self.discard:
                    if self._is_gen:
                        for obj in out:
                            if bool(obj) or not self.discard_empty:
                                self.output_q.put(obj)
                    elif bool(out) or not self.discard_empty:
                        self.output_q.put(out)

            except Exception:
                log.error(f'exception in worker thread', exc_info=True)
                self.halt()


class QueueProcessor(HaltMixin):
    '''
    A basic scraper with multiple workers.
    '''

    def __init__(self, *, input_q, work_func, n_workers, output_q=None,
                 unpack=False, input_limit=0, output_limit=0, prio=False,
                 discard=False, **kwargs):
        '''
        Arguments:
            input_limit (int): limit argument for the expander.
            prio (bool): use a priority queue on the input.
        '''

        super().__init__(**kwargs)

        if not prio:
            self.input_q = ensure_type(input_q, Queue)
        else:
            self.input_q = ensure_type(input_q, PriorityQueue)

        self.output_q = ensure_type(output_q, Queue, maxsize=output_limit)

        self.expander = QueueExpander(
            limit=input_limit,
            halt=self._halt,
            input_q=self.input_q,
            n_outputs=n_workers,
            balance='rr',
        )

        self.work_pipes = [
            WorkPipe(input_q=win_q, halt=self._halt, discard=discard,
                     work_func=work_func, unpack=unpack)
            for win_q in self.expander.output_qs
        ]

        self.condenser = QueueCondenser(
            input_qs=[wp.output_q for wp in self.work_pipes],
            output_q=self.output_q,
            halt=self._halt,
        )

    def run(self):
        self.expander.run()
        for wp in self.work_pipes:
            wp.run()
        self.condenser.run()

    def join(self, timeout=None):
        self.expander.join(timeout=timeout)
        for wp in self.work_pipes:
            wp.join(timeout=timeout)
        self.condenser.join(timeout=timeout)


class ConcurrentProcessor(HaltMixin, ControlThreadMixin):

    def __init__(self, *, input_q, work_func, output_q=None,
                 collect_output=True, collect_empty=False,
                 n_workers=8, expand_arg=False,
                 timeout=1., **kwargs):
        '''
        Arguments:
            input_q (Queue): the work functions get inputs from this queue
            work_func (Callable): the function to perform on the inputs
            output_q: (None | Queue): the output Queue. If None and
                `collect_output`, a new Queue will be instantiated.
            collect_output (bool): if True, `work_func` output will be put
                on `output_q` in order of completion. If false, it will be
                discarded.
            collect_emtpy (bool): if False, `work_func` output that evaluates
                to False (e.g. empty dicts) will not be added to the output
                queue. Has no effect if `collect_output` is `False`.
            n_workers (int): number of work threads to spawn
            expand_arg (bool): if True, the work thread will call
                `work_func(*arg)` where arg is the object `get`ted from the
                input_q. Otherwise will call `work_func(arg)`.
            timeout (float): timeout for reads on the input_q. Only affects
                the delay between issuing a halt and cessation of processing,
                so is defaulted to a fairly large value to minimize
                busywaiting.
        '''

        super().__init__(**kwargs)

        self.work_func = work_func
        self.input_q = ensure_type(input_q, Queue)

        self.collect_output = collect_output
        if self.collect_output:
            self.output_q = ensure_type(output_q, Queue)
        else:
            self.output_q = None
        self.collect_empty = collect_empty

        self.timeout = timeout
        self.n_workers = n_workers
        self.expand_arg = expand_arg

        self._work_threads = [
            Thread(daemon=True, target=self._thread_target)
            for x in range(n_workers)
        ]

    def _control_loop(self):
        for t in self._work_threads:
            t.start()
        for t in self._work_threads:
            t.join()

    def _thread_target(self):
        while not self._halt.is_set():
            try:
                arg = self.input_q.get(timeout=self.timeout)
            except Empty:
                continue

            if self.expand_arg:
                out = self.work_func(*arg)
            else:
                out = self.work_func(arg)

            if self.collect_output and (self.collect_empty or bool(out)):
                print(f'collecting {out}')
                self.output_q.put(out)


class PoolThrottle:
    '''
    Function call throttle, "X calls in Y seconds" style.

    Constrains call flow to adhere to a "max X requests in Y seconds" schema.
    Calls made in excess of this rate will block until there is room in the
    pool. The pool is implemented as a sliding window.

    Functions to be throttled should be decorated directly by an instance of
    this class. Multiple functions decorated by the same instance will share
    the same pool.

    Call order is preserved.
    '''

    def __init__(self, *, pool: int, window: int, strict=False) -> None:
        '''
        Args:
            pool:
                max requests allowed in a sliding window of `window` seconds
            window:
                sliding span of time in which `pool` requests are allowed
            res:
                time resolution with which call attempts are made if the pool
                is full
            strict:
                if `True`, the `window`-second timer after a call will start
                only after it returns, else it will start right before the call
                is made. A strict pool effectively makes X calls every (Y +
                average_call_time) seconds.
        '''
        self.pool = pool
        self.window = window
        self.strict = strict

        self._sched = sched.scheduler()
        # current number of calls "in flight"
        # calls will block unless a synchronized read of _ifl returns < Y
        # in which case _ifl will be incremented, and a decrement scheduled to
        # occur in Y seconds
        self._ifl = 0
        self._now_serving = 0
        self._ifl_lock = Lock()

        self._next_number = 0
        self._next_number_lock = Lock()

    def _dec_ifl(self):
        with self._ifl_lock:
            self._ifl -= 1

    def _take_a_number(self):
        with self._next_number_lock:
            n = self._next_number
            self._next_number += 1
            return n

    def __call__(self, f):
        @wraps(f)
        def throttled(*args, **kwargs):
            number = self._take_a_number()
            while True:
                # check for decrements
                wait = self._sched.run(blocking=False)
                # don't do actual function call with the lock

                with self._ifl_lock:
                    do_call = (self._ifl < self.pool and
                               number == self._now_serving)
                    if do_call:
                        self._ifl += 1
                        self._now_serving += 1
                        if not self.strict:
                            self._sched.enter(self.window, 1, self._dec_ifl)
                if do_call:
                    try:
                        return f(*args, **kwargs)
                    finally:
                        if self.strict:
                            self._sched.enter(self.window, 1, self._dec_ifl)
                else:
                    time.sleep(1e-2 if wait is None else wait)
        return throttled
