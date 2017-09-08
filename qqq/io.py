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
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Tuple, TypeVar, Generic, Union)

from .qlog import get_logger
from .util import ensure_type

LOG = get_logger(__file__)

T = TypeVar('T')
A = TypeVar('A', bound=Hashable)
B = TypeVar('B', bound=Hashable)
V = TypeVar('V')


def iterable_to_q(iterable: Iterable[T]):
    q = Queue()  # type: Queue[T]
    for item in iterable:
        q.put(item)
    return q


class Controller:
    '''
    Adds a control loop and halt handle.

    The control loop runs in a separate thread, starting when the `run`
    method is invoked. It will stop once the `halt` method is called.

    The control loop can be joined with the `join` method. Note this will
    block indefinitely unless the loop has been halted.
    '''

    def __init__(
            self, *,
            halt: Event=None,
            min_delay: float=0.001,
            **kwargs) -> None:
        '''
        Arguments:
            halt:
                The Event object to use as the halting handle. If `None`,
                a new Event is instantiated.
            min_delay:
                Minimum number of seconds to wait between control loop passes.
        '''
        self._control_thread = Thread(target=self.control_loop, daemon=True)
        self._halt = ensure_type(halt, Event)
        self.min_delay = min_delay

        if not hasattr(self, 'subordinates'):
            self.subordinates = []  # type: List[Controller]

    def halt(self):
        '''
        Halt the control loop and all subordinates.

        Subordinates are halted after the main loop.
        '''

        LOG.verbose(f'halting {self}')
        self._halt.set()

        for subordinate in reversed(self.subordinates):
            subordinate.halt()

    def run(self):
        '''
        Starts the control thread and subordinate controllers.

        Subordinate controllers are started after the control loop.
        '''

        LOG.verbose(f'starting {self}')
        self._control_thread.start()

        for subordinate in self.subordinates:
            subordinate.run()

    def join(self, timeout=None):
        '''
        Joins the subordinate controllers and the control loop.

        Subordinates are joined first, in reverse order from their start
        order.
        '''

        for subordinate in reversed(self.subordinates):
            subordinate.join()

        LOG.verbose(f'waiting to join {self}')
        self._control_thread.join(timeout=timeout)

    def control_loop(self):
        '''
        Initiates the control loop.
        '''

        while not self._halt.is_set():
            t = time.time()

            self._control_loop()

            d = self.min_delay + t - time.time()
            if d > 0:
                time.sleep(d)

    @abstractmethod
    def _control_loop(self):
        '''
        Single pass of the control loop to be run in the control thread.
        '''


class ManyToManyQueueMux(Controller, Generic[A, B, V]):
    '''
    Moves output from mulitple queues to multiple queues by given function.
    '''

    def __init__(
            self, *,
            input_qs: Union[Iterable[Queue], Dict[A, Queue]],
            output_qs: Union[Iterable[Queue], Dict[B, Queue]],
            cond_func: Callable[[A, V], Optional[Tuple[B, V]]],
            greedy: bool=True,
            delay: float=0.1,
            timeout: float=0.03,
            **kwargs) -> None:
        '''
        Arguments:
            input_qs:
                An input queue or collection of input Queues.  If an iterable,
                the keys will be the integer indices of the iterable.  If
                single queue will be treated like a singleton iterable.
            output_qs:
                An output queue or collection of output Queues.  If an
                iterable, the keys will be the integer indices of the iterable.
                If single queue will be treated like a singleton iterable.
            cond_func:
                A function which takes an input key and a value from the queue
                and produces an output key and a list of new value. Should not
                be an expensive computation, as it will be run in the single
                work thread. If `None` is returned as the key, the output
                will be placed in the output queue with the fewest elements.
            greedy:
                If `True`, will use a greedy strategy, emptying each input
                queue fully before moving on to the next. Otherwise will
                alternate strictly between input Queues, up to timeout.
            timeout:
                Timetout used for internal queue operations. Smaller values
                increase responsiveness at the expense of busywait CPU usage.
            delay:
                If an output queue is full, will attempt reinsertion after
                `delay` seconds.
        '''

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

        super().__init__(**kwargs)

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


class WorkPipe(Controller):
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
        self.unpack = unpack
        self.work_func = work_func
        self.discard = discard
        self.discard_empty = discard_empty

        self.input_q = ensure_type(input_q, Queue)
        self.output_q = ensure_type(output_q, Queue, maxsize=limit)

        self._is_gen = isgenerator(work_func)

        super().__init__(**kwargs)

    def _control_loop(self):
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
            LOG.error(f'exception in worker thread', exc_info=True)


class ConcurrentProcessor(Controller):
    '''
    Runs an operation on inputs from a Queue in multiple threaded workers.

    The output is optionally collected to an output Queue.
    '''

    def __init__(
            self, *,
            input_q: Queue,
            work_func: Queue,
            output_q: Queue=None,
            discard: bool=False,
            collect_empty: bool=False,
            unpack: bool=False,
            n_workers: int=8,
            buffer_size: int=5,
            timeout: float=1.,
            **kwargs) -> None:
        '''
        Arguments:
            input_q:
                Queue from which inputs to the work function are drawn.
            work_func:
                The function to call on the inputs.
            output_q:
                The output Queue. If `None` and `collect_output`, a new
                Queue will be instantiated.
            dicard:
                If true, `work_func` output will be discarded. Otherwise,
                it is put in `output_q` in order of completion.
            collect_emtpy:
                If true, and `collect_output` is true, function output that
                evaluates to `False` (e.g. empty dicts) will also be collected.
                By default, such output is discarded.
            n_workers:
                Number of work threads to spawn.
            buffer_size:
                Number of inputs to buffer, per worker thread.
            unpack:
                If true, the work thread will call `work_func(*arg)` where
                arg is the object retreived from `input_q`. Otherwise
                will call `work_func(arg)`.
            timeout (float): timeout for reads on the input_q. Only affects
                the delay between issuing a halt and cessation of processing,
                so is defaulted to a fairly large value to minimize
                busywaiting.
        '''

        super().__init__(**kwargs)

        self.work_func = work_func
        self.input_q = ensure_type(input_q, Queue)

        self.discard = discard
        if not self.discard:
            self.output_q = ensure_type(output_q, Queue)
        else:
            self.output_q = None
        self.collect_empty = collect_empty

        self.timeout = timeout
        self.n_workers = n_workers
        self.buffer_size = buffer_size
        self.unpack = unpack

        self.expander = QueueExpander(
            limit=buffer_size,
            input_q=self.input_q,
            n_outputs=n_workers,
            balance='rr',
        )

        self.work_pipes = [
            WorkPipe(
                input_q=win_q,
                discard=discard,
                work_func=work_func,
                unpack=unpack
            )
            for win_q in self.expander.output_qs
        ]

        self.condenser = QueueCondenser(
            input_qs=[wp.output_q for wp in self.work_pipes],
            output_q=self.output_q,
        )

        self.subordinates =\
            [self.expander] + self.work_pipes + [self.condenser]

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
