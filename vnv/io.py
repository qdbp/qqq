'''
Cooperative multithreading is the superi...

Imma let you finish but, preemptive is where it's at.

Preemptive threads are the original containers.
'''
import sched
import time
from abc import abstractmethod, abstractproperty
from collections import deque
from functools import wraps
from itertools import cycle
from queue import Empty, Full, PriorityQueue, Queue
from random import randrange
from threading import Event, Lock, Thread
from typing import (Any, Callable, Dict, Iterable, List, Optional,
                    Tuple, TypeVar, Generic, Union, Deque, Iterator)

import pdb
from .log import get_logger
from .util import ensure_type

LOG = get_logger(__file__)

T = TypeVar('T')
# XXX: https://github.com/python/mypy/issues/3150
A = TypeVar('A')  # , bound=Hashable)
B = TypeVar('B')  # , bound=Hashable)
V = TypeVar('V')

X = TypeVar('X')
Y = TypeVar('Y')


def iterable_to_q(iterable: Iterable[T]):
    q = Queue()  # type: Queue
    for item in iterable:
        q.put(item)
    return q


def iter_q(q: Queue) -> Iterator[Any]:
    while True:
        try:
            yield q.get_nowait()
        except Empty:
            return


class Controller:
    '''
    Mixin. Adds a control loop and halt handle.

    The control loop runs in a separate thread, starting when the `run`
    method is invoked. It will stop once the `halt` method is called.

    The control loop can be joined with the `join` method. Note this will
    block indefinitely unless the loop has been halted.

    A controller object may have any number of subordinate controllers. These
    will be automatically run and halted after and before the parent's main
    loop, respectively.
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
        self.min_delay = min_delay

        self._control_thread = Thread(target=self.control_loop, daemon=True)
        self._halt = ensure_type(halt, Event)
        self._subordinates: List[Controller] = []

    def add_subordinate(self, subordinate: "Controller") -> None:
        self._subordinates.append(subordinate)

    def halt(self) -> None:
        '''
        Halt the control loop and all subordinates.

        Subordinates are halted before the main loop.
        '''

        for subordinate in reversed(self._subordinates):
            subordinate.halt()

        LOG.verbose(f'halting {self}')
        self._halt.set()

    def run(self) -> None:
        '''
        Starts the control thread and subordinate controllers.

        Subordinate controllers are started after the control loop.
        '''

        LOG.verbose(f'starting {self}')
        self._control_thread.start()

        for subordinate in self._subordinates:
            subordinate.run()

    def join(self, timeout=None) -> None:
        '''
        Joins the subordinate controllers and the control loop.

        Subordinates are joined first, in reverse order from their start
        order.
        '''

        for subordinate in reversed(self._subordinates):
            subordinate.join()

        LOG.verbose(f'waiting to join {self}')
        self._control_thread.join(timeout=timeout)

    def control_loop(self) -> None:
        '''
        Initiates the control loop.
        '''

        while not self._halt.is_set():
            t = time.time()

            try:
                self._control_loop()
            except Exception:
                LOG.error(f'exception in worker thread', exc_info=True)

            dt = time.time() - t
            if dt < self.min_delay:
                time.sleep(self.min_delay - dt)

    def _control_loop(self) -> None:
        '''
        Single pass of the control loop to be run in the control thread.

        The default is suitable for "container" Controllers which do work
        through subordinates only.
        '''
        time.sleep(0.1)


class ManyToManyQueueMux(Controller, Generic[A, B, V]):
    '''
    Moves output from mulitple queues to multiple queues by given function.
    '''

    def __init__(
            self, *,
            input_q_map: Dict[A, Queue],
            output_q_map: Dict[B, Queue],
            cond_func: Callable[[A, V], Iterable[Tuple[Optional[B], V]]],
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
                and produces a list of output keys (output key, new value)
                pairs. Should not be an expensive computation, as it will be
                run in the single work thread. If `None` is returned as the
                key, the output will be placed in the output queue with the
                fewest elements.
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

        self.input_q_map = input_q_map
        self.output_q_map = output_q_map

        self.cond_func = cond_func

        self.greedy = greedy

        self.delay = delay
        self.timeout = timeout

        self._in_cycle = iter(cycle(self.input_q_map.items()))

        self._prio = PriorityQueue()
        self._prio_max = 2 * max(len(self.input_q_map), len(self.output_q_map))

        Controller.__init__(self, **kwargs)

    def __output_q_from_key(self, out_key: Optional[B]) -> Queue:
        if out_key is not None:
            return self.output_q_map[out_key]
        else:
            return sorted(
                self.output_q_map.values(), key=lambda oq: oq.qsize()
            )[0]

    def _control_loop(self) -> None:
        # read phase
        while True:
            if self._prio.qsize() > self._prio_max:
                break
            key, in_q = next(self._in_cycle)
            try:
                obj = in_q.get(timeout=self.timeout)
            except Empty:
                break

            for out_key, out in self.cond_func(key, obj):
                self._prio.put_nowait((time.time(), (out_key, out)))

        # flush phase
        while True:
            try:
                due, (out_key, val) = self._prio.get_nowait()
                self.__output_q_from_key(out_key).put(
                    val, timeout=self.timeout,
                )
            except Empty:
                break
            except Full:
                self._prio.put_nowait((due + self.delay, (out_key, val)))

    def set_input_q(self, key: A, q: Queue) -> None:
        '''
        Replaces the input queue at `key` with `q`.
        '''
        if key not in self.input_q_map:
            raise ValueError('can only replace existing queues')
        self.input_q_map[key] = q
        self._in_cycle = iter(cycle(self.input_q_map.items()))

    def set_output_q(self, key: B, q: Queue) -> None:
        '''
        Replaces the input queue at `key` with `q`.
        '''
        if key not in self.output_q_map:
            raise ValueError('can only replace existing queues')
        self.output_q_map[key] = q

    def __getitem__(self, key: B) -> Queue:
        return self.output_q_map[key]


class QueueExpander(ManyToManyQueueMux[int, int, V], Generic[V]):
    '''
    Splits a queue evenly.
    '''

    def __init__(
            self, *,
            input_q: Queue, n_outputs: int,
            balance='min', limit=0, **kwargs) -> None:
        '''
        Arguments:
            limit (int): maxsize for the output_qs.
            balance ('rr' | 'min' | 'rand'): If 'rr', distributes intput
                to outputs by a round robin scheme. If 'rand', assigns
                outputs independently uniformly at random. If 'min', assigns
                inputs to a queue with the minimum number of items.
        '''

        self._rr_ix = 0

        def expander(key: int, obj: V) -> List[Tuple[Optional[int], V]]:
            out: Tuple[Optional[int], V]
            if balance == 'min':
                out = None, obj
            elif balance == 'rand':
                out = randrange(n_outputs), obj
            elif balance == 'rr':
                out = self._rr_ix, obj
                self._rr_ix = (self._rr_ix + 1) % n_outputs
            else:
                raise ValueError(f'unknown balancing scheme {balance}')

            return [out]

        ManyToManyQueueMux.__init__(
            self,
            input_q_map={0: input_q},
            output_q_map={x: Queue(maxsize=limit) for x in range(n_outputs)},
            cond_func=expander, **kwargs,
        )


class QueueCondenser(ManyToManyQueueMux[int, int, V], Generic[V]):
    '''
    Condenses many queues to one.
    '''

    @property
    def output_q(self):
        return self[0]

    def __init__(
            self, *,
            input_qs: List[Queue],
            output_q=None, limit=0, **kwargs) -> None:

        output_q = ensure_type(output_q, Queue, maxsize=limit)

        def condenser(key: int, obj: V) -> List[Tuple[int, V]]:
            return [(0, obj)]

        ManyToManyQueueMux.__init__(
            self,
            input_q_map={x: q for x, q in enumerate(input_qs)},
            output_q_map={0: output_q},
            cond_func=condenser,
            **kwargs,
        )


class QueueTee(ManyToManyQueueMux[int, int, V], Generic[V]):
    '''
    Copies output to multiple queues.
    '''
    def __init__(
            self, *,
            input_q: Queue,
            output_qs: Union[List[Queue], int],
            maxsize=10,
            **kwargs) -> None:

        if isinstance(output_qs, int):
            output_qs = [Queue(maxsize=maxsize) for i in range(output_qs)]

        self._l = len(output_qs)

        def cond_func(key: int, obj: V) -> List[Tuple[int, V]]:
            return [(ix, obj) for ix in range(self._l)]

        ManyToManyQueueMux.__init__(
            self,
            input_q_map={0: input_q},
            output_q_map={ix: q for ix, q in enumerate(output_qs)},
            cond_func=cond_func, **kwargs,
        )


class QueueReader(Controller):
    '''
    Reads and iterable into a queue.
    '''

    def __init__(
            self, in_iter: Iterable[Any], output_q=None, maxsize=None) -> None:
        Controller.__init__(self)

        self.iter = iter(in_iter)
        self.output_q = ensure_type(output_q, Queue, maxsize=maxsize)

    def _control_loop(self):
        try:
            self.output_q.put(next(self.iter))
        except StopIteration:
            self.halt()


class PipeWorker(Generic[X, Y]):
    '''
    The work component of a WorkPipe.

    Meant to run in a single thread.
    '''

    class EmptyEmit(Exception):
        pass

    @abstractmethod
    def can_absorb(self) -> bool:
        '''
        Indicates if the worker is ready to absorb.

        This MUST return True if the next emit would always raise EmptyEmit
        in the absence of an absorb.

        The absorb method will be called until either the input queue is
        drainer or this method return false, at which point an emit will
        be attempted.
        '''

    @abstractmethod
    def absorb(self, args: X) -> None:
        '''
        Take an object into internal state.
        '''

    @abstractmethod
    def _emit(self) -> Optional[Y]:
        '''
        Implementation of emit, allows for None output which will be ignored.
        '''

    def emit(self) -> Y:
        '''
        Produce an output from internal state.
        '''
        out = self._emit()
        while out is None:
            out = self._emit()
        return out


class FIFOWorker(PipeWorker[X, Y]):
    '''
    Wraps a simple function in the PipeWorker interface.

    This class is not thread safe and is meant to be used within a single
    WorkPipe thread.
    '''

    @classmethod
    def mk_factory(cls, func, *args, **kwargs):
        def factory():
            return cls(func, *args, **kwargs)
        return factory

    def __init__(
            self,
            func: Union[
                    Callable[[X], Optional[Y]],
                    Callable[[X], Iterator[Optional[Y]]]
                ],
            *, max_buffer=10,
            ) -> None:
        '''
        TODO
        '''

        self.max_buffer = max_buffer
        # no need to set the actual max size on the deques
        self.pending_input: Deque[X] = deque([])
        self.pending_output: Deque[Y] = deque([])
        self.func = func

    def can_absorb(self):
        return len(self.pending_input) < self.max_buffer

    def absorb(self, args: X) -> None:
        self.pending_input.append(args)

    def _emit(self) -> Optional[Y]:
        
        if len(self.pending_output) > 0:
            return self.pending_output.pop()

        if len(self.pending_input) == 0:
            raise PipeWorker.EmptyEmit

        out = self.func(self.pending_input.popleft())

        if out is None:
            return None
        # XXX: mypy doesn't infer function type properly
        # if we just check whether the function is a genfunc or not
        # even though it should work since the Union is on the outside
        elif isinstance(out, Iterator):
            for x in out:
                # XXX why does mypy think x can also be Any?
                self.pending_output.appendleft(x)  # type: ignore
            return None
        else:
            return out


class WorkPipe(Controller, Generic[X, Y]):
    '''
    Turns a function into a work pipe between two queues.

    Detects and drains generators automatically.
    '''

    def __init__(
            self, name, *,
            input_q, worker: PipeWorker[X, Y], output_q=None,
            input_limit=10, output_limit=50,
            discard=False, **kwargs) -> None:
        '''
        Arguments:
            input_q: input Queue from which arguments are read
            output_q: output Queue in which to place results. If None, a new
                Queue is created.
            worker:
                A PipeWorker instance to use as the driver of the pipe.
                A callable can also be passed, in which case it will be
                converted into a FIFOWorker.
            halt: Event object which when set will stop execution
            discard: if `True`, function outputs will not be collected at all.
            output_limit:
                the function will block if there are this many or more outputs
                in the output queue already. Same semantics as for
                `Queue(maxsize=)`
        '''
        self.name = name

        if callable(worker):
            self.worker: PipeWorker[X, Y] = FIFOWorker(worker)
        elif isinstance(worker, PipeWorker):
            self.worker: PipeWorker[X, Y] = worker
        else:
            raise ValueError(
                f'{worker.__class__.__name__} is not a valid worker')

        self.discard = discard

        self.input_q = ensure_type(input_q, Queue, maxsize=input_limit)
        self.output_q = ensure_type(output_q, Queue, maxsize=output_limit)

        Controller.__init__(self, **kwargs)

    def _control_loop(self):

        while self.worker.can_absorb():
            try:
                self.worker.absorb(self.input_q.get_nowait())
            except Empty:
                break

        while True:
            try:
                output = self.worker.emit()
                if output is None or self.discard:
                    continue
                self.output_q.put(output, timeout=0.1)
            except (Full, PipeWorker.EmptyEmit):
                return


class ConcurrentProcessor(Controller, Generic[X, Y]):
    '''
    Runs an operation on inputs from a Queue in multiple threaded workers.

    The output is optionally collected to an output Queue.
    '''

    def __init__(
            self, name, *,
            worker_factory: Callable[[], PipeWorker[X, Y]],
            input_q: Queue,
            output_q: Queue=None,
            discard: bool=False,
            unpack: bool=False,
            eager_absorb=False,
            n_workers: int=8,
            input_limit: int=10,
            output_limit: int=10,
            buffer_size: int=10,
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
            n_workers:
                Number of work threads to spawn.
            input_limit:
                Maximum number of elements the input queue will hold.
            output_limit:
                Maximum number of elements the output queue will hold.
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

        Controller.__init__(self, **kwargs)

        self.name = name
        self.input_q = ensure_type(input_q, Queue, maxsize=input_limit)

        self.discard = discard
        if not self.discard:
            self.output_q = ensure_type(output_q, Queue, maxsize=output_limit)
        else:
            self.output_q = None

        self.timeout = timeout
        self.n_workers = n_workers
        self.buffer_size = buffer_size
        self.unpack = unpack

        self.expander: QueueExpander[X] = QueueExpander(
            limit=buffer_size,
            input_q=self.input_q,
            n_outputs=n_workers,
            balance='rr',
        )

        self.work_pipes: List[WorkPipe[X, Y]] = [
            WorkPipe(
                self.name + f'_wp_{wx}',
                input_q=win_q,
                discard=discard,
                eager_absorb=eager_absorb,
                worker=worker_factory(),
                unpack=unpack
            )
            for wx, win_q in enumerate([
                q for k, q in sorted(self.expander.output_q_map.items())
            ])
        ]

        self.condenser: QueueCondenser[Y] = QueueCondenser(
            input_qs=[wp.output_q for wp in self.work_pipes],
            output_q=self.output_q,
            limit=output_limit,
        )

        self.add_subordinate(self.expander)
        for wp in self.work_pipes:
            self.add_subordinate(wp)
        self.add_subordinate(self.condenser)

    def set_input_q(self, q: Queue):
        self.input_q = q
        self.expander.set_input_q(0, q)

    def set_output_q(self, q: Queue):
        self.output_q = q
        self.condenser.set_output_q(0, q)

    def __str__(self):
        return f'<{self.name.upper()} at {id(self):#x}>'


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
        # calls will block unless a synchronized read of _ifl returns < `pool`
        # in which case _ifl will be incremented, and a decrement scheduled to
        # occur in `window` seconds
        self._ifl = 0
        self._now_serving = 0
        self._ifl_lock = Lock()

        self._next_number = 0
        self._next_number_lock = Lock()

    def _dec_ifl(self) -> None:
        with self._ifl_lock:
            self._ifl -= 1

    def _take_a_number(self) -> int:
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
