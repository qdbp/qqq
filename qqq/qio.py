from abc import abstractmethod
from random import randrange
import sched
import time
from inspect import isgeneratorfunction as isgenerator
from functools import wraps
from queue import Empty, Queue
from threading import Event, Lock, Thread

from .qlog import get_logger

log = get_logger(__file__)


def ensure_type(obj, t, *args, **kwargs):
    if obj is None:
        return t(*args, **kwargs)
    elif not isinstance(obj, t):
        raise ValueError(f'need a {t.__name__} object, got {obj}')
    else:
        return obj


def iterable_to_q(iterable):
    q = Queue()
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

        super().__init__(**kwargs)

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

        super().__init__(**kwargs)

    def run(self):
        '''
        Starts the control thread.
        '''
        self._control_thread.start()

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
                 greedy=True, timeout=0.02, **kwargs):
        '''
        Arguments:
            input_qs (Queue | Iterable[Queue] | Dict[IKey, Queue]):
                An input queue or collection of input Queues.  If an iterable,
                the keys will be integer indices.  If single queue will be
                treated like a singleton iterable.
            output_qs (Queue | Iterable[Queue] | Dict[OKey, Queue]):
                An output queue or collection of output Queues.  If an
                iterable, the keys will be integer indices.  If single queue
                will be treated like a singleton iterable.
            cond_func (Callable[Tuple[IKey, Value], Tuple[OKey, Value]]):
                A function which takes an input key and a value from the queue
                and produces an output key and a list of new value. Should not
                be an expensive computation, as it will be run in the single
                work thread.
            greedy (bool):
                If `True`, will use a greedy strategy, emptying each input
                queue fully before moving on to the next. Otherwise will
                alternate strictly between input Queues, up to timeout.
            timeout (float):
                If no objects are received on an input queue in `timeout`, will
                move on to the next.
        '''
        super().__init__(**kwargs)

        self.input_qs, self.input_q_map = self.__rectify_q_arg(input_qs)
        self.output_qs, self.output_q_map = self.__rectify_q_arg(output_qs)

        self.cond_func = cond_func

        self.greedy = greedy
        self.timeout = timeout

    def __rectify_q_arg(self, q_arg):
        if isinstance(q_arg, dict):
            out = list(q_arg.values())
            out_map = q_arg
        elif isinstance(q_arg, Queue):
            out = [q_arg]
            out_map = {0: q_arg}
        elif isinstance(q_arg, (list, tuple)):
            out = q_arg
            out_map = {ix: q for ix, q in enumerate(q_arg)}

        return out, out_map

    def _control_loop(self):
        while not self._halt.is_set():
            for key, in_q in self.input_q_map.items():
                while True:
                    try:
                        obj = in_q.get(timeout=self.timeout)
                        k, out = self.cond_func(key, obj)
                        self.output_q_map[k].put(out)
                    except Empty:
                        break

                    if not self.greedy:
                        break


class QueueExpander(ManyToManyQueueMux):
    '''
    Splits a queue evenly.
    '''

    def __init__(self, *, input_q, n_outputs, balance, **kwargs):

        output_qs = [Queue() for x in range(n_outputs)]

        def expander(key, obj):
            if balance:
                ix = None
                m = float('inf')
                for qx, q in enumerate(output_qs):
                    test_m = q.qsize()
                    if test_m < m:
                        ix = qx
                        m = test_m
                return ix, obj
            else:
                return randrange(n_outputs), obj

        super().__init__(input_qs=[input_q],
                         output_qs=output_qs,
                         cond_func=expander,
                         **kwargs)


class QueueCondenser(ManyToManyQueueMux):
    '''
    Condenses many queues to one.
    '''

    def __init__(self, *, input_qs, output_q=None, **kwargs):

        output_q = ensure_type(output_q, Queue)

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
                 unpack=False, discard=False, limit=0, **kwargs):
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
            limit: the function will block if there are this many or more
                outputs in the output queue already. Same semantics as for
                `Queue(maxsize=)`
        '''

        super().__init__(**kwargs)

        self.unpack = unpack
        self.work_func = work_func
        self.discard = discard

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
                            self.output_q.put(obj)
                    else:
                        self.output_q.put(out)
            except Exception:
                log.error(f'exception in worker thread', exc_info=True)
                self.halt()


class QueueProcessor(HaltMixin):
    '''
    A basic scraper with multiple workers.
    '''

    def __init__(self, *, input_q, work_func, n_workers,
                 output_q=None, halt=None, unpack=False, limit=0,
                 discard=False, **mux_kwargs):

        self.input_q = ensure_type(input_q, Queue)
        self.output_q = ensure_type(output_q, Queue)

        self.expander = QueueExpander(
            input_q=self.input_q,
            n_outputs=n_workers,
            balance=True,
            **mux_kwargs,
        )

        self.work_pipes = [
            WorkPipe(input_q=win_q, halt=halt, discard=discard,
                     work_func=work_func, unpack=unpack)
            for win_q in self.expander.output_qs
        ]

        self.condenser = QueueCondenser(
            input_qs=[wp.output_q for wp in self.work_pipes],
            output_q=self.output_q,
            **mux_kwargs,
        )

    def run(self):
        self.expander.run()
        for wp in self.work_pipes:
            wp.run()
        self.condenser.run()


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

    def __init__(self, pool_size, window, strict=False):
        '''
        Args:
            pool_size:
                max requests allowed in a sliding window of `window` seconds
            window:
                sliding span of time in which `pool_size` requests are allowed
            res:
                time resolution with which call attempts are made if the pool
                is full
            strict:
                if `True`, the `window`-second timer after a call will start
                only after it returns, else it will start right before the call
                is made. A strict pool effectively makes X calls every (Y +
                average_call_time) seconds.
        '''
        self.pool_size = pool_size
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
                    do_call = (self._ifl < self.pool_size and
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
