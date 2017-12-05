import asyncio as aio
import sched
import time
from collections import deque
from functools import wraps
from threading import Lock
from typing import Deque  # noqa


class PoolThrottle:
    '''
    Function call throttle, "X calls in Y seconds" style.

    Wraps a callable and constrains call flow to adhere to a "max X requests in
    Y seconds" schema.  Calls made in excess of this rate will block until
    there is room in the pool. The pool is implemented as a sliding window.

    Functions to be throttled should be decorated directly by an instance of
    this class. Multiple functions decorated by the same instance will share
    the same pool.

    Call order is preserved.

    Thread-safe.
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

                # NOTE don't do actual function call with the lock
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


class PoolThrottleAsync:
    '''
    Performs the same function as PoolThrottle, except intended for use
    in single-threaded asyncio applications.

    Instead of blocking rate-limited functions, they are instead made to
    yield control.

    Categorically not thread safe.
    '''

    def __init__(
            self, *, pool: int, window: int, loop, granularity=1e-2) -> None:

        self.pool = pool
        self.window = window
        self.loop = loop
        self.granularity = granularity

        # current number of calls "in flight"
        # calls will block unless a synchronized read of _ifl returns < `pool`
        # in which case _ifl will be incremented, and a decrement scheduled to
        # occur in `window` seconds
        self._ifl = 0
        self._now_serving = 0
        self._next_number = 0
        self._next_slots: Deque[float] = deque([], maxlen=self.pool)

    def _dec_ifl(self) -> None:
        self._ifl -= 1

    def _take_a_number(self) -> int:
        n = self._next_number
        self._next_number += 1
        return n

    def __call__(self, f):

        @wraps(f)
        async def throttled(*args, **kwargs):
            number = self._take_a_number()

            while True:
                # if there's room for us to call
                if self._ifl < self.pool:
                    # and we're the next in line, we call
                    if number == self._now_serving:
                        self._ifl += 1
                        self._now_serving += 1
                        try:
                            return await f(*args, **kwargs)
                        finally:
                            mark = self.loop.time() + self.window
                            self.loop.call_at(mark, self._dec_ifl)
                            self._next_slots.append(mark)
                    # if we're not the next in line, we check again soon
                    else:
                        await aio.sleep(self.granularity)
                        continue
                # if there's no room for us, we wait until the next plausible
                # opportunity
                else:
                    try:
                        await aio.sleep(max(
                            self.granularity,
                            self._next_slots[0] - self.loop.time() -
                            self.granularity
                        ))
                    except Exception as e:
                        print(e)

        return throttled
