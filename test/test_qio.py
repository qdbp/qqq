import concurrent.futures as cfu
from queue import Queue
import time
from threading import Event
import threading

import numpy.random as npr
import pytest

from qqq.qio import Scraper, SplitQueue
from qqq.qio import PoolThrottle


def no_test_collatz():
    outputs = []

    def collatz_mul(arg):
        i, n0, n = arg
        return (i + 1, n0, 3*n + 1)
    
    def collatz_done(arg):
        i, n0, n = arg
        assert n == 1
        outputs.append(arg)
    
    def collatz_div(arg):
        i, n0, n = arg
        return (i + 1, n0, n//2)
    
    def collatz_sort(arg):
        n = arg[2]
        if n == 1:
            return 2
        elif n % 2:
            return 1
        else:
            return 0

    inq = Queue()
    inp = [(0, i, i) for i in range(1, 25)]
    for i in inp:
        inq.put(i)
    
    halt = Event()
    even, odd, done = SplitQueue(inq, 3, collatz_sort, halt=halt)#, halt=halt)
    
    s1 = Scraper(even, collatz_div, output_q=inq, halt_on_empty=False,
                 verbose=True, halt=halt).run()
    s2 = Scraper(odd, collatz_mul, output_q=inq, halt_on_empty=False,
                 verbose=True, halt=halt).run()
    s3 = Scraper(done, collatz_done, collect_output=False, halt_on_empty=False,
                 verbose=True, halt=halt).run()

    halt.set()

    assert len(outputs) == 24


def test_poolthrottle():
    pool = PoolThrottle(1, 0.1)
    exe = cfu.ThreadPoolExecutor(max_workers=3)

    out = []

    @pool
    def pooled_append(x):
        out.append(x)

    def fighter1():
        for i in range(10):
            pooled_append(i)

    def fighter2():
        for i in range(10):
            pooled_append(i)

    def fighter3():
        for i in range(10):
            pooled_append(i)

    exe.submit(fighter1)
    exe.submit(fighter2)
    exe.submit(fighter3)
    exe.shutdown(True)

    print('\n\n\n\n')
    print(out)


if __name__ == '__main__':
    # pytest.main([__file__])
    test_poolthrottle()
