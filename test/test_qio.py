import concurrent.futures as cfu
from queue import Queue
import time
from threading import Event
import threading

import numpy.random as npr
import pytest

import qqq.qio as qio


def test_expander():
    
    nums = list(range(10))
    
    in_q = qio.iterable_to_q(nums)
    
    expander = qio.QueueExpander(input_q=in_q, n_outputs=10, balance='rr')
    expander.run()
    expander.halt()
    expander.join()

    for ix, oq in enumerate(expander.output_qs):
        assert oq.get_nowait() == ix


def test_mtmqm():

    in_qs = qio.iterable_to_q(list(range(18)))
    out_qs = [qio.Queue(maxsize=2) for i in range(6)]

    i = 0

    def cond_func(key, val):
        nonlocal i
        j = i
        i += 1
        i = i % 6
        return j, val

    qp = qio.ManyToManyQueueMux(
         input_qs=in_qs, output_qs=out_qs, cond_func=cond_func,
    )
    qp.run()
    time.sleep(1)
    qp.halt()
    qp.join()

    for oq in qp.output_qs:
        while not oq.empty():
            print(oq.get())

    print([oq.qsize() for oq in qp.output_qs])


def test_poolthrottle():
    pool = qio.PoolThrottle(1, 0.1)
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
    

if __name__ == '__main__':
    # pytest.main([__file__])
    test_poolthrottle()
