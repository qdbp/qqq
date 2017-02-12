import concurrent.futures as cfu
from queue import Queue
import time
from threading import Event
import threading

import numpy.random as npr
import pytest

from qqq.qio import ManyToManyQueueMux, Scraper
from qqq.qio import PoolThrottle


def test_mtmqm():

    in_0 = Queue()
    in_1 = Queue()
    in_2 = Queue()

    out_0 = Queue()
    out_1 = Queue()

    ins = [in_0, in_1, in_2]
    outs = [out_0, out_1]

    mtmqm = 

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
