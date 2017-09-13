import concurrent.futures as cfu
from queue import Queue, Empty
import time
from threading import Event
import threading

import numpy.random as npr
import pytest

import qqq.io as qio


def test_mtmqm_core():

    s = 'abcdefg'
    s2 = 'abc'

    in_q1 = qio.iterable_to_q(list(range(6)))
    in_q2 = qio.iterable_to_q(s)

    def cond_func(key, obj):
        if isinstance(key, str) and key in s:
            if obj in s2:
                out = [('a', 2 * obj), ('b', 2 * obj)]
            else:
                out = [('a', obj)]
        else:
            out = [(3, obj % 2)]
        print(key, obj, out)
        return out

    qp = qio.ManyToManyQueueMux(
            input_q_map={0: in_q1, 'a': in_q2},
            output_q_map={
                'a': Queue(), 'b': Queue(), 3: Queue()
            },
            cond_func=cond_func,
    )

    qp.run()
    time.sleep(0.1)
    qp.halt()
    qp.join()

    a = sorted(qio.iter_q(qp['a']))
    b = sorted(qio.iter_q(qp['b']))
    c = sorted(qio.iter_q(qp[3]))

    assert a == sorted(['aa', 'bb', 'cc', 'd', 'e', 'f', 'g'])
    assert b == sorted(['aa', 'bb', 'cc'])
    assert c == sorted([0] * 3 + [1] * 3)


def test_mtmqm_stress():

    in_q_map = {
        ix: qio.iterable_to_q(npr.randint(0, 10, size=10000))
        for ix in range(10)
    }
    out_q_map = {ix: Queue() for ix in range(10)}

    def cond_func(key, obj):
        return [(obj, obj)]

    m = qio.ManyToManyQueueMux(
        input_q_map=in_q_map,
        output_q_map=out_q_map,
        cond_func=cond_func,
    )
    m.run()
    time.sleep(0.1)
    m.halt()
    m.join()
    for k, v in m.output_q_map.items():
        print(v)
        assert set(qio.iter_q(v)) == {k}
        

def test_expander():
    
    nums = list(range(10))
    
    in_q = qio.iterable_to_q(nums)
    
    expander = qio.QueueExpander(input_q=in_q, n_outputs=10, balance='rr')
    expander.run()
    expander.halt()
    expander.join()

    for ix, oq in sorted(expander.output_q_map.items()):
        assert oq.get_nowait() == ix


def test_condenser():

    qs = [qio.iterable_to_q(range(10 * i, 10 * i + 10)) for i in range(5)]

    cd = qio.QueueCondenser(input_qs=qs)
    cd.run()
    time.sleep(0.1)
    cd.halt()
    cd.join()

    assert sorted(qio.iter_q(cd.output_q)) == sorted(range(50))


def test_QueueTee():
    in_q = qio.iterable_to_q(range(10))

    qt = qio.QueueTee(input_q=in_q, output_qs=2)
    qt.run()
    time.sleep(0.1)
    qt.halt()
    qt.join()

    out = []
    for v in qt.output_q_map.values():
        out += qio.iter_q(v)

    assert sorted(out) == sorted([*range(10)] * 2)


def test_fifo_worker():
    def foo(x):
        return x + x

    fifw = qio.FIFOWorker(foo)

    test_arr = ['a', 'b', 'c']
    for char in test_arr:
        fifw.absorb(char)

    print(fifw.pending_input)

    for i in test_arr:
        assert fifw.emit() == i + i

    with pytest.raises(Empty):
        fifw.emit()

    def bar(a):
        for i in range(2):
            yield a * (i + 1)

    fifw = qio.FIFOWorker(bar)
    for char in test_arr:
        fifw.absorb(char)

    for char in test_arr:
        assert fifw.emit() == char
        assert fifw.emit() == char * 2


def test_work_pipe():

    int_q = qio.iterable_to_q(range(20))
    def foo(x):
        return -x if x % 2 else None

    worker = qio.FIFOWorker(foo)

    wp = qio.WorkPipe(
        'test', input_q=int_q,
        worker=worker, eager_absorb=True, output_limit=3,
    )
    wp.run()
    time.sleep(0.1)

    assert int_q.empty()
    assert wp.output_q.qsize() == 3

    for i in [-i for i in range(20) if i % 2]:
        assert wp.output_q.get(timeout=0.02) == i

    with pytest.raises(Empty):
        wp.output_q.get(timeout=0.02)


def test_concurrent_processor():

    input_q = qio.iterable_to_q(range(100))

    def dummyworker(x):
        time.sleep(0.1)
        return 69

    worker_factory = qio.FIFOWorker.mk_factory(dummyworker)

    cp = qio.ConcurrentProcessor(
        'test_processor',
        input_q=input_q,
        worker_factory=worker_factory,
        n_workers=100,
    )

    cp.run()

    t = time.time()
    for i in range(100):
        assert cp.output_q.get(0.05) == 69
    assert time.time() - t < 0.5
    
    with pytest.raises(Empty):
        cp.output_q.get(timeout=0.02)

    t = time.time()
    cp.halt()
    cp.join()
    dt = time.time() - t
    print(dt)
    assert dt < 0.5


def test_poolthrottle():
    pool = qio.PoolThrottle(pool=1, window=0.1)
    exe = cfu.ThreadPoolExecutor(max_workers=3)

    out = []

    @pool
    def pooled_append(x):
        out.append(x)

    def fighter1():
        for i in range(5):
            pooled_append(i)

    def fighter2():
        for i in range(5):
            pooled_append(i)

    def fighter3():
        for i in range(5):
            pooled_append(i)

    exe.submit(fighter1)
    exe.submit(fighter2)
    exe.submit(fighter3)
    t = time.time()
    exe.shutdown(True)
    dt = time.time() - t
    assert 1.25 < dt < 1.75
    

if __name__ == '__main__':
    # pytest.main([__file__])
    test_poolthrottle()
