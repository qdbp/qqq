import concurrent.futures as cfu
import time


import vnv.io as qio


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


def test_aio_poolthrottle(event_loop):

    apool = qio.PoolThrottleAsync(pool=3, window=2, loop=event_loop)
    out = []

    @apool
    async def pooled_append():
        out.append(time.time())

    mark = time.time() + 2

    async def runner():
        while time.time() < mark:
            await pooled_append()
            await pooled_append()
            await pooled_append()

    event_loop.run_until_complete(runner())

    assert 5 <= len(out) <= 7
    assert 1.8 <= (out[-1] - out[0]) <= 2.2
