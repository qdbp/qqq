import concurrent.futures as cfu
from queue import Queue
import time
from threading import Event
import threading

import numpy.random as npr
import pytest

from qqq.qio import Scraper, SplitQueue
from qqq.qio import FunctionPrinter


def test_collatz():
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

    time.sleep(10)
    halt.set()

    assert len(outputs) == 24

# def test_functionprinter():
#     fp = FunctionPrinter()
# 
#     @fp.decorate
#     def func():
#         sys.stdout.write('printing func!\n')
# 
#     @fp.decorate
#     def gunc():
#         sys.stdout.write('printing gunc 1!\n')
#         func()
#         sys.stdout.write('printing gunc 2 without newline...')
#         sys.stdout.write('... still new stuff ...')
#         sys.stdout.write('... and done!\n') 
#         print('using print in gunc!')
# 
#     @fp.decorate
#     def hunc():
#         sys.stdout.write('printing hunc 1!\n')
#         gunc()
#         junc()
#         func()
#         raise ValueError()
#         gunc()
#         sys.stdout.write('printing hunc 2!\n')
# 
#     def junc():
#         sys.stdout.write('unwrapped function junc!\n')
# 
#     def kunc():
#         sys.stdout.write('unwrapped function kunc 1, will call hunc, junc!\n')
#         try:
#             hunc()
#         except Exception as e:
#             print('got exception {}'.format(e))
#         junc()
#         gunc()
#         sys.stdout.write('unwrapped function kunc 2\n')

    kunc()

if __name__ == '__main__':
    pytest.main([__file__])
