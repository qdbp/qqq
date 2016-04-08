import concurrent.futures as cfu
import os.path as osp
import pickle as pkl
import queue as que
import time
import traceback as trc


def wr(*args):
    '''
    Convenience function to print something in the style of
    `sys.stdout.write(something+'\r')
    '''
    print(*args, end='\r', flush=True)


def p(fn, f, args=(), kwargs=None, d='.', ow=False, owa=(), owk=None):
    """ pickle cache manager. will cache the results of
        a particular computation to a file with the given
        name if it doesn't exist, else return the contents
        of the file with that name.

        No attempt to match computation to file is
        made in any way, that is for the user to track. """
    if kwargs is None:
        kwargs = {}
    if owk is None:
        owk = {}

    if callable(ow):
        do_ow = ow(*owa, **owk)
    elif isinstance(ow, bool):
        do_ow = ow

    assert isinstance(do_ow, bool),\
        'overwrite condition {} untenable'.format(do_ow)

    fp = osp.join(d, fn)
    if osp.isfile(fp) and not do_ow:
        with open(fp, 'rb') as f:
            return pkl.load(f)
    else:
        res = f(*args, **kwargs)
        with open(fp, 'wb') as f:
            pkl.dump(res, f)
        return res


def scrape(get_func, args, process_func, max_workers=64, sleep=0.05,
           allow_fail=[], verbose=True):
    '''
    Function to abstract a scraping process wherein a slow, parallelizable
    I/O operation feeds a fast processor. Many instances of the I/O operation
    are spawned, with their outputs fed (in arbitrary order) to the processor.

    Arguments:
        get_func: function taking a single positional argument, returning
            an object `process_func` can accept.
        args: list of arguments to `get_func`. A single instance will be
            spawned for each arg in `args`.
        process_func: function which takes the output of `get_func` and does
            something useful with it, like storing it in a database.
        max_workers: number of instances of `get_func` to keep spawned at
            any time.
        sleep: time to sleep each time no data from `get_func`s is available
            for process_func
    '''
    q = que.Queue()

    def queuer(arg):
        q.put(get_func(arg))

    with cfu.ThreadPoolExecutor(max_workers=max_workers) as x:
        futs = set()
        for arg in args:
            futs.add(x.submit(queuer, arg))

        while True:
            try:
                res = q.get_nowait()
                try:
                    process_func(res)
                except Exception as e:
                    if verbose:
                        trc.print_exc()
                    if not type(e) in allow_fail:
                        for f in futs:
                            f.cancel()
                        break
            except que.Empty:
                if all(f.done() for f in futs):
                    return
                else:
                    time.sleep(sleep)


def rq_json(base_url, params):
    import requests as rqs
    return rqs.get(base_url, params=params).json()
