import os.path as osp
import pickle as pkl
import sqlite3 as sql


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
