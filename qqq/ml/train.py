import argparse as arg
import concurrent.futures as cfu
import glob
import os.path as osp
import itertools as itr
import threading

import mmh3
import numpy as np
import numpy.random as npr

from qqq.qlog import get_logger
log = get_logger(__file__)


def get_wgh(ys):
    # mean of weights since uniform sampling in iter_batches!
    wgh = np.mean([1/np.sum(y, axis=0) for y in ys], axis=0)
    return wgh*(len(wgh)/np.sum(wgh))


class TrainUI:





# FIXME: this is only useful when we can't load all of the data into memory
# make it work in that case
class DirFeeder:
    '''
    Reads a directory of data.
    '''

    hsh = mmh3.hash_bytes
    hmax = 4096

    def __init__(self, d, suffix, hash_data=False, hash_names=True):
        '''
        Argumerns:
            d: directory from which to draw files
            suffix: file suffix of data files
            salt: salt to use for the hashing operation. if `None`, a random
                salt is chosen every time
            hash_data: up to 1 MiB of file data will be hashed if `True`.
                if both `hash_names` and `hash_data` are `True`, the name
                and data will be concatenated before hashing.
            hash_names: if `True`, the file basenames will be hashed.

        '''
        if not (hash_data or hash_names):
            log.warning('both name and data hashing disabled. '
                        'this is invalid; will hash data')
            hash_data = True

        self.d = d
        self.suf = suffix

        self.fns = sorted(glob.glob(osp.join(d, '*' + suffix)))
        self._map = {}

        for fn in self.fns:
            to_hash = b''
            if hash_names:
                to_hash += fn.encode('utf-8')
            if hash_data:
                with open(fn, 'rb') as f:
                    to_hash += fn.read(1 << 20)

            self._values[fn] = sum(self.hash(to_hash))

    def bagged_kfold(self, load_fun, k=3):
        for kx in range(k):
            train = set()
            val = set()
            for fn, val in self._map:
                if kx/k < val/self.range <= (kx + 1)/k:
                    val.add(fn)
                else:
                    train.add(fn)



if __name__ == '__main__':

    # from keras.datasets import mnist
    # from keras.utils import np_utils
    # import matplotlib.pyplot as plt
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train.reshape(X_train.shape[0], 28, 28)
    # X_test = X_test.reshape(X_test.shape[0], 28, 28)
    # Y_train = np_utils.to_categorical(y_train, 10)
    # Y_test = np_utils.to_categorical(y_test, 10)

    # data = {'x': [X_train, X_test], 'y': [Y_train, Y_test]}

    # gen = iter_batches(data, 128, seqlen=3, seq_mode={'x': 'concat',
    #                                                   'y': 'last'},
    #                    concat_axis={'x': 1},
    #                    rand=True)

    # out = next(gen)[0]
    # print(out['x'][21])
    # print(out['y'][21])
    # plt.matshow(out['x'][21])

    # print(out['x'][22])
    # print(out['y'][22])
    # plt.matshow(out['x'][22])

    # plt.show()
