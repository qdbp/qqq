import concurrent.futures as cfu
import itertools as itr
import threading

import numpy as np
import numpy.random as npr


def iter_batches(data, bs, rand=True, excl_dict=None,
                 trans=None, wgh_key='y', seqlen=1, seq_mode=None,
                 concat_axis=None,
                 workers=8):
    """ Given a dictionary of lists of lists, each list of lists
        with all lists of lists having the same length, and
        all ith sublists in the list of lists having the same length.
        Yields a dictionary with the same keys as `data`, but with
        values being arrays of the given batch size, with corresponding
        elements in each array taken from the elements of the sublists.

        As an example, given {'x': [[1, 2], [7, 8, 9],
                              'y': [[0, 1], [1, -1, 0]}
        a batch might be {'x': [9, 2], 'y': [0, 1]}

        Sublists are selected uniformly, implying samples from smaller
        sublists will be overrepresented. This is intentional.

        Assumes y is categorical for the purpose of calculating
        weights.

        Arguments:
            data: dictionary containing lists of sequences of features
                or labels
            bs: desired batch size to be output with every
                yield.
            rand:
                whether to randomize. If true, to generate every
                sample in every batch, first a list x will be picked
                uniformly from xs, then a sample will be picked
                uniformly at random from x. If false, samples will be
                yielded in the order xs[0][0], xs[0][1], ...,
                xs[-1][-2], xs[-1][-1]. If `ret` is true,
                the generator will return immediately on seeing the last
                sample. Consequently, the last (#samples)%bs points will
                never be yielded.
            excl_dict:
                dictionary of functions called on their respective raw input to check
                whether this input should be excluded
            trans:
                dictionary of transformation
                transformation function arbitrarily mapping a sample
                x to either a numpy array of a different shape, or a
                dictionary of such arrays. The output shape of batches
                will be inferred automatically from the output of this
                function on the first sample. If a dictionary is returned,
                the output will aggregate the transformations of bs samples
                in a single dictionary, grouped by key.
            seqlen:
                number of consecutive data points to place in a sequence,
                according to the mode specified by seq_mode
            seq_mode:
                dict specifying how to, for each output, handle sequence
                elements. Can be one of 'concat', 'newdim' or 'last'.
                'concat' concatenates along the axis given in concat_axis,
                which must be given if 'concat' is used. 'newdim' creates a new
                dimension in the batch array and orders the sequence elements
                therealong.
                'last' selects the last element of the sequence.
                Defaults to 'newaxis'.
            concat_axis:
                dictionary giving axes along which to concatenate the given
                outputs for which seq_mode is concat.

    """
    
    if seq_mode is None:
        seq_mode = {}
    if concat_axis is None:
        concat_axis = {}

    try:
        assert all(k in concat_axis for k, v in seq_mode.items()
                   if v == 'concat')
    except AssertionError:
        raise ValueError('did not specify concat axis for some data!')

    if trans is None:
        trans = {}
    if excl_dict is None:
        excl_dict = {}

    # lens of master lists
    lens = set()
    sublens = []
    for k, v in data.items():
        lens.add(len(v))
        if not sublens:
            for subv in v:
                sublens.append(len(subv))
        else:
            for subl, subv in zip(sublens, v):
                assert len(subv) == subl
        req_shape = v[0][0].shape
        for subv in v:
            assert all(pt.shape == req_shape for pt in subv)
    assert len(lens) == 1

    l = lens.pop()

    batch = {}
    for k in data.keys():
        sm = seq_mode.get(k, 'newdim')
        data_shape = trans.get(k, lambda x: x)(data[k][0][0]).shape

        if sm == 'concat':
            ca = concat_axis[k]
            data_shape = list(data_shape)
            data_shape[ca] *= seqlen
            data_shape = tuple(data_shape)

        batch[k] = np.zeros((bs,) +
                            ((seqlen,)
                             if (seqlen > 1 and sm == 'newdim')
                             else ()) +
                            data_shape)

    batch_w = {wgh_key: np.zeros(bs)}
    wgh = get_wgh(data[wgh_key])

    def do_sample(args):
        bx, ix, jx, sx = args
        incl = True
        for k, v in data.items():
            d = v[ix][jx]
            sm = seq_mode.get(k, 'newdim')
            incl &= True if k not in excl_dict else not excl_dict[k](d)
            if not incl:
                batch_w[wgh_key][bx] = 0
                return

            val = d if k not in trans else trans[k](d)

            if sm == 'newdim':
                batch[k][bx, sx] = val
            elif sm == 'concat':
                ca = concat_axis[k]
                index = [bx] + [slice(None, None, None) for _ in val.shape]
                index[ca+1] = slice(sx*val.shape[ca],
                                    (sx+1)*val.shape[ca],
                                    None)
                index = tuple(index)
                batch[k][index] = val
            elif sm == 'last':
                if sx == seqlen-1:
                    batch[k][bx] = val
            else:
                raise ValueError('unrecognized seq_mode {}'.format(sm))

        batch_w[wgh_key][bx] = wgh[np.argmax(data[wgh_key][ix][jx])]

    exe = cfu.ThreadPoolExecutor(max_workers=workers)

    det_ix = 0
    det_jx = 0

    def inc(i, j):
        j += 1
        if j > sublens[i] - seqlen:
            j = 0
            i += 1
        if i == l:
            i = 0
        return i, j
        
    while True:
        ixes, jxes = [], []
        if rand:
            _ixes = npr.randint(l, size=bs)
            for _ix in _ixes:
                ixes += [_ix for _ in range(seqlen)]
                jx = npr.randint(sublens[_ix] - seqlen + 1)
                jxes += [jx + i for i in range(seqlen)]
        else:
            for _ in range(bs):
                ix, jx = det_ix, det_jx
                ixes.append(ix)
                jxes.append(jx)
                det_ix, det_jx = inc(ix, jx)

        sxes = itr.cycle([i for i in range(seqlen)])

        list(exe.map(do_sample, zip(range(bs), ixes, jxes, sxes)))

        yield batch, batch_w


def get_wgh(ys):
    # mean of weights since uniform sampling in iter_batches!
    wgh = np.mean([1/np.sum(y, axis=0) for y in ys], axis=0)
    return wgh*(len(wgh)/np.sum(wgh))

if __name__ == '__main__':
    x = [np.zeros((100*(i+1), 3, 100, 100)) + i for i in range(10)]
    y = [np.ones((100*(i+1), 5)) + i for i in range(10)]
    z = [np.zeros((100*(i+1), 3, 25, 25)) + i for i in range(10)]

    data = {'x': x, 'y': y, 'z': z}

    gen = iter_batches(data, 256, seqlen=3, seq_mode={'x': 'last',
                                                      'y': 'concat'},
                       concat_axis={'y': 0})
    out = next(gen)[0]
    print(out)
    print('x', out['x'].shape)
    print('y', out['y'].shape)
    print('z', out['z'].shape)

    # import time
    # t = time.time()
    # for i, ix in zip(gen, range(1000)):
    #     print(i[0]['x'][:, 0, 5, 5])

    # print('{:.3f}'.format(time.time() - t))
