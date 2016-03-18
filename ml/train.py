import numpy as np
import numpy.random as npr


def iter_batches(data, bs, rand=True, excl_func=None,
                 trans=None, ret=False, wgh_key='y', seqlen=1):
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
            ret:
                Whether to return when the the input lists have been exhausted.
                Only applied when rand is False
            excl_func:
                a callable taking a single sample and returning a boolean.
                Called on every sample, causing the sample to be skipped
                if the return value is False.
            trans:
                dictionary of transformation
                transformation function arbitrarily mapping a sample
                x to either a numpy array of a different shape, or a
                dictionary of such arrays. The output shape of batches
                will be inferred automatically from the output of this
                function on the first sample. If a dictionary is returned,
                the output will aggregate the transformations of bs samples
                in a single dictionary, grouped by key.

    """

    if trans is None:
        trans = {}

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

    batch = {k: np.zeros((bs,) + ((seqlen,) if seqlen > 1 else ()) +
                         trans.get(k, lambda x: x)(data[k][0][0]).shape
                         )
             for k in data.keys()
             }

    batch_w = {wgh_key: np.zeros(bs)}
    wgh = get_wgh(data[wgh_key])

    def do_sample(ix, jx):
        pass
        
    det_ix = 0
    det_jx = 0
    while True:
        if rand:
            ixes = npr.randint(l, size=bs)
            
            
            jxes = [npr.randint(sublens[ix]) for ix in ixes]
        else:
            ixes = [] 
            jxes = []
            jx = det_jx
            ix = det_ix
            for i in range(bs):
                jxes.append(jx)
                ixes.append(ix)
                jx += 1
                if jx == sublens[ix]:
                    jx = 0
                    ix += 1
                if ix == l:
                    ix = 0
                
        for b in range(bs):
            # TODO: suboptimal implementation
            # can't pick
            while rand and jx + seqlen >= sublens[ix]:
                jx = npr.randint(sublens[ix])

            for sx in range(seqlen):
                for k, v in data.items():
                    d = v[ix][jx]
                    if seqlen > 1:
                        batch[k][b, sx] = d if k not in trans else trans[k](d)
                    else:
                        batch[k][b] = d if k not in trans else trans[k](d)
                jx += 1

            det_jx = jx
            if not rand:
                if det_jx >= sublens[ix] - seqlen:
                    det_jx = 0
                    det_ix += 1
                if det_ix == l:
                    if ret:
                        return
                    else:
                        det_ix = 0
                        det_jx = 0
            batch_w[wgh_key][b] = wgh[np.argmax(data[wgh_key][ix][jx])]

        yield batch, batch_w


def get_wgh(ys):
    # mean of weights since uniform sampling in iter_batches!
    wgh = np.mean([1/np.sum(y, axis=0) for y in ys], axis=0)
    return wgh*(len(wgh)/np.sum(wgh))

if __name__ == '__main__':
    x = [np.zeros((1000, 1, 100, 100)) for i in range(10)]
    y = [np.ones((1000, 5)) for i in range(10)]
    z = [np.zeros((1000, 1, 25, 25)) for i in range(10)]

    data = {'x': x, 'y': y, 'z': z}

    gen = iter_batches(data, 256)
    gen = iter_batches(data, 256, seqlen=5)

    import time
    t = time.time()
    for i, ix in zip(gen, range(1000)):
        print(i[0]['x'].shape)

    print('{:.3f}'.format(time.time() - t))
