import numpy as np
import numpy.random as npr


def gen_random_labels(X, n_classes, pvec=None):
    '''
    Returns a random labelling of dataset X.

    Labels are one-hot encoded, with `n_classes` dimensions.

    Args:
        X: number of labels to generate, integer or array-like.
            If array-like, `X.shape[0]` labels will be generated.
        n_classes: number of output classes
        pvec: array-like, gives probabilities of each class. If `None`, all
            classes are equiprobable.

    '''

    try:
        num = X.shape[0]
    except AttributeError:
        num = X

    pvec = np.ones((n_classes,)) / n_classes

    return npr.multinomial(1, pvec, size=num)
