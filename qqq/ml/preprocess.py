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


def get_k_of_each(y, k):
    '''
    Returns min(#members, k) members of each class of y.

    Args:
        y: binary-encoded labels

    Returns:
        ixes: indices of the selected elements

    Examples:
        >>> y_sub = y[get_k_of_each(10, y)]
    '''
    if len(y.shape) != 2:
        raise ValueError('This function expects a 2D array.')

    ixes = {}
    ymax = np.argmax(y, axis=1)

    for i in range(y.shape[1]):
        ixes[i] = np.where(ymax == i)[0]

    return np.concatenate([*ixes.values()])


def complement_ixes(y, ixes):
    try:
        y = len(y)
    except Exception:
        pass

    all_ixes = np.ones(y, dtype=np.uint8)
    all_ixes[ixes] = 0

    return np.where(all_ixes == 1)[0]
