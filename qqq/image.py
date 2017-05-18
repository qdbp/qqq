import numpy as np
import numpy.fft as fft
import numpy.random as npr

from .np import hilbert_ixes


def subpixel_shift(img_arr, dx=0.5, inplace=True):
    '''
    Shifts an image by a fraction of a pixel in a random direction,
    with circular boundary conditions.

    Arguments:
        img_arr (2D ndarray): array to subpixel shift in a random direction
        dx (float): distance in pixel to shift by
        inplace (bool): if True, will modify the array inplace

    '''

    f = fft.fft2(img_arr, axes=(0, 1))
    g = np.meshgrid(
        fft.fftfreq(img_arr.shape[-3]),
        fft.fftfreq(img_arr.shape[-2])
    )

    u = npr.normal(0, 1, size=2)
    fk = g[0] * u[0] + g[1] * u[1]
    phi = np.exp(2j * np.pi * dx * fk)

    out = np.real(fft.ifft2(phi[..., np.newaxis] * f, axes=(0, 1)))
    if inplace:
        img_arr[:] = out[:]
    else:
        return out


def cartesian_shift(img_arr):
    '''
    Shifts an image along axes by one pixel. Inplace opration.
    '''
    zeros = [None, None]
    slices_s = [..., None, None, slice(None, None)]
    slices_d = [..., None, None, slice(None, None)]

    do_transform = False
    for i in range(1, 3):
        if npr.random() > 0.667:
            continue
        do_transform = True
        if npr.random() < 0.5:
            slices_s[i] = slice(1, None)
            slices_d[i] = slice(None, -1)
            zeros[i - 1] = -1
        else:
            slices_s[i] = slice(None, -1)
            slices_d[i] = slice(1, None)
            zeros[i - 1] = 0

    if not do_transform:
        return

    img_arr[tuple(slices_d)] = img_arr[tuple(slices_s)]

    if zeros[0] is not None:
        img_arr[..., zeros[0], :, :] = 0.
    if zeros[1] is not None:
        img_arr[..., :, zeros[1], :] = 0.


def image_flip(img_arr):
    '''
    Flips an image at random along x and y. Inplace operation.
    '''
    x = npr.randint(4)
    if x == 0:
        img_arr[..., :, :, :] = img_arr[..., ::-1, ::-1, :]
    elif x == 1:
        img_arr[..., :, :, :] = img_arr[..., :, ::-1, :]
    elif x == 2:
        img_arr[..., :, :, :] = img_arr[..., ::-1, :, :]
    # x == 3 -> nop


def chunk_image(img_arr, chunk_size=32, random=True):
    '''
    Chunks an image into a sequence of smaller images.

    Assumes the image is evenly divisible, discards lower-right remainder
    if not.
    '''

    cw = img_arr.shape[1] // chunk_size
    ch = img_arr.shape[0] // chunk_size

    n_c = cw * ch

    out = np.zeros((n_c, chunk_size, chunk_size, img_arr.shape[2]))

    if random:
        ixes = np.arange(n_c, dtype=np.uint32)
    else:
        ixes = npr.permutation(n_c)

    for i in range(ch):
        for j in range(cw):
            out[ixes[i * cw + j], :] =\
                img_arr[
                    chunk_size * i:chunk_size * (i + 1),
                    chunk_size * j:chunk_size * (j + 1),
                    ...]

    return out


def chunk_image_shape(shape, chunk_size=32):
    cw = shape[1] // chunk_size
    ch = shape[0] // chunk_size
    n_c = cw * ch
    return (n_c, chunk_size, chunk_size, shape[2])


def shift_colors(img_arr, *, components, scale=0.1):
    '''
    Adjust colors a la Krizhevsky et al (2012).

    Arguments:
        components: matrix of lambda-scaled principal components, as rows.
        scale: std of pc coefficients
    '''

    noise = npr.normal(0, scale, img_arr.shape[-1])
    img_arr += noise @ components


def hilbertize_image(img_arr):
    w = img_arr.shape[0]

    o = np.zeros((w**2, img_arr.shape[-1]), dtype=np.float16)
    hix = hilbert_ixes(w)

    for i in range(w):
        for j in range(w):
            o[hix[i, j]] = img_arr[i, j]

    return o


def hilbertize_image_shape(img_shape):
    return (img_shape[0] * img_shape[1], img_shape[2])


def color_pca(img_arr, *, components):
    '''
    Transforms the image colorspace into the space defined by the given
    components.
    '''

    return img_arr @ components


def color_pca_shape(shape, *, components):
    return shape[:-1] + components.shape[-1:]
