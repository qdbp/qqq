import numpy as np
import matplotlib.pyplot as plt

def plot_pixels(x, xl, yl):
    f = plt.figure()
    ax = f.add_subplot(111)

    x = x.astype(np.float)
    x /= np.max(np.abs(x))
    c = [(0, 0, 0, max(a, 0)) for a in x]
    flat = np.array((np.tile(np.linspace (0, yl, num=yl, endpoint=False, dtype=np.int), xl),
                     np.repeat(np.linspace(xl, 0, num=xl, endpoint=False, dtype=np.int), yl)))
    print(c)
    ax.scatter(flat[0], flat[1], c=c, marker='s', s=300, lw=0)

    return f

if __name__ == '__main__':
    from data import get_raw_mnist

    Xl, Yl, Xt, Yt, XLEN, YLEN = get_raw_mnist()

    f = plot_pixels(Xl.reshape((Xl.shape[0], XLEN*YLEN))[5], XLEN, YLEN)
    plt.show()
