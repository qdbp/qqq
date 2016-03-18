from keras.datasets import mnist
import keras.utils.np_utils as kun
import numpy as np

def get_raw_mnist():
    (Xl, yl), (Xt, yt) = mnist.load_data()

    XLEN = Xl.shape[1]
    YLEN = Xl.shape[2]
    LEN = XLEN*YLEN

    Xl = Xl.reshape(Xl.shape[0], LEN)
    Xt = Xt.reshape(Xt.shape[0], LEN)

    Yl = kun.to_categorical(yl)
    Yt = kun.to_categorical(yt)

    return Xl, Yl, Xt, Yt, XLEN, YLEN

def get_norm_mnist():

    Xl, Yl, Xt, Yt, XLEN, YLEN = get_raw_mnist() 

    IX = np.where(np.sum(Xl, axis=0) > 0)[0]
    
    Xl = Xl[:,IX]
    Xt = Xt[:,IX]
    
    m, s = np.mean(Xl, axis=0), np.std(Xl, axis=0)
    Xl = (Xl - m)/s
    Xt = (Xt - m)/s

    return Xl, Yl, Xt, Yt, XLEN, YLEN

