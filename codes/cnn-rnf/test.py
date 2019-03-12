import numpy as np

from keras import backend as K
from keras.models import Model
from keras.engine import Layer
from keras.layers import *
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.convolutional import *
from keras.utils import np_utils

def max(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.max(x, axis=axis, keepdims=keepdims)

def test_max():
    x = np.array([
        [1, 2, 1, 4],
        [2, 1, 5, 1],
        [3, 1, 1, 1]
    ])

    print(max(x, 0))
    """ [3 2 5 4] """

def lstm_base():
    ## LSTM
    sequence = Input(shape=(4, 5), dtype='float32')
    print("lstm_base:", sequence)
    outputs = LSTM(6)(sequence)
    print("lstm_base:", outputs)

def lstm_time():
    ## LSTM TimeDistributed
    sequence = Input(shape=(3, 4, 5), dtype='float32')
    print("lstm_time:", sequence)
    outputs = TimeDistributed(LSTM(6))(sequence)
    print("lstm_time:", outputs)

def conv_base():
    ## Conv2D
    sequence = Input(shape=(299, 299, 3), dtype='float32')
    print("conv_time:", sequence)
    outputs = Conv2D(64, (3, 3))(sequence)
    print("conv_time:", outputs)

def conv_time():
    ## Conv2D TimeDistributed
    sequence = Input(shape=(10, 299, 299, 3), dtype='float32')
    print("conv_time:", sequence)
    outputs = TimeDistributed(Conv2D(64, (3, 3)))(sequence)
    print("conv_time:", outputs)

def test_model():
    # lstm_base()
    # lstm_time()
    conv_base()
    # conv_time()

if __name__=="__main__":
    # test_max()
    test_model()