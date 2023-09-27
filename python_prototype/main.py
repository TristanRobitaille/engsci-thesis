"""
Python prototype of transdformer model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#--- Notes ---#
#Assume input data is a single sequential vector of voltages: x = [x0, x1, ... , xn] (can augment to multichannel later)
#There are only a few sleep stage classes, and y = [y0, y1, ... , yn] is the sleep stage at each timestep

#--- Constants ---#
WINDOW_LENGTH = 100 #Number of voltage samples (and corresponding sleep stage outputs)
VOLTAGE_EMBEDDING_DEPTH = 25 #Length of voltage embedding vector for each timestep ('d')

#---Encoder---#
#1) Voltage embedding

#2) Positional encoding
def positional_encoder(window_length=WINDOW_LENGTH, embed_depth=VOLTAGE_EMBEDDING_DEPTH):
    """
    Return a matrix of dimensions WINDOW_LENGTH, VOLTAGE_EMBEDDING_DEPTH where each row represents
    the position encoding for a particular position in the input sequence voltage vector.
    """
    voltage_pos = np.arange(stop=window_length)[:, np.newaxis]
    embedding_pos = 2 * np.arange(stop=embed_depth) // 2

    angle_matrix_rads = voltage_pos / 10000**(embedding_pos/embed_depth)

    angle_matrix_rads[:, 0::2] = np.sin(angle_matrix_rads[:, 0::2])
    angle_matrix_rads[:, 1::2] = np.cos(angle_matrix_rads[:, 1::2])

    return angle_matrix_rads

#3) Multi-head attention

#4) Add & Norm 

#5) Feedforward on attention vectors

#6) Add & Norm

#---Decoder---#
#1) Output embedding

#2) Positional encoding

#3) Masked multi-head attention

#4) Add & Norm

#5) Multi-head attention (with encoder input)

#6) Feedforward on attention vectors

#5) Add & Norm

#6) Linear layer

#7) Softmax

def main():
    pos_encoding = positional_encoder()
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.xlabel('Voltage position in input vector')
    plt.xlim((0, VOLTAGE_EMBEDDING_DEPTH))
    plt.ylabel('Position in voltage embedding')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()