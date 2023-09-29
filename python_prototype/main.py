"""
Python prototype of transdformer model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#--- Notes ---#
#Assume input data is a single sequential vector of voltages: x = [x0, x1, ... , xn] (can augment to multichannel later)
#Assume x[0] is the oldest measurement.
#There are only a few sleep stage classes, and y = [sleep stage for given clip]

#--- Constants ---#
WINDOW_LENGTH = 7680 #Number of voltage samples @ 256Hz -> 30s clips
VOLTAGE_EMBEDDING_DEPTH = 25 #Length of voltage embedding vector for each timestep

PRINT_POSITION_EMBEDDING = False 

#---Encoder---#
#1) Voltage embedding

def positional_encoder(window_length:int=WINDOW_LENGTH, embed_depth:int=VOLTAGE_EMBEDDING_DEPTH) -> np.ndarray:
    """
    Return a matrix of dimensions WINDOW_LENGTH, VOLTAGE_EMBEDDING_DEPTH where each row represents
    the position encoding for a particular position in the input sequence voltage vector.
    """
    voltage_pos = np.arange(stop=window_length)[:, np.newaxis]
    embedding_pos = 2 * np.arange(stop=embed_depth) // 2

    angle_matrix_rads = voltage_pos / 10000**(embedding_pos/embed_depth)

    angle_matrix_rads[:, 0::2] = np.sin(angle_matrix_rads[:, 0::2])
    angle_matrix_rads[:, 1::2] = np.cos(angle_matrix_rads[:, 1::2])

    angle_matrix_rads = tf.cast(angle_matrix_rads, dtype=tf.float16)
    
    if PRINT_POSITION_EMBEDDING:
        plt.pcolormesh(angle_matrix_rads, cmap='RdBu')
        plt.title('Position encoding matrix')
        plt.xlabel('Position in voltage embedding')
        plt.xlim((0, VOLTAGE_EMBEDDING_DEPTH))
        plt.ylabel('Voltage position in input vector')
        plt.colorbar()
        plt.show()

    return angle_matrix_rads

def truncate(input_sequence:tf.Tensor) -> tf.Tensor:
    """
    Truncate input sequence to (WINDOW_LENGTH) if it is too long. Return unchanged if not too long.
    """

    if input_sequence.shape[0] > WINDOW_LENGTH:
        return input_sequence[-WINDOW_LENGTH-1:-1]
    else:
        return input_sequence


def padding_mask(input:tf.Tensor) -> tf.Tensor:
    """
    Returns a mask to avoid attending to padded values (if input sequence is too short). 1 to attend, 0 to ignore.
    We ignore values in the oldest positions (lower indices)
    """

    mask_keep = tf.constant(1, dtype=tf.int16, shape=(input.shape[0]))
    
    if input.shape[0] > WINDOW_LENGTH: #No need to pad
        return mask_keep[0:WINDOW_LENGTH]
    else:
        return tf.concat([tf.constant(0, dtype=tf.int16, shape=(WINDOW_LENGTH-input.shape[0])), mask_keep], axis=0)

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
    input = tf.constant(3, dtype=tf.int16, shape=(WINDOW_LENGTH-5))
    print(padding_mask(input))

if __name__ == "__main__":
    main()