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
CLIP_LENGTH = 7680 #Number of voltage samples @ 256Hz -> 30s clips
VOLTAGE_EMBEDDING_DEPTH = 32 #Length of voltage embedding vector for each timestep. I let model dimensionality = embedding depth
DATA_TYPE = tf.float32

PRINT_POSITION_EMBEDDING = False
PRINT_SELF_ATTENTION = False

#--- Helpers ---#
def plot_matrix_colours(matrix, title:str, xlabel:str="X", ylabel:str="Y", block_execution:bool=True) -> None:
    fig = plt.figure()    
    plt.pcolormesh(matrix, cmap='RdBu')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.show(block=block_execution)

#--- Encoder ---#
def positional_encoder() -> np.ndarray:
    """
    Return a matrix of dimensions CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH where each row represents
    the position encoding for a particular position in the input sequence voltage vector.
    """
    voltage_pos = np.arange(stop=CLIP_LENGTH)[:, np.newaxis]
    embedding_pos = 2 * np.arange(stop=VOLTAGE_EMBEDDING_DEPTH) // 2

    angle_matrix_rads = voltage_pos / 10000**(embedding_pos/VOLTAGE_EMBEDDING_DEPTH)

    angle_matrix_rads[:, 0::2] = np.sin(angle_matrix_rads[:, 0::2])
    angle_matrix_rads[:, 1::2] = np.cos(angle_matrix_rads[:, 1::2])

    angle_matrix_rads = tf.cast(angle_matrix_rads, dtype=DATA_TYPE)

    if PRINT_POSITION_EMBEDDING:
        plot_matrix_colours(angle_matrix_rads, title='Position encoding matrix', xlabel='Position in voltage embedding', ylabel='Position in voltage embedding')

    return angle_matrix_rads

def truncate(input_sequence:tf.Tensor) -> tf.Tensor:
    """
    Truncate input sequence to (CLIP_LENGTH) if it is too long. Return unchanged if not too long.
    """

    if input_sequence.shape[0] > CLIP_LENGTH:
        return input_sequence[-CLIP_LENGTH-1:-1]
    else:
        return input_sequence

def padding_mask(input:tf.Tensor) -> tf.Tensor:
    """
    Returns a mask to avoid attending to padded values (if input sequence is too short). 1 to attend, 0 to ignore.
    We ignore values in the oldest positions (lower indices)
    """

    mask_keep = tf.constant(1, shape=(input.shape[0]), dtype=DATA_TYPE)

    if input.shape[0] > CLIP_LENGTH: #No need to pad
        return mask_keep[0:CLIP_LENGTH]
    else:
        return tf.concat([tf.constant(0, dtype=DATA_TYPE, shape=(CLIP_LENGTH-input.shape[0])), mask_keep], axis=0)

def self_attention(q:tf.Tensor, k:tf.Tensor, v:tf.Tensor, mask) -> (tf.Tensor, tf.Tensor):
    """
    Compute the self-attention given query (q), key (k) and value (v) matrices.
    Dimensions:
        q -> (CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH)
        k -> (CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH)
        v -> (CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH)
        mask -> (CLIP_LENGTH). Specifies which values in the input clip we should attend to
    """

    qk_t = tf.matmul(q, k, transpose_b=True) #QK^T
    qk_t = qk_t / np.sqrt(VOLTAGE_EMBEDDING_DEPTH)
    qk_t_masked = qk_t + (1 - mask)*-1e9
    self_attention_weights = tf.nn.softmax(qk_t_masked, axis=-1)
    self_attention = tf.matmul(self_attention_weights, v)

    if PRINT_SELF_ATTENTION:
        plot_matrix_colours(self_attention, title="Self-attention", block_execution=False)
        plot_matrix_colours(self_attention_weights, title="Self-attention weights")

    return self_attention, self_attention_weights

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
    query = tf.random.normal(shape=(CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH), mean=0, stddev=0.5, dtype=DATA_TYPE)
    key =   tf.random.normal(shape=(CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH), mean=0, stddev=0.5, dtype=DATA_TYPE)
    value = tf.random.normal(shape=(CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH), mean=0, stddev=0.5, dtype=DATA_TYPE)

    mask = padding_mask(tf.constant(value=10425, dtype=DATA_TYPE, shape=(CLIP_LENGTH)))

    self_attention_out, self_attention_weights = self_attention(query, key, value, mask)

if __name__ == "__main__":
    main()