"""
Python prototype of transdformer model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#TODO
#-Consider using multiple encoder and decoder layers

#--- Notes ---#
#Assume input data is a single sequential vector of voltages: x = [x0, x1, ... , xn] (can augment to multichannel later)
#Assume x[0] is the oldest measurement.
#There are only a few sleep stage classes, and y = [sleep stage for given clip]

#--- Constants and Hyperparameters ---#
CLIP_LENGTH = int(3840/4) #Number of voltage samples @ 256Hz -> 30s clips
OUTPUT_SEQUENCE_LENGTH = 1
MAX_VOLTAGE = 2**16 #Maximum ADC code output
NUM_SLEEP_STAGES = 5
DATA_TYPE = tf.float32
INITIAL_SLEEP_STAGE = 0

VOLTAGE_EMBEDDING_DEPTH = 32 #Length of voltage embedding vector for each timestep. I let model dimensionality = embedding depth
BATCH_SIZE = 16
FULLY_CONNECTED_DIM = 64
MHA_NUM_HEADS = 8
LAYER_NORM_EPSILON = 0.5
DROPOUT_RATE = 0.1

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

def random_input_batch(length:int=CLIP_LENGTH) -> tf.Tensor:
    random_values = tf.random.normal((BATCH_SIZE, length), mean=0.0, stddev=1.0, dtype=tf.float32)
    return tf.math.abs(random_values)

#--- Encoder ---#
def positional_encoder(length:int=CLIP_LENGTH) -> tf.Tensor:
    """
    Return a matrix of dimensions (BATCH_SIZE, length, VOLTAGE_EMBEDDING_DEPTH) where each row represents
    the position encoding for a particular position in the input sequence voltage vector.
    """
    voltage_pos = np.arange(stop=length)[:, np.newaxis]
    embedding_pos = 2 * np.arange(stop=VOLTAGE_EMBEDDING_DEPTH) // 2

    angle_matrix_rads = voltage_pos / 10000**(embedding_pos/VOLTAGE_EMBEDDING_DEPTH)

    angle_matrix_rads[:, 0::2] = np.sin(angle_matrix_rads[:, 0::2])
    angle_matrix_rads[:, 1::2] = np.cos(angle_matrix_rads[:, 1::2])

    angle_matrix_rads = tf.cast(angle_matrix_rads, dtype=DATA_TYPE)

    angle_matrix_rads = tf.expand_dims(angle_matrix_rads, axis=0)
    angle_matrix_rads = tf.tile(angle_matrix_rads, multiples=[BATCH_SIZE, 1, 1])

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

def create_padding_mask(input:tf.Tensor) -> tf.Tensor:
    """
    Returns a mask to avoid attending to padded values (if input sequence is too short). 1 to attend, 0 to ignore.
    We ignore values in the oldest positions (lower indices)
    """

    mask_keep = tf.constant(1, shape=(input.shape[1]), dtype=DATA_TYPE)

    if input.shape[1] > CLIP_LENGTH: #No need to pad
        return mask_keep[0:CLIP_LENGTH]
    else:
        return tf.concat([tf.constant(0, dtype=DATA_TYPE, shape=(CLIP_LENGTH-input.shape[1])), mask_keep], axis=0)

def self_attention(q:tf.Tensor, k:tf.Tensor, v:tf.Tensor, mask:tf.Tensor) -> (tf.Tensor, tf.Tensor):
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

def FullyConnected():
    return tf.keras.Sequential([tf.keras.layers.Dense(FULLY_CONNECTED_DIM, activation='relu'), tf.keras.layers.Dense(VOLTAGE_EMBEDDING_DEPTH)])

def encoder_layer(input:tf.Tensor, mask:tf.Tensor, training:bool=False) -> tf.keras.layers.Layer:
    """
    Single encoder layer
    Dimensions:
        input -> (BATCH_SIZE, CLIP_LENGTH, FULLY_CONNECTED_DIM)
        mask -> (1, CLIP_LENGTH, 1)
    """
    #1 - Multi-head (self) attention
    mha_output = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)(input, input, input, mask)

    #2 - Skip connection to Add & Norm
    skip_mha = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(input + mha_output)

    #3 - Feed MHA output to Fully Connected
    fc_output = FullyConnected()(skip_mha)

    #4 - [Training only] Apply dropout to FC
    fc_output = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(fc_output, training=training)

    #5 - Normalize output
    return tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(skip_mha + fc_output)

def encoder_fwd_pass(input:tf.Tensor, mask:tf.Tensor, training:bool=False):
    """
    Performs one forward propagation through the encoder block
    Dimensions:
        -input: (BATCH_SIZE, CLIP_LENGTH)
        -mask: (1, CLIP_LENGTH, 1)
        -output: (BATCH_SIZE, CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH)
    """

    #1 - Voltage embedding
    voltage_embedding = tf.keras.layers.Embedding(MAX_VOLTAGE, VOLTAGE_EMBEDDING_DEPTH)(input)
    voltage_embedding *= tf.math.sqrt(tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32))

    #2 - Positonal encoding
    encoder_input = voltage_embedding + positional_encoder()

    #3 - [Training only] Apply dropout
    encoder_input = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(encoder_input, training=training)

    #4 - Pass through encoder
    return encoder_layer(encoder_input, mask, training)

#--- Decoder ---#
def decoder_layer(input:tf.Tensor, encoder_output:tf.Tensor, padding_mask:tf.Tensor, training:bool=False):
    """
    Single decoder layer
    Dimensions:
        input -> (BATCH_SIZE, CLIP_LENGTH, FULLY_CONNECTED_DIM)
        encoder_output -> (BATCH_SIZE, CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH)
        padding_mask -> (1, OUTPUT_SEQUENCE_LENGTH, 1)
    """

    #1 - Multi-head (self) attention
    mha1_output, mha1_attention_weights = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)(input, input, input, None, return_attention_scores=True)

    #2 - Skip connection to Add & Norm
    skip_mha1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(input + mha1_output)

    #3 - Multi-head attention with encoder output
    mha2_output, mha2_attention_weights = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)(skip_mha1, encoder_output, encoder_output, padding_mask, return_attention_scores=True)

    #4 - Skip connection to Add & Norm
    skip_mha2 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(skip_mha1 + mha2_output)

    #5 - Feed MHA output to Fully Connected
    fc_output = FullyConnected()(skip_mha2)

    #6 - [Training only] Apply dropout to FC
    fc_output = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(fc_output, training=training)

    #7 - Normalize output
    decoder_output = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)(skip_mha2 + fc_output)

    return decoder_output, mha1_attention_weights, mha2_attention_weights

def decoder_fwd_pass(input:tf.Tensor, encoder_output:tf.Tensor, padding_mask:tf.Tensor, training:bool=False):
    """
    Performs one forward propagation through the encoder block
    Dimensions:
        input -> (BATCH_SIZE, CLIP_LENGTH, FULLY_CONNECTED_DIM)
        encoder_output -> (BATCH_SIZE, CLIP_LENGTH, VOLTAGE_EMBEDDING_DEPTH)
        padding_mask -> (1, OUTPUT_SEQUENCE_LENGTH, 1)
    """
    attention_weights = {}

    #1 - Voltage embedding
    embedding = tf.keras.layers.Embedding(NUM_SLEEP_STAGES, VOLTAGE_EMBEDDING_DEPTH)(input)
    embedding *= tf.math.sqrt(tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32))

    #2 - Positonal encoding
    decoder_input = embedding + positional_encoder(OUTPUT_SEQUENCE_LENGTH)

    #3 - [Training only] Apply dropout
    decoder_input = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(decoder_input, training=training)

    #4 - Go through decoder
    decoder_output, block1, block2 = decoder_layer(decoder_input, encoder_output, padding_mask, training=training)
    attention_weights[f'decoder_layer1_block1_self_att'] = block1
    attention_weights[f'decoder_layer1_block2_decenc_att'] = block2

    return decoder_output, attention_weights

#--- Transformer ---#
def transformer(eeg_clip:tf.Tensor, sleep_stage:tf.Tensor, encoder_padding_mask:tf.Tensor, decoder_padding_mask:tf.Tensor, training:bool=False):
    #Go through encoder and decoder
    encoder_output = encoder_fwd_pass(eeg_clip, encoder_padding_mask, training)
    decoder_output, attention_weights = decoder_fwd_pass(sleep_stage, encoder_output, decoder_padding_mask, training)

    #Final softmax
    final_softmax = tf.keras.layers.Dense(NUM_SLEEP_STAGES, activation='softmax')(decoder_output)

    return final_softmax, attention_weights

def main():
    eeg_clip = random_input_batch()
    padding_mask = create_padding_mask(eeg_clip)
    padding_mask = tf.reshape(padding_mask, (1, CLIP_LENGTH, 1))

    sleep_stage_softmax, attention_weights = transformer(eeg_clip=eeg_clip, sleep_stage=tf.constant(INITIAL_SLEEP_STAGE, shape=(1)), 
                                                        encoder_padding_mask=padding_mask, decoder_padding_mask=tf.constant(True, shape=(1, 1, 1)), training=False)
    
if __name__ == "__main__":
    main()