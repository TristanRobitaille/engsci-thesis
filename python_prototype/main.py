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

#--- Encoder ---#
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.positional_encoding = positional_encoder(OUTPUT_SEQUENCE_LENGTH)
        
        #Layers
        self.voltage_embedding_layer = tf.keras.layers.Embedding(MAX_VOLTAGE, VOLTAGE_EMBEDDING_DEPTH)
        self.dropout = tf.keras.layers.Dropout(rate=DROPOUT_RATE)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.dense_relu = tf.keras.layers.Dense(FULLY_CONNECTED_DIM, activation='relu')
        self.dense = tf.keras.layers.Dense(VOLTAGE_EMBEDDING_DEPTH)

    def call(self, input:tf.Tensor, mask:tf.Tensor, training:bool=False):
        #1 - Voltage embedding
        voltage_embedding = self.voltage_embedding_layer(input)
        voltage_embedding *= tf.math.sqrt(tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32))

        #2 - Positonal encoding
        encoder_input = voltage_embedding + positional_encoder()

        #3 - [Training only] Apply dropout
        encoder_input = self.dropout(encoder_input, training=training)

        #4 - Run through encoder
        mha_output = self.mha(encoder_input, encoder_input, encoder_input, mask) #4.1 - Multi-head (self) attention
        skip_mha = self.norm(encoder_input + mha_output) #4.2 - Skip connection to Add & Norm
        fc_output = self.dense_relu(skip_mha) #4.3 - Feed MHA output to 2 Dense layers
        fc_output = self.dense(fc_output)
        fc_output = self.dropout(fc_output, training=training) #4.4 - [Training only] Apply dropout to FC
        return self.norm(skip_mha + fc_output) #4.5 - Normalize output

#--- Decoder ---#
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = positional_encoder(OUTPUT_SEQUENCE_LENGTH)
        self.attention_weights = {}

        #Layers
        self.embedding_layer = tf.keras.layers.Embedding(NUM_SLEEP_STAGES, VOLTAGE_EMBEDDING_DEPTH)
        self.dropout1 = tf.keras.layers.Dropout(rate=DROPOUT_RATE)
        self.dropout2 = tf.keras.layers.Dropout(rate=DROPOUT_RATE)
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.dense_relu = tf.keras.layers.Dense(FULLY_CONNECTED_DIM, activation='relu')
        self.dense = tf.keras.layers.Dense(VOLTAGE_EMBEDDING_DEPTH)

    def call(self, sleep_stage:tf.Tensor, encoder_output:tf.Tensor, padding_mask:tf.Tensor, training:bool=False):
        #1 - Voltage embedding
        embedding = self.embedding_layer(sleep_stage)
        embedding *= tf.math.sqrt(tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32))

        #2 - Positonal encoding
        decoder_input = embedding + self.positional_encoding

        #3 - [Training only] Apply dropout
        decoder_input = self.dropout1(decoder_input, training=training)

        #4 - Go through decoder
        mha1_output, mha1_attention_weights = self.mha1(decoder_input, decoder_input, decoder_input, None, return_attention_scores=True) #4.1 - Multi-head (self) attention
        skip_mha1 = self.norm1(sleep_stage + mha1_output) #4.2 - Skip connection to Add & Norm
        mha2_output, mha2_attention_weights = self.mha2(skip_mha1, encoder_output, encoder_output, padding_mask, return_attention_scores=True) #4.3 - Multi-head attention with encoder output
        skip_mha2 = self.norm2(skip_mha1 + mha2_output) #4.4 - Skip connection to Add & Norm
        fc_output = self.dense_relu(skip_mha2) #4.5 - Feed MHA output to 2 Dense layers
        fc_output = self.dense(fc_output)
        fc_output = self.dropout2(fc_output, training=training) #4.6 - [Training only] Apply dropout to FC
        decoder_output = self.norm3(skip_mha2 + fc_output) #4.7 - Normalize output

        self.attention_weights[f'decoder_layer1_block1_self_att'] = mha1_attention_weights
        self.attention_weights[f'decoder_layer1_block2_decenc_att'] = mha2_attention_weights

        return decoder_output, self.attention_weights

#--- Transformer ---#
class Transformer(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_dense_layer = tf.keras.layers.Dense(NUM_SLEEP_STAGES, activation='softmax')
    
    def call(self, inputs, training:int=False):
        eeg_clip, sleep_stage = inputs

        #Masks
        encoder_padding_mask = create_padding_mask(eeg_clip)
        encoder_padding_mask = tf.reshape(encoder_padding_mask, (1, CLIP_LENGTH, 1))
        decoder_padding_mask = tf.constant(True, shape=(1, OUTPUT_SEQUENCE_LENGTH, 1))

        encoder_output = self.encoder(eeg_clip, encoder_padding_mask, training)
        sleep_stage, attention_weights = self.decoder(sleep_stage, encoder_output, padding_mask=decoder_padding_mask, training=training)

        final_softmax = self.final_dense_layer(sleep_stage)

        return final_softmax

#--- Learning rate schedule ---#
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=4000):
    super().__init__()

    self.d_model = tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def main():
    eeg_clip_batch = random_input_batch()
    sleep_stage = tf.zeros((OUTPUT_SEQUENCE_LENGTH, 1))

    model = Transformer()
    output = model((eeg_clip_batch, sleep_stage))
    model.summary()
    
    #Training
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

if __name__ == "__main__":
    main()