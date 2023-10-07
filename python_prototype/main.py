"""
Python prototype of transdformer model
"""

import os
import datetime
import pkg_resources
import tensorflow as tf
import numpy as np

#TODO
#-Consider using multiple encoder and decoder layers
#-Proper masking and padding
#-Fix batch sizes
#-No global constants; pass in parameters instead

#--- Notes ---#
#Assume input data is a single sequential vector of voltages: x = [x0, x1, ... , xn] (can augment to multichannel later)
#Assume x[0] is the oldest measurement.
#There are only a few sleep stage classes, and y = [sleep stage for given clip]

#--- Constants and Hyperparameters ---#
CLIP_LENGTH = int(7680) #Number of voltage samples @ 256Hz -> 30s clips
OUTPUT_SEQUENCE_LENGTH = 1
MIN_VOLTAGE = 0
MAX_VOLTAGE = 2**16-1 #Maximum ADC code output
NUM_SLEEP_STAGES = 5 + 1 #5 stages + unknown
DATA_TYPE = tf.float32

VOLTAGE_EMBEDDING_DEPTH = 32 #Length of voltage embedding vector for each timestep. I let model dimensionality = embedding depth
BATCH_SIZE = 1
FULLY_CONNECTED_DIM = 64
MHA_NUM_HEADS = 8
LAYER_NORM_EPSILON = 0.5
DROPOUT_RATE = 0.1

PRINT_POSITION_EMBEDDING = False
PRINT_SELF_ATTENTION = False

if PRINT_POSITION_EMBEDDING or PRINT_SELF_ATTENTION:
    import matplotlib.pyplot as plt

#--- Helpers ---#
def plot_matrix_colours(matrix, title:str, xlabel:str="X", ylabel:str="Y", block_execution:bool=True) -> None:
    fig = plt.figure()
    plt.pcolormesh(matrix, cmap='RdBu')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.show(block=block_execution)

def random_dataset(input_sequence_length:int=CLIP_LENGTH, output_sequence_length:int=OUTPUT_SEQUENCE_LENGTH, num_sequences:int=1000) -> (tf.Tensor, tf.Tensor):
    """
    Returns a tuple of NumPy arrays: (input sequence, sleep stage).
    The give some sort of relationship between the input and output, we generate input data from a different range based on the output.
    Dimensions:
        -input_sequence: (num_sequences, input_sequence_length)
        -sleep_stage: (num_sequences, output_sequence_length)
    """
    sleep_stage_dataset = tf.random.uniform(shape=[num_sequences], minval=1, maxval=NUM_SLEEP_STAGES-1) #Sleep stage of zero means uninitialized
    sleep_stage_dataset = tf.math.round(sleep_stage_dataset)
    input_dataset = tf.zeros(shape=(0, input_sequence_length))
   
    for sleep_stage in sleep_stage_dataset:
        mean = (2**16-1) * sleep_stage/NUM_SLEEP_STAGES

        new_input_sequence = tf.random.normal(shape=(1, input_sequence_length), mean=mean, stddev=5000)
        new_input_sequence = tf.clip_by_value(new_input_sequence, clip_value_min=MIN_VOLTAGE, clip_value_max=MAX_VOLTAGE-1)
        new_input_sequence = tf.round(new_input_sequence)
        input_dataset = tf.concat([input_dataset, new_input_sequence], axis=0)

    return input_dataset, sleep_stage_dataset

def positional_encoder(num_clips:int=1, length:int=CLIP_LENGTH) -> tf.Tensor:
    """
    Return a matrix of dimensions (num_clips, length, VOLTAGE_EMBEDDING_DEPTH) where each row represents
    the position encoding for a particular position in the input sequence voltage vector.
    """
    voltage_pos = np.arange(stop=length)[:, np.newaxis]
    embedding_pos = 2 * np.arange(stop=VOLTAGE_EMBEDDING_DEPTH) // 2

    angle_matrix_rads = voltage_pos / 10000**(embedding_pos/VOLTAGE_EMBEDDING_DEPTH)

    angle_matrix_rads[:, 0::2] = np.sin(angle_matrix_rads[:, 0::2])
    angle_matrix_rads[:, 1::2] = np.cos(angle_matrix_rads[:, 1::2])

    angle_matrix_rads = tf.cast(angle_matrix_rads, dtype=DATA_TYPE)

    angle_matrix_rads = tf.expand_dims(angle_matrix_rads, axis=0)
    angle_matrix_rads = tf.tile(angle_matrix_rads, multiples=[num_clips, 1, 1])

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

    return tf.ones(shape=(1, CLIP_LENGTH), dtype=DATA_TYPE)

    if input.shape[0] is None: input_length = input.shape[1]
    else: input_length = input.shape[0]

    mask_keep = tf.ones(shape=(1, input_length), dtype=DATA_TYPE)

    if input_length > CLIP_LENGTH: #No need to pad
        return mask_keep[0:CLIP_LENGTH]
    else:
        return tf.concat([tf.zeros(shape=(1, CLIP_LENGTH-input_length), dtype=DATA_TYPE), mask_keep], axis=0)

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
        #Layers
        self.voltage_embedding_layer = tf.keras.layers.Embedding(input_dim=MAX_VOLTAGE, output_dim=VOLTAGE_EMBEDDING_DEPTH, input_length=CLIP_LENGTH)
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
        if (input.shape == (CLIP_LENGTH,)): encoder_input = voltage_embedding + positional_encoder(1, OUTPUT_SEQUENCE_LENGTH)
        else: encoder_input = voltage_embedding + positional_encoder(input.shape[0], OUTPUT_SEQUENCE_LENGTH)

        #3 - [Training only] Apply dropout
        encoder_input = self.dropout(encoder_input, training=training) #encoder_input: (batch size, clip_length, embedding_depth)


        #4 - Run through encoder
        mha_output = self.mha(encoder_input, encoder_input, encoder_input, mask) #4.1 - Multi-head (self) attention
        skip_mha = self.norm(encoder_input + mha_output) #4.2 - Skip connection to Add & Norm
        fc_output = self.dense_relu(skip_mha) #4.3 - Feed MHA output to 2 Dense layers
        fc_output = self.dense(fc_output)
        fc_output = self.dropout(fc_output, training=training) #4.4 - [Training only] Apply dropout to FC
        out = self.norm(skip_mha + fc_output) #4.5 - Normalize output
        return out #out: (batch size, clip_length, embedding_depth)

#--- Decoder ---#
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
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
        #sleep_stage: (batch size, output_sequence_length)
        #encoder_output: (batch_size, input_sequence_length, embedding depth)

        #1 - Voltage embedding
        embedding = self.embedding_layer(sleep_stage)
        embedding *= tf.math.sqrt(tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32))

        #2 - Positonal encoding
        if (embedding.shape == (1,VOLTAGE_EMBEDDING_DEPTH)): decoder_input = embedding + positional_encoder(1, OUTPUT_SEQUENCE_LENGTH)
        else: decoder_input = embedding + positional_encoder(encoder_output.shape[0], OUTPUT_SEQUENCE_LENGTH)

        #3 - [Training only] Apply dropout
        decoder_input = self.dropout1(decoder_input, training=training) #decoder_input: (batch size, output_sequence_length, embedding_depth)

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

        return decoder_output, self.attention_weights #decoder_output: (batch size, output_sequence_length, embedding depth)

#--- Transformer ---#
class Transformer(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_dense_layer = tf.keras.layers.Dense(NUM_SLEEP_STAGES, activation='softmax')

    @tf.function
    def call(self, inputs, training):
        eeg_clip, sleep_stage = inputs

        eeg_clip = tf.reshape(eeg_clip, (1, CLIP_LENGTH))
        sleep_stage = tf.reshape(sleep_stage, (1, OUTPUT_SEQUENCE_LENGTH))

       #Masks
        encoder_padding_mask = create_padding_mask(eeg_clip)
        encoder_padding_mask = tf.reshape(encoder_padding_mask, (1, CLIP_LENGTH, 1))
        decoder_padding_mask = tf.constant(True, shape=(1, OUTPUT_SEQUENCE_LENGTH, 1))

        encoder_output = self.encoder(eeg_clip, encoder_padding_mask, training)        
        sleep_stage, attention_weights = self.decoder(sleep_stage, encoder_output, padding_mask=decoder_padding_mask, training=training)

        final_softmax = self.final_dense_layer(sleep_stage)

        return final_softmax #final_softmax: (batch size, output_sequence_length, num_sleep_stages+1)

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

#--- Loss and Accuracy ---#
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

def main():
    # Load data
    if (pkg_resources.get_distribution("tensorflow").version == "2.8.0+computecanada"):
        data = tf.data.experimental.load(os.getcwd() + '/SS3_EDF_Tensorized')
    else:
        data = tf.data.Dataset.load(os.getcwd() + '/SS3_EDF_Tensorized')
    iterator = iter(data)
    x, y, z = next(iterator)

    x = x[0:4999]
    y = y[0:4999]

    x = tf.cast(x, dtype=DATA_TYPE)
    y = tf.cast(y, dtype=DATA_TYPE)

    clip_train_cutoff = int(0.95*x.shape[0])
    x_train = x[0:clip_train_cutoff, 0, :]
    y_train = y[0:clip_train_cutoff:, 0, :]
    x_test = x[clip_train_cutoff:-1, 0, :]
    y_test = y[clip_train_cutoff:-1:, 0, :]

    y_uninitialized = tf.zeros(shape=[x_train.shape[0]])

    # Model
    model = Transformer()
    model.compile(
        loss=masked_loss,
        optimizer=tf.keras.optimizers.Adam(CustomSchedule(), beta_1=0.9, beta_2=0.98, epsilon=1e-9),
        metrics=[masked_accuracy]
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=(x_train, y_uninitialized), y=y_train, epochs=20, batch_size=1,
              validation_split=0.1, callbacks=[tensorboard_callback])

    #Manual validation
    total_correct = 0
    total = 0

    for x,y in zip(x_test, y_test):
        sleep_stage = model.call(inputs=(x, tf.zeros(shape=[1])), training=False)
        total += 1
        if (tf.argmax(sleep_stage[0][0]) == tf.cast(y, dtype=tf.int64)): total_correct += 1
        print(f"Ground truth: {y}, sleep stage pred: {tf.argmax(sleep_stage[0][0])}, accuracy: {total_correct/total:.4f}")

if __name__ == "__main__":
    main()
