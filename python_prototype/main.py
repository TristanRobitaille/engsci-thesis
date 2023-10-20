"""
Python prototype of transformer model

Arguments:
    --clip_length_s: Clip length (in sec). Must be one of 3.25, 5, 7.5, 10, 15, 30. Must match input dataset clip length. Default = 30s.
    --input_dataset: Filepath of the dataset used for training and validation.
    --num_training_clips: Number of clips to use for training. Defaults to 3000.
"""

import datetime
import pkg_resources
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from io import StringIO 
import sys

#TODO
#-Consider using multiple encoder and decoder layers, understand it 
#-Proper masking and padding
#-Fix batch sizes
#-No global constants; pass in parameters instead

#--- Notes ---#
#Assume input data is a single sequential vector of voltages: x = [x0, x1, ... , xn] (can augment to multichannel later)
#Assume x[0] is the oldest measurement.
#There are only a few sleep stage classes, and y = [sleep stage for given clip]

#--- Constants and Hyperparameters ---#
DEFAULT_CLIP_LENGTH_S = int(30)
SAMPLING_FREQUENCY_HZ = int(256)
OUTPUT_SEQUENCE_LENGTH = 1
MIN_VOLTAGE = 0
MAX_VOLTAGE = 2**16-1 #Maximum ADC code output
NUM_SLEEP_STAGES = 5 #Excludes 'unknown'
DATA_TYPE = tf.float32
USE_HISTORICAL_LOOKBACK = False
HISTORICAL_LOOKBACK_LENGTH = 8
NUM_EPOCHS = 10
TEST_SET_RATIO = 0.05 #Percentage of training data reserved for validation

VOLTAGE_EMBEDDING_DEPTH = 32 #Length of voltage embedding vector for each timestep. I let model dimensionality = embedding depth
BATCH_SIZE = 1
FULLY_CONNECTED_DIM = 64
MHA_NUM_HEADS = 8
LAYER_NORM_EPSILON = 0.5
DROPOUT_RATE = 0.1

PRINT_POSITION_EMBEDDING = False
PRINT_SELF_ATTENTION = False
VERBOSITY = 'NORMAL' #'QUIET', 'NORMAL', 'DETAILED'

if PRINT_POSITION_EMBEDDING or PRINT_SELF_ATTENTION:
    import matplotlib.pyplot as plt

#--- Helpers ---#
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def plot_matrix_colours(matrix, title:str, xlabel:str="X", ylabel:str="Y", block_execution:bool=True) -> None:
    fig = plt.figure()
    plt.pcolormesh(matrix, cmap='RdBu')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.show(block=block_execution)

def random_dataset(input_sequence_length:int, output_sequence_length:int=OUTPUT_SEQUENCE_LENGTH, num_sequences:int=1000) -> (tf.Tensor, tf.Tensor):
    """
    Returns a tuple of NumPy arrays: (input sequence, sleep stage).
    The give some sort of relationship between the input and output, we generate input data from a different range based on the output.
    Dimensions:
        -input_sequence: (num_sequences, input_sequence_length)
        -sleep_stage: (num_sequences, output_sequence_length)
    """
    sleep_stage_dataset = tf.random.uniform(shape=[num_sequences], minval=1, maxval=NUM_SLEEP_STAGES) #Sleep stage of zero means uninitialized
    sleep_stage_dataset = tf.math.round(sleep_stage_dataset)
    input_dataset = tf.zeros(shape=(0, input_sequence_length))
   
    for sleep_stage in sleep_stage_dataset:
        mean = (2**16-1) * sleep_stage/NUM_SLEEP_STAGES

        new_input_sequence = tf.random.normal(shape=(1, input_sequence_length), mean=mean, stddev=5000)
        new_input_sequence = tf.clip_by_value(new_input_sequence, clip_value_min=MIN_VOLTAGE, clip_value_max=MAX_VOLTAGE-1)
        new_input_sequence = tf.round(new_input_sequence)
        input_dataset = tf.concat([input_dataset, new_input_sequence], axis=0)

    return input_dataset, sleep_stage_dataset

def positional_encoder(sequence_length:int, batch_size:int=1) -> tf.Tensor:
    """
    Return a matrix of dimensions (batch_size, sequence_length, VOLTAGE_EMBEDDING_DEPTH) where each row represents
    the position encoding for a particular position in the input sequence voltage vector.
    """
    voltage_pos = np.arange(stop=sequence_length)[:, np.newaxis]
    embedding_pos = 2 * np.arange(stop=VOLTAGE_EMBEDDING_DEPTH) // 2

    angle_matrix_rads = voltage_pos / 10000**(embedding_pos/VOLTAGE_EMBEDDING_DEPTH)

    angle_matrix_rads[:, 0::2] = np.sin(angle_matrix_rads[:, 0::2])
    angle_matrix_rads[:, 1::2] = np.cos(angle_matrix_rads[:, 1::2])

    angle_matrix_rads = tf.cast(angle_matrix_rads, dtype=DATA_TYPE)

    angle_matrix_rads = tf.expand_dims(angle_matrix_rads, axis=0)
    angle_matrix_rads = tf.tile(angle_matrix_rads, multiples=[batch_size, 1, 1])

    if PRINT_POSITION_EMBEDDING:
        plot_matrix_colours(angle_matrix_rads, title='Position encoding matrix', xlabel='Position in sequence', ylabel='Position in voltage embedding')

    return angle_matrix_rads

def maskify(input:tf.Tensor) -> tf.Tensor:
    """
    Returns a Tensor of boolean values of the same shape as input, where an element is 
    False if the corresponding input element is 0, else is True.
    """
    nonzero_mask = tf.math.not_equal(input, 0)

    return tf.cast(nonzero_mask, tf.bool)

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

def export_summary(parser, model, accuracy, sleep_stages_count_training, sleep_stages_count_pred, num_training_clips, num_test_clips) -> None:
    """
    Saves model and training summary to file
    """
    with Capturing() as model_summary:
        model.summary()
    model_summary = "\n".join(model_summary)
    
    log = "TRANSFORMER MODEL TRAINING SUMMARY\n"
    log += model_summary
    log += f"\nTest set accuracy: {accuracy:.4f}\n"
    log += f"Dataset: {parser.input_dataset}\n"
    log += f"Channel: {parser.input_channel}\n"
    log += f"CLIP_LENGTH (s): {int(parser.clip_length_s)/SAMPLING_FREQUENCY_HZ:.2f}\n"
    log += f"NUM_TRAINING_CLIPS: {int(parser.num_training_clips)}\n"
    log += f"OUTPUT_SEQUENCE_LENGTH: {OUTPUT_SEQUENCE_LENGTH}\n"
    log += f"NUM_SLEEP_STAGES: {NUM_SLEEP_STAGES}\n"
    log += f"DATA_TYPE: {DATA_TYPE}\n"
    log += f"VOLTAGE_EMBEDDING_DEPTH: {VOLTAGE_EMBEDDING_DEPTH}\n"
    log += f"BATCH_SIZE: {BATCH_SIZE}\n"
    log += f"FULLY_CONNECTED_DIM: {FULLY_CONNECTED_DIM}\n"
    log += f"MHA_NUM_HEADS: {MHA_NUM_HEADS}\n"
    log += f"LAYER_NORM_EPSILON: {LAYER_NORM_EPSILON}\n"
    log += f"DROPOUT_RATE: {DROPOUT_RATE}\n"
    log += f"Sleep stages count in training data: {sleep_stages_count_training}\n"
    log += f"Sleep stages count in prediction: {sleep_stages_count_pred}\n"
    log += f"Number of training clips: {num_training_clips}\n"
    log += f"Number of test clips: {num_test_clips}\n"
    log += f"Takes in historical sleep stages: {USE_HISTORICAL_LOOKBACK}\n"

    output_log_filename = "/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".txt"
    with open(output_log_filename, 'w') as file:
        file.write(log)

def load_from_dataset(args) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
    """
    Loads data from dataset and returns training and test tensor pairs, along with a return code.
    """
    if (pkg_resources.get_distribution("tensorflow").version == "2.8.0+computecanada"):
        data = tf.data.experimental.load(args.input_dataset)
    else:
        data = tf.data.Dataset.load(args.input_dataset)
    iterator = iter(data)

    #x -> (num_clips, num_channels, num_samples_per_clips)
    #y -> (num_clips, 1, historical_lookback_length)
    #z -> (#channels in dataset)
    x, y, z = next(iterator)
    
    if int(args.num_training_clips) < (x.shape[0] + 1):
        x = x[0:int(args.num_training_clips)]
        y = y[0:int(args.num_training_clips)]

    x = tf.cast(x, dtype=DATA_TYPE)
    y = tf.cast(y, dtype=DATA_TYPE)

    # Select data from input channel parameter
    channel_index = -1
    channel_list = []
    channels_list_numpy = z.numpy()
    count = 0
    for s in channels_list_numpy:
        temp_string = s.decode()
        if temp_string == args.input_channel : channel_index = count
        count += 1

    if channel_index == -1:
        print(f"Could not find requested input channel ('{args.input_channel}') in input dataset ('{args.input_dataset}'). Available channels are: {channel_list}. Aborting!")
        return 0, 0, 0, 0, -1

    clip_train_cutoff = int((1-TEST_SET_RATIO)*x.shape[0])

    x_train = x[0:clip_train_cutoff, channel_index, :]
    x_test = x[clip_train_cutoff:-1, channel_index, :]

    if (USE_HISTORICAL_LOOKBACK):
        y_train = y[0:clip_train_cutoff:, 0, 0:HISTORICAL_LOOKBACK_LENGTH]
        y_test = y[clip_train_cutoff:-1:, 0, 0:HISTORICAL_LOOKBACK_LENGTH]
    else:
        y_train = y[0:clip_train_cutoff:, 0, 0]
        y_test = y[clip_train_cutoff:-1:, 0, 0]

    return x_train, x_test, y_train, y_test, 0

#--- Encoder ---#
class Encoder(tf.keras.layers.Layer):
    def __init__(self, clip_length:int):
        super().__init__()
        #Layers
        self.clip_length = clip_length
        self.voltage_embedding_layer = tf.keras.layers.Embedding(input_dim=MAX_VOLTAGE, output_dim=VOLTAGE_EMBEDDING_DEPTH, input_length=self.clip_length)
        self.dropout = tf.keras.layers.Dropout(rate=DROPOUT_RATE)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.dense_relu = tf.keras.layers.Dense(FULLY_CONNECTED_DIM, activation='relu')
        self.dense = tf.keras.layers.Dense(VOLTAGE_EMBEDDING_DEPTH)

    def call(self, input:tf.Tensor, training:bool=False):
        #1 - Voltage embedding
        voltage_embedding = self.voltage_embedding_layer(input)
        voltage_embedding *= tf.math.sqrt(tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32))

        #2 - Positonal encoding
        encoder_input = voltage_embedding + positional_encoder(sequence_length=self.clip_length, batch_size=BATCH_SIZE)

        #3 - [Training only] Apply dropout
        encoder_input = self.dropout(encoder_input, training=training) #encoder_input: (batch size, clip_length, embedding_depth)

        #4 - Run through encoder
        mha_output = self.mha(encoder_input, encoder_input, encoder_input, attention_mask=tf.ones(shape=input.shape, dtype=input.dtype)) #4.1 - Multi-head (self) attention
        skip_mha = self.norm(encoder_input + mha_output) #4.2 - Skip connection to Add & Norm
        fc_output = self.dense_relu(skip_mha) #4.3 - Feed MHA output to 2 Dense layers
        fc_output = self.dense(fc_output)
        fc_output = self.dropout(fc_output, training=training) #4.4 - [Training only] Apply dropout to FC
        out = self.norm(skip_mha + fc_output) #4.5 - Normalize output
        return out #out: (batch size, clip_length, embedding_depth)

#--- Decoder ---#
class Decoder(tf.keras.layers.Layer):
    def __init__(self, clip_length:int):
        super(Decoder, self).__init__()
        self.attention_weights = {}

        #Layers
        self.embedding_layer = tf.keras.layers.Embedding(NUM_SLEEP_STAGES+1, VOLTAGE_EMBEDDING_DEPTH, mask_zero=True) #+1 needed in input_dim because 0 is reserved as a padding value
        self.dropout1 = tf.keras.layers.Dropout(rate=DROPOUT_RATE)
        self.dropout2 = tf.keras.layers.Dropout(rate=DROPOUT_RATE)
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=MHA_NUM_HEADS, key_dim=VOLTAGE_EMBEDDING_DEPTH, dropout=DROPOUT_RATE)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON)
        self.dense_relu = tf.keras.layers.Dense(FULLY_CONNECTED_DIM, activation='relu')
        self.dense = tf.keras.layers.Dense(VOLTAGE_EMBEDDING_DEPTH)
        self.clip_length = clip_length

    def call(self, sleep_stage:tf.Tensor, encoder_output:tf.Tensor, padding_mask:tf.Tensor, training:bool=False):
        #sleep_stage: (batch size, output_sequence_length)
        #encoder_output: (batch_size, input_sequence_length, embedding depth)

        #1 - Voltage embedding
        embedding = self.embedding_layer(sleep_stage)
        embedding *= tf.math.sqrt(tf.cast(VOLTAGE_EMBEDDING_DEPTH, tf.float32))

        #2 - Positonal encoding
        decoder_input = embedding + positional_encoder(sequence_length=OUTPUT_SEQUENCE_LENGTH, batch_size=BATCH_SIZE)

        #3 - [Training only] Apply dropout
        decoder_input = self.dropout1(decoder_input, training=training) #decoder_input: (batch size, output_sequence_length, embedding_depth)

        #4 - Go through decoder
        #TODO: mask here
        mha1_output, mha1_attention_weights = self.mha1(decoder_input, decoder_input, decoder_input, attention_mask=padding_mask, return_attention_scores=True) #4.1 - Multi-head (self) attention
        skip_mha1 = self.norm1(sleep_stage + mha1_output) #4.2 - Skip connection to Add & Norm
        #TODO what mask should go here?
        mha2_output, mha2_attention_weights = self.mha2(skip_mha1, encoder_output, encoder_output, attention_mask=None, return_attention_scores=True) #4.3 - Multi-head attention with encoder output
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
    def __init__(self, clip_length:int=DEFAULT_CLIP_LENGTH_S) -> None:
        super().__init__()
        self.encoder = Encoder(clip_length)
        self.decoder = Decoder(clip_length)
        self.final_dense_layer = tf.keras.layers.Dense(NUM_SLEEP_STAGES+1, activation='softmax') #+1 needed because 0 is reserved as an padding 
        self.clip_length = clip_length

    @tf.function
    def call(self, inputs, training):
        #eeg_clip = (batch_size, num_samples_per_clip)
        #sleep_stage = (batch_size, 1)
        eeg_clip, sleep_stage = inputs

        print(eeg_clip)
        print(sleep_stage)

        eeg_clip = tf.reshape(eeg_clip, (1, self.clip_length))
        sleep_stage = tf.reshape(sleep_stage, (1, OUTPUT_SEQUENCE_LENGTH))

        #Masks #TODO: Fix this
        decoder_padding_mask = maskify(sleep_stage)
        decoder_padding_mask = tf.expand_dims(decoder_padding_mask, axis=1)
        # decoder_padding_mask = tf.constant(True, shape=(1, OUTPUT_SEQUENCE_LENGTH, 1))

        encoder_output = self.encoder(eeg_clip, training)        
        sleep_stage, attention_weights = self.decoder(sleep_stage, encoder_output, padding_mask=decoder_padding_mask, training=training)

        final_softmax = self.final_dense_layer(sleep_stage)

        try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
            del final_softmax._keras_mask
        except AttributeError:
            pass

        return final_softmax #final_softmax: (batch size, output_sequence_length, num_sleep_stages)

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
    # Parse arguments
    parser = ArgumentParser(description='Transformer model Tensorflow prototype.')
    parser.add_argument('--clip_length_s', help='Clip length (in sec). Must match input dataset clip length. Must be one of 3.25, 5, 7.5, 10, 15, 30.', default='30')
    parser.add_argument('--num_training_clips', help='Number of clips to use for training. Defaults to 3000.', default=3000)
    parser.add_argument('--input_dataset', help='Filepath of the dataset used for training and validation.')
    parser.add_argument('--input_channel', help='Name of the channel to use for training and validation.')
    args = parser.parse_args()
    num_samples_per_clip = int(SAMPLING_FREQUENCY_HZ * float(args.clip_length_s))

    # Load data
    x_train, x_test, y_train, y_test, success = load_from_dataset(args=args)
    if (success == -1): return

    #Count sleep stages in dataset
    sleep_stages_count_training = [0, 0, 0, 0, 0, 0]
    for sleep_stage_number in range(len(y_train)):
        sleep_stages_count_training[int(y_train[sleep_stage_number])] += 1

    y_uninitialized = tf.zeros(shape=[x_train.shape[0]])

    # Model
    model = Transformer(num_samples_per_clip)
    model.compile(
        loss=masked_loss,
        optimizer=tf.keras.optimizers.Adam(CustomSchedule(), beta_1=0.9, beta_2=0.98, epsilon=1e-9),
        metrics=[masked_accuracy]
    )

    tensorboard_log_dir = "logs/fit/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    model.fit(x=(x_train, y_uninitialized), y=y_train, epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE, callbacks=[tensorboard_callback])

    #Manual validation
    total_correct = 0
    total = 0

    sleep_stages_count_pred = [0, 0, 0, 0, 0, 0]
    for x,y in zip(x_test, y_test):
        y_uninitialized = tf.zeros(shape=(1, 1))
        sleep_stage = model.call(inputs=(tf.reshape(x, (1, x.shape[0])), y_uninitialized), training=False)
        sleep_stage = tf.argmax(sleep_stage[0][0])
        total += 1
        if (sleep_stage) == tf.cast(y, dtype=tf.int64): total_correct += 1
        print(f"Ground truth: {y}, sleep stage pred: {sleep_stage}, accuracy: {total_correct/total:.4f}")
        sleep_stages_count_pred[int(sleep_stage)] += 1
    
    #Save accuracy and model details to log file
    export_summary(args, model, total_correct/total,
                   sleep_stages_count_training, sleep_stages_count_pred,
                   int(x_train.shape[0]), int(x_test.shape[0]))

if __name__ == "__main__":
    main()
