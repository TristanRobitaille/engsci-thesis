import datetime
import pkg_resources
import git
import sys
import os

import tensorflow as tf
import tensorflow_addons as tfa

from io import StringIO
from argparse import ArgumentParser
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Add
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

import utilities

MIN_VOLTAGE = 0
MAX_VOLTAGE = 2**16-1 #Maximum ADC code output
DEFAULT_CLIP_LENGTH_S = int(30)
SAMPLING_FREQUENCY_HZ = int(256)
DEFAULT_CLIP_LENGTH_NUM_SAMPLES = DEFAULT_CLIP_LENGTH_S * SAMPLING_FREQUENCY_HZ
NUM_SLEEP_STAGES = 5 + 1 # 'unknown'
DROPOUT_RATE = 0.1
DATA_TYPE = tf.float32
TEST_SET_RATIO = 0.1 #Percentage of training data reserved for validation
VERBOSITY = 'QUIET' #'QUIET', 'NORMAL', 'DETAILED'
AUTOTUNE = tf.data.experimental.AUTOTUNE

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

def reduce_bias(signals, sleep_stage):

    # Determine indices of each sleep stage
    sleep_stage_indices = list()
    for _ in range(NUM_SLEEP_STAGES): sleep_stage_indices.append([]) 

    for i in range(len(sleep_stage)):
        sleep_stage_indices[int(sleep_stage[i,:])].append(i)

    # Delete 65% of sleep stages #2
    #TODO: Should randomly sample the clips to be deleted since removing them from the end is introducing a bias in the validation data (that is taken from the end)
    sleep_stage_indices[2].sort(reverse=True)

    for i in range(int(0.65*len(sleep_stage_indices[2]))):
        sleep_stage = tf.concat([sleep_stage[:sleep_stage_indices[2][i]], sleep_stage[sleep_stage_indices[2][i]+1:]], axis=0)
        signals = tf.concat([signals[:sleep_stage_indices[2][i]], signals[sleep_stage_indices[2][i]+1:]], axis=0)

    return signals, sleep_stage

def trim_clips(args, signals:tf.Tensor, sleep_stages:tf.Tensor):
    # Cast data type
    signals = tf.cast(signals, dtype=DATA_TYPE)
    sleep_stages = tf.cast(sleep_stages, dtype=DATA_TYPE)

    # Dataset split
    num_training_clips = int(signals.shape[0]*(1-TEST_SET_RATIO))

    signals_train = signals[0:num_training_clips]
    signals_val = signals[num_training_clips-1:-1]
    sleep_stages_train = sleep_stages[0:num_training_clips]
    sleep_stages_val = sleep_stages[num_training_clips-1:-1]

    # Trim tensors to be a multiple of batch_size
    if (signals_train.shape[0] % args.batch_size != 0):
        max_training_length = signals_train.shape[0] - signals_train.shape[0] % args.batch_size
        signals_train = signals_train[0:max_training_length]
        sleep_stages_train = sleep_stages_train[0:max_training_length]

    if (signals_val.shape[0] % args.batch_size != 0):
        max_validation_length = signals_val.shape[0] - signals_val.shape[0] % args.batch_size
        signals_val = signals_val[0:max_validation_length]
        sleep_stages_val = sleep_stages_val[0:max_validation_length]

    return signals_train, signals_val, sleep_stages_train, sleep_stages_val

def load_from_dataset(args):
    """
    Loads data from dataset and returns batched, shuffled dataset of correct channel
    """
    global NUM_TRAINING_CLIPS

    if (pkg_resources.get_distribution("tensorflow").version == "2.8.0+computecanada"):
        data = tf.data.experimental.load(args.input_dataset)
    else:
        data = tf.data.Dataset.load(args.input_dataset)

    data = (data.cache().shuffle(args.num_clips).prefetch(AUTOTUNE))
    data = next(iter(data))

    sleep_stages = data['sleep_stage']

    # Check corner cases
    if args.input_channel not in data.keys():
        print(f"Requested input chanel {args.input_channel} not found in input dataset ({args.input_dataset}). Available channels are {data.keys()}. Aborting.")
        return 0, 0, 0, 0, -1
    else:
        signals = data[args.input_channel]

    if (args.num_clips > signals.shape[0]):
        print(f"Requested number of clips ({args.num_clips}) larger than number of clips in dataset ({signals.shape[0]})! Will use {signals.shape[0]} clips.")
        NUM_TRAINING_CLIPS = signals.shape[0]
    else:
        signals = signals[0:args.num_clips-args.num_clips%args.batch_size, :]
        sleep_stages = sleep_stages[0:args.num_clips-args.num_clips%args.batch_size, :]

    signals, sleep_stages = reduce_bias(signals, sleep_stages)
    signals_train, signals_val, sleep_stages_train, sleep_stages_val = trim_clips(args, signals, sleep_stages)

    return signals_train, signals_val, sleep_stages_train, sleep_stages_val, 0

def export_summary(parser, model, accuracy:float, sleep_stages_count_training:list, sleep_stages_count_validation:list, sleep_stages_count_pred:list, num_training_clips:int) -> None:
    """
    Saves model and training summary to file
    """
    with Capturing() as model_summary:
        model.summary()
    model_summary = "\n".join(model_summary)

    repo = git.Repo(search_parent_directories=True)

    # Count relative number of stages
    num_clips_training = sum(sleep_stages_count_training)
    num_clips_validation = sum(sleep_stages_count_validation)
    num_clips_pred = sum(sleep_stages_count_pred)

    log = "VISION TRANSFORMER MODEL TRAINING SUMMARY\n"
    log += model_summary
    log += f"\nTest set accuracy: {accuracy:.4f}\n"
    log += f"Git hash: {repo.head.object.hexsha}\n"
    log += f"Dataset: {parser.input_dataset}\n"
    log += f"Channel: {parser.input_channel}\n"
    log += f"CLIP_LENGTH (s): {parser.clip_length_s}\n"
    log += f"NUM_SLEEP_STAGES (includes unknown): {NUM_SLEEP_STAGES}\n"
    log += f"DATA_TYPE: {DATA_TYPE}\n"
    log += f"VOLTAGE_EMBEDDING_DEPTH: {parser.embedding_depth}\n"
    log += f"BATCH_SIZE: {parser.batch_size}\n"
    log += f"MHA_NUM_HEADS: {parser.num_heads}\n"
    log += f"NUM_LAYERS: {parser.num_layers}\n"
    log += f"MLP_DIMENSION: {parser.mlp_dim}\n"
    log += f"DROPOUT_RATE: {DROPOUT_RATE}\n"
    log += f"Sleep stages count in training data: {sleep_stages_count_training} ({[round(num / num_clips_training, 4) for num in sleep_stages_count_training]})\n"
    log += f"Sleep stages count in validation data: {sleep_stages_count_validation} ({[round(num / num_clips_validation, 4) for num in sleep_stages_count_validation]})\n"
    log += f"Sleep stages count in prediction data: {sleep_stages_count_pred} ({[round(num / num_clips_pred, 4) for num in sleep_stages_count_pred]})\n"
    log += f"Number of training clips used: {num_training_clips}\n"
    log += f"Number of epochs: {parser.num_epochs}\n"
    log += f"Learning rate: {parser.learning_rate:.6f}\n"

    output_log_filename = os.getcwd() + "/python_prototype/results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_vision.txt"

    with open(output_log_filename, 'w') as file:
        file.write(log)

#--- Multi-Head Attention ---#
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension:int, num_heads:int=8):
        super(MultiHeadSelfAttention, self).__init__()

        # Hyperparameters
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.projection_dimension = self.embedding_dimension // self.num_heads

        if self.embedding_dimension % num_heads != 0:
            raise ValueError(f"Embedding dimension = {self.embedding_dimension} should be divisible by number of heads = {self.num_heads}")

        # Layers
        self.query_dense = Dense(self.embedding_dimension)
        self.key_dense = Dense(self.embedding_dimension)
        self.value_dense = Dense(self.embedding_dimension)
        self.combine_heads = Dense(self.embedding_dimension)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], dtype=DATA_TYPE)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dimension))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, inputs):
        batch_size = inputs.shape[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0,2,1,3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embedding_dimension))
        output = self.combine_heads(concat_attention)

        return output

#--- Encoder ---#
class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_depth:int, num_heads:int, mlp_dim:int, dropout_rate:int=0.1):
        super(Encoder, self).__init__()

        # Hyperparameters
        self.embedding_depth = embedding_depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        # Layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.mhsa = MultiHeadSelfAttention(self.embedding_depth, self.num_heads)
        self.dropout1 = Dropout(self.dropout_rate)
        self.add = Add()

        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            Dense(mlp_dim, activation=tfa.activations.gelu),
            Dropout(self.dropout_rate),
            Dense(self.embedding_depth),
            Dropout(self.dropout_rate),
        ])
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.mhsa(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.add([attn_output, inputs])

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return self.add([mlp_output, out1])

#--- Vision Transformer ---#
class VisionTransformer(tf.keras.Model):
    def __init__(self, clip_length_num_samples:int, patch_length_num_samples:int, num_layers:int, num_classes:int, embedding_depth:int, num_heads:int, mlp_dim:int, dropout_rate:float=0.1):
        super(VisionTransformer, self).__init__()

        # Hyperparameters
        self.clip_length_num_samples = clip_length_num_samples
        self.patch_length_num_samples = patch_length_num_samples
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding_depth = embedding_depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.num_patches = int(self.clip_length_num_samples / self.patch_length_num_samples)

        # Layers
        self.rescale = Rescaling(1.0 / MAX_VOLTAGE)
        self.patch_projection = Dense(self.embedding_depth)
        self.positional_embedding = self.add_weight("pos_emb", shape=(1, self.num_patches+1, self.embedding_depth)) #+1 is for the trainable classification token prepended to input sequence of patches
        self.class_embedding = self.add_weight("class_emb", shape=(1, 1, self.embedding_depth))
        self.add = tf.keras.layers.Add()
        self.encoder_layers = [Encoder(embedding_depth=self.embedding_depth, num_heads=self.num_heads, mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)]
        self.mlp_head = tf.keras.Sequential([
            LayerNormalization(epsilon=1e-6),
            Dense(self.mlp_dim, activation=tfa.activations.gelu),
            Dropout(self.dropout_rate),
            Dense(self.num_classes, activation='softmax')
        ])

    def extract_patches(self, batch_size:int, clips):
        patches = tf.reshape(clips, [batch_size, -1, self.patch_length_num_samples])
        return patches

    def call(self, clip, training:bool):
        batch_size = clip.shape[0]

        # Normalize to [0,1]
        clip = self.rescale(clip)

        # Extract patches
        patches = self.extract_patches(batch_size, clip)

        # Linear projection
        clip = self.patch_projection(patches) #clip = (num_patches, embedding_depth)

        # Classification token
        class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.embedding_depth])

        # Construct sequence
        clip = tf.concat([class_embedding, clip], axis=1)
        clip = clip + self.positional_embedding

        # Go through encoder
        for layer in self.encoder_layers:
            clip = layer(clip, training)

        # Classify with first token
        prediction = self.mlp_head(clip[:, 0])

        return prediction

#--- Misc ---#
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def main():
    # Parse arguments
    parser = ArgumentParser(description='Transformer model Tensorflow prototype.')
    parser.add_argument('--num_clips', help='Number of clips to use for training + validation. Defaults to 3000.', default=3000, type=int)
    parser.add_argument('--input_dataset', help='Filepath of the dataset used for training and validation.')
    parser.add_argument('--input_channel', help='Name of the channel to use for training and validation.')
    parser.add_argument('--clip_length_s', help='Clip length (in sec). Must match input dataset clip length. Must be one of 3.25, 5, 7.5, 10, 15, 30. Defaults to 15s.', default=15, type=float)
    parser.add_argument('--patch_length_s', help='Patch length (in sec). Must be integer multiple of clip_length_s. Defaults to 0.5s.', default=0.5, type=float)
    parser.add_argument('--num_layers', help='Number of encoder layer. Defaults to 8.', default=8, type=int)
    parser.add_argument('--embedding_depth', help='Depth of the embedding layer. Defaults to 32.', default=32, type=int)
    parser.add_argument('--num_heads', help='Number of multi-attention heads. Defaults to 8.', default=8, type=int)
    parser.add_argument('--mlp_dim', help='Dimension of the MLP layer. Defaults to 32.', default=32, type=int)
    parser.add_argument('--num_epochs', help='Number of training epochs. Defaults to 25.', default=25, type=int)
    parser.add_argument('--batch_size', help='Batch size for training. Defaults to 8.', default=8, type=int)
    parser.add_argument('--learning_rate', help='Learning rate for training. Defaults to 1e-4.', default=1e-4, type=float)

    args = parser.parse_args()
    if args.clip_length_s % args.patch_length_s != 0:
        raise ValueError(f"patch_length_s ({args.patch_length_s}s) should be an integer multiple of clip_length_s ({args.clip_length_s}s))")

    # Hyperparameters
    clip_length_num_samples = int(args.clip_length_s * SAMPLING_FREQUENCY_HZ)
    patch_length_num_samples = int(args.patch_length_s * SAMPLING_FREQUENCY_HZ)

    # Load data
    signals_train, signals_val, sleep_stages_train, sleep_stages_val, success = load_from_dataset(args=args)
    if (success == -1): return

    # Train model
    print(f"Starting training with {int((1 - TEST_SET_RATIO)*signals_train.shape[0])} clips")
    model = VisionTransformer(clip_length_num_samples=clip_length_num_samples, patch_length_num_samples=patch_length_num_samples, num_layers=args.num_layers, num_classes=NUM_SLEEP_STAGES,
                                embedding_depth=args.embedding_depth, num_heads=args.num_heads, mlp_dim=args.mlp_dim, dropout_rate=DROPOUT_RATE)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(CustomSchedule(args.embedding_depth), beta_1=0.9, beta_2=0.98, epsilon=1e-9),
        metrics=["accuracy"],
    )

    tensorboard_log_dir = "logs/fit/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    model.fit(x=signals_train, y=sleep_stages_train, epochs=int(args.num_epochs), batch_size=args.batch_size, callbacks=[tensorboard_callback], )

    # Manual validation
    total_correct = 0
    total = 0
    sleep_stages_count_pred = [0, 0, 0, 0, 0, 0]

    print(f"Now commencing manual validation with {signals_val.shape[0]} clips")
    for x,y in zip(signals_val, sleep_stages_val):
        x = tf.reshape(x, shape=[1, x.shape[0]])
        sleep_stage = model.call(x, training=False)
        sleep_stage = tf.argmax(sleep_stage[0])
        total += 1
        if (sleep_stage) == tf.cast(y, dtype=tf.int64): total_correct += 1
        if (VERBOSITY == 'Normal'): print(f"Ground truth: {y}, sleep stage pred: {sleep_stage}, accuracy: {total_correct/total:.4f}")
        sleep_stages_count_pred[int(sleep_stage)] += 1

    # Count sleep stages in training and validation datasets
    sleep_stages_count_training = utilities.count_sleep_stages(sleep_stages_train, NUM_SLEEP_STAGES)
    sleep_stages_count_validation = utilities.count_sleep_stages(sleep_stages_val, NUM_SLEEP_STAGES)

    # Save accuracy and model details to log file
    export_summary(args, model, total_correct/total, sleep_stages_count_training, sleep_stages_count_validation, sleep_stages_count_pred, num_training_clips=int(signals_train.shape[0]))

    print("Done. Good bye.")

if __name__ == "__main__":
    main()