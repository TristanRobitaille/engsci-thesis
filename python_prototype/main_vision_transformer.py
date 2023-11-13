import datetime
import pkg_resources
import git
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from io import StringIO
from argparse import ArgumentParser
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Add
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

import utilities

MIN_VOLTAGE = 0
MAX_VOLTAGE = 2**16-1 #Maximum ADC code output
DEFAULT_CLIP_LENGTH_S = int(30)
SAMPLING_FREQUENCY_HZ = int(256)
DEFAULT_CLIP_LENGTH_NUM_SAMPLES = DEFAULT_CLIP_LENGTH_S * SAMPLING_FREQUENCY_HZ
NUM_SLEEP_STAGES = 5 + 1 #Includes 'unknown'
DROPOUT_RATE = 0.1
DATA_TYPE = tf.float32
TEST_SET_RATIO = 0.1 #Percentage of training data reserved for validation
RANDOM_SEED = 42
VERBOSITY = 'QUIET' #'QUIET', 'NORMAL', 'DETAILED'
AUTOTUNE = tf.data.experimental.AUTOTUNE
RESAMPLE_TRAINING_DATASET = True
RESAMPLE_VALIDATION_DATASET = False

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

def trim_clips(args, signals_train:tf.Tensor, signals_val:tf.Tensor, sleep_stages_train:tf.Tensor, sleep_stages_val:tf.Tensor):
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

    global NUM_SLEEP_STAGES

    if (pkg_resources.get_distribution("tensorflow").version == "2.8.0+computecanada"):
        data = tf.data.experimental.load(args.input_dataset)
    else:
        data = tf.data.Dataset.load(args.input_dataset)

    # data = (data.cache().shuffle(args.num_clips).prefetch(AUTOTUNE))
    data = next(iter(data))

    sleep_stages = data['sleep_stage']
    NUM_SLEEP_STAGES = int(args.input_dataset.split("-stg")[0].split("_")[-1]) + 1 # Extract number of sleep stages (+1 for unknown)
    
    # Check corner cases
    if args.input_channel not in data.keys():
        raise ValueError(f"Requested input channel {args.input_channel} not found in input dataset ({args.input_dataset}).\nAvailable channels are {data.keys()}.\nAborting.")
    else:
        signals = data[args.input_channel]

    if (args.num_clips > signals.shape[0]):
        print(f"Requested number of clips ({args.num_clips}) larger than number of clips in dataset ({signals.shape[0]})! Will use {signals.shape[0]} clips.")
    else:
        signals = signals[0:args.num_clips-args.num_clips%args.batch_size, :]
        sleep_stages = sleep_stages[0:args.num_clips-args.num_clips%args.batch_size, :]

    # Convert to numpy arrays and shuffle
    signals = signals.numpy()
    sleep_stages = sleep_stages.numpy()
    indices = np.random.permutation(len(signals))
    signals = signals[indices]
    sleep_stages = sleep_stages[indices]

    # Split into training and validation sets
    signals_train, signals_val, sleep_stages_train, sleep_stages_val = train_test_split(signals, sleep_stages, test_size=TEST_SET_RATIO, random_state=RANDOM_SEED)

    # Undersample clips such that all classes in minority class have same number of clips
    resampler = RandomUnderSampler(sampling_strategy=args.dataset_resample_strategy, random_state=RANDOM_SEED, replacement=args.dataset_resample_replacement)

    if RESAMPLE_TRAINING_DATASET:
        signals_train, sleep_stages_train = resampler.fit_resample(signals_train, sleep_stages_train)

    if RESAMPLE_VALIDATION_DATASET:
        signals_val, sleep_stages_val = resampler.fit_resample(signals_val, sleep_stages_val)

    # Trim clips to be a multiple of batch_size
    signals_train, signals_val, sleep_stages_train, sleep_stages_val = trim_clips(args, signals_train, signals_val, sleep_stages_train, sleep_stages_val)

    return signals_train, signals_val, sleep_stages_train, sleep_stages_val, resampler

def export_summary(parser, model, fit_history, resampler, accuracy:float, sleep_stages_count_training:list, sleep_stages_count_validation:list, sleep_stages_count_pred:list, completion_time:float) -> None:
    """
    Saves model and training summary to file
    """
    try:
        with Capturing() as model_summary:
            model.summary()
        model_summary = "\n".join(model_summary)

        repo = git.Repo(search_parent_directories=True)

        # Count relative number of stages
        num_clips_training = sum(sleep_stages_count_training)
        num_clips_validation = sum(sleep_stages_count_validation)
        num_clips_pred = sum(sleep_stages_count_pred)

        log = "VISION TRANSFORMER MODEL TRAINING SUMMARY\n"
        log += f"Git hash: {repo.head.object.hexsha}\n"
        log += f"Time to complete: {completion_time:.2f}s\n"
        log += model_summary
        log += f"\nDataset: {parser.input_dataset}\n"
        log += f"Channel: {parser.input_channel}\n"
        log += f"Validation set accuracy: {accuracy:.4f}\n"
        log += f"Training accuracy: {[round(accuracy, 4) for accuracy in fit_history.history['accuracy']]}\n"
        log += f"Training loss: {[round(loss, 4) for loss in fit_history.history['loss']]}\n"
        log += f"Number of epochs: {parser.num_epochs}\n\n"

        log += f"Dataset resample strategy: {parser.dataset_resample_strategy}\n"
        log += f"Dataset resample replacement: {parser.dataset_resample_replacement}\n"
        log += f"Dataset resampler: {resampler}\n\n"

        log += f"Requested number of training clips: {int(TEST_SET_RATIO*parser.num_clips)}\n"
        log += f"Sleep stages count in training data ({num_clips_training}): {sleep_stages_count_training} ({[round(num / num_clips_training, 4) for num in sleep_stages_count_training]})\n"
        log += f"Sleep stages count in validation set input ({num_clips_validation}): {sleep_stages_count_validation} ({[round(num / num_clips_validation, 4) for num in sleep_stages_count_validation]})\n"
        log += f"Sleep stages count in validation set prediction ({num_clips_pred}): {sleep_stages_count_pred} ({[round(num / num_clips_pred, 4) for num in sleep_stages_count_pred]})\n\n"

        log += f"CLIP_LENGTH (s): {parser.clip_length_s}\n"
        log += f"NUM_SLEEP_STAGES (includes unknown): {NUM_SLEEP_STAGES}\n"
        log += f"DATA_TYPE: {DATA_TYPE}\n"
        log += f"VOLTAGE_EMBEDDING_DEPTH: {parser.embedding_depth}\n"
        log += f"BATCH_SIZE: {parser.batch_size}\n"
        log += f"MHA_NUM_HEADS: {parser.num_heads}\n"
        log += f"NUM_LAYERS: {parser.num_layers}\n"
        log += f"MLP_DIMENSION: {parser.mlp_dim}\n"
        log += f"DROPOUT_RATE: {DROPOUT_RATE}\n"
        log += f"INITIAL_LEARNING_RATE: {parser.learning_rate:.6f}\n"
        log += f"RESAMPLE_TRAINING_DATASET: {RESAMPLE_TRAINING_DATASET}\n"
        log += f"RESAMPLE_VALIDATION_DATASET: {RESAMPLE_VALIDATION_DATASET}\n"
        log += f"RANDOM_SEED: {RANDOM_SEED}\n"

        log += f"Model loss: {model.loss.name}\n"
        log += f"Model optimizer: {model.optimizer.name} (beta_1: {model.optimizer.beta_1}, beta_2: {model.optimizer.beta_2}, epsilon: {model.optimizer.epsilon})\n"

        # Check whether files with the same name already exist and append counter if necessary
        candidate_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_vision.txt"
        output_log_filename = utilities.find_file_name(candidate_file_name, "/home/trobitaille/engsci-thesis/python_prototype/results/")
        
        # Save to file
        with open(output_log_filename, 'w') as file:
            file.write(log)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to export summary.")

def parse_arguments():
    """"
    Parses command line arguments and return parser object
    """

    # Resampling documentation: https://imbalanced-learn.org/stable/introduction.html
    resampling_type_choices = ['RandomOverSampler', 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SMOTENC', 'SMOTEN', 'KMeansSMOTE', 'SVMSMOTE', 'ClusterCentroids', 'RandomUnderSampler', 'TomekLinks']

    parser = utilities.ArgumentParserWithError(description='Transformer model Tensorflow prototype.')
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
    parser.add_argument('--class_weights', help='List of weights to apply in loss calculation.', nargs='+', default=[1, 1, 1, 1, 1, 1], type=float)
    parser.add_argument('--dataset_resample_algo', help="Which dataset resampling algorithm to use. Currently using 'imblearn' package.", choices='RandomUnderSampler', default='', type=str)
    parser.add_argument('--dataset_resample_strategy', help='Defines which strategy to use when resampling dataset with RandomUnderSampler(). Defaults to "auto"', choices=['majority', 'not minority', 'not majority', 'all', 'auto'], default='auto', type=str)
    parser.add_argument('--dataset_resample_replacement', help='Whether replacement is allowed when resampling dataset with RandomUnderSampler(). Defaults to false', default=False, type=bool)

    # Parse arguments
    try:
        args = parser.parse_args()
    except Exception as e:
        utilities.log_error_and_exit(e)

    # Check validity of arguments
    if args.clip_length_s % args.patch_length_s != 0:
        raise ValueError(f"patch_length_s ({args.patch_length_s}s) should be an integer multiple of clip_length_s ({args.clip_length_s}s))")

    if len(args.class_weights) != NUM_SLEEP_STAGES:
        raise ValueError(f"Number of class weights ({len(args.class_weights)}) should be equal to number of sleep stages ({NUM_SLEEP_STAGES})")
    
    return args

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
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1

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
    # Start clock
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Hyperparameters
    clip_length_num_samples = int(args.clip_length_s * SAMPLING_FREQUENCY_HZ)
    patch_length_num_samples = int(args.patch_length_s * SAMPLING_FREQUENCY_HZ)

    # Load data
    try: signals_train, signals_val, sleep_stages_train, sleep_stages_val, resampler = load_from_dataset(args=args)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to load data from dataset.")

    # Train model
    print(f"Starting training with {int((1 - TEST_SET_RATIO)*signals_train.shape[0])} clips")
    try:
        model = VisionTransformer(clip_length_num_samples=clip_length_num_samples, patch_length_num_samples=patch_length_num_samples, num_layers=args.num_layers, num_classes=NUM_SLEEP_STAGES,
                                  embedding_depth=args.embedding_depth, num_heads=args.num_heads, mlp_dim=args.mlp_dim, dropout_rate=DROPOUT_RATE)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to initialize model.")

    try:
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(CustomSchedule(args.embedding_depth), beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            metrics=["accuracy"],
        )
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to compile model.")

    tensorboard_log_dir = "logs/fit/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    try: fit_history = model.fit(x=signals_train, y=sleep_stages_train, epochs=int(args.num_epochs), batch_size=args.batch_size, callbacks=[tensorboard_callback], class_weight=args.class_weights)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to fit model.")

    # Manual validation
    total_correct = 0
    total = 0
    sleep_stages_count_pred = [0 for _ in range(NUM_SLEEP_STAGES)]

    print(f"Now commencing manual validation with {signals_val.shape[0]} clips")

    # Make batches
    signals_val_batches = [signals_val[i:i + args.batch_size] for i in range(0, len(signals_val), args.batch_size)]
    sleep_stages_val_batches = [sleep_stages_val[i:i + args.batch_size] for i in range(0, len(sleep_stages_val), args.batch_size)]
    try:
        for x_batch, y_batch in zip(signals_val_batches, sleep_stages_val_batches):
            sleep_stages = model(x_batch, training=False)
            sleep_stages = tf.argmax(sleep_stages, axis=1).numpy()

            for i in range(args.batch_size): total_correct += (sleep_stages[i] == y_batch[i,0])
            total += len(y_batch)

            if (VERBOSITY == 'Normal'): print(f"Ground truth: {y_batch}, sleep stage pred: {sleep_stages}, accuracy: {total_correct/total:.4f}")
            for sleep_stage in sleep_stages:
                sleep_stages_count_pred[int(sleep_stage)] += 1
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to manually validate model.")

    # Count sleep stages in training and validation datasets
    sleep_stages_count_training = utilities.count_sleep_stages(sleep_stages_train, NUM_SLEEP_STAGES)
    sleep_stages_count_validation = utilities.count_sleep_stages(sleep_stages_val, NUM_SLEEP_STAGES)

    # Save accuracy and model details to log file
    completion_time = time.time() - start_time

    export_summary(args, model, fit_history, resampler, total_correct/total, sleep_stages_count_training,
                   sleep_stages_count_validation, sleep_stages_count_pred, completion_time=completion_time)

    print("Done. Good bye.")

if __name__ == "__main__":
    main()