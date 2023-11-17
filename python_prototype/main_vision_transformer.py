import time
start_time = time.time()

import os
import git
import sys
import socket
import imblearn
import datetime

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from io import StringIO
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from sklearn.model_selection import train_test_split

import utilities

MAX_VOLTAGE = 2**16-1 #Maximum ADC code output
DEFAULT_CLIP_LENGTH_S = int(30)
SAMPLING_FREQUENCY_HZ = int(256)
NUM_SLEEP_STAGES = 5 + 1 #Includes 'unknown'
USE_SLEEP_STAGE_HISTORY = True
NUM_SLEEP_STAGE_HISTORY = -1
DROPOUT_RATE = 0.1
DATA_TYPE = tf.float32
TEST_SET_RATIO = 0.1 #Percentage of training data reserved for validation
RANDOM_SEED = 42
VERBOSITY = 'QUIET' #'QUIET', 'NORMAL', 'DETAILED'
RESAMPLE_TRAINING_DATASET = False
WHOLE_NIGHT_VALIDATION = True or (NUM_SLEEP_STAGE_HISTORY > 0) #Whether to validate individual nights (sequentially) or a random subset of clips (if we use the preduiction history, we need whole nights for validation)

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

def resample_clips(args, signals_train, labels_train):
    # Check if we want resample targets
    args.training_set_target_count = {i+1: weight for i, weight in enumerate(args.training_set_target_count)}
    if args.training_set_target_count[1] == -1: args.training_set_target_count = 'auto'

    original_sleep_stage_count = utilities.count_instances_per_class(labels_train, NUM_SLEEP_STAGES)

    # Remove some of majority class
    resampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={1:4200, 2:10000, 3:6700, 4:9250, 5:5600}, replacement=args.dataset_resample_replacement)
    signals_train, labels_train = resampler.fit_resample(signals_train, labels_train)

    #Undersampling
    if args.dataset_resample_algo == 'RandomUnderSampler':
        resampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=args.training_set_target_count, replacement=args.dataset_resample_replacement)

    elif args.dataset_resample_algo == 'TomekLinks':
        resampler = imblearn.under_sampling.TomekLinks(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'ClusterCentroids':
        resampler = imblearn.under_sampling.ClusterCentroids(sampling_strategy=args.training_set_target_count)

    #Oversampling
    elif args.dataset_resample_algo == 'SMOTE':
        resampler = imblearn.over_sampling.SMOTE(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'SMOTENC':
        resampler = imblearn.over_sampling.SMOTENC(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'SMOTEN':
        resampler = imblearn.over_sampling.SMOTEN(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'ADASYN':
        resampler = imblearn.over_sampling.ADASYN(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'BorderlineSMOTE':
        resampler = imblearn.over_sampling.BorderlineSMOTE(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'KMeansSMOTE':
        resampler = imblearn.over_sampling.KMeansSMOTE(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'SVMSMOTE':
        resampler = imblearn.over_sampling.SVMSMOTE(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'RandomOverSampler':
        resampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy=args.training_set_target_count)

    #Combination of undersampling and oversampling
    elif args.dataset_resample_algo == 'SMOTEENN':
        resampler = imblearn.combine.SMOTEENN(sampling_strategy=args.training_set_target_count)

    elif args.dataset_resample_algo == 'SMOTETomek':
        resampler = imblearn.combine.SMOTETomek(sampling_strategy=args.training_set_target_count)

    signals_train, labels_train = resampler.fit_resample(signals_train, labels_train)

    return signals_train, labels_train, original_sleep_stage_count

def split_whole_night_validation_set(args, signals, sleep_stages, whole_night_markers):
    """
    Split the dataset into whole nights for validation
    """

    split_index = int((1-TEST_SET_RATIO) * len(sleep_stages))
    whole_night_indices = [i for i, x in enumerate(whole_night_markers) if x == True]
    cutoff_night = min(whole_night_indices, key=lambda x: abs(x - split_index))

    signals_train = signals[0:cutoff_night]
    signals_val = signals[cutoff_night:]
    sleep_stages_train = sleep_stages[0:cutoff_night]
    sleep_stages_val = sleep_stages[cutoff_night:]
    
    return signals_train, signals_val, sleep_stages_train, sleep_stages_val

def load_from_dataset(args):
    """
    Loads data from dataset and returns batched, shuffled dataset of correct channel
    """

    global NUM_SLEEP_STAGES
    global NUM_SLEEP_STAGE_HISTORY
    global WHOLE_NIGHT_VALIDATION

    #Extract number of sleep stage history
    dataset_name = os.path.basename(args.input_dataset)
    if "history_" in dataset_name and USE_SLEEP_STAGE_HISTORY:
        NUM_SLEEP_STAGE_HISTORY = int(dataset_name.split("history_")[1].split("-")[0])
        WHOLE_NIGHT_VALIDATION = True

    # Load dataset
    if socket.gethostname() == "claude-ryzen":              data = tf.data.experimental.load(args.input_dataset)
    elif socket.gethostname() == "MBP_Tristan":             data = tf.data.Dataset.load(args.input_dataset)
    elif "cedar.computecanada.ca" in socket.gethostname():  data = tf.data.Dataset.load(args.input_dataset)

    data = next(iter(data))

    sleep_stages = data['sleep_stage']
    start_of_night_markers = data['new_night_marker']
    if NUM_SLEEP_STAGE_HISTORY > 0: sleep_stages_history = data[f'history_{NUM_SLEEP_STAGE_HISTORY}-steps']
    NUM_SLEEP_STAGES = int(args.input_dataset.split("-stg")[0].split("_")[-1]) + 1 # Extract number of sleep stages (+1 for unknown)

    # Check corner cases
    if args.input_channel not in data.keys():
        raise ValueError(f"[{(time.time()-start_time):.2f}s] Requested input channel {args.input_channel} not found in input dataset ({args.input_dataset}).\nAvailable channels are {data.keys()}.\nAborting.")
    else:   signals = data[args.input_channel]

    if (args.num_clips > signals.shape[0]):
        print(f"[{(time.time()-start_time):.2f}s] Requested number of clips ({args.num_clips}) larger than number of clips in dataset ({signals.shape[0]})! Will use {signals.shape[0]} clips.")
        args.num_clips = signals.shape[0]
    else:
        # Select subset of clips
        signals = signals[0:args.num_clips, :]
        sleep_stages = sleep_stages[0:args.num_clips, :]
        if NUM_SLEEP_STAGE_HISTORY > 0: sleep_stages_history = sleep_stages_history[0:args.num_clips, :]

    # Convert to numpy arrays
    signals = signals.numpy()
    sleep_stages = sleep_stages.numpy()
    if NUM_SLEEP_STAGE_HISTORY > 0: sleep_stages_history = sleep_stages_history.numpy()

    # Concatenate signals and sleep stages history for easier manual manipulation
    if NUM_SLEEP_STAGE_HISTORY > 0: signals = np.concatenate((signals, sleep_stages_history), axis=1)

    # Split into training and validation sets
    if WHOLE_NIGHT_VALIDATION:
        signals_train, signals_val, sleep_stages_train, sleep_stages_val = split_whole_night_validation_set(args, signals, sleep_stages, start_of_night_markers)
        start_of_night_markers = start_of_night_markers[len(signals_train):args.num_clips]
    else: signals_train, signals_val, sleep_stages_train, sleep_stages_val = train_test_split(signals, sleep_stages, test_size=TEST_SET_RATIO, random_state=RANDOM_SEED, shuffle=True)

    # Resamples clips from the training dataset
    if RESAMPLE_TRAINING_DATASET:
        print(f"[{(time.time()-start_time):.2f}s] Data loaded. Starting resample.")
        signals_train, sleep_stages_train, original_sleep_stage_count = resample_clips(args, signals_train, sleep_stages_train)
    else:
        original_sleep_stage_count = -1

    # Trim clips to be a multiple of batch_size
    signals_train, signals_val, sleep_stages_train, sleep_stages_val = trim_clips(args, signals_train, signals_val, sleep_stages_train, sleep_stages_val)

    return signals_train, signals_val, sleep_stages_train, sleep_stages_val, start_of_night_markers, original_sleep_stage_count

def export_summary(parser, model, fit_history, accuracy:float, original_sleep_stage_count:list, sleep_stages_count_training:list, sleep_stages_count_validation:list, sleep_stages_count_pred:list) -> None:
    """
    Saves model and training summary to file
    """
    try:
        with Capturing() as model_summary:
            model.summary()
        model_summary = "\n".join(model_summary)

        repo = git.Repo(search_parent_directories=True)

        # Count relative number of stages
        if (original_sleep_stage_count != -1):  num_clips_original_dataset = sum(original_sleep_stage_count)
        num_clips_training = sum(sleep_stages_count_training)
        num_clips_validation = sum(sleep_stages_count_validation)

        log = "VISION TRANSFORMER MODEL TRAINING SUMMARY\n"
        log += f"Git hash: {repo.head.object.hexsha}\n"
        log += f"Time to complete: {(time.time()-start_time):.2f}s\n"
        log += model_summary
        log += f"\nDataset: {parser.input_dataset}\n"
        log += f"Channel: {parser.input_channel}\n"
        log += f"Validation set accuracy: {accuracy:.4f}\n"
        log += f"Training accuracy: {[round(accuracy, 4) for accuracy in fit_history.history['accuracy']]}\n"
        log += f"Training loss: {[round(loss, 4) for loss in fit_history.history['loss']]}\n"
        log += f"Number of epochs: {parser.num_epochs}\n\n"

        log += f"Training set resampling: {RESAMPLE_TRAINING_DATASET}\n"
        log += f"Training set resampling replacement: {parser.dataset_resample_replacement}\n"
        log += f"Training set resampler: {parser.dataset_resample_algo}\n"
        log += f"Training set target count: {parser.training_set_target_count}\n"
        log += f"Use sleep stage history: {USE_SLEEP_STAGE_HISTORY}\n"
        log += f"Number of historical sleep stages: {NUM_SLEEP_STAGE_HISTORY}\n"
        log += f"Dataset split random seed: {RANDOM_SEED}\n\n"

        log += f"Requested number of training clips: {int(TEST_SET_RATIO*parser.num_clips)}\n"
        if (original_sleep_stage_count != -1): log += f"Sleep stages count in original dataset ({num_clips_original_dataset}): {original_sleep_stage_count} ({[round(num / num_clips_training, 4) for num in original_sleep_stage_count]})\n"
        log += f"Sleep stages count in training data ({num_clips_training}): {sleep_stages_count_training} ({[round(num / num_clips_training, 4) for num in sleep_stages_count_training]})\n"
        log += f"Sleep stages count in validation set input ({num_clips_validation}): {sleep_stages_count_validation} ({[round(num / num_clips_validation, 4) for num in sleep_stages_count_validation]})\n"
        log += f"Sleep stages count in validation set prediction ({num_clips_validation}): {sleep_stages_count_pred} ({[round(num / num_clips_validation, 4) for num in sleep_stages_count_pred]})\n\n"

        log += f"Clip length (s): {parser.clip_length_s}\n"
        log += f"Number of sleep stages (includes unknown): {NUM_SLEEP_STAGES}\n"
        log += f"Data type: {DATA_TYPE}\n"
        log += f"Batch size: {parser.batch_size}\n"
        log += f"Embedding depth: {parser.embedding_depth}\n"
        log += f"MHA number of heads: {parser.num_heads}\n"
        log += f"Number of layers: {parser.num_layers}\n"
        log += f"MLP dimensions: {parser.mlp_dim}\n"
        log += f"Dropout rate: {DROPOUT_RATE}\n"
        log += f"Class training weights: {parser.class_weights}\n"
        log += f"Initial learning rate: {parser.learning_rate:.6f}\n"

        log += f"Model loss: {model.loss.name}\n"

        # Check whether files with the same name already exist and append counter if necessary
        candidate_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_vision.txt"

        if socket.gethostname() == "claude-ryzen":              output_log_filename = utilities.find_file_name(candidate_file_name, "/home/trobitaille/engsci-thesis/python_prototype/results/")
        elif socket.gethostname() == "MBP_Tristan":             output_log_filename = utilities.find_file_name(candidate_file_name, "/Users/tristan/Desktop/engsci-thesis/python_prototype/results/")
        elif "cedar.computecanada.ca" in socket.gethostname():  output_log_filename = utilities.find_file_name(candidate_file_name, "/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/results/")

        # Save to file
        with open(output_log_filename, 'w') as file:
            file.write(log)

        print(f"[{(time.time()-start_time):.2f}s] Saved model summary to {output_log_filename}.")

        return output_log_filename

    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to export summary.")

def parse_arguments():
    """"
    Parses command line arguments and return parser object
    """

    # Resampling documentation: https://imbalanced-learn.org/stable/introduction.html
    resampling_type_choices = ['RandomOverSampler', 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SMOTENC', 'SMOTEN', 'KMeansSMOTE', 'SVMSMOTE', 'ClusterCentroids', 'RandomUnderSampler', 'TomekLinks', 'SMOTEENN', 'SMOTETomek']

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
    parser.add_argument('--dataset_resample_algo', help="Which dataset resampling algorithm to use. Currently using 'imblearn' package.", choices=resampling_type_choices, default='RandomUnderSampler', type=str)
    parser.add_argument('--dataset_resample_replacement', help='Whether replacement is allowed when resampling dataset. Defaults to False', default=False, type=bool)
    parser.add_argument('--training_set_target_count', help='Target number of clips per class in training set. Defaults to [3500, 5000, 4000, 4250, 3750].', nargs='+', default=[3500, 5000, 4000, 4250, 3750], type=int)

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

def train_model(args, signals_train, sleep_stages_train, clip_length_num_samples, patch_length_num_samples):
    try:
        model = VisionTransformer(clip_length_num_samples=clip_length_num_samples, patch_length_num_samples=patch_length_num_samples, num_layers=args.num_layers, num_classes=NUM_SLEEP_STAGES,
                                  embedding_depth=args.embedding_depth, num_heads=args.num_heads, mlp_dim=args.mlp_dim, dropout_rate=DROPOUT_RATE, history_length=NUM_SLEEP_STAGE_HISTORY)
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

    args.class_weights = {i: weight for i, weight in enumerate(args.class_weights)}
    try:
        fit_history = model.fit(x=signals_train, y=sleep_stages_train, epochs=int(args.num_epochs), batch_size=args.batch_size,
                                callbacks=[tensorboard_callback], class_weight=args.class_weights, verbose=2)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to fit model.")

    return model, fit_history

def manual_validation(args, model, signals_val, sleep_stages_val, whole_night_indices):
    total_correct = 0
    total = 0
    sleep_stages_count_pred = [0 for _ in range(NUM_SLEEP_STAGES)]
    if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.zeros(shape=(1, NUM_SLEEP_STAGE_HISTORY), dtype=DATA_TYPE)

    print(f"[{(time.time()-start_time):.2f}s] Now commencing manual validation with {signals_val.shape[0]} clips.")

    try:
        for x, y in zip(signals_val, sleep_stages_val):
            x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1

            if NUM_SLEEP_STAGE_HISTORY > 0:
                x = tf.concat([x[:,:-NUM_SLEEP_STAGE_HISTORY], historical_pred], axis=1) # Concatenate historical prediction to input
                if whole_night_indices[total].numpy()[0] == 1.0: historical_pred = tf.zeros(shape=(1, NUM_SLEEP_STAGE_HISTORY)) # Reset historical prediction at 0 (unknown) if at the start a new night

            sleep_stage_pred = model(x, training=False)
            sleep_stage_pred = tf.cast(tf.argmax(sleep_stage_pred, axis=1), dtype=DATA_TYPE)
            if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:NUM_SLEEP_STAGE_HISTORY-1]], axis=1)

            # Count number of correct predictions
            total_correct += (sleep_stage_pred == y[0]).numpy()[0]
            total += 1

            if (VERBOSITY == 'Normal'): print(f"Ground truth: {y}, sleep stage pred: {sleep_stage_pred}, accuracy: {total_correct/total:.4f}")
            sleep_stages_count_pred[int(sleep_stage_pred)] += 1

    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to manually validate model.")

    return total_correct, sleep_stages_count_pred

def plot_single_night_prediction(args, model, single_night_filename, log_file_path):
    #Single night to compare validation and prediction
    if socket.gethostname() == "claude-ryzen":              data = tf.data.experimental.load(single_night_filename)
    elif socket.gethostname() == "MBP_Tristan":             data = tf.data.Dataset.load(single_night_filename)
    elif "cedar.computecanada.ca" in socket.gethostname():  data = tf.data.Dataset.load(single_night_filename)

    data = next(iter(data))
    total_correct = 0
    sleep_stages_single = data['sleep_stage']
    signals = data[args.input_channel]
    if NUM_SLEEP_STAGE_HISTORY > 0: sleep_stages_history = data[f'history_{NUM_SLEEP_STAGE_HISTORY}-steps']

    # Concatenate signals and sleep stages history for easier manual manipulation
    if NUM_SLEEP_STAGE_HISTORY > 0: signals = tf.concat((signals, sleep_stages_history), axis=1)

    signals_single = signals.numpy()
    sleep_stages_single = [sleep_stage[0] for sleep_stage in sleep_stages_single.numpy()]

    sleep_stages_count_single = [0, 0, 0, 0, 0, 0]
    sleep_stages_single_pred = []
    if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.zeros(shape=(1, NUM_SLEEP_STAGE_HISTORY), dtype=DATA_TYPE)

    for x, y in zip(signals_single, sleep_stages_single):
        x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1

        if NUM_SLEEP_STAGE_HISTORY > 0:
            x = tf.concat([x[:,:-NUM_SLEEP_STAGE_HISTORY], historical_pred], axis=1) # Concatenate historical prediction to input

        sleep_stage_pred = model(x, training=False)
        sleep_stage_pred = tf.cast(tf.argmax(sleep_stage_pred, axis=1), dtype=DATA_TYPE)
        sleep_stages_single_pred.append(sleep_stage_pred.numpy()[0])
        if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:NUM_SLEEP_STAGE_HISTORY-1]], axis=1)

        # Count number of correct predictions
        total_correct += (sleep_stage_pred == y).numpy()[0]

        sleep_stages_count_single[int(sleep_stage_pred)] += 1

    print(f"[{(time.time()-start_time):.2f}s] Single night inference complete. Starting plot export.")

    # Plot results
    plt.figure(figsize=(10, 6)) # Width, height in inches
    plt.plot(list(map(int, sleep_stages_single)), label='sleep_stages_single', linewidth=0.5)
    plt.plot(list(map(int, sleep_stages_single_pred)), label='sleep_stages_single_pred', linewidth=0.5)
    plt.legend()

    # Set labels and title
    plt.xlabel('Clip count')
    plt.ylabel('Sleep stage')
    plt.title(f"Ground truth vs prediction for a single night. Accuracy: {(total_correct/len(sleep_stages_single_pred)):.4}")

    # Add ticks
    plt.xticks()
    plt.yticks()

    # Add stage count
    plt.text(0, -1, f'Number of sleep stages: {sleep_stages_count_single}')
    plt.text(0, -1, f'Number of sleep stages: {sleep_stages_count_single}')

    # Export the plot
    plt.savefig(log_file_path.replace(".txt", ".png"))

    # Export interactive HTML
    trace1 = go.Scatter(y=sleep_stages_single_pred, mode='lines', name='sleep_stages_single_pred')
    trace2 = go.Scatter(y=sleep_stages_single, mode='lines', name='sleep_stages_single')

    # Create a layout
    layout = go.Layout(title=f"Ground truth vs prediction for a single night. Accuracy: {(total_correct/len(sleep_stages_single_pred)):.4}",
                       xaxis=dict(title='Clip count'),
                       yaxis=dict(title='Sleep stage'))

    # Create a Figure and add the traces
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Save the figure as an HTML file
    fig.write_html(log_file_path.replace(".txt", ".html"))

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
    def __init__(self, embedding_depth:int, num_heads:int, mlp_dim:int, dropout_rate:int=0.1, history_length:int=NUM_SLEEP_STAGE_HISTORY):
        super(Encoder, self).__init__()

        # Hyperparameters
        self.embedding_depth = embedding_depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.history_length = history_length

        # Layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.mhsa = MultiHeadSelfAttention(self.embedding_depth, self.num_heads)
        self.dropout1 = Dropout(self.dropout_rate)

        self.historical_feedback_dense = Dense(self.embedding_depth)

        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            Dense(mlp_dim, activation=tfa.activations.gelu),
            Dropout(self.dropout_rate),
            Dense(self.embedding_depth),
            Dropout(self.dropout_rate),
        ])
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, inputs, pred_history, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.mhsa(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)

        batch_size = attn_output.shape[0]

        if pred_history != None:
            # print(f"pred_history: {pred_history.shape}")
            pred_history = self.historical_feedback_dense(pred_history)
            # print(f"pred_history: {pred_history.shape}")
            pred_history = tf.reshape(pred_history, [batch_size, 1, self.embedding_depth])
            # print(f"pred_history: {pred_history.shape}")

            out1 = attn_output + inputs + pred_history
            # print(f"out1: {out1.shape}")
        else: out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)

        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1

#--- Vision Transformer ---#
class VisionTransformer(tf.keras.Model):
    def __init__(self, clip_length_num_samples:int, patch_length_num_samples:int, num_layers:int, num_classes:int,
                 embedding_depth:int, num_heads:int, mlp_dim:int, dropout_rate:float=0.1, history_length:int=NUM_SLEEP_STAGE_HISTORY):
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
        self.history_length = history_length

        # Layers
        self.rescale = Rescaling(1.0 / MAX_VOLTAGE)
        self.patch_projection = Dense(self.embedding_depth)
        self.positional_embedding = self.add_weight("pos_emb", shape=(1, self.num_patches+1, self.embedding_depth)) #+1 is for the trainable classification token prepended to input sequence of patches
        self.class_embedding = self.add_weight("class_emb", shape=(1, 1, self.embedding_depth))
        self.encoder_layers = [Encoder(embedding_depth=self.embedding_depth, num_heads=self.num_heads, mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate, history_length=self.history_length) for _ in range(self.num_layers)]
        self.mlp_head = tf.keras.Sequential([
            LayerNormalization(epsilon=1e-6),
            Dense(self.mlp_dim, activation=tfa.activations.gelu),
            Dropout(self.dropout_rate),
            Dense(self.num_classes, activation='softmax')
        ])

    def extract_patches(self, batch_size:int, clips):
        patches = tf.reshape(clips, [batch_size, -1, self.patch_length_num_samples])
        return patches

    def call(self, input, training:bool):
        # Extract historical lookback (if present)
        if self.history_length > 0:
            clip, historical_lookback = tf.split(input, [self.clip_length_num_samples, self.history_length], axis=1)
        else:
            clip = input
            historical_lookback = None

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
            clip = layer(inputs=clip, pred_history=historical_lookback, training=training)

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
    args = parse_arguments()
    print(f"[{(time.time()-start_time):.2f}s] Arguments parsed; starting dataset load.")

    # Hyperparameters
    clip_length_num_samples = int(args.clip_length_s * SAMPLING_FREQUENCY_HZ)
    patch_length_num_samples = int(args.patch_length_s * SAMPLING_FREQUENCY_HZ)

    # Load data
    try: signals_train, signals_val, sleep_stages_train, sleep_stages_val, start_of_night_markers, original_sleep_stage_count = load_from_dataset(args=args)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to load data from dataset.")

    # Train model
    print(f"[{(time.time()-start_time):.2f}s] Dataset ready. Starting training with {int((1 - TEST_SET_RATIO)*signals_train.shape[0])} clips.")
    model, fit_history = train_model(args, signals_train, sleep_stages_train, clip_length_num_samples, patch_length_num_samples)

    # Manual validation
    total_correct, sleep_stages_count_pred = manual_validation(args, model, signals_val, sleep_stages_val, start_of_night_markers)

    # Count sleep stages in training and validation datasets
    sleep_stages_count_training = utilities.count_instances_per_class(sleep_stages_train, NUM_SLEEP_STAGES)
    sleep_stages_count_validation = utilities.count_instances_per_class(sleep_stages_val, NUM_SLEEP_STAGES)

    # Save accuracy and model details to log file
    log_file_path = export_summary(args, model, fit_history, total_correct/sum(sleep_stages_count_validation), original_sleep_stage_count,
                                   sleep_stages_count_training, sleep_stages_count_validation, sleep_stages_count_pred)

    #Single night to compare validation and prediction
    print(f"[{(time.time()-start_time):.2f}s] Manual validation done. Starting validation on single night.")
    plot_single_night_prediction(args, model, "/mnt/data/tristan/engsci_thesis_python_prototype_data/single_night/SS3_EDF_Tensorized_5-stg_30s_history_4-steps_01-03-0046", log_file_path)

    print(f"[{(time.time()-start_time):.2f}s] Done. Good bye.")

if __name__ == "__main__":
    main()