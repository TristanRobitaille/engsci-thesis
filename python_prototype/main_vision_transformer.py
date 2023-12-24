import time
start_time = time.time()

import os
import io
import git
import sys
import json
import socket
import sklearn
import datetime
import imblearn

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import utilities

MAX_VOLTAGE = 2**16-1 #Maximum ADC code output
DEFAULT_CLIP_LENGTH_S = int(30)
SAMPLING_FREQUENCY_HZ = int(256)
NUM_SLEEP_STAGES = 5 + 1 #Includes 'unknown'
USE_SLEEP_STAGE_HISTORY = False
NUM_SLEEP_STAGE_HISTORY = -1
NUM_OUTPUT_FILTERING = 0
DATA_TYPE = tf.float32
TEST_SET_RATIO = 0.04 #Percentage of training data reserved for validation
RANDOM_SEED = 42
NUM_WARMUP_STEPS = 4000
VERBOSITY = 'QUIET' #'QUIET', 'NORMAL', 'DETAILED'
RESAMPLE_TRAINING_DATASET = False
WHOLE_NIGHT_VALIDATION = True or (NUM_SLEEP_STAGE_HISTORY > 0) #Whether to validate individual nights (sequentially) or a random subset of clips (if we use the preduiction history, we need whole nights for validation)
NUM_CLIPS_PER_FILE_EDGETPU = 250

#--- Helpers ---#
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
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
    resampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={1:4200, 2:10000, 3:6700, 4:9250, 5:5600}, replacement=args.enable_dataset_resample_replacement)
    signals_train, labels_train = resampler.fit_resample(signals_train, labels_train)

    #Undersampling
    if args.dataset_resample_algo == 'RandomUnderSampler':
        resampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=args.training_set_target_count, replacement=args.enable_dataset_resample_replacement)

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
    global SAMPLING_FREQUENCY_HZ

    # Extract from dataset metadata
    with open(args.input_dataset + ".json", 'r') as json_file:
        dataset_metadata = json.load(json_file)

    SAMPLING_FREQUENCY_HZ = dataset_metadata["sampling_freq_Hz"]
    NUM_SLEEP_STAGE_HISTORY = dataset_metadata["historical_lookback_length"]
    WHOLE_NIGHT_VALIDATION = (NUM_SLEEP_STAGE_HISTORY > 0)
    NUM_SLEEP_STAGES = dataset_metadata["num_stages"] + 1 # +1 needed to account for "unknown" sleep stage

    # Load dataset
    data = tf.data.Dataset.load(args.input_dataset)
    data = next(iter(data))

    sleep_stages = data['sleep_stage']
    start_of_night_markers = data['new_night_marker']
    if NUM_SLEEP_STAGE_HISTORY > 0: sleep_stages_history = data[f'history_{NUM_SLEEP_STAGE_HISTORY}-steps']

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
    else: signals_train, signals_val, sleep_stages_train, sleep_stages_val = sklearn.model_selection.train_test_split(signals, sleep_stages, test_size=TEST_SET_RATIO, random_state=RANDOM_SEED, shuffle=True)

    # Resamples clips from the training dataset
    if RESAMPLE_TRAINING_DATASET:
        print(f"[{(time.time()-start_time):.2f}s] Data loaded. Starting resample.")
        signals_train, sleep_stages_train, original_sleep_stage_count = resample_clips(args, signals_train, sleep_stages_train)
    else:
        original_sleep_stage_count = -1

    # Trim clips to be a multiple of batch_size
    signals_train, signals_val, sleep_stages_train, sleep_stages_val = trim_clips(args, signals_train, signals_val, sleep_stages_train, sleep_stages_val)

    # Cast
    signals_train = tf.cast(signals_train, dtype=DATA_TYPE)
    signals_val = tf.cast(signals_val, dtype=DATA_TYPE)
    sleep_stages_train = tf.cast(sleep_stages_train, dtype=DATA_TYPE)
    sleep_stages_val = tf.cast(sleep_stages_val, dtype=DATA_TYPE)

    if (args.output_edgetpu_data):
        for file in range(signals_val.shape[0] // NUM_CLIPS_PER_FILE_EDGETPU):
            data = np.expand_dims(signals_val[NUM_CLIPS_PER_FILE_EDGETPU*file: NUM_CLIPS_PER_FILE_EDGETPU*file + NUM_CLIPS_PER_FILE_EDGETPU], axis=1)
            np.save(f"python_prototype/edgetpu_data/{file}_{SAMPLING_FREQUENCY_HZ}Hz.npy", data)

    return signals_train, signals_val, sleep_stages_train, sleep_stages_val, start_of_night_markers, original_sleep_stage_count

def export_summary(output_log_filename, parser, model, fit_history, accuracy:float, original_sleep_stage_count:list, sleep_stages_count_training:list, sleep_stages_count_validation:list, sleep_stages_count_pred:list) -> None:
    """
    Saves model and training summary to file
    """
    try:
        with Capturing() as model_summary:
            model.summary()
        model_summary = "\n".join(model_summary)

        repo = git.Repo(search_parent_directories=True)
        if not USE_SLEEP_STAGE_HISTORY:
            parser.historical_lookback_DNN_depth = -1

        # Count relative number of stages
        if (original_sleep_stage_count != -1):  num_clips_original_dataset = sum(original_sleep_stage_count)
        num_clips_training = sum(sleep_stages_count_training)
        num_clips_validation = sum(sleep_stages_count_validation)

        log = "VISION TRANSFORMER MODEL TRAINING SUMMARY\n"
        log += f"Git hash: {repo.head.object.hexsha}\n"
        log += f"Time to complete: {(time.time()-start_time):.2f}s\n"
        log += model_summary
        log += f"\nDataset: {parser.input_dataset}\n"
        log += f"Save model: {parser.save_model}\n"
        log += f"File path of model to load (model trained if empty string): {parser.load_model_filepath}\n"
        log += f"Channel: {parser.input_channel}\n"
        log += f"Validation set accuracy: {accuracy:.4f}\n"
        log += f"Training accuracy: {[round(accuracy, 4) for accuracy in fit_history.history['accuracy']]}\n"
        log += f"Training loss: {[round(loss, 4) for loss in fit_history.history['loss']]}\n"
        log += f"Number of epochs: {parser.num_epochs}\n\n"

        log += f"Training set resampling: {RESAMPLE_TRAINING_DATASET}\n"
        log += f"Training set resampling replacement: {parser.enable_dataset_resample_replacement}\n"
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
        log += f"Patch length (s): {parser.patch_length_s}\n"
        log += f"Number of sleep stages (includes unknown): {NUM_SLEEP_STAGES}\n"
        log += f"Data type: {DATA_TYPE}\n"
        log += f"Batch size: {parser.batch_size}\n"
        log += f"Embedding depth: {parser.embedding_depth}\n"
        log += f"MHA number of heads: {parser.num_heads}\n"
        log += f"Number of layers: {parser.num_layers}\n"
        log += f"MLP dimensions: {parser.mlp_dim}\n"
        log += f"Dropout rate: {parser.dropout_rate:.3f}\n"
        log += f"Historical prediction lookback DNN depth: {parser.historical_lookback_DNN_depth}\n"
        log += f"Class training weights: {parser.class_weights}\n"
        log += f"Initial learning rate: {parser.learning_rate:.6f}\n"
        log += f"Rescale layer enabled: {parser.enable_input_rescale}\n"
        log += f"Number of samples in output filtering: {NUM_OUTPUT_FILTERING}\n"

        log += f"Model loss: {model.loss.name}\n"

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
    parser.add_argument('--input_dataset', help='Filepath of the dataset used for training and validation.', type=str)
    parser.add_argument('--input_channel', help='Name of the channel to use for training and validation.', type=str)
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
    parser.add_argument('--enable_dataset_resample_replacement', help='Whether replacement is allowed when resampling dataset.', action='store_true')
    parser.add_argument('--training_set_target_count', help='Target number of clips per class in training set. Defaults to [3500, 5000, 4000, 4250, 3750].', nargs='+', default=[3500, 5000, 4000, 4250, 3750], type=int)
    parser.add_argument('--enable_input_rescale', help='Enables layer rescaling inputs between [0, 1] at input. Defaults to True.', action='store_false')
    parser.add_argument('--dropout_rate', help='Dropout rate for all dropout layers. Defaults to 0.1.', default=0.1 , type=float)
    parser.add_argument('--save_model', help='Saves model to disk.', action='store_true')
    parser.add_argument('--load_model_filepath', help='Indicates, if not empty, the filepath of a model to load rather than training it. Defaults to None.', default=None, type=str)
    parser.add_argument('--historical_lookback_DNN_depth', help='Internal size of the output DNN for historical lookback. Defaults to 64.', default=64, type=int)
    parser.add_argument('--output_edgetpu_data', help='Select whether to output Numpy arrays when parsing input data to run on edge TPU.', action='store_true')

    # Parse arguments
    try:
        args = parser.parse_args()
    except Exception as e:
        utilities.log_error_and_exit(e)

    # Print arguments received
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Check validity of arguments
    if args.clip_length_s % args.patch_length_s != 0:
        raise ValueError(f"patch_length_s ({args.patch_length_s}s) should be an integer multiple of clip_length_s ({args.clip_length_s}s))")

    if len(args.class_weights) != NUM_SLEEP_STAGES:
        raise ValueError(f"Number of class weights ({len(args.class_weights)}) should be equal to number of sleep stages ({NUM_SLEEP_STAGES})")

    return args

def train_model(args, signals_train, sleep_stages_train, clip_length_num_samples, patch_length_num_samples):
    try:
        model = VisionTransformer(clip_length_num_samples=clip_length_num_samples, patch_length_num_samples=patch_length_num_samples, num_layers=args.num_layers, num_classes=NUM_SLEEP_STAGES, historical_lookback_DNN_depth=args.historical_lookback_DNN_depth,
                                  embedding_depth=args.embedding_depth, num_heads=args.num_heads, mlp_dim=args.mlp_dim, dropout_rate=args.dropout_rate, history_length=NUM_SLEEP_STAGE_HISTORY, enable_scaling=args.enable_input_rescale)
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

def export_plots(pred:list, ground_truth:list, accuracy:float, log_file_path:str):
    """
    Exports PNG plot and HTML web plot of the prediction and ground truth
    """

    # Plot results
    plt.figure(figsize=(10, 6)) # Width, height in inches
    plt.plot(list(map(int, ground_truth)), label='sleep_stages_single', linewidth=0.5)
    plt.plot(list(map(int, pred)), label='sleep_stages_single_pred', linewidth=0.5)
    plt.legend()

    # Set labels and title
    plt.xlabel('Clip count')
    plt.ylabel('Sleep stage')
    plt.title(f"Ground truth vs prediction for a single night. Accuracy: {accuracy:.4}")

    # Add ticks
    plt.xticks()
    plt.yticks()

    # Export the plot
    plt.savefig(log_file_path.replace(".extension", ".png"))

    # Export interactive HTML
    trace1 = go.Scatter(y=pred, mode='lines', name='sleep_stages_single_pred')
    trace2 = go.Scatter(y=ground_truth, mode='lines', name='sleep_stages_single')

    # Create a layout
    layout = go.Layout(title=f"Ground truth vs prediction for a single night. Accuracy: {accuracy:.4}",
                       xaxis=dict(title='Clip count'),
                       yaxis=dict(title='Sleep stage'))

    # Create a Figure and add the traces
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Save the figure as an HTML file
    fig.write_html(log_file_path.replace(".extension", ".html"))

def manual_validation(model, signals_val, sleep_stages_val, whole_night_indices):
    total_correct = 0
    total = 0
    sleep_stages_count_pred = [0 for _ in range(NUM_SLEEP_STAGES)]
    if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.zeros(shape=(1, NUM_SLEEP_STAGE_HISTORY), dtype=DATA_TYPE)

    print(f"[{(time.time()-start_time):.2f}s] Now commencing manual validation with {signals_val.shape[0]} clips.")

    sleep_stages_pred = []
    sleep_stages_ground_truth = []

    output_filter = utilities.MovingAverage(NUM_OUTPUT_FILTERING)

    try:
        for x, y in zip(signals_val, sleep_stages_val):
            x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1
            sleep_stages_ground_truth.append(y[0])

            if NUM_SLEEP_STAGE_HISTORY > 0:
                x = tf.concat([x[:,:-NUM_SLEEP_STAGE_HISTORY], historical_pred], axis=1) # Concatenate historical prediction to input
                if whole_night_indices[total].numpy()[0] == 1.0: historical_pred = tf.zeros(shape=(1, NUM_SLEEP_STAGE_HISTORY)) # Reset historical prediction at 0 (unknown) if at the start a new night

            sleep_stage_pred = model(x, training=False)
            sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)

            # Filter sleep stage
            sleep_stage_pred = tf.cast(output_filter.filter(sleep_stage_pred), dtype=DATA_TYPE)

            if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:NUM_SLEEP_STAGE_HISTORY-1]], axis=1)

            # Count number of correct predictions
            total_correct += (sleep_stage_pred[0] == y[0]).numpy()
            total += 1

            sleep_stages_pred.append(sleep_stage_pred[0])

            if (VERBOSITY == 'Normal'): print(f"Ground truth: {y}, sleep stage pred: {sleep_stage_pred}, accuracy: {total_correct/total:.4f}")
            sleep_stages_count_pred[int(sleep_stage_pred)] += 1

    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description="Failed to manually validate model.")

    return total_correct, sleep_stages_count_pred, sleep_stages_pred, sleep_stages_ground_truth

def plot_single_night_prediction(args, model, single_night_filename, log_file_path):
    #Single night to compare validation and prediction
    data = tf.data.Dataset.load(single_night_filename)

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
    output_filter = utilities.MovingAverage(NUM_OUTPUT_FILTERING)

    if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.zeros(shape=(1, NUM_SLEEP_STAGE_HISTORY), dtype=DATA_TYPE)

    for x, y in zip(signals_single, sleep_stages_single):
        x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1

        if NUM_SLEEP_STAGE_HISTORY > 0:
            x = tf.concat([x[:,:-NUM_SLEEP_STAGE_HISTORY], historical_pred], axis=1) # Concatenate historical prediction to input

        sleep_stage_pred = model(x, training=False)
        sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)
        sleep_stage_pred = tf.cast(output_filter.filter(sleep_stage_pred), dtype=DATA_TYPE)
        sleep_stages_single_pred.append(sleep_stage_pred[0])
        if NUM_SLEEP_STAGE_HISTORY > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:NUM_SLEEP_STAGE_HISTORY-1]], axis=1)

        # Count number of correct predictions
        total_correct += (sleep_stage_pred == y).numpy()[0]

        sleep_stages_count_single[int(sleep_stage_pred)] += 1

    print(f"[{(time.time()-start_time):.2f}s] Single night inference complete. Starting plot export.")
    export_plots(pred=sleep_stages_single_pred, ground_truth=sleep_stages_single, accuracy=total_correct/len(sleep_stages_single_pred), log_file_path=log_file_path)

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
        self.query_dense = tf.keras.layers.Dense(self.embedding_dimension, name="mhsa_dense1")
        self.key_dense = tf.keras.layers.Dense(self.embedding_dimension, name="mhsa_dense2")
        self.value_dense = tf.keras.layers.Dense(self.embedding_dimension, name="mhsa_dense3")
        self.combine_heads = tf.keras.layers.Dense(self.embedding_dimension, name="mhsa_dense4")

    def attention(self, query, key, value):
        query = tf.cast(query, dtype=DATA_TYPE)
        key = tf.cast(key, dtype=DATA_TYPE)
        value = tf.cast(value, dtype=DATA_TYPE)
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
    def __init__(self, embedding_depth:int, num_heads:int, mlp_dim:int, dropout_rate:float=0.1, history_length:int=NUM_SLEEP_STAGE_HISTORY):
        super(Encoder, self).__init__()

        # Hyperparameters
        self.embedding_depth = embedding_depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.history_length = history_length

        # Layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layerNorm_encoder")
        self.mhsa = MultiHeadSelfAttention(self.embedding_depth, self.num_heads)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="dropout_encoder")

        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_layerNorm_encoder")
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=tfa.activations.gelu, name="mlp_dense1_encoder"),
            tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="mlp_dropout1_encoder"),
            tf.keras.layers.Dense(self.embedding_depth, name="mlp_dense2_encoder"),
            tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="mlp_dropout2_encoder"),
        ], name="mlp_encoder")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="dropout2_encoder")

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
    def __init__(self, clip_length_num_samples:int, patch_length_num_samples:int, num_layers:int, num_classes:int, embedding_depth:int, historical_lookback_DNN_depth:int,
                 num_heads:int, mlp_dim:int, dropout_rate:float=0.1, enable_scaling:bool=True, history_length:int=NUM_SLEEP_STAGE_HISTORY):
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
        self.enable_scaling = enable_scaling
        self.historical_lookback_DNN = False and (True if self.history_length > 0 else False)
        self.historical_lookback_DNN_depth = historical_lookback_DNN_depth

        # Layers
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / MAX_VOLTAGE)
        self.patch_projection = tf.keras.layers.Dense(self.embedding_depth, name="patch_projection_dense")
        if self.history_length > 0:
            self.positional_embedding = self.add_weight("pos_emb", shape=(1, self.num_patches+2, self.embedding_depth)) #+2 is for the trainable classification token the historical lookback prepended to input sequence of patches
        else:
            self.positional_embedding = self.add_weight("pos_emb", shape=(1, self.num_patches+1, self.embedding_depth)) #+1 is for the trainable classification token prepended to input sequence of patches
        self.class_embedding = self.add_weight("class_emb", shape=(1, 1, self.embedding_depth))
        self.encoder_layers = [Encoder(embedding_depth=self.embedding_depth, num_heads=self.num_heads, mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate, history_length=self.history_length) for _ in range(self.num_layers)]
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_head_layerNorm"),
            tf.keras.layers.Dense(self.mlp_dim, activation=tfa.activations.gelu, name="mlp_head_dense1"),
            tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="mlp_head_dropout"),
            tf.keras.layers.Dense(self.num_classes, activation='softmax', name="mlp_head_dense2")
        ], name="mlp_head")
        # self.historical_lookback = tf.keras.Sequential([
        #     tf.keras.layers.Dense(self.historical_lookback_DNN_depth, activation=tf.keras.activations.relu),
        #     tf.keras.layers.Dense(2 * self.historical_lookback_DNN_depth, activation=tf.keras.activations.relu),
        #     tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED),
        #     tf.keras.layers.Dense(self.num_classes, activation='softmax')
        # ], name="historical_lookback")

    def extract_patches(self, batch_size:int, clips):
        patches = tf.reshape(clips, [batch_size, -1, self.patch_length_num_samples])
        return patches

    def call(self, input, training:bool):
        # Extract historical lookback (if present)
        if self.history_length > 0:
            clip, historical_lookback = tf.split(input, [self.clip_length_num_samples, self.history_length], axis=1)
            historical_lookback_padding = tf.zeros(shape=(clip.shape[0], self.patch_length_num_samples - self.history_length), dtype=DATA_TYPE)
            historical_lookback_patch = tf.concat([historical_lookback, historical_lookback_padding], axis=1)
            historical_lookback_patch = tf.expand_dims(historical_lookback_patch, axis=1)
        else:
            clip = input
            historical_lookback = None

        batch_size = clip.shape[0]
        if batch_size == None: batch_size = 1

        # Normalize to [0,1]
        if self.enable_scaling: clip = self.rescale(clip)

        # Extract patches
        patches = self.extract_patches(batch_size, clip)
        if self.history_length > 0:
            patches = tf.concat([patches, historical_lookback_patch], axis=1)

        # Linear projection
        clip = self.patch_projection(patches) #clip = (num_patches, embedding_depth)

        # Classification token
        class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.embedding_depth])

        # Construct sequence
        clip = tf.concat([class_embedding, clip], axis=1)
        clip = clip + self.positional_embedding

        # Go through encoder
        for layer in self.encoder_layers:
            clip = layer(inputs=clip, training=training)

        # Classify with first token
        prediction = self.mlp_head(clip[:, 0])

        if self.historical_lookback_DNN:
            historical_lookback = tf.cast(historical_lookback, tf.uint8)
            historical_lookback = tf.one_hot(historical_lookback, depth=self.num_classes)
            prediction = tf.expand_dims(prediction, axis=1)
            concatenated = tf.concat([prediction, historical_lookback], axis=1)
            concatenated = tf.reshape(concatenated, [batch_size, -1])
            prediction = self.historical_lookback(concatenated)

        return prediction

#--- Misc ---#
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_depth, warmup_steps=NUM_WARMUP_STEPS):
        super().__init__()
        self.embedding_depth = embedding_depth
        self.embedding_depth = tf.cast(self.embedding_depth, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embedding_depth) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'embedding_depth': int(self.embedding_depth),
            'warmup_steps': int(self.warmup_steps),
        }
        return config

def main():
    # Parse arguments
    args = parse_arguments()
    print(f"[{(time.time()-start_time):.2f}s] Arguments parsed; starting dataset load.")

    # Load data
    try: signals_train, signals_val, sleep_stages_train, sleep_stages_val, start_of_night_markers, original_sleep_stage_count = load_from_dataset(args=args)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to load data from dataset.")

    # Train or load model
    clip_length_num_samples = int(args.clip_length_s * SAMPLING_FREQUENCY_HZ)
    patch_length_num_samples = int(args.patch_length_s * SAMPLING_FREQUENCY_HZ)
    if args.load_model_filepath == None:
        print(f"[{(time.time()-start_time):.2f}s] Dataset ready. Starting training with {int(signals_train.shape[0])} clips.")
        model, fit_history = train_model(args, signals_train, sleep_stages_train, clip_length_num_samples, patch_length_num_samples)
    else: model = tf.keras.models.load_model(args.load_model_filepath, custom_objects={"CustomSchedule": CustomSchedule})

    # Manual validation
    total_correct, sleep_stages_count_pred, manual_validation_pred, manual_validation_ground_truth = manual_validation(model, signals_val, sleep_stages_val, start_of_night_markers)

    # Count sleep stages in training and validation datasets
    sleep_stages_count_training = utilities.count_instances_per_class(sleep_stages_train, NUM_SLEEP_STAGES)
    sleep_stages_count_validation = utilities.count_instances_per_class(sleep_stages_val, NUM_SLEEP_STAGES)

    # Make results directory. Check whether folder with the same name already exist and append counter if necessary
    time_of_export = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if socket.gethostname() == "claude-ryzen":              folder_base_path = f"/home/trobitaille/engsci-thesis/python_prototype/results/"
    elif socket.gethostname() == "MBP_Tristan":             folder_base_path = f"/Users/tristan/Desktop/engsci-thesis/python_prototype/results/"
    elif "cedar.computecanada.ca" in socket.gethostname():  folder_base_path = f"/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/results/"

    output_folder_path = utilities.find_folder_path(folder_base_path+f"{time_of_export}_vision", folder_base_path)

    os.makedirs(output_folder_path, exist_ok=True)

    # Save accuracy and model details to log file
    export_summary(f"{output_folder_path}/{time_of_export}_vision.txt", args, model, fit_history, total_correct/sum(sleep_stages_count_validation), original_sleep_stage_count,
                   sleep_stages_count_training, sleep_stages_count_validation, sleep_stages_count_pred)

    # Export plot of prediction and ground truth
    export_plots(pred=manual_validation_pred, ground_truth=manual_validation_ground_truth, accuracy=total_correct/len(manual_validation_pred), log_file_path=f"{output_folder_path}/{time_of_export}_vision_manual_validation.extension")

    #Single night to compare validation and prediction
    print(f"[{(time.time()-start_time):.2f}s] Manual validation done. Starting validation on single night.")
    plot_single_night_prediction(args, model, single_night_filename="/mnt/data/tristan/engsci_thesis_python_prototype_data/single_night/SS3_EDF_Tensorized_5-stg_30s_200Hz_01-03-0046", log_file_path=f"{output_folder_path}/{time_of_export}_vision.extension")

    # Save model to disk
    if args.save_model:
        model.save(f"{output_folder_path}/{time_of_export}_vision.tf", save_format="tf")
        print(f"[{(time.time()-start_time):.2f}s] Saved model to {output_folder_path}/{time_of_export}_vision.tf.")

        # Convert to Tensorflow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save .tflite model
        with open(f"{output_folder_path}/{time_of_export}_vision.tflite", "wb") as f:
            f.write(tflite_model)
        print(f"[{(time.time()-start_time):.2f}s] Saved TensorFlow Lite model to {output_folder_path}/{time_of_export}_vision.tflite.")

    print(f"[{(time.time()-start_time):.2f}s] Done. Good bye.")

if __name__ == "__main__":
    main()