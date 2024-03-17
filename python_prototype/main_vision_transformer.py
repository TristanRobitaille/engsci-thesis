import time
start_time = time.time()

import os
import io
import csv
import git
import sys
import json
import socket
import shutil
import datetime
import imblearn
import utilities

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import plotly.graph_objects as go

MAX_VOLTAGE = 2**16-1 # Maximum ADC code output
DEFAULT_CLIP_LENGTH_S = int(30)
SAMPLING_FREQUENCY_HZ = int(256)
USE_SLEEP_STAGE_HISTORY = False
NUM_SLEEP_STAGE_HISTORY = -1
DATA_TYPE = tf.float32
NUM_NIGHTS_VALIDATION = 2 # Number of nights used for validation
RANDOM_SEED = 42
RESAMPLE_TRAINING_DATASET = False
SHUFFLE_TRAINING_CLIPS = True
NUM_CLIPS_PER_FILE_EDGETPU = 500 # 500 is only valid for 256Hz
K_FOLD_OUTPUT_TO_FILE = False # If true, will write validation accuracy to a CSV for k-fold sweep validation
K_FOLD_SETS_MANUAL_PRUNE = [4]

AVAILABLE_OPTIMIZERS = ["Adam", "AdamW"]
AVAILABLE_RESAMPLERS = ['RandomOverSampler', 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SMOTENC', 'SMOTEN', 'KMeansSMOTE', 'SVMSMOTE', 'ClusterCentroids', 'RandomUnderSampler', 'TomekLinks', 'SMOTEENN', 'SMOTETomek'] # https://imbalanced-learn.org/stable/introduction.html

DEBUG = False
OUTPUT_CSV = False

train_signals_representative_dataset = None
sleep_map = utilities.SleepStageMap()

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

    original_sleep_stage_count = utilities.count_instances_per_class(labels_train, sleep_map.get_num_stages()+1) #+1 is to account for 'Unknown'

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

def split_whole_night_validation_set(signals, args, sleep_stages, whole_night_markers):
    """
    Split the dataset into whole nights for validation
    """

    whole_night_indices = [i for i, x in enumerate(whole_night_markers.numpy()) if x == 1]
    last_k_fold_set = (NUM_NIGHTS_VALIDATION*args.k_fold_val_set + NUM_NIGHTS_VALIDATION) >= len(whole_night_indices) # Last k-fold validation set
    start_of_1st_val_night = whole_night_indices[NUM_NIGHTS_VALIDATION*args.k_fold_val_set] # Index of the first clip in the first validation night

    if (not last_k_fold_set):
        end_of_last_val_night = whole_night_indices[NUM_NIGHTS_VALIDATION*args.k_fold_val_set + NUM_NIGHTS_VALIDATION] # Index of the first clip of the night after the last validation night
        signals_train = np.concatenate((np.array(signals[0:start_of_1st_val_night]), signals[end_of_last_val_night:]))
        sleep_stages_train = np.concatenate((np.array(sleep_stages[0:start_of_1st_val_night]), sleep_stages[end_of_last_val_night:]))
        signals_val = signals[start_of_1st_val_night : end_of_last_val_night]
        sleep_stages_val = sleep_stages[start_of_1st_val_night : end_of_last_val_night]
    else:
        signals_train = signals[0:start_of_1st_val_night]
        sleep_stages_train = sleep_stages[0:start_of_1st_val_night]
        signals_val = signals[start_of_1st_val_night:]
        sleep_stages_val = sleep_stages[start_of_1st_val_night:]

    # Update new night markers to match indices of validation data
    whole_night_indices = whole_night_indices[NUM_NIGHTS_VALIDATION*args.k_fold_val_set : NUM_NIGHTS_VALIDATION*args.k_fold_val_set + NUM_NIGHTS_VALIDATION]
    whole_night_indices = [index - start_of_1st_val_night for index in whole_night_indices]

    return signals_train, signals_val, sleep_stages_train, sleep_stages_val, whole_night_indices

def load_from_dataset(args):
    """
    Loads data from dataset and returns batched, shuffled dataset of correct channel
    """

    global NUM_SLEEP_STAGE_HISTORY
    global SAMPLING_FREQUENCY_HZ
    global CLIP_LENGTH_NUM_SAMPLES

    # Extract from dataset metadata
    with open(args.input_dataset + ".json", 'r') as json_file:
        dataset_metadata = json.load(json_file)
        print(f"Dataset metadata {json.dumps(dataset_metadata, indent=4)}\n\n")

    SAMPLING_FREQUENCY_HZ = dataset_metadata["sampling_freq_Hz"]
    NUM_SLEEP_STAGE_HISTORY = dataset_metadata["historical_lookback_length"]
    CLIP_LENGTH_NUM_SAMPLES = int(dataset_metadata["clip_length_s"] * SAMPLING_FREQUENCY_HZ)
    sleep_map.set_map_name(dataset_metadata["sleep_stages_map_name"])

    # Check validity of arguments
    val_nights = range(args.k_fold_val_set*NUM_NIGHTS_VALIDATION, args.k_fold_val_set*NUM_NIGHTS_VALIDATION+NUM_NIGHTS_VALIDATION)
    if len(args.class_weights) != (sleep_map.get_num_stages()+1):
        raise ValueError(f"Number of class weights ({len(args.class_weights)}) should be equal to number of sleep stages ({sleep_map.get_num_stages()+1})")
    if (dataset_metadata["num_files_used"] % NUM_NIGHTS_VALIDATION != 0):
        raise ValueError(f"Number of nights in dataset ({dataset_metadata['num_files_used']}) not a multiple of number of nights to use in validation ({NUM_NIGHTS_VALIDATION})!")
    if not os.path.exists(os.path.dirname(args.k_fold_val_results_fp)):
        raise ValueError(f"Base path for k-fold validation results ({os.path.dirname(args.k_fold_val_results_fp)}) doesn't exist!")
    if max(val_nights) >= dataset_metadata["num_files_used"]:
        raise ValueError(f"One of more nights used for validation ({list(val_nights)}) exceeds number of files in dataset ({dataset_metadata['num_files_used']})!")
    if (CLIP_LENGTH_NUM_SAMPLES % args.patch_length != 0):
        raise ValueError(f"Patch length (# of samples; {int(args.patch_length)}) needs to be an integer divisor of clip length (# of samples; {CLIP_LENGTH_NUM_SAMPLES}s! Aborting.)")

    # Load dataset
    if "cedar.computecanada.ca" in socket.gethostname(): # Compute Canada (aka running on GPU needs Tensorflow 2.8.0) needs a Tensorflow downgrade (or gets a compilation error)
        data = tf.data.experimental.load(args.input_dataset)
    else: data = tf.data.Dataset.load(args.input_dataset)

    data = next(iter(data))

    sleep_stages = data['sleep_stage']
    start_of_night_markers = data['new_night_marker']
    if NUM_SLEEP_STAGE_HISTORY > 0:
        sleep_stages_history = data[f'history_{NUM_SLEEP_STAGE_HISTORY}-steps']
        raise ValueError("Don't use historical lookback. Script isn't set up correctly for that (clips are shuffled, history not handled in model, etc.)")

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
    signals_train, signals_val, sleep_stages_train, sleep_stages_val, start_of_val_night_indices = split_whole_night_validation_set(signals, args, sleep_stages, start_of_night_markers)

    # Shuffle training data
    if SHUFFLE_TRAINING_CLIPS: signals_train, sleep_stages_train = utilities.shuffle(signals_train, sleep_stages_train, RANDOM_SEED)

    # Resamples clips from the training dataset
    if RESAMPLE_TRAINING_DATASET:
        print(f"[{(time.time()-start_time):.2f}s] Data loaded. Starting resample.")
        signals_train, sleep_stages_train, original_sleep_stage_count = resample_clips(args, signals_train, sleep_stages_train)
    else:
        original_sleep_stage_count = -1

    # Trim clips to be a multiple of batch_size
    signals_train, signals_val, sleep_stages_train, sleep_stages_val = trim_clips(args, signals_train, signals_val, sleep_stages_train, sleep_stages_val)

    # Output edge TPU data
    if (args.output_edgetpu_data):
        for file in range(signals_val.shape[0] // NUM_CLIPS_PER_FILE_EDGETPU):
            data = np.expand_dims(signals_val[NUM_CLIPS_PER_FILE_EDGETPU*file: NUM_CLIPS_PER_FILE_EDGETPU*file + NUM_CLIPS_PER_FILE_EDGETPU], axis=1)
            np.save(f"python_prototype/edgetpu_data/{file}_{SAMPLING_FREQUENCY_HZ}Hz.npy", data)

    # Cast
    signals_train = tf.cast(signals_train, dtype=DATA_TYPE)
    signals_val = tf.cast(signals_val, dtype=DATA_TYPE)
    sleep_stages_train = tf.cast(sleep_stages_train, dtype=DATA_TYPE)
    sleep_stages_val = tf.cast(sleep_stages_val, dtype=DATA_TYPE)
    data_dict = {"signals_train":signals_train, "signals_val":signals_val, "sleep_stages_train":sleep_stages_train, "sleep_stages_val":sleep_stages_val}

    # Save data_dict to disk for future use in accuracy study
    if (args.reference_night_fp != ""): np.save("asic/fixed_point_accuracy_study/ref_data.npy", data_dict)

    return data_dict, start_of_val_night_indices, original_sleep_stage_count, dataset_metadata

def export_summary(out_fp, parser, model, fit_history, acc:dict, original_sleep_stage_count:list, sleep_stages_count_training:list,
                   sleep_stages_count_val:list, pred_cnt:dict, mlp_dense_activation, dataset_metadata:dict, model_specific_only:bool=False) -> None:
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
        num_clips_validation = sum(sleep_stages_count_val)

        log = "VISION TRANSFORMER MODEL TRAINING SUMMARY\n"
        log += f"Git hash: {repo.head.object.hexsha}\n"
        if not model_specific_only: log += f"Time to complete: {(time.time()-start_time):.2f}s\n"
        log += model_summary
        log += f"\nDataset: {parser.input_dataset}\n"
        log += f"Number of operations: {model.calculate_ops()}\n"
        log += f"Save model: {parser.save_model}\n"
        log += f"Save k-fold validation file: {K_FOLD_OUTPUT_TO_FILE}\n"
        log += f"File path of model to load (model trained if empty string): {parser.load_model_filepath}\n"
        log += f"Channel: {parser.input_channel}\n"
        if not model_specific_only:
            log += f"k-fold validation set: {parser.k_fold_val_set}\n"
            for model_type, acc in acc.items(): log += f"Validation set accuracy ({model_type}): {acc[-1]}\n" # Last items in acc is always accuracy for this run
            log += f"Training accuracy: {[round(accuracy, 4) for accuracy in fit_history.history['accuracy']]}\n"
            log += f"Validation accuracy (while training): {[round(accuracy, 4) for accuracy in fit_history.history['val_accuracy']]}\n"
            log += f"Training loss: {[round(loss, 4) for loss in fit_history.history['loss']]}\n"
            log += f"Validation loss (while training): {[round(loss, 4) for loss in fit_history.history['val_loss']]}\n"
        log += f"Number of epochs: {parser.num_epochs}\n\n"
        log += f"# of nights in validation: {NUM_NIGHTS_VALIDATION}\n"

        log += f"Training set resampling: {RESAMPLE_TRAINING_DATASET}\n"
        log += f"Training set resampling replacement: {parser.enable_dataset_resample_replacement}\n"
        log += f"Training set shuffled: {SHUFFLE_TRAINING_CLIPS}\n"
        log += f"Training set resampler: {parser.dataset_resample_algo}\n"
        log += f"Training set target count: {parser.training_set_target_count}\n"
        log += f"Use sleep stage history: {USE_SLEEP_STAGE_HISTORY}\n"
        log += f"Number of historical sleep stages: {NUM_SLEEP_STAGE_HISTORY}\n"
        log += f"Dataset split random seed: {RANDOM_SEED}\n"
        log += f"Dataset metadata {json.dumps(dataset_metadata, indent=4)}\n\n"

        if (original_sleep_stage_count != -1): log += f"Sleep stages count in original dataset ({num_clips_original_dataset}): {original_sleep_stage_count} ({[round(num / num_clips_training, 4) for num in original_sleep_stage_count]})\n"
        if not model_specific_only: log += f"Sleep stages count in training data ({num_clips_training}): {sleep_stages_count_training} ({[round(num / num_clips_training, 4) for num in sleep_stages_count_training]})\n"
        if not model_specific_only: log += f"Sleep stages count in validation set input ({num_clips_validation}): {sleep_stages_count_val} ({[round(num / num_clips_validation, 4) for num in sleep_stages_count_val]})\n"
        if not model_specific_only:
            for model_type, pred_cnt in pred_cnt.items(): log += f"Sleep stages count in validation set prediction ({num_clips_validation}, {model_type}): {pred_cnt} ({[round(num / num_clips_validation, 4) for num in pred_cnt[-1]]})\n"

        log += f"\nClip length (s): {dataset_metadata['clip_length_s']}\n"
        log += f"Patch length (# of samples): {parser.patch_length}\n"
        log += f"Number of sleep stages (includes unknown): {sleep_map.get_num_stages()+1}\n"
        log += f"Data type: {DATA_TYPE}\n"
        log += f"Batch size: {parser.batch_size}\n"
        log += f"Embedding depth: {parser.embedding_depth}\n"
        log += f"MHA number of heads: {parser.num_heads}\n"
        log += f"Number of layers: {parser.num_layers}\n"
        log += f"MLP dimension: {parser.mlp_dim}\n"
        log += f"Number of dense (+ dropout) layers in MLP head before softmax: {parser.mlp_head_num_dense}\n"
        log += f"Dropout rate: {parser.dropout_rate_percent:.3f}\n"
        log += f"Historical prediction lookback DNN depth: {parser.historical_lookback_DNN_depth}\n"
        log += f"Activation function of first dense layer in MLP layer and MLP head: {mlp_dense_activation}\n"
        log += f"Class training weights: {parser.class_weights}\n"
        log += f"Initial learning rate: {parser.learning_rate:.6f}\n"
        log += f"Optimizer: {parser.optimizer}\n"
        log += f"Rescale layer enabled: {parser.enable_input_rescale}\n"
        log += f"Use classification token: {parser.use_class_embedding}\n"
        log += f"Positional embedding enabled: {parser.enable_positional_embedding}\n"
        log += f"Output filter type: {parser.out_filter_type}\n"
        log += f"Number of samples in output filter: {parser.num_out_filter}\n"
        log += f"Output filter self-reset threshold: {parser.filter_self_reset_threshold}\n"
        log += f"Model loss type: {model.loss.name}\n"
        log += f"Note: {parser.note}\n"

        # Save to file
        with open(out_fp, 'w') as file: file.write(log)

    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to export summary.")

def parse_arguments():
    """"
    Parses command line arguments and return parser object
    """

    parser = utilities.ArgumentParserWithError(description='Transformer model Tensorflow prototype.')
    parser.add_argument('--num_clips', help='Number of clips to use for training + validation. Defaults to 3000.', default=3000, type=int)
    parser.add_argument('--input_dataset', help='Filepath of the dataset used for training and validation.', type=str)
    parser.add_argument('--input_channel', help='Name of the channel to use for training and validation.', type=str)
    parser.add_argument('--patch_length', help='Patch length (in # of samples). Must be integer divisor of sampling_freq*clip_length_s. Defaults to 256.', default=256, type=int)
    parser.add_argument('--num_layers', help='Number of encoder layer. Defaults to 8.', default=8, type=int)
    parser.add_argument('--embedding_depth', help='Depth of the embedding layer. Defaults to 32.', default=32, type=int)
    parser.add_argument('--num_heads', help='Number of multi-attention heads. Defaults to 8.', default=8, type=int)
    parser.add_argument('--mlp_dim', help='Dimension of the MLP layer. Defaults to 32.', default=32, type=int)
    parser.add_argument('--mlp_head_num_dense', help="Number of dense layers (with its dropout) to add before softmax layer in MLP head. Defaults to 1.", default=1, type=int)
    parser.add_argument('--num_epochs', help='Number of training epochs. Defaults to 25.', default=25, type=int)
    parser.add_argument('--batch_size', help='Batch size for training. Defaults to 8.', default=8, type=int)
    parser.add_argument('--learning_rate', help='Learning rate for training. Defaults to 1e-4.', default=1e-4, type=float)
    parser.add_argument('--class_weights', help='List of weights to apply in loss calculation.', nargs='+', default=[1, 1, 1, 1, 1, 1], type=float)
    parser.add_argument('--dataset_resample_algo', help="Which dataset resampling algorithm to use. Currently using 'imblearn' package.", choices=AVAILABLE_RESAMPLERS, default='RandomUnderSampler', type=str)
    parser.add_argument('--enable_dataset_resample_replacement', help='Whether replacement is allowed when resampling dataset.', action='store_true')
    parser.add_argument('--training_set_target_count', help='Target number of clips per class in training set. Defaults to [3500, 5000, 4000, 4250, 3750].', nargs='+', default=[3500, 5000, 4000, 4250, 3750], type=int)
    parser.add_argument('--enable_input_rescale', help='Enables layer rescaling inputs between [0, 1] at input.', action='store_true')
    parser.add_argument('--enable_positional_embedding', help='Enables positional embedding.', action='store_true')
    parser.add_argument('--dropout_rate_percent', help='Dropout rate for all dropout layers (in integer %). Defaults to 10%.', default=10 , type=int)
    parser.add_argument('--save_model', help='Saves model to disk.', action='store_true')
    parser.add_argument('--load_model_filepath', help='Indicates, if not empty, the filepath of a model to load rather than training it. Defaults to None.', default=None, type=str)
    parser.add_argument('--historical_lookback_DNN_depth', help='Internal size of the output DNN for historical lookback. Defaults to 64.', default=64, type=int)
    parser.add_argument('--output_edgetpu_data', help='Select whether to output Numpy arrays when parsing input data to run on edge TPU.', action='store_true')
    parser.add_argument('--use_class_embedding', help='Select whether to use classification token in model.', action='store_true')
    parser.add_argument('--k_fold_val_set', help='Set number used for k-fold validation. Starts at and defaults to 0th set.', default=0, type=int)
    parser.add_argument('--num_out_filter', help='Number of averages in output moving average filter. Set to 0 to disable. Defaults to 0.', default=0, type=int)
    parser.add_argument('--k_fold_val_results_fp', help='Filepath of CSV file used for k-fold validation results. Also exports model details to .txt of the same path.', type=str)
    parser.add_argument('--out_filter_type', help="Selects whether to filter the model output before the argmax operation on the softmax layer's output. Default to 'post_argmax'.", choices=["pre_argmax", "post_argmax"], default="post_argmax", type=str)
    parser.add_argument('--filter_self_reset_threshold', help="Number of samples for which the sleep stage prediction must be constant for the filter to self-reset. Set to -1 to disable. Default to -1.", default=-1, type=int)
    parser.add_argument('--note', help="Optional note to write info textfile. Defaults to None.", default="", type=str )
    parser.add_argument('--num_runs', help="Number of training runs to perform. Defaults to 1.", default=1, type=int)
    parser.add_argument('--optimizer', help=f"Optimizer to use. May be one of {AVAILABLE_OPTIMIZERS}. Defaults to Adam.", choices=AVAILABLE_OPTIMIZERS, default="Adam", type=str)
    parser.add_argument('--reference_night_fp', help="Filepath of the reference night used for validation.", default="", type=str)

    # Parse arguments
    try:
        args = parser.parse_args()
    except Exception as e:
        utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to parse arguments.")

    # Scale arguments
    args.dropout_rate_percent /= 100

    # Print arguments received
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Prune bad set
    if args.k_fold_val_set in K_FOLD_SETS_MANUAL_PRUNE:
        print(f"Receive k-fold set {args.k_fold_val_set}, which is in the pruned set ({K_FOLD_SETS_MANUAL_PRUNE}). Exiting.")
        exit()

    return args

def train_model(args, data:dict, mlp_dense_activation:str):
    try:
        model = VisionTransformer(args, mlp_dense_activation)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to initialize model.")

    try:
        if args.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(CustomSchedule(args.embedding_depth), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        elif args.optimizer == "AdamW": # AdamW is used in 'MultiChannelSleepNet: A Transformer-Based Model for Automatic Sleep Stage Classification With PSG'
            if "cedar.computecanada.ca" in socket.gethostname(): # Compute Canada (aka running on GPU needs Tensorflow 2.8.0) doesn't have AdamW in Keras
                print("CAUTION: You are using the AdamW optimizer on Cedar (so with TensorFlow 2.8), which isn't working (accuracy stuck during training)")
                optimizer = tfa.optimizers.AdamW(learning_rate=CustomSchedule(args.embedding_depth), weight_decay=0.004, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            else:
                optimizer = tf.keras.optimizers.AdamW(learning_rate=CustomSchedule(args.embedding_depth), weight_decay=0.004, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=optimizer, metrics=["accuracy"])

    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to compile model.")

    tensorboard_log_dir = "logs/fit/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    args.class_weights = {i: weight for i, weight in enumerate(args.class_weights)}

    try:
        fit_history = model.fit(x=data["signals_train"], y=data["sleep_stages_train"], validation_data=(data["signals_val"], data["sleep_stages_val"]),
                                epochs=args.num_epochs, batch_size=args.batch_size, callbacks=[tensorboard_callback], class_weight=args.class_weights, verbose=2)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to fit model.")

    return model, fit_history

def export_plots(pred:list, ground_truth:list, accuracy:float, log_file_path:str, ds_metadata:dict):
    """
    Exports PNG plot and HTML web plot of the prediction and ground truth
    """
    x_axis_label = f'Clip ({ds_metadata["clip_length_s"]:.1f}s) count'

    # Export PNG
    data = {"Prediction":list(map(int, pred)), "Ground truth":list(map(int, ground_truth))}
    plt.figure(figsize=(11, 6), dpi=300) # Width, height in inches
    plt.plot('Ground truth', data=data, linewidth=0.5)
    plt.plot('Prediction', data=data, linewidth=0.5)
    plt.xlabel(x_axis_label)
    plt.title(f"Ground truth vs prediction. Accuracy: {accuracy:.4}")
    plt.legend()
    plt.grid()
    plt.yticks(np.arange(0, sleep_map.get_num_stages()+1, step=1), sleep_map.get_name_map())
    plt.savefig(log_file_path.replace(".ext", ".png")) # Export the plot

    # Export interactive HTML
    trace1 = go.Scatter(y=data['Prediction'], mode='lines', name='Model prediction')
    trace2 = go.Scatter(y=data['Ground truth'], mode='lines', name='Ground truth')
    layout = go.Layout(title=f"Ground truth vs prediction. Accuracy: {accuracy:.4}", xaxis=dict(title=x_axis_label), yaxis=dict(title='Sleep stage')) # Create a layout
    fig = go.Figure(data=[trace1, trace2], layout=layout) # Create a Figure and add the traces
    fig.update_layout(yaxis=dict(tickmode='array', tickvals=np.arange(0, sleep_map.get_num_stages()+1, step=1), ticktext=sleep_map.get_name_map()))
    fig.write_html(log_file_path.replace(".ext", ".html")) # Save the figure as an HTML file

def export_training_val_plot(out_fp:str, fit_history):
    plt.figure(figsize=(10, 6), dpi=300) # Width, height in inches
    plt.title(f"Validation set accuracy and loss during training")
    plt.plot(fit_history.history['val_accuracy'], label='Validation accuracy')
    plt.plot(fit_history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    max_y1_tick = utilities.round_up_to_nearest_tenth(max(fit_history.history["val_accuracy"]))
    max_y2_tick = utilities.round_up_to_nearest_tenth(max(fit_history.history["val_loss"]))
    plt.yticks(np.arange(0,max([max_y1_tick+0.1, max_y2_tick+0.1, 1+0.1]), step=0.1))
    plt.savefig(f"{out_fp}/models/train_val_accuracy.png")

def manual_val(model, args, type:str, data:dict, whole_night_indices:list, out_fp:str, ds_metadata:dict):
    total_correct = 0
    accuracy = 0
    pred_cnt = [0 for _ in range(sleep_map.get_num_stages()+1)]
    filter_post_argmax = (args.out_filter_type == 'post_argmax')

    print(f"[{(time.time()-start_time):.2f}s] Starting manual validation for model type '{type}'.")

    # Run model
    try:
        if (type == "tf"):
            sleep_stages_pred, ground_truth = utilities.run_model(model, data=data, whole_night_indices=whole_night_indices, data_type=DATA_TYPE, num_output_filtering=args.num_out_filter,
                                                    filter_post_argmax=filter_post_argmax, self_reset_threshold=args.filter_self_reset_threshold, num_sleep_stage_history=NUM_SLEEP_STAGE_HISTORY)
        elif (type == "tflite") or (type == "tflite (quant)") or (type == "tflite (16bx8b full quant)"):
            sleep_stages_pred = utilities.run_tflite_model(model, data=data, whole_night_indices=whole_night_indices, data_type=DATA_TYPE, num_output_filtering=args.num_out_filter,
                                                           filter_post_argmax=filter_post_argmax, self_reset_threshold=args.filter_self_reset_threshold, num_sleep_stage_history=NUM_SLEEP_STAGE_HISTORY)
        elif type == "tflite (full quant)":
            sleep_stages_pred = utilities.run_tflite_model(model, data=data, whole_night_indices=whole_night_indices, data_type=tf.uint8, num_output_filtering=args.num_out_filter,
                                                           filter_post_argmax=filter_post_argmax, self_reset_threshold=args.filter_self_reset_threshold, num_sleep_stage_history=NUM_SLEEP_STAGE_HISTORY)

        # Check accuracy
        for i in range(len(sleep_stages_pred)):
            total_correct += (sleep_stages_pred[i] == data["sleep_stages_val"][i][0].numpy())
            pred_cnt[sleep_stages_pred[i]] += 1
        accuracy = round(total_correct/len(sleep_stages_pred), ndigits=4)

        # Export plots
        export_plots(pred=sleep_stages_pred, ground_truth=data["sleep_stages_val"].numpy(), accuracy=accuracy, log_file_path=f"{out_fp}/models/{type}_man_val.ext", ds_metadata=ds_metadata)
    except Exception as e: print(f"Failed to manually validate model type '{type}'. Exception: {e} Moving on.")

    print(f"[{(time.time()-start_time):.2f}s] Done manually validating model type '{type}'.")
    return accuracy, pred_cnt

def representative_dataset():
    for data in train_signals_representative_dataset[0:500]:
        yield {"input_1": data}

def increment_ops(ops_dict:dict, in_shape, out_shape, layer_type:str, activation:bool=False):
    """
    Updates operations count dictionary for different types of layers.
    For MatrixMultiply, enter the leftmost matrix as input (i.e. in Y = AX, in_shape should come from A)
    """

    y_in, x_in, z_in = in_shape
    y_out, x_out, z_out = out_shape

    if layer_type == "Dense":
        ops_dict["mults"] += x_in * y_in * x_out
        ops_dict["adds"] += x_in * y_in * (x_out-1) + x_out * y_in
        ops_dict["incrs"] += x_in * y_in * (x_out-1) + x_out
        if activation: ops_dict["acts"] += x_out * y_in

    elif layer_type == "LayerNorm":
        ops_dict["adds"] += y_in * (3*x_in - 1)
        ops_dict["divs"] += y_in * (x_in+2)
        ops_dict["subs"] += x_in * y_in
        ops_dict["mults"] += 2 * x_in * y_in
        ops_dict["sqrts"] += y_in
        ops_dict["incrs"] += 3 * y_in * (x_in-1) + 3 * (y_in-1)

    elif layer_type == "MatrixMultiply":
        if (z_in == 0): # 2D matrix
            ops_dict["mults"] += x_in * y_in * x_out
            ops_dict["adds"] += x_in * (y_in - 1) * y_out
            ops_dict["incrs"] += x_in * y_in * x_out
        else: # 3D matrix
            ops_dict["mults"] += x_in * x_in * y_in * z_in
            ops_dict["adds"] += x_in * x_in * y_in * (z_in - 1)
            ops_dict["incrs"] += x_in * x_in * y_in * z_in

    elif layer_type == "nnSoftmax":
        ops_dict["divs"] += x_in * y_in
        ops_dict["exps"] += x_in * y_in
        ops_dict["adds"] += y_in * (x_in - 1)
        ops_dict["incrs"] += 2 * y_in * (x_in - 1)

def extract_stats_accuracies(rows:dict):
    """"
    Returns a tuple of dictionaries containing the average and median accuracies for k-fold validation
    """
    acc = {}
    for key in rows[0].keys():
        if key != 'k-fold set': acc.update({key:[]}) # Reset accuracies

    for row in rows:
        if row[key] == '': break # We've hit the blank row, so stop
        for key in acc.keys(): acc[key].append(float(row[key]))

    avg = {}
    med = {}
    for key, value in acc.items():
        acc = np.array(value)
        avg.update({key:np.average(acc)})
        med.update({key:np.median(acc)})

    avg.update({'k-fold set': 'Average'})
    med.update({'k-fold set': 'Median'})

    return avg, med

def export_k_fold_results(args, acc:dict):
    fp = args.k_fold_val_results_fp + ".csv"
    rows = []
    data_to_write = {'k-fold set':args.k_fold_val_set}
    for model_type in acc.keys(): acc[model_type] = max(acc[model_type]) # Keep max. accuracy
    data_to_write.update(acc)
    blank_row = dict.fromkeys(data_to_write.keys())

    # If file doesn't exist, create it
    if not os.path.isfile(fp):
        with open(fp, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data_to_write.keys())
            writer.writeheader()
            writer.writerow(data_to_write)
            writer.writerow(blank_row)
            acc.update({'k-fold set': 'Average'}) # Sole k-fold set in results file, so average is itself
            writer.writerow(acc)
            acc['k-fold set'] = 'Median' # Same for median
            writer.writerow(acc)
        return

    # File already exists. Keep trying to open it if it's already opened by another process.
    while True:
        try:
            with open(fp, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                rows = list(reader)

                # Check if row already exists, and update it if so
                row_found = False
                for row in rows:
                    if row['k-fold set'] == str(args.k_fold_val_set): # Current k-fold set is already in CSV -> Just update accuracies on that line
                        row.update(data_to_write)
                        row_found = True
                        break
                if not row_found: rows.insert(0, data_to_write)
                break
        except PermissionError: time.sleep(1)

    with open(fp, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_to_write.keys())
        writer.writeheader()

        for row in rows:
            if row["k-fold set"] == 'Average': break # We've hit the beginning of the stats rows, stop writting
            writer.writerow(row)

        # Writing additional blank row and average accuracies row
        avg_row, med_row = extract_stats_accuracies(rows) # Function to calculate average accuracies
        writer.writerow(avg_row)
        writer.writerow(med_row)

def save_models(model, all_models:dict, out_fp:str):
    model.save(f"{out_fp}/models/model.tf", save_format="tf")
    model.save_weights(filepath=f"{out_fp}/models/model_weights.h5")
    print(f"[{(time.time()-start_time):.2f}s] Saved model to {out_fp}/models/model.tf")

    if "cedar.computecanada.ca" not in socket.gethostname(): # Only export model if not running on Cedar (aka running TF 2.8) since it doesn't support it
        tf.keras.utils.plot_model(model.build_graph(), to_file=f"{out_fp}/model_architecture.png", expand_nested=True, show_trainable=True, show_shapes=True, show_layer_activations=True, dpi=300, show_dtype=True)

    # Convert to Tensorflow Lite model and save
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    all_models["tflite"] = f"{out_fp}/models/model.tflite"
    with open(f"{out_fp}/models/model.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"[{(time.time()-start_time):.2f}s] Saved TensorFlow Lite model to {out_fp}/models/model.tflite.")

    # Convert to quantized Tensorflow Lite model and save
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    all_models["tflite (quant)"] = f"{out_fp}/models/model_quant.tflite"
    with open(f"{out_fp}/models/model_quant.tflite", "wb") as f:
        f.write(tflite_quant_model)
    print(f"[{(time.time()-start_time):.2f}s] Saved quantized TensorFlow Lite model to {out_fp}/models/model_quant.tflite.")

    # Convert to full quantized Tensorflow Lite model and save
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_full_quant_model = converter.convert()
    all_models["tflite (full quant)"] = f"{out_fp}/models/model_full_quant.tflite"
    with open(f"{out_fp}/models/model_full_quant.tflite", "wb") as f:
        f.write(tflite_full_quant_model)
    print(f"[{(time.time()-start_time):.2f}s] Saved fully quantized TensorFlow Lite model to {out_fp}/models/model_full_quant.tflite.")

    # Convert to full quantized Tensorflow Lite model with 16b activations and 8b weights and save
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    tflite_16x8_full_quant_model = converter.convert()
    all_models["tflite (16bx8b full quant)"] = f"{out_fp}/models/model_full_quant_16bx8b.tflite"
    with open(f"{out_fp}/models/model_full_quant_16bx8b.tflite", "wb") as f:
        f.write(tflite_16x8_full_quant_model)
    print(f"[{(time.time()-start_time):.2f}s] Saved fully quantized TensorFlow Lite model (16b activations and 8b weights) to {out_fp}/models/model_full_quant_16bx8b.tflite.")

#--- APTx activation ---#
def aptx(x):
    return (1 + tf.keras.activations.tanh(0.5*x)) * (0.5*x)
tf.keras.utils.get_custom_objects().update({'aptx': aptx})

#--- Multi-Head Attention ---#
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, args, embedding_depth:int, num_heads:int, **kwargs):
        super(MultiHeadSelfAttention, self).__init__( **kwargs)

        # Hyperparameters
        self.embedding_depth = embedding_depth
        self.num_heads = num_heads
        self.projection_dimension = int(self.embedding_depth // self.num_heads)
        self.clip_length_num_samples = CLIP_LENGTH_NUM_SAMPLES
        self.patch_length = args.patch_length
        self.num_patches = int(self.clip_length_num_samples / self.patch_length)
        self.use_class_embedding = args.use_class_embedding

        if self.embedding_depth % num_heads != 0:
            raise ValueError(f"Embedding dimension = {self.embedding_depth} should be divisible by number of heads = {self.num_heads}")

        # Layers
        self.query_dense = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_query_dense")
        self.key_dense = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_key_dense")
        self.value_dense = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_value_dense")
        self.combine_heads = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_combine_head_dense")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True) #score = (batch_size, num_heads, num_patches+1, num_patches+1)
        dim_key = tf.cast(tf.shape(key)[-1], dtype=DATA_TYPE)
        scaled_score = score / tf.math.sqrt(dim_key) #scaled_score = (batch_size, num_heads, num_patches+1, num_patches+1)
        weights = tf.nn.softmax(logits=scaled_score, axis=-1) #weights = (batch_size, num_heads, num_patches+1, num_patches+1)
        output = tf.matmul(weights, value) #output = (batch_size, num_heads, num_patches+1, embedding_depth/num_heads)

        if OUTPUT_CSV:
            for i in range(self.num_heads):
                np.savetxt(f"python_prototype/reference_data/enc_scaled_score_{i}.csv", scaled_score[0][i], delimiter=",")
                np.savetxt(f"python_prototype/reference_data/enc_softmax_{i}.csv", weights[0][i], delimiter=",")

            output_list = tf.unstack(output[0], axis=0)
            output_stack = tf.concat(output_list, axis=1)
            np.savetxt(f"python_prototype/reference_data/enc_softmax_mult_V.csv", output_stack, delimiter=",")

        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dimension))
        return tf.transpose(x, perm=[0,2,1,3])

    @tf.function(autograph=not DEBUG)
    def call(self, inputs):
        batch_size = inputs.shape[0]

        query = self.query_dense(inputs) #query = (batch_size, num_patches+1, embedding_depth)
        key = self.key_dense(inputs) #key = (batch_size, num_patches+1, embedding_depth)
        value = self.value_dense(inputs) #value = (batch_size, num_patches+1, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_Q_dense.csv", query[0], delimiter=",")
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_K_dense.csv", key[0], delimiter=",")
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_V_dense.csv", value[0], delimiter=",")

        query = self.separate_heads(query, batch_size) #query = (batch_size, num_heads, num_patches+1, embedding_depth/num_heads)
        key = self.separate_heads(key, batch_size) #key = (batch_size, num_heads, num_patches+1, embedding_depth/num_heads)
        value = self.separate_heads(value, batch_size) #value = (batch_size, num_heads, num_patches+1, embedding_depth/num_heads)

        attention, weights = self.attention(query, key, value) #attention = (batch_size, num_heads, num_patches+1, embedding_depth/num_heads)
        attention = tf.transpose(attention, perm=[0,2,1,3]) #attention = (batch_size, num_patches+1, num_heads, embedding_depth/num_heads)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embedding_depth)) #concat_attention = (batch_size, num_patches+1, embedding_depth)
        output = self.combine_heads(concat_attention) #output = (batch_size, num_patches+1, embedding_depth)

        return output

    def calculate_ops(self):
        """
        Return a dict of operation counts. Takes into account matrix operations and indexing into the matrices. Doesn't discriminate between float and fixed-point.
        Ignores historical lookback.
        """
        ops = {"adds":0, "subs":0, "mults":0, "divs":0, "acts":0, "incrs":0, "exps":0, "sqrts":0}

        y_in, x_in = self.num_patches + self.use_class_embedding, self.embedding_depth
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_in,x_in,0), layer_type="Dense", activation=False) # Query Dense
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_in,x_in,0), layer_type="Dense", activation=False) # Key Dense
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_in,x_in,0), layer_type="Dense", activation=False) # Value Dense

        y_in, x_in, z_in = self.embedding_depth/self.num_heads, self.num_patches + self.use_class_embedding, self.num_heads
        y_out, x_out, z_out = self.embedding_depth/self.num_heads, self.num_patches + self.use_class_embedding, self.num_patches + self.use_class_embedding
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,z_in), out_shape=(y_out,x_out,z_out), layer_type="MatrixMultiply") # Matrix multiply between Query and Value
        # Divide by sqrt(# heads)
        ops["sqrts"] += 1
        ops["divs"] += x_out * y_out * z_out
        ops["incrs"] += (x_out - 1) * y_out * z_out
        increment_ops(ops_dict=ops, in_shape=(y_out,x_out,z_out), out_shape=(y_out,x_out,z_out), layer_type="nnSoftmax") # Softmax
        increment_ops(ops_dict=ops, in_shape=(y_out,x_out,z_out), out_shape=(y_out,x_out,self.num_heads), layer_type="MatrixMultiply") # Matrix multiply with values
        increment_ops(ops_dict=ops, in_shape=(x_in,self.embedding_depth,0), out_shape=(x_in,self.embedding_depth,0), layer_type="Dense", activation=False) # Output Dense

        return ops

    def get_config(self):
        config = super().get_config()
        return config

#--- Encoder ---#
class Encoder(tf.keras.layers.Layer):
    def __init__(self, args, mlp_dense_activation, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # Hyperparameters
        self.args = args
        self.embedding_depth = args.embedding_depth
        self.num_heads = args.num_heads
        self.mlp_dim = args.mlp_dim
        self.dropout_rate_percent = args.dropout_rate_percent
        self.history_length = NUM_SLEEP_STAGE_HISTORY
        self.mlp_dense_activation = mlp_dense_activation
        self.clip_length_num_samples = CLIP_LENGTH_NUM_SAMPLES
        self.patch_length = args.patch_length
        self.num_patches = int(self.clip_length_num_samples / self.patch_length)
        self.use_class_embedding = args.use_class_embedding

        # Layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layerNorm1_encoder")
        self.mhsa = MultiHeadSelfAttention(self.args, self.embedding_depth, self.num_heads)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate_percent, seed=RANDOM_SEED, name="dropout1_encoder")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layerNorm2_encoder")

        self.mlp_dense1 = tf.keras.layers.Dense(self.mlp_dim, activation=self.mlp_dense_activation, name="mlp_dense1_encoder")
        self.mlp_dropout1 = tf.keras.layers.Dropout(self.dropout_rate_percent, seed=RANDOM_SEED, name="mlp_dropout1_encoder")
        self.mlp_dense2 = tf.keras.layers.Dense(self.embedding_depth, name="mlp_dense2_encoder")
        self.mlp_dropout2 = tf.keras.layers.Dropout(self.dropout_rate_percent, seed=RANDOM_SEED, name="mlp_dropout2_encoder")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate_percent, seed=RANDOM_SEED, name="dropout2_encoder")

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs) #inputs_norm = (batch_size, num_patches+1, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_layernorm1.csv", inputs_norm[0], delimiter=",")
        attn_output = self.mhsa(inputs_norm) #attn_output = (batch_size, num_patches+1, embedding_depth)
        attn_output = self.dropout1(attn_output, training=training) #attn_output = (batch_size, num_patches+1, embedding_depth)

        out1 = attn_output + inputs #out1 = (batch_size, num_patches+1, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_res_sum_1.csv", out1[0], delimiter=",")
        out1_norm = self.layernorm2(out1) #out1_norm = (batch_size, num_patches+1, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_layernorm2.csv", out1_norm[0], delimiter=",")

        mlp = self.mlp_dense1(out1_norm) #mlp = (batch_size, num_patches+1, mlp_dim)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_mlp_dense1.csv", mlp[0], delimiter=",")
        mlp = self.mlp_dropout1(mlp, training=training) #mlp = (batch_size, num_patches+1, mlp_dim)
        mlp_output = self.mlp_dense2(mlp) #mlp_output = (batch_size, num_patches+1, embedding_depth)
        mlp_output = self.mlp_dropout2(mlp_output, training=training) #mlp_output = (batch_size, num_patches+1, embedding_depth)

        mlp_output = self.dropout2(mlp_output, training=training) #mlp_output = (batch_size, num_patches+1, embedding_depth)
        # enc_out = mlp_output + out1 # TODO: Should this not be inputs instead of out1?
        enc_out = mlp_output + inputs # TODO: Should this not be inputs instead of out1?
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/enc_output.csv", enc_out[0], delimiter=",")
        return enc_out

    def calculate_ops(self):
        """
        Return a dict of operation counts. Takes into account matrix operations and indexing into the matrices. Doesn't discriminate between float and fixed-point.
        Ignores historical lookback.
        """
        ops = {"adds":0, "subs":0, "mults":0, "divs":0, "acts":0, "incrs":0, "exps":0, "sqrts":0}

        y_in, x_in = self.num_patches + self.use_class_embedding, self.embedding_depth

        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_in,x_in,0), layer_type="LayerNorm") # LayerNorm
        for op, num in self.mhsa.calculate_ops().items(): ops[op] += num # MHSA
        ops["adds"] += y_in * x_in # Add inputs
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_in,x_in,0), layer_type="LayerNorm") # LayerNorm

        y_out, x_out = self.num_patches + self.use_class_embedding, self.mlp_dim
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_out,x_out,0), layer_type="Dense", activation=True) # MLP Dense 1
        increment_ops(ops_dict=ops, in_shape=(y_out,x_out,0), out_shape=(y_in,x_in,0), layer_type="Dense", activation=True) # MLP Dense 2
        ops["adds"] += y_in * x_in # Add

        return ops

    def get_config(self):
        config = super().get_config()
        config.update({"layers":[
            self.layernorm1,
            self.mhsa,
            self.dropout1,
            self.layernorm2,
            # self.mlp,
            self.dropout2,]})
        return config

#--- Vision Transformer ---#
class VisionTransformer(tf.keras.Model):
    def __init__(self, args, mlp_dense_activation):
        super(VisionTransformer, self).__init__()

        # Hyperparameters
        self.clip_length_num_samples = CLIP_LENGTH_NUM_SAMPLES
        self.patch_length = int(args.patch_length)
        self.num_encoder_layers = args.num_layers
        self.num_classes = sleep_map.get_num_stages()+1
        self.embedding_depth = args.embedding_depth
        self.num_heads = args.num_heads
        self.mlp_dim = args.mlp_dim
        self.dropout_rate_percent = args.dropout_rate_percent
        self.num_patches = int(self.clip_length_num_samples / self.patch_length)
        self.history_length = NUM_SLEEP_STAGE_HISTORY
        self.historical_lookback_DNN = False and (True if self.history_length > 0 else False)
        self.historical_lookback_DNN_depth = args.historical_lookback_DNN_depth
        self.enable_scaling = args.enable_input_rescale
        self.enable_positional_embedding = args.enable_positional_embedding
        self.use_class_embedding = args.use_class_embedding
        self.mlp_dense_activation = mlp_dense_activation
        self.mlp_head_num_dense = args.mlp_head_num_dense
        self.num_out_filter = args.num_out_filter

        # Layers
        self.patch_projection = tf.keras.layers.Dense(self.embedding_depth, name="patch_projection_dense")
        if self.enable_scaling: self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / MAX_VOLTAGE)
        if self.use_class_embedding: self.class_embedding = self.add_weight("class_emb", shape=(1, 1, self.embedding_depth))
        if self.enable_positional_embedding: self.positional_embedding = self.add_weight("pos_emb", shape=(1, self.num_patches+self.use_class_embedding+(self.history_length > 0), self.embedding_depth)) #+1 for the trainable classification token, +1 for historical lookback
        self.encoder_layers = [Encoder(args, mlp_dense_activation=self.mlp_dense_activation, name=f"Encoder_{i+1}") for i in range(self.num_encoder_layers)]
        self.mlp_head_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_head_layerNorm")
        self.mlp_head = []
        for i in range(self.mlp_head_num_dense):
            self.mlp_head.append(tf.keras.layers.Dense(self.mlp_dim, activation=self.mlp_dense_activation, name=f"mlp_head_dense{i+1}"))
            self.mlp_head.append(tf.keras.layers.Dropout(self.dropout_rate_percent, seed=RANDOM_SEED, name=f"mlp_head_dropout{i+1}"))
        self.mlp_head = tf.keras.Sequential(self.mlp_head, name="mlp_head")
        self.mlp_head_softmax = tf.keras.layers.Dense(self.num_classes, activation="softmax", name="mlp_head_softmax")

    def extract_patches(self, batch_size:int, clips, training:bool=False):
        patches = tf.reshape(clips, [batch_size, -1, self.patch_length])
        return patches

    def call(self, input, training:bool=False):
        # Extract historical lookback (if present)
        if self.history_length > 0:
            clip, historical_lookback = tf.split(input, [self.clip_length_num_samples, self.history_length], axis=1)
            historical_lookback_padding = tf.zeros(shape=(clip.shape[0], self.patch_length - self.history_length), dtype=DATA_TYPE)
            historical_lookback_patch = tf.concat([historical_lookback, historical_lookback_padding], axis=1)
            historical_lookback_patch = tf.expand_dims(historical_lookback_patch, axis=1)
        else:
            clip = input
            historical_lookback = None

        batch_size = clip.shape[0]
        if batch_size == None: batch_size = 1

        # Normalize to [0, 1]
        if self.enable_scaling: clip = self.rescale(clip)

        # Extract patches
        patches = self.extract_patches(batch_size, clip) #patches = (batch_size, num_patches, patch_length)
        if self.history_length > 0:
            patches = tf.concat([patches, historical_lookback_patch], axis=1)

        # Linear projection
        clip = self.patch_projection(patches) #clip = (batch_size, num_patches, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/patch_proj.csv", clip[0], delimiter=",")

        # Classification token
        if self.use_class_embedding:
            class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.embedding_depth]) #class_embedding = (batch_size, 1, embedding_depth)
            clip = tf.concat([class_embedding, clip], axis=1) #clip = (batch_size, num_patches+1, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/class_emb.csv", clip[0], delimiter=",")

        if self.enable_positional_embedding: clip = clip + self.positional_embedding #clip = (batch_size, num_patches+1, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/pos_emb.csv", clip[0], delimiter=",")

        # Go through encoder
        for layer in self.encoder_layers:
            clip = layer(inputs=clip, training=training) #clip = (batch_size, num_patches+1, embedding_depth)

        # Classify with first token
        mlp_head_layernorm = self.mlp_head_layernorm(clip[:, 0]) #mlp_head_layernorm = (batch_size, embedding_depth)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/mlp_head_layernorm.csv", mlp_head_layernorm[0], delimiter=",")
        mlp_head = self.mlp_head(mlp_head_layernorm) # Select the first row of each batch's encoder output. prediction = (batch_size, sleep_map.get_num_stages()+1)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/mlp_head_out.csv", mlp_head[0], delimiter=",")
        prediction = self.mlp_head_softmax(mlp_head) #prediction = (batch_size, sleep_map.get_num_stages()+1)
        if OUTPUT_CSV: np.savetxt("python_prototype/reference_data/mlp_head_softmax.csv", prediction[0], delimiter=",")

        if self.historical_lookback_DNN:
            historical_lookback = tf.cast(historical_lookback, tf.uint8)
            historical_lookback = tf.one_hot(historical_lookback, depth=self.num_classes)
            prediction = tf.expand_dims(prediction, axis=1)
            concatenated = tf.concat([prediction, historical_lookback], axis=1)
            concatenated = tf.reshape(concatenated, [batch_size, -1])
            prediction = self.historical_lookback(concatenated)

        # Numpy array with dummy softmax to validate averaging unit in ASIC functional simulation
        if OUTPUT_CSV:
            for i in range(self.num_out_filter):
                dummy_softmax = np.random.rand(1,sleep_map.get_num_stages()+1)
                np.savetxt(f"python_prototype/reference_data/dummy_softmax_{i}.csv", dummy_softmax, delimiter=",")
        return prediction

    def build_graph(self):
        x = tf.keras.Input(shape=(1,self.clip_length_num_samples))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))

    def get_config(self):
        config = {
            'name': 'VisionTransformer',
            'clip_length_num_samples': self.clip_length_num_samples,
            'patch_length': self.patch_length,
            'num_encoder_layers': self.num_encoder_layers,
            'num_classes': self.num_classes,
            'embedding_depth': self.embedding_depth,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'dropout_rate_percent': self.dropout_rate_percent,
            'num_patches': self.num_patches,
            'history_length': self.history_length,
            'historical_lookback_DNN': self.historical_lookback_DNN,
            'historical_lookback_DNN_depth': self.historical_lookback_DNN_depth,
            'enable_scaling': self.enable_scaling,
            'enable_positional_embedding': self.enable_positional_embedding,
            'use_class_embedding': self.use_class_embedding,
            'mlp_dense_activation': self.mlp_dense_activation
        }
        return config

    def calculate_ops(self):
        """
        Return a dict (# adds, # mults, # divs, # acts, # incrs). Takes into account matrix operations and indexing into the matrices. Doesn't discriminate between float and fixed-point.
        Ignores historical lookback.
        """
        ops = {"adds":0, "subs":0, "mults":0, "divs":0, "acts":0, "incrs":0, "exps":0, "sqrts":0}

        # Rescale
        if self.enable_scaling:
            ops["divs"] += self.clip_length_num_samples
            ops["incrs"] += self.clip_length_num_samples

        # Patch embeddings
        y_in, x_in = self.num_patches, self.patch_length
        y_out, x_out = self.num_patches, self.embedding_depth
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_out,x_out,0), layer_type="Dense", activation=False)

        # Positional embedding
        if self.enable_positional_embedding:
            ops["adds"] += (self.num_patches + self.use_class_embedding) * self.embedding_depth

        # Encoder layers
        for op, num in self.encoder_layers[0].calculate_ops().items():
            ops[op] += self.num_encoder_layers * num

        # MLP head
        y_in, x_in = self.num_patches+self.use_class_embedding, self.embedding_depth
        increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_in,x_in,0), layer_type="LayerNorm")
        for _ in range(self.mlp_head_num_dense):
            increment_ops(ops_dict=ops, in_shape=(y_in,x_in,0), out_shape=(y_in,self.mlp_dim,0), layer_type="Dense", activation=True)
        increment_ops(ops_dict=ops, in_shape=(y_in,self.mlp_dim,0), out_shape=(1, sleep_map.get_num_stages()+1,0), layer_type="Dense", activation=True)

        for op in ops.keys(): ops[op] = int(ops[op])
        return ops

#--- Misc ---#
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_depth, warmup_steps_exponent=-1.5, warmup_steps=4000):
        super().__init__()
        self.embedding_depth = tf.cast(embedding_depth, tf.float32)
        self.warmup_steps_exponent = warmup_steps_exponent
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** self.warmup_steps_exponent)

        return tf.math.rsqrt(self.embedding_depth) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'embedding_depth': int(self.embedding_depth),
            'warmup_steps': int(self.warmup_steps),
            'warmup_steps_exponent': self.warmup_steps_exponent,
        }
        return config

def main():
    # Parse arguments
    args = parse_arguments()
    print(f"[{(time.time()-start_time):.2f}s] Arguments parsed; starting dataset load.\n")

    # Load data
    try: data, start_of_val_night_indices, original_sleep_stage_cnt, dataset_metadata = load_from_dataset(args=args)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to load data from dataset.")
    global train_signals_representative_dataset
    train_signals_representative_dataset = data["signals_train"]

    print(f"[{(time.time()-start_time):.2f}s] Dataset ready.")
    mlp_dense_act = "swish"
    time_of_export = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_out_fp = utilities.find_folder_path(utilities.folder_base_path()+f"results/{time_of_export}_vision", utilities.folder_base_path()+"results")
    os.makedirs(base_out_fp, exist_ok=True)

    acc = {"tf":[], "tflite":[], "tflite (quant)":[], "tflite (full quant)":[], "tflite (16bx8b full quant)":[]}
    pred_cnt = {"tf":[], "tflite":[], "tflite (quant)":[], "tflite (full quant)":[], "tflite (16bx8b full quant)":[]}

    for run_num in range(args.num_runs):
        print(f"[{(time.time()-start_time):.2f}s] Starting training with {int(data['signals_train'].shape[0])} clips for run {run_num+1}.")
        run_out_fp = base_out_fp+f"/run_{run_num+1}"

        # Train or load model
        if args.load_model_filepath == None:
            model, fit_history = train_model(args, data, mlp_dense_act)
        else: model = tf.keras.models.load_model(args.load_model_filepath, custom_objects={"CustomSchedule": CustomSchedule})

        # Make results directory. Check whether folder with the same name already exist and append counter if necessary
        os.makedirs(run_out_fp, exist_ok=True)
        os.makedirs(run_out_fp+"/models", exist_ok=True)

        # Save models to disk
        all_models = {"tf":model, "tflite":-1, "tflite (quant)":-1, "tflite (full quant)":-1, "tflite (16bx8b full quant)":-1}
        if args.save_model: save_models(model=model, all_models=all_models, out_fp=run_out_fp)

        # Manual validation
        for model_type, model in all_models.items():
            current_acc, current_pred_cnt = manual_val(model, args=args, type=model_type, data=data, whole_night_indices=start_of_val_night_indices, out_fp=run_out_fp, ds_metadata=dataset_metadata)
            acc[model_type].append(current_acc)
            pred_cnt[model_type].append(current_pred_cnt)

        # Count sleep stages in training and validation datasets
        sleep_stages_cnt_train = utilities.count_instances_per_class(data['sleep_stages_train'], sleep_map.get_num_stages()+1)
        sleep_stages_cnt_val = utilities.count_instances_per_class(data['sleep_stages_val'], sleep_map.get_num_stages()+1)

        # Save accuracy and model details to log file
        export_summary(run_out_fp+"/info.txt", args, all_models["tf"], fit_history, acc, original_sleep_stage_cnt, sleep_stages_cnt_train, sleep_stages_cnt_val, pred_cnt, mlp_dense_act, dataset_metadata)
        print(f"[{(time.time()-start_time):.2f}s] Saved model summary to {run_out_fp}/info.txt.")

        # Plot validation accuracy and loss during training
        export_training_val_plot(run_out_fp, fit_history=fit_history)

    # Write to k-fold validation file
    if K_FOLD_OUTPUT_TO_FILE:
        export_summary(args.k_fold_val_results_fp+".txt", args, all_models["tf"], fit_history, acc, original_sleep_stage_cnt, sleep_stages_cnt_train,
                       sleep_stages_cnt_val, pred_cnt, mlp_dense_act, dataset_metadata, model_specific_only=True)
        export_k_fold_results(args, acc)
        print(f"[{(time.time()-start_time):.2f}s] Wrote to k-fold results file.")

    # Run reference night to help validate C++ functional model (make sure the parameters match the dataset)
    if (args.reference_night_fp != ""):
        global DEBUG
        global OUTPUT_CSV

        DEBUG = True
        OUTPUT_CSV = True
        tf.config.run_functions_eagerly(True) # Allows step-by-step debugging of tf.functions
        ref_data = utilities.edf_to_h5(edf_fp=args.reference_night_fp, channel="EEG Cz-LER", clip_length_s=30, sampling_freq_hz=128, full_night=True, h5_filename="python_prototype/reference_data/eeg.h5")
        all_models["tf"](ref_data[0], training=False) # Run with the first clip
        shutil.copy2(base_out_fp+"/run_1/models/model.tflite", "python_prototype/reference_data/model.tflite")
        shutil.copy2(base_out_fp+"/run_1/models/model_quant.tflite", "python_prototype/reference_data/model_quant.tflite")
        shutil.copy2(base_out_fp+"/run_1/models/model_weights.h5", "python_prototype/reference_data/model_weights.h5")
        shutil.rmtree("python_prototype/reference_data/model.tf", ignore_errors=True)
        shutil.copytree(base_out_fp+"/run_1/models/model.tf", "python_prototype/reference_data/model.tf")

    print(f"[{(time.time()-start_time):.2f}s] Done. Good bye.")

if __name__ == "__main__":
    main()