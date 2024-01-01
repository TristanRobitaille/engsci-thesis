import time
start_time = time.time()

import os
import io
import git
import sys
import json
# import math
import sklearn
import datetime
import imblearn
import utilities

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

MAX_VOLTAGE = 2**16-1 #Maximum ADC code output
DEFAULT_CLIP_LENGTH_S = int(30)
SAMPLING_FREQUENCY_HZ = int(256)
USE_SLEEP_STAGE_HISTORY = False
NUM_SLEEP_STAGE_HISTORY = -1
NUM_OUTPUT_FILTERING = 0
DATA_TYPE = tf.float32
TEST_SET_RATIO = 0.04 #Percentage of training data reserved for validation
RANDOM_SEED = 42
NUM_WARMUP_STEPS = 4000
VERBOSITY = 'QUIET' #'QUIET', 'NORMAL', 'DETAILED'
RESAMPLE_TRAINING_DATASET = False
WHOLE_NIGHT_VALIDATION = True #Whether to validate individual nights (sequentially) or a random subset of clips (if we use the prediction history, we need whole nights for validation)
SHUFFLE_TRAINING_CLIPS = True
NUM_CLIPS_PER_FILE_EDGETPU = 500 # Only valid for 256Hz

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

def split_whole_night_validation_set(signals, sleep_stages, whole_night_markers):
    """
    Split the dataset into whole nights for validation
    """

    split_index = int((1-TEST_SET_RATIO) * len(sleep_stages))
    whole_night_indices = [i for i, x in enumerate(whole_night_markers.numpy()) if x == True]
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

    global NUM_SLEEP_STAGE_HISTORY
    global WHOLE_NIGHT_VALIDATION
    global SAMPLING_FREQUENCY_HZ
    global CLIP_LENGTH_NUM_SAMPLES

    # Extract from dataset metadata
    with open(args.input_dataset + ".json", 'r') as json_file:
        dataset_metadata = json.load(json_file)
        print(f"Dataset metadata {json.dumps(dataset_metadata, indent=4)}\n\n")

    SAMPLING_FREQUENCY_HZ = dataset_metadata["sampling_freq_Hz"]
    NUM_SLEEP_STAGE_HISTORY = dataset_metadata["historical_lookback_length"]
    WHOLE_NIGHT_VALIDATION = WHOLE_NIGHT_VALIDATION or (NUM_SLEEP_STAGE_HISTORY > 0)
    CLIP_LENGTH_NUM_SAMPLES = int(dataset_metadata["clip_length_s"] * SAMPLING_FREQUENCY_HZ)
    sleep_map.set_map_name(dataset_metadata["sleep_stages_map_name"])

    # Check validity of arguments
    if (dataset_metadata["clip_length_s"] % args.patch_length_s != 0):
        raise ValueError(f"Patch length ({args.patch_length_s}s) needs to be a multiple of clip length ({dataset_metadata['clip_length_s']}s! Aborting.)")
    if len(args.class_weights) != (sleep_map.get_num_stages()+1):
        raise ValueError(f"Number of class weights ({len(args.class_weights)}) should be equal to number of sleep stages ({sleep_map.get_num_stages()+1})")

    # Load dataset
    data = tf.data.Dataset.load(args.input_dataset)
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
    if WHOLE_NIGHT_VALIDATION:
        signals_train, signals_val, sleep_stages_train, sleep_stages_val = split_whole_night_validation_set(signals, sleep_stages, start_of_night_markers)
        start_of_night_markers = start_of_night_markers[len(signals_train):args.num_clips]
    else: signals_train, signals_val, sleep_stages_train, sleep_stages_val = sklearn.model_selection.train_test_split(signals, sleep_stages, test_size=TEST_SET_RATIO, random_state=RANDOM_SEED)

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

    return data_dict, start_of_night_markers, original_sleep_stage_count, dataset_metadata

def export_summary(out_fp, parser, model, fit_history, acc:dict, original_sleep_stage_count:list, sleep_stages_count_training:list,
                   sleep_stages_count_val:list, pred_cnt:dict, mlp_dense_activation, dataset_metadata:dict, note:str="") -> None:
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
        log += f"Time to complete: {(time.time()-start_time):.2f}s\n"
        log += model_summary
        log += f"\nDataset: {parser.input_dataset}\n"
        log += f"Number of operations: {model.calculate_ops()}\n"
        log += f"Save model: {parser.save_model}\n"
        log += f"File path of model to load (model trained if empty string): {parser.load_model_filepath}\n"
        log += f"Channel: {parser.input_channel}\n"
        for model_type, acc in acc.items(): log += f"Validation set accuracy ({model_type}): {acc:.4f}\n"
        log += f"Training accuracy: {[round(accuracy, 4) for accuracy in fit_history.history['accuracy']]}\n"
        log += f"Validation accuracy (while training): {[round(accuracy, 4) for accuracy in fit_history.history['val_accuracy']]}\n"
        log += f"Training loss: {[round(loss, 4) for loss in fit_history.history['loss']]}\n"
        log += f"Validation loss (while training): {[round(loss, 4) for loss in fit_history.history['val_loss']]}\n"
        log += f"Number of epochs: {parser.num_epochs}\n\n"

        log += f"Training set resampling: {RESAMPLE_TRAINING_DATASET}\n"
        log += f"Training set resampling replacement: {parser.enable_dataset_resample_replacement}\n"
        log += f"Training set shuffled: {SHUFFLE_TRAINING_CLIPS}\n"
        log += f"Training set resampler: {parser.dataset_resample_algo}\n"
        log += f"Training set target count: {parser.training_set_target_count}\n"
        log += f"Use sleep stage history: {USE_SLEEP_STAGE_HISTORY}\n"
        log += f"Number of historical sleep stages: {NUM_SLEEP_STAGE_HISTORY}\n"
        log += f"Dataset split random seed: {RANDOM_SEED}\n"
        log += f"Whole-night validation: {WHOLE_NIGHT_VALIDATION}\n"
        log += f"Dataset metadata {json.dumps(dataset_metadata, indent=4)}\n\n"

        log += f"Requested number of training clips: {int(TEST_SET_RATIO*parser.num_clips)}\n"
        if (original_sleep_stage_count != -1): log += f"Sleep stages count in original dataset ({num_clips_original_dataset}): {original_sleep_stage_count} ({[round(num / num_clips_training, 4) for num in original_sleep_stage_count]})\n"
        log += f"Sleep stages count in training data ({num_clips_training}): {sleep_stages_count_training} ({[round(num / num_clips_training, 4) for num in sleep_stages_count_training]})\n"
        log += f"Sleep stages count in validation set input ({num_clips_validation}): {sleep_stages_count_val} ({[round(num / num_clips_validation, 4) for num in sleep_stages_count_val]})\n"
        for model_type, pred_cnt in pred_cnt.items(): log += f"Sleep stages count in validation set prediction ({num_clips_validation}, {model_type}): {pred_cnt} ({[round(num / num_clips_validation, 4) for num in pred_cnt]})\n"

        log += f"\nClip length (s): {dataset_metadata['clip_length_s']}\n"
        log += f"Patch length (s): {parser.patch_length_s}\n"
        log += f"Number of sleep stages (includes unknown): {sleep_map.get_num_stages()+1}\n"
        log += f"Data type: {DATA_TYPE}\n"
        log += f"Batch size: {parser.batch_size}\n"
        log += f"Embedding depth: {parser.embedding_depth}\n"
        log += f"MHA number of heads: {parser.num_heads}\n"
        log += f"Number of layers: {parser.num_layers}\n"
        log += f"MLP dimensions: {parser.mlp_dim}\n"
        log += f"Dropout rate: {parser.dropout_rate:.3f}\n"
        log += f"Historical prediction lookback DNN depth: {parser.historical_lookback_DNN_depth}\n"
        log += f"Activation function of first dense layer in MLP layer and MLP head: {mlp_dense_activation}\n"
        log += f"Class training weights: {parser.class_weights}\n"
        log += f"Initial learning rate: {parser.learning_rate:.6f}\n"
        log += f"Rescale layer enabled: {parser.enable_input_rescale}\n"
        log += f"Use classification token: {parser.use_class_embedding}\n"
        log += f"Positional embedding enabled: {parser.enable_positional_embedding}\n"
        log += f"Number of samples in output filtering: {NUM_OUTPUT_FILTERING}\n"
        log += f"Model loss type: {model.loss.name}\n"
        log += f"Note: {note}\n"

        # Save to file
        with open(out_fp+"/info.txt", 'w') as file: file.write(log)
        print(f"[{(time.time()-start_time):.2f}s] Saved model summary to {out_fp}/info.txt.")

    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to export summary.")

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
    parser.add_argument('--patch_length_s', help='Patch length (in sec). Must be integer multiple of clip length. Defaults to 0.5s.', default=0.5, type=float)
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
    parser.add_argument('--enable_input_rescale', help='Enables layer rescaling inputs between [0, 1] at input.', action='store_true')
    parser.add_argument('--enable_positional_embedding', help='Enables positional embedding.', action='store_true')
    parser.add_argument('--dropout_rate', help='Dropout rate for all dropout layers. Defaults to 0.1.', default=0.1 , type=float)
    parser.add_argument('--save_model', help='Saves model to disk.', action='store_true')
    parser.add_argument('--load_model_filepath', help='Indicates, if not empty, the filepath of a model to load rather than training it. Defaults to None.', default=None, type=str)
    parser.add_argument('--historical_lookback_DNN_depth', help='Internal size of the output DNN for historical lookback. Defaults to 64.', default=64, type=int)
    parser.add_argument('--output_edgetpu_data', help='Select whether to output Numpy arrays when parsing input data to run on edge TPU.', action='store_true')
    parser.add_argument('--use_class_embedding', help='Select whether to use classification token in model.', action='store_true')

    # Parse arguments
    try:
        args = parser.parse_args()
    except Exception as e:
        utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to parse arguments.")

    # Print arguments received
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    return args

def train_model(args, data:dict, mlp_dense_activation:str):
    try:
        model = VisionTransformer(args, mlp_dense_activation)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to initialize model.")

    try:
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(CustomSchedule(args.embedding_depth), beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            metrics=["accuracy"],
        )
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

def manual_val(model, type:str, data:dict, whole_night_indices:dict, out_fp:str, ds_metadata:dict):
    total_correct = 0
    accuracy = 0
    pred_cnt = [0 for _ in range(sleep_map.get_num_stages()+1)]

    print(f"[{(time.time()-start_time):.2f}s] Starting manual validation for model type '{type}'.")

    # Run model
    try:
        if (type == "tf"):
            sleep_stages_pred = utilities.run_model(model, data=data, whole_night_indices=whole_night_indices, data_type=DATA_TYPE, num_output_filtering=NUM_OUTPUT_FILTERING, num_sleep_stage_history=NUM_SLEEP_STAGE_HISTORY)
        elif (type == "tflite") or (type == "tflite (quant)") or (type == "tflite (16bx8b full quant)"):
            sleep_stages_pred = utilities.run_tflite_model(model, data=data, whole_night_indices=whole_night_indices, data_type=DATA_TYPE, num_output_filtering=NUM_OUTPUT_FILTERING, num_sleep_stage_history=NUM_SLEEP_STAGE_HISTORY)
        elif type == "tflite (full quant)":
            sleep_stages_pred = utilities.run_tflite_model(model, data=data, whole_night_indices=whole_night_indices, data_type=tf.uint8, num_output_filtering=NUM_OUTPUT_FILTERING, num_sleep_stage_history=NUM_SLEEP_STAGE_HISTORY)

        # Check accuracy
        for i in range(len(sleep_stages_pred)):
            total_correct += (sleep_stages_pred[i] == data["sleep_stages_val"][i][0].numpy())
            pred_cnt[sleep_stages_pred[i]] += 1
        accuracy = total_correct/len(sleep_stages_pred)

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

    x_in, y_in, z_in = in_shape
    x_out, y_out, z_out = out_shape

    if layer_type == "Dense":
        ops_dict["mults"] += x_in**2 * y_in
        ops_dict["adds"] += x_in * y_in * (x_in-1) + x_out * y_in
        ops_dict["incrs"] += x_in * y_in * (x_in-1) + x_out * y_in
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
            ops_dict["incrs"] += x_in * y_out * y_in
        else: # 3D matrix
            ops_dict["mults"] += x_in * y_in * y_out * z_out
            ops_dict["adds"] += x_in * y_in * z_in * (y_out - 1) * z_out
            ops_dict["incrs"] += x_in * y_in * z_out * y_out

    elif layer_type == "nnSoftmax":
        ops_dict["divs"] += x_in * y_in
        ops_dict["exps"] += x_in * y_in
        ops_dict["adds"] += y_in * (x_in - 1)
        ops_dict["incrs"] += 2 * y_in * (x_in - 1)

#--- APTx activation ---#
def aptx(x):
    return (1 + tf.keras.activations.tanh(0.5*x)) * (0.5*x)
tf.keras.utils.get_custom_objects().update({'aptx': aptx})

#--- Multi-Head Attention ---#
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, args, embedding_depth:int, num_heads:int=8):
        super(MultiHeadSelfAttention, self).__init__()

        # Hyperparameters
        self.embedding_depth = embedding_depth
        self.num_heads = num_heads
        self.projection_dimension = self.embedding_depth // self.num_heads
        self.clip_length_num_samples = CLIP_LENGTH_NUM_SAMPLES
        self.patch_length_num_samples = int(args.patch_length_s * SAMPLING_FREQUENCY_HZ)
        self.num_patches = int(self.clip_length_num_samples / self.patch_length_num_samples)
        self.use_class_embedding = args.use_class_embedding

        if self.embedding_depth % num_heads != 0:
            raise ValueError(f"Embedding dimension = {self.embedding_depth} should be divisible by number of heads = {self.num_heads}")

        # Layers
        self.query_dense = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_query_dense")
        self.key_dense = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_key_dense")
        self.value_dense = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_value_dense")
        self.combine_heads = tf.keras.layers.Dense(self.embedding_depth, name="mhsa_combine_head_dense")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True) #score = (batch_size, embedding_depth/num_heads, num_patches+1, num_patches+1)
        dim_key = tf.cast(tf.shape(key)[-1], dtype=DATA_TYPE)
        assert (self.num_heads == 16) or (self.num_heads == 4), "num_heads not 4 or 16, as needed to simplify attention calculations."
        scaled_score = score / tf.math.sqrt(dim_key) #scaled_score = (batch_size, embedding_depth/num_heads, num_patches+1, num_patches+1)
        weights = tf.nn.softmax(logits=scaled_score, axis=-1) #weights = (batch_size, embedding_depth/num_heads, num_patches+1, num_patches+1)
        output = tf.matmul(weights, value) #output = (batch_size, embedding_depth/num_heads, num_patches+1, num_heads)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dimension))
        return tf.transpose(x, perm=[0,2,1,3])

    @tf.function
    def call(self, inputs):
        batch_size = inputs.shape[0]

        query = self.query_dense(inputs) #query = (batch_size, num_patches+1, embedding_depth)
        key = self.key_dense(inputs) #key = (batch_size, num_patches+1, embedding_depth)
        value = self.value_dense(inputs) #value = (batch_size, num_patches+1, embedding_depth)

        query = self.separate_heads(query, batch_size) #query = (batch_size, embedding_depth/num_heads, num_patches+1, num_heads)
        key = self.separate_heads(key, batch_size) #key = (batch_size, embedding_depth/num_heads, num_patches+1, num_heads)
        value = self.separate_heads(value, batch_size) #value = (batch_size, embedding_depth/num_heads, num_patches+1, num_heads)

        attention, weights = self.attention(query, key, value) #attention = (batch_size, embedding_depth/num_heads, num_patches+1, num_heads)
        attention = tf.transpose(attention, perm=[0,2,1,3]) #attention = (batch_size, num_patches+1, embedding_depth/num_heads, num_heads)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embedding_depth)) #concat_attention = (batch_size, num_patches+1, embedding_depth)
        output = self.combine_heads(concat_attention) #output = (batch_size, num_patches+1, embedding_depth)

        return output

    def calculate_ops(self):
        """
        Return a dict of operation counts. Takes into account matrix operations and indexing into the matrices. Doesn't discriminate between float and fixed-point.
        Ignores historical lookback.
        """
        ops = {"adds":0, "subs":0, "mults":0, "divs":0, "acts":0, "incrs":0, "exps":0, "sqrts":0}

        x_in, y_in = self.num_patches + self.use_class_embedding, self.embedding_depth
        increment_ops(ops_dict=ops, in_shape=(x_in,y_in,0), out_shape=(x_in,y_in,0), layer_type="Dense", activation=False) # Query Dense
        increment_ops(ops_dict=ops, in_shape=(x_in,y_in,0), out_shape=(x_in,y_in,0), layer_type="Dense", activation=False) # Key Dense
        increment_ops(ops_dict=ops, in_shape=(x_in,y_in,0), out_shape=(x_in,y_in,0), layer_type="Dense", activation=False) # Value Dense

        x_in, y_in, z_in = self.embedding_depth/self.num_heads, self.num_patches + self.use_class_embedding, self.num_heads
        x_out, y_out, z_out = self.embedding_depth/self.num_heads, self.num_patches + self.use_class_embedding, self.num_patches + self.use_class_embedding
        increment_ops(ops_dict=ops, in_shape=(x_in,y_in,z_in), out_shape=(x_out,y_out,z_out), layer_type="MatrixMultiply") # Matrix multiply between Query and Value
        # Divide by sqrt(# heads)
        ops["sqrts"] += 1
        ops["divs"] += x_out * y_out * z_out
        ops["incrs"] += (x_out - 1) * y_out * z_out
        increment_ops(ops_dict=ops, in_shape=(x_out,y_out,z_out), out_shape=(x_out,y_out,z_out), layer_type="nnSoftmax") # Softmax
        increment_ops(ops_dict=ops, in_shape=(x_out,y_out,z_out), out_shape=(x_in,y_in,self.num_heads), layer_type="MatrixMultiply") # Matrix multiply with values
        increment_ops(ops_dict=ops, in_shape=(y_out,self.embedding_depth,0), out_shape=(y_out,self.embedding_depth,0), layer_type="Dense", activation=False) # Output Dense

        return ops

#--- Encoder ---#
class Encoder(tf.keras.layers.Layer):
    def __init__(self, args, mlp_dense_activation):
        super(Encoder, self).__init__()

        # Hyperparameters
        self.args = args
        self.embedding_depth = args.embedding_depth
        self.num_heads = args.num_heads
        self.mlp_dim = args.mlp_dim
        self.dropout_rate = args.dropout_rate
        self.history_length = NUM_SLEEP_STAGE_HISTORY
        self.mlp_dense_activation = mlp_dense_activation
        self.clip_length_num_samples = CLIP_LENGTH_NUM_SAMPLES
        self.patch_length_num_samples = int(args.patch_length_s * SAMPLING_FREQUENCY_HZ)
        self.num_patches = int(self.clip_length_num_samples / self.patch_length_num_samples)
        self.use_class_embedding = args.use_class_embedding

        # Layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layerNorm1_encoder")
        self.mhsa = MultiHeadSelfAttention(self.args, self.embedding_depth, self.num_heads)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="dropout1_encoder")

        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layerNorm2_encoder")
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.mlp_dim, activation=self.mlp_dense_activation, name="mlp_dense1_encoder"),
            tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="mlp_dropout1_encoder"),
            tf.keras.layers.Dense(self.embedding_depth, name="mlp_dense2_encoder"),
            tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="mlp_dropout2_encoder"),
        ], name="mlp_encoder")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="dropout2_encoder")

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs) #inputs_norm = (batch_size, num_patches+1, embedding_depth)
        attn_output = self.mhsa(inputs_norm) #attn_output = (batch_size, num_patches+1, embedding_depth)
        attn_output = self.dropout1(attn_output, training=training) #attn_output = (batch_size, num_patches+1, embedding_depth)

        out1 = attn_output + inputs  #out1 = (batch_size, num_patches+1, embedding_depth)
        out1_norm = self.layernorm2(out1) #out1_norm = (batch_size, num_patches+1, embedding_depth)
        mlp_output = self.mlp(out1_norm) #mlp_output = (batch_size, num_patches+1, embedding_depth)

        mlp_output = self.dropout2(mlp_output, training=training) #mlp_output = (batch_size, num_patches+1, embedding_depth)
        return mlp_output + out1

    def calculate_ops(self):
        """
        Return a dict of operation counts. Takes into account matrix operations and indexing into the matrices. Doesn't discriminate between float and fixed-point.
        Ignores historical lookback.
        """
        ops = {"adds":0, "subs":0, "mults":0, "divs":0, "acts":0, "incrs":0, "exps":0, "sqrts":0}

        x, y = self.num_patches + self.use_class_embedding, self.embedding_depth

        increment_ops(ops_dict=ops, in_shape=(x,y,0), out_shape=(x,y,0), layer_type="LayerNorm") # LayerNorm
        for op, num in self.mhsa.calculate_ops().items(): ops[op] += num # MHSA
        ops["adds"] += x * y # Add inputs
        increment_ops(ops_dict=ops, in_shape=(x,y,0), out_shape=(x,y,0), layer_type="LayerNorm") # LayerNorm
        increment_ops(ops_dict=ops, in_shape=(x,y,0), out_shape=(x,y,0), layer_type="Dense", activation=True) # MLP Dense 1
        increment_ops(ops_dict=ops, in_shape=(x,y,0), out_shape=(x,y,0), layer_type="Dense", activation=True) # MLP Dense 2
        ops["adds"] += x * y # Add

        return ops

#--- Vision Transformer ---#
class VisionTransformer(tf.keras.Model):
    def __init__(self, args, mlp_dense_activation):
        super(VisionTransformer, self).__init__()

        # Hyperparameters
        self.clip_length_num_samples = CLIP_LENGTH_NUM_SAMPLES
        self.patch_length_num_samples = int(args.patch_length_s * SAMPLING_FREQUENCY_HZ)
        self.num_encoder_layers = args.num_layers
        self.num_classes = sleep_map.get_num_stages()+1
        self.embedding_depth = args.embedding_depth
        self.num_heads = args.num_heads
        self.mlp_dim = args.mlp_dim
        self.dropout_rate = args.dropout_rate
        self.num_patches = int(self.clip_length_num_samples / self.patch_length_num_samples)
        self.history_length = NUM_SLEEP_STAGE_HISTORY
        self.historical_lookback_DNN = False and (True if self.history_length > 0 else False)
        self.historical_lookback_DNN_depth = args.historical_lookback_DNN_depth
        self.enable_scaling = args.enable_input_rescale
        self.enable_positional_embedding = args.enable_positional_embedding
        self.use_class_embedding = args.use_class_embedding
        self.mlp_dense_activation = mlp_dense_activation

        # Layers
        self.patch_projection = tf.keras.layers.Dense(self.embedding_depth, name="patch_projection_dense")
        if self.enable_scaling: self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / MAX_VOLTAGE)
        if self.use_class_embedding: self.class_embedding = self.add_weight("class_emb", shape=(1, 1, self.embedding_depth))
        if self.enable_positional_embedding: self.positional_embedding = self.add_weight("pos_emb", shape=(1, self.num_patches+self.use_class_embedding+(self.history_length > 0), self.embedding_depth)) #+1 for the trainable classification token, +1 for historical lookback
        self.encoder_layers = [Encoder(args, mlp_dense_activation=self.mlp_dense_activation) for _ in range(self.num_encoder_layers)]
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_head_layerNorm"),
            tf.keras.layers.Dense(self.mlp_dim, activation=self.mlp_dense_activation, name="mlp_head_dense1"),
            tf.keras.layers.Dropout(self.dropout_rate, seed=RANDOM_SEED, name="mlp_head_dropout"),
            tf.keras.layers.Dense(self.num_classes, activation="softmax", name="mlp_head_dense2"),
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

    def call(self, input, training:bool=False):
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

        # Normalize to [0, 1]
        if self.enable_scaling: clip = self.rescale(clip)

        # Extract patches
        patches = self.extract_patches(batch_size, clip) #patches = (batch_size, num_patches, patch_length_num_samples)
        if self.history_length > 0:
            patches = tf.concat([patches, historical_lookback_patch], axis=1)

        # Linear projection
        clip = self.patch_projection(patches) #clip = (batch_size, num_patches, embedding_depth)

        # Classification token
        if self.use_class_embedding:
            class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.embedding_depth]) #class_embedding = (batch_size, 1, embedding_depth)
            clip = tf.concat([class_embedding, clip], axis=1) #clip = (batch_size, num_patches+1, embedding_depth)

        if self.enable_positional_embedding: clip = clip + self.positional_embedding #clip = (batch_size, num_patches+1, embedding_depth)

        # Go through encoder
        for layer in self.encoder_layers:
            clip = layer(inputs=clip, training=training) #clip = (batch_size, num_patches+1, embedding_depth)

        # Classify with first token
        prediction = self.mlp_head(clip[:, 0]) #prediction = (batch_size, sleep_map.get_num_stages()+1)

        if self.historical_lookback_DNN:
            historical_lookback = tf.cast(historical_lookback, tf.uint8)
            historical_lookback = tf.one_hot(historical_lookback, depth=self.num_classes)
            prediction = tf.expand_dims(prediction, axis=1)
            concatenated = tf.concat([prediction, historical_lookback], axis=1)
            concatenated = tf.reshape(concatenated, [batch_size, -1])
            prediction = self.historical_lookback(concatenated)

        return prediction

    def build_graph(self):
        x = tf.keras.Input(shape=(1,self.clip_length_num_samples))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))

    def get_config(self):
        config = {
            'name': 'VisionTransformer',
            'clip_length_num_samples': self.clip_length_num_samples,
            'patch_length_num_samples': self.patch_length_num_samples,
            'num_encoder_layers': self.num_encoder_layers,
            'num_classes': self.num_classes,
            'embedding_depth': self.embedding_depth,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'dropout_rate': self.dropout_rate,
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
        x_in, y_in = self.num_patches, self.patch_length_num_samples
        x_out, y_out = self.num_patches, self.embedding_depth
        increment_ops(ops_dict=ops, in_shape=(x_in,y_in,0), out_shape=(x_out,y_out,0), layer_type="Dense", activation=False)

        # Positional embedding
        if self.enable_positional_embedding:
            ops["adds"] += (self.num_patches + self.use_class_embedding) * self.embedding_depth

        # Encoder layers
        for op, num in self.encoder_layers[0].calculate_ops().items():
            ops[op] += self.num_encoder_layers * num

        # MLP head
        x_in, y_in = self.num_patches+self.use_class_embedding, y_out
        increment_ops(ops_dict=ops, in_shape=(x_in,y_in,0), out_shape=(x_in,y_in,0), layer_type="LayerNorm")
        increment_ops(ops_dict=ops, in_shape=(x_in,y_in,0), out_shape=(y_in,self.mlp_dim,0), layer_type="Dense", activation=True)
        increment_ops(ops_dict=ops, in_shape=(y_in,self.mlp_dim,0), out_shape=(1, sleep_map.get_num_stages()+1,0), layer_type="Dense", activation=True)

        for op in ops.keys(): ops[op] = int(ops[op])
        return ops

#--- Misc ---#
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_depth, warmup_steps=NUM_WARMUP_STEPS):
        super().__init__()
        self.embedding_depth = tf.cast(embedding_depth, tf.float32)
        self.warmup_steps_exponent = -1.5
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
    print(f"[{(time.time()-start_time):.2f}s] Arguments parsed; starting dataset load.")

    # Load data
    try: data, start_of_night_markers, original_sleep_stage_cnt, dataset_metadata = load_from_dataset(args=args)
    except Exception as e: utilities.log_error_and_exit(exception=e, manual_description=f"[{(time.time()-start_time):.2f}s] Failed to load data from dataset.")
    global train_signals_representative_dataset
    train_signals_representative_dataset = data["signals_train"]

    # Train or load model
    mlp_dense_act = "swish"
    if args.load_model_filepath == None:
        print(f"[{(time.time()-start_time):.2f}s] Dataset ready. Starting training with {int(data['signals_train'].shape[0])} clips.")
        model, fit_history = train_model(args, data, mlp_dense_act)
    else: model = tf.keras.models.load_model(args.load_model_filepath, custom_objects={"CustomSchedule": CustomSchedule})

    # Make results directory. Check whether folder with the same name already exist and append counter if necessary
    time_of_export = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_fp = utilities.find_folder_path(utilities.folder_base_path()+f"{time_of_export}_vision", utilities.folder_base_path())
    os.makedirs(out_fp, exist_ok=True)
    os.makedirs(out_fp+"/models", exist_ok=True)

    # Save models to disk
    models = {"tf":model, "tflite":-1, "tflite (quant)":-1, "tflite (full quant)":-1, "tflite (16bx8b full quant)":-1}
    if args.save_model:
        model.save(f"{out_fp}/models/model.tf", save_format="tf")
        print(f"[{(time.time()-start_time):.2f}s] Saved model to {out_fp}/models/model.tf")

        tf.keras.utils.plot_model(model.build_graph(), to_file=f"{out_fp}/model_architecture.png", expand_nested=True, show_trainable=True, show_shapes=True, show_layer_activations=True, dpi=300, show_dtype=True)

        # Convert to Tensorflow Lite model and save
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        models["tflite"] = f"{out_fp}/models/model.tflite"
        with open(f"{out_fp}/models/model.tflite", "wb") as f:
            f.write(tflite_model)
        print(f"[{(time.time()-start_time):.2f}s] Saved TensorFlow Lite model to {out_fp}/models/model.tflite.")

        # Convert to quantized Tensorflow Lite model and save
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        models["tflite (quant)"] = f"{out_fp}/models/model_quant.tflite"
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
        models["tflite (full quant)"] = f"{out_fp}/models/model_full_quant.tflite"
        with open(f"{out_fp}/models/model_full_quant.tflite", "wb") as f:
            f.write(tflite_full_quant_model)
        print(f"[{(time.time()-start_time):.2f}s] Saved fully quantized TensorFlow Lite model to {out_fp}/models/model_full_quant.tflite.")

        # Convert to full quantized Tensorflow Lite model with 16b activations and 8b weights and save
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        tflite_16x8_full_quant_model = converter.convert()
        models["tflite (16bx8b full quant)"] = f"{out_fp}/models/model_full_quant_16bx8b.tflite"
        with open(f"{out_fp}/models/model_full_quant_16bx8b.tflite", "wb") as f:
            f.write(tflite_16x8_full_quant_model)
        print(f"[{(time.time()-start_time):.2f}s] Saved fully quantized TensorFlow Lite model (16b activations and 8b weights) to {out_fp}/models/model_full_quant_16bx8b.tflite.")

    # Manual validation
    acc = {"tf":-1, "tflite":-1, "tflite (quant)":-1, "tflite (full quant)":-1, "tflite (16bx8b full quant)":-1}
    pred_cnt = {"tf":-1, "tflite":-1, "tflite (quant)":-1, "tflite (full quant)":-1, "tflite (16bx8b full quant)":-1}
    for model_type, model in models.items():
        acc[model_type], pred_cnt[model_type] = manual_val(model, type=model_type, data=data, whole_night_indices=start_of_night_markers, out_fp=out_fp, ds_metadata=dataset_metadata)

    # Count sleep stages in training and validation datasets
    sleep_stages_cnt_train = utilities.count_instances_per_class(data['sleep_stages_train'], sleep_map.get_num_stages()+1)
    sleep_stages_cnt_val = utilities.count_instances_per_class(data['sleep_stages_val'], sleep_map.get_num_stages()+1)

    # Save accuracy and model details to log file
    note = ""
    export_summary(out_fp, args, models["tf"], fit_history, acc, original_sleep_stage_cnt, sleep_stages_cnt_train, sleep_stages_cnt_val, pred_cnt, mlp_dense_act, dataset_metadata, note)

    # Plot validation accuracy and loss during training
    export_training_val_plot(out_fp, fit_history=fit_history)

    print(f"[{(time.time()-start_time):.2f}s] Done. Good bye.")

if __name__ == "__main__":
    main()