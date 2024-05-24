import csv
import h5py
import glob
import socket
import argparse
import datetime

import numpy as np
import glob as glob
import tensorflow as tf
from pyedflib import EdfReader
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

"""
Some utility function
"""

#--- GLOBALS ---#
PRUNE_THRESHOLD = 1e-4 # To reduce ASIC power consumption, we prune weights below this threshold and avoid computation if one of the input is 0
SLEEP_STAGE_ANNOTATONS_CHANNEL = 2 #Channel of sleep stages in annotations file
global_min = np.inf
global_max = -np.inf
global_closest_to_zero = np.inf
global_total_params = 0
global_prunable_params = 0

#--- CLASSES ---#
class ArgumentParserWithError(argparse.ArgumentParser):
    def error(self, message):
        raise Exception(message)

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

class MovingAverage():
    def __init__(self, num_samples:int, self_reset_threshold:int=-1):
        self.num_samples = num_samples # Number of desired samples in moving average
        self.self_reset_enabled = (self_reset_threshold > 0)
        self.self_reset_threshold = self_reset_threshold # After this number of samples with a constant sleep stage, we self reset the filter. This is to provide a sharp edge on transitions if the output has been stable.

        self.output_samples = []
        self.sleep_stage_samples = []

    def filter(self, new_output:tf.Tensor):
        """ Returns a filtered (self.num_samples.shape[0], 1) tensor."""
        if self.num_samples == 0: # No filtering
            return new_output

        if self.self_reset_enabled and (len(self.sleep_stage_samples) >= self.self_reset_threshold) and (self.is_sleep_stage_buffer_constant()): self.reset()

        self.output_samples.append(new_output)
        if len(self.output_samples) > self.num_samples: self.output_samples.pop(0)

        return tf.reduce_mean(self.output_samples, axis=0)

    def reset(self):
        self.output_samples = []

    def append_sleep_stage(self, new_sleep_stage:int):
        self.sleep_stage_samples.append(new_sleep_stage)
        if len(self.sleep_stage_samples) > self.self_reset_threshold: self.sleep_stage_samples.pop(0)

    def is_sleep_stage_buffer_constant(self):
        """
        Returns whether the sleep stage prediction buffer contains only the same values, up to the self reset threshold
        """

        constant = True

        for i in range(len(self.sleep_stage_samples)):
            if self.sleep_stage_samples[-i-1] != self.sleep_stage_samples[-1]:
                constant = False
                break

        return constant

class SleepStageMap():
    """
    Class serves as easy way to map sleep stages between raw value found from data annotations, numerical values used in model and name used in plots.
    Light sleep stages: 1 and 2
    Deep sleep stages: 3 and 4
    Others: REM, wake and unknown
    """

    _allowed_maps = ["no_combine", "light_only_combine", "deep_only_combine", "both_light_deep_combine"]

    def __init__(self, map_name:str="no_combine"):
        assert (map_name in self._allowed_maps), (f"Mapping name '{map_name}' not in allowed mappings ({self._allowed_maps})")
        self.map_name = map_name

    def set_map_name(self, map_name:str):
        self.map_name = map_name

    def get_map_name(self):
        return self.map_name

    def get_numerical_map(self):
        """
        Returns a dictionary mapping the label name found in data to numerical value used in model.
        """
        if   (self.map_name == "no_combine"):               return {"Sleep stage 1":4, "Sleep stage 2":3, "Sleep stage 3":2, "Sleep stage 4":1, "Sleep stage R":5, "Sleep stage W":6, "Sleep stage ?":0}
        elif (self.map_name == "light_only_combine"):       return {"Sleep stage 1":3, "Sleep stage 2":3, "Sleep stage 3":2, "Sleep stage 4":1, "Sleep stage R":4, "Sleep stage W":5, "Sleep stage ?":0}
        elif (self.map_name == "deep_only_combine"):        return {"Sleep stage 1":3, "Sleep stage 2":2, "Sleep stage 3":1, "Sleep stage 4":1, "Sleep stage R":4, "Sleep stage W":5, "Sleep stage ?":0}
        elif (self.map_name == "both_light_deep_combine"):  return {"Sleep stage 1":2, "Sleep stage 2":2, "Sleep stage 3":1, "Sleep stage 4":1, "Sleep stage R":3, "Sleep stage W":4, "Sleep stage ?":0}
        else: raise Exception(f"Map name '{self.map_name}' not recognized.")

    def get_name_map(self):
        """
        Returns list of sleep stages name (useful for plot labels). It is ordered from deepest to lightest stage.
        """
        if   (self.map_name == "no_combine"):               return ["Unknown", "N4 (deep)",   "N3 (deep)",    "N2 (light)",   "N1 (light)", "REM", "Wake"]
        elif (self.map_name == "light_only_combine"):       return ["Unknown", "N4 (deep)",   "N3 (deep)",    "N1/2 (light)", "REM",        "Wake"]
        elif (self.map_name == "deep_only_combine"):        return ["Unknown", "N3/4 (deep)", "N2 (light)",   "N1 (light)",   "REM",        "Wake"]
        elif (self.map_name == "both_light_deep_combine"):  return ["Unknown", "N3/4 (deep)", "N1/2 (light)", "REM", "Wake"]
        else: raise Exception(f"Map name '{self.map_name}' not recognized.")

    def get_num_stages(self):
        """
        Returns the number of sleep stages (excluding the 'unknown' stage)
        """
        if   (self.map_name == "no_combine"):               return 6
        elif (self.map_name == "both_light_deep_combine"):  return 4
        else:                                               return 5 # One of light or deep combined

#--- FUNCTIONS ---#
def plot_1D_tensor(input_tensor: tf.Tensor) -> None:
    """
    Simply plots a given tensor and prints its min, max and dimensions.
    Plot blocks execution of script until the window is closed.
    """

    data = np.array(input_tensor)

    plt.plot(data)
    plt.title(f"min: {min(data)}, max: {max(data)}, shape: {data.shape}")
    plt.show(block=True)

def random_dataset(clip_length_num_samples:int, max_min:tuple, num_clips:int=1000) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Returns a tuple of Tensors: (input clip, sleep stage).
    The give some sort of relationship between the input and output, we generate input data from a different range based on the output.
    Dimensions:
        -input_sequence: (num_clips, clip_length_num_samples)
        -sleep_stage: (num_clips, 1)
    """

    clip_max, clip_min, sleep_max, sleep_min = max_min
    clip_length_num_samples = int(clip_length_num_samples)
    num_clips = int(num_clips)

    sleep_stage_dataset = tf.random.uniform(shape=[num_clips], minval=sleep_min, maxval=sleep_max) #Sleep stages
    sleep_stage_dataset = tf.math.round(sleep_stage_dataset)
    input_dataset = tf.zeros(shape=(0, clip_length_num_samples))

    for sleep_stage in sleep_stage_dataset:
        mean = (clip_max) * sleep_stage/sleep_max
        stddev = 2000 # Observed in real dataset

        new_input_sequence = tf.random.normal(shape=(1, clip_length_num_samples), mean=mean, stddev=stddev)
        new_input_sequence = tf.clip_by_value(new_input_sequence, clip_value_min=clip_min, clip_value_max=clip_max)
        new_input_sequence = tf.round(new_input_sequence)
        input_dataset = tf.concat([input_dataset, new_input_sequence], axis=0)

    return input_dataset, sleep_stage_dataset

def find_folder_path(candidate_folder_path:str, directory:str) -> str:
    """
    Finds a folder path that does not already exist in the given directory.
    If the candidate folder path already exists, a number is appended to the end of the folder path.
    """

    existing_folder_paths = glob.glob(directory)

    if (candidate_folder_path in existing_folder_paths):
        counter = 2
        candidate_folder_path += f"_{counter}"
        while candidate_folder_path in existing_folder_paths:
            counter += 1
            candidate_folder_path += f"_{counter}"

    return candidate_folder_path

def find_txt_file_name(candidate_file_name:str, directory:str) -> str:
    """
    Finds a file name that does not already exist in the given directory.
    If the candidate file name already exists, a number is appended to the end of the file name.
    """

    candidate_filepath = directory + candidate_file_name
    existing_filepaths = glob.glob(directory+"/*")

    if (candidate_filepath in existing_filepaths):
        counter = 2
        candidate_filepath = candidate_filepath[:-4] + f"_{counter}.txt"
        while candidate_filepath in existing_filepaths:
            counter += 1
            candidate_filepath = candidate_filepath[:-6] + f"_{counter}.txt"

    return candidate_filepath

def log_error_and_exit(exception, manual_description:str="", additional_msg:str="") -> None:
    print(f"socket.gethostname(): {socket.gethostname()}")
    if socket.gethostname() == "claude-ryzen":                      error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/trobitaille/engsci-thesis/python_prototype/error_logs/")
    elif socket.gethostname() == "Tristans-MacBook-Pro-3.local":    error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/Users/tristan/Developer/engsci-thesis/python_prototype/error_logs/")
    elif "cedar.computecanada.ca" in socket.gethostname():          error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/error_logs/")

    print(f"{manual_description} Received error: {exception}")

    with open(error_log, 'w') as f:
        f.write(f"{manual_description} ")
        f.write(f"Exception: {str(exception)}")
        f.write(additional_msg)
    exit()

def get_weight_distribution(model:tf.keras.Model) -> None:
    """
    Plots the distribution of weights in a given model.
    """

    for layer in model.layers:
        # Plot distribution of weights
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]
            weights = weights.flatten()
            plt.clf() # Clear the current figure
            plt.hist(list(weights), bins=100)
            plt.title(f"Layer {layer.name}")
            plt.savefig(f"/home/trobitaille/engsci-thesis/python_prototype/scratch/{layer.name}.png")
        else:
            print(f"No weights found for layer {layer.name}")

def count_instances_per_class(input, num_classes) -> list:
    count = [0 for _ in range(num_classes)]
    for elem in input:
        count[int(elem)] += 1
    return count

def run_model(model, data:dict, whole_night_indices:list, data_type:tf.DType, num_output_filtering:int=0, filter_post_argmax:bool=True, self_reset_threshold:int=-1, num_sleep_stage_history:int=0):
    # Load saved model (if argument is string)
    if (isinstance(model, str)):
        model = tf.keras.models.load_model(model, custom_objects={"CustomSchedule": CustomSchedule})

    sleep_stages_pred = []
    ground_truth = []
    total = 0
    output_filter = MovingAverage(num_output_filtering, self_reset_threshold=self_reset_threshold)
    if num_sleep_stage_history > 0: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history), dtype=data_type)

    try:
        for x, y in zip(data["signals_val"], data["sleep_stages_val"]):
            x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1
            if num_sleep_stage_history > 0:
                x = tf.concat([x[:,:-num_sleep_stage_history], historical_pred], axis=1) # Concatenate historical prediction to input
                if total in whole_night_indices: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history)) # Reset historical prediction at 0 (unknown) if at the start a new night
            if filter_post_argmax and (total in whole_night_indices): output_filter.reset() # Reset filter if starting a new night

            sleep_stage_pred = model(x, training=False)
            if not filter_post_argmax: sleep_stage_pred = output_filter.filter(sleep_stage_pred) # Filter softmax output
            sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)

            if filter_post_argmax: sleep_stage_pred = tf.cast(output_filter.filter(sleep_stage_pred), dtype=data_type) # Filter sleep stage
            else: sleep_stage_pred = tf.cast(sleep_stage_pred, dtype=data_type)

            if num_sleep_stage_history > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:num_sleep_stage_history-1]], axis=1)
            sleep_stages_pred.append(int(sleep_stage_pred[0].numpy()))
            output_filter.append_sleep_stage(int(sleep_stage_pred[0].numpy()))
            total += 1
            ground_truth.append(int(y))
    except Exception as e: log_error_and_exit(exception=e, manual_description="Failed to manually run model.")

    return sleep_stages_pred, ground_truth

def run_tflite_model(model_fp:str, data:str, whole_night_indices:list, data_type:tf.DType, num_output_filtering:int=0, filter_post_argmax:bool=True, self_reset_threshold:int=-1, num_sleep_stage_history:int=0) -> list:
    interpreter = tf.lite.Interpreter(model_path=model_fp)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    total = 0
    output_filter = MovingAverage(num_output_filtering, self_reset_threshold=self_reset_threshold)
    if num_sleep_stage_history > 0: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history), dtype=data_type)

    for x, y in zip(data["signals_val"], data["sleep_stages_val"]):
        if (data_type == tf.int8):
            x = x / 2**15 * 2**7
        x = tf.cast(x=x, dtype=data_type)
        x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1

        if num_sleep_stage_history > 0:
            x = tf.concat([x[:,:-num_sleep_stage_history], historical_pred], axis=1) # Concatenate historical prediction to input
            if total in whole_night_indices: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history)) # Reset historical prediction at 0 (unknown) if at the start a new night
        if total in whole_night_indices: output_filter.reset() # Reset filter if starting a new night

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        sleep_stage_pred = interpreter.get_tensor(output_details[0]['index'])
        if not filter_post_argmax: sleep_stage_pred = output_filter.filter(sleep_stage_pred) # Filter softmax output
        sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)

        if filter_post_argmax: tf.cast(output_filter.filter(sleep_stage_pred), dtype=data_type) # Filter sleep stage
        else: sleep_stage_pred = tf.cast(sleep_stage_pred, dtype=data_type)

        if num_sleep_stage_history > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:num_sleep_stage_history-1]], axis=1)
        predictions.append(int(sleep_stage_pred[0].numpy()))
        output_filter.append_sleep_stage(int(sleep_stage_pred[0].numpy()))
        total += 1

    return predictions

def folder_base_path() -> str:
    if socket.gethostname() == "claude-ryzen":                      return f"/home/trobitaille/engsci-thesis/python_prototype/"
    elif socket.gethostname() == "Tristans-MacBook-Pro-3.local":    return f"/Users/tristan/Developer/engsci-thesis/python_prototype/"
    elif "cedar.computecanada.ca" in socket.gethostname():          return f"/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/"

def shuffle(signal:tf.Tensor, sleep_stages:tf.Tensor, random_seed:int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Shuffles two input tensors in the same way s.t. corresponding pairs remain
    """
    size = tf.shape(signal)[0]
    indices = tf.range(size)
    shuffled_indices = tf.random.shuffle(value=indices, seed=random_seed)
    signal_shuffled = tf.gather(signal, shuffled_indices)
    sleep_stages_shuffled = tf.gather(sleep_stages, shuffled_indices)
    return signal_shuffled, sleep_stages_shuffled

def round_up_to_nearest_tenth(input) -> float:
    rounded_input = round(input, 1) # Round to nearest 0.1
    rounded_up_input = rounded_input if (rounded_input > input) else (rounded_input + 0.1) # Round up to nearest tenth
    return rounded_up_input

def edf_to_h5(edf_fp:str, h5_filename:str, sleep_map_name:str, channel:str, clip_length_s=30, full_night:bool=False, sampling_freq_hz:int=128) -> tf.Tensor:
    """ Reads .edf input signal file, saves to .h5 and return the signal as a Tensor.
        If full_night is True, the entire night is saved. Otherwise, only the first clip_length_s seconds are saved.
    """
    # Ground truth
    sleep_map = SleepStageMap(sleep_map_name)
    sleep_stage = EdfReader(edf_fp.replace("PSG", "Base")).readAnnotations()[SLEEP_STAGE_ANNOTATONS_CHANNEL]
    for i in range(len(sleep_stage)): sleep_stage[i] = sleep_map.get_numerical_map()[sleep_stage[i]]
    sleep_stage = sleep_stage.astype(int)

    # Signals
    signal_reader = EdfReader(edf_fp)
    channels = list(signal_reader.getSignalLabels())
    channel_number = channels.index(channel)
    signal = signal_reader.readSignal(channel_number, digital=True)

    # Resampling
    original_indices = np.arange(0, len(signal))
    target_length = int(len(signal) * (sampling_freq_hz / 256)) # 256 Hz is the original sampling frequency
    target_indices = np.linspace(0, len(signal) - 1, target_length)
    interpolator = interp1d(original_indices, signal, kind='linear', fill_value="extrapolate")
    resample = interpolator(target_indices)
    resample = resample.astype(int)

    # Grab 1 clip or all night. If full night, will save as a matrix instead of a vector.
    if (full_night == False): resample = resample[0:clip_length_s*sampling_freq_hz]

    for i in range(len(resample)):
        resample[i] += 2**15 # Offset by 15b

    with h5py.File(h5_filename, 'w') as file:
        if (full_night == True):
            length = min(len(resample), len(sleep_stage)) - 1
            resample = [resample[i : i+clip_length_s*sampling_freq_hz] for i in range(0, len(resample), clip_length_s*sampling_freq_hz)]
            file.create_dataset('eeg', data=resample[0:length])
            file.create_dataset('sleep_stage', data=sleep_stage[0:length])
            tensor = tf.convert_to_tensor(resample[0:length], dtype=tf.float32, name="input_1")
            tensor = tf.expand_dims(tensor, axis=0)
            return tensor
        else:
            file.create_dataset('eeg', data=[resample, resample]) # Need a 2D dataset
            file.create_dataset('sleep_stage', data=[sleep_stage, sleep_stage]) # Need a 2D dataset
            tensor = tf.convert_to_tensor(resample, dtype=tf.float32, name="input_1")
            tensor = tf.expand_dims(tensor, axis=0)
            return tensor

def export_layer_outputs(model:tf.keras.Model, input_signal:tf.Tensor):
    model.build(input_shape=input_signal.shape)
    model(input_signal, training=False)
    for i, layer in enumerate(model.layers):
        layer_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        layer_output = layer_model.predict(input_signal)
        print(layer_output)

def visit_h5_datasets(node):
    global global_min, global_max, global_closest_to_zero, global_total_params, global_prunable_params
    if isinstance(node, h5py.Dataset):
        data = node[:]
        if np.min(data) < global_min: global_min = np.min(data)
        if np.max(data) > global_max: global_max = np.max(data)
        if np.min(abs(data)) < global_closest_to_zero: global_closest_to_zero = np.min(abs(data))
        global_total_params += data.size
        global_prunable_params += np.sum(abs(data) < PRUNE_THRESHOLD)

def print_stats_from_h5(h5_fp:str) -> None:
    global global_min, global_max
    with h5py.File(h5_fp, 'r') as f:
        f.visititems(visit_h5_datasets)
    print(f'Global Min = {global_min:.5f}, Global Max = {global_max:.5f}, Closest to zero (abs.) = {global_closest_to_zero:.8f}')
    print(f'Total prunable params (abs. < {PRUNE_THRESHOLD}) = {global_prunable_params/global_total_params*100:.2f}%')

def run_accuracy_study(model_fp:str, eeg_fp:str, results_fp:str, num_clips:int):
    model = tf.keras.models.load_model(model_fp, custom_objects={"CustomSchedule": CustomSchedule})

    data = {'signals_val': tf.Tensor(), 'sleep_stages_val': tf.Tensor()}
    with h5py.File(eeg_fp, 'r') as f:
        data['signals_val'] = tf.convert_to_tensor(f['eeg'][:], dtype=tf.float32)
        data['sleep_stages_val'] = tf.convert_to_tensor(f['sleep_stage'][:], dtype=tf.float32)

    sleep_stages_pred, ground_truth = run_model(model, data, whole_night_indices=[0], data_type=tf.float32, num_output_filtering=3, filter_post_argmax=True)

    # Save results to CSV
    with open(results_fp, "r", newline="") as file: # Read the existing data from the file
        reader = csv.reader(file)
        data = list(reader)

    # Insert the new rows into the correct position in the list
    for i, (sleep_stage, truth) in enumerate(zip(sleep_stages_pred, ground_truth)):
        if (i == num_clips): break
        # If there's a row, modify it. Else, append a new row.
        if (i < (len(data)-2)):
            row = data[i+2]
            row[0] = i
            row[1] = truth
            row[2] = sleep_stage
            for i in range(20): row.append("")
            data[i+2] = row
        else:
            row = [i, truth, sleep_stage]
            data.insert(i+2, row)

    # Write the list back to the file
    with open(results_fp, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    print("Done computing results for accuracy study.")

def main():
    run_accuracy_study(model_fp="asic/fixed_point_accuracy_study/model.tf", eeg_fp="asic/fixed_point_accuracy_study/eeg.h5", results_fp="asic/fixed_point_accuracy_study/results_test_1.csv", num_clips=2000)

if __name__ == "__main__":
    main()
