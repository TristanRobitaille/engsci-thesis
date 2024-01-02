import glob
import socket
import argparse
import datetime

import numpy as np
import glob as glob
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Some utility function
"""

#--- CLASSES ---#
class ArgumentParserWithError(argparse.ArgumentParser):
    def error(self, message):
        raise Exception(message)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_depth, warmup_steps=40000):
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

class MovingAverage():
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.samples = []

    def filter(self, new_sample:tf.Tensor):
        """ Returns a filtered (self.num_samples.shape[0], 1) tensor."""
        if self.num_samples == 0: # No filtering
            return new_sample
        
        self.samples.append(new_sample)
        if len(self.samples) > self.num_samples:
            self.samples.pop(0)
        return tf.reduce_mean(self.samples, axis=0)
    
    def reset(self):
        self.samples = []

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
        if   (self.map_name == "no_combine"):           return {"Sleep stage 1":4, "Sleep stage 2":3, "Sleep stage 3":2, "Sleep stage 4":1, "Sleep stage R":5, "Sleep stage W":6, "Sleep stage ?":0}
        elif (self.map_name == "light_only_combine"):   return {"Sleep stage 1":3, "Sleep stage 2":3, "Sleep stage 3":2, "Sleep stage 4":1, "Sleep stage R":4, "Sleep stage W":5, "Sleep stage ?":0}
        elif (self.map_name == "deep_only_combine"):    return {"Sleep stage 1":3, "Sleep stage 2":2, "Sleep stage 3":1, "Sleep stage 4":1, "Sleep stage R":4, "Sleep stage W":5, "Sleep stage ?":0}
        else:                                           return {"Sleep stage 1":2, "Sleep stage 2":2, "Sleep stage 3":1, "Sleep stage 4":1, "Sleep stage R":3, "Sleep stage W":4, "Sleep stage ?":0}

    def get_name_map(self):
        """
        Returns list of sleep stages name (useful for plot labels). It is ordered from deepest to lightest stage.
        """
        if   (self.map_name == "no_combine"):           return ["Unknown", "N4 (deep)",   "N3 (deep)",    "N2 (light)",   "N1 (light)", "REM", "Wake"]
        elif (self.map_name == "light_only_combine"):   return ["Unknown", "N4 (deep)",   "N3 (deep)",    "N1/2 (light)", "REM",        "Wake"]
        elif (self.map_name == "deep_only_combine"):    return ["Unknown", "N3/4 (deep)", "N2 (light)",   "N1 (light)",   "REM",        "Wake"]
        else:                                           return ["Unknown", "N3/4 (deep)", "N1/2 (light)", "REM", "Wake"]
    
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

def random_dataset(clip_length_num_samples:int, max_min:tuple, num_clips:int=1000) -> (tf.Tensor, tf.Tensor):
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
    if socket.gethostname() == "claude-ryzen":              error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/trobitaille/engsci-thesis/python_prototype/error_logs/")
    elif socket.gethostname() == "MBP_Tristan":             error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/Users/tristan/Desktop/engsci-thesis/python_prototype/error_logs/")
    elif "cedar.computecanada.ca" in socket.gethostname():  error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/error_logs/")

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

def run_model(model, data:dict, whole_night_indices:list, data_type:tf.DType, num_output_filtering:int=0, num_sleep_stage_history:int=0) -> list:
    # Load saved model (if argument is string)
    if (isinstance(model, str)):
        model = tf.keras.models.load_model(model, custom_objects={"CustomSchedule": CustomSchedule})

    sleep_stages_pred = []
    total = 0
    output_filter = MovingAverage(num_output_filtering)
    if num_sleep_stage_history > 0: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history), dtype=data_type)

    try:
        for x, y in zip(data["signals_val"], data["sleep_stages_val"]):
            x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1

            if num_sleep_stage_history > 0:
                x = tf.concat([x[:,:-num_sleep_stage_history], historical_pred], axis=1) # Concatenate historical prediction to input
                if whole_night_indices[total].numpy()[0] == 1.0: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history)) # Reset historical prediction at 0 (unknown) if at the start a new night

            sleep_stage_pred = model(x, training=False)
            sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)

            if whole_night_indices[total].numpy()[0] == 1.0: output_filter.reset() # Reset filter if starting a new night
            sleep_stage_pred = tf.cast(output_filter.filter(sleep_stage_pred), dtype=data_type) # Filter sleep stage
            
            if num_sleep_stage_history > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:num_sleep_stage_history-1]], axis=1)
            sleep_stages_pred.append(int(sleep_stage_pred[0].numpy()))
            total += 1
    except Exception as e: log_error_and_exit(exception=e, manual_description="Failed to manually run model.")

    return sleep_stages_pred

def run_tflite_model(model_fp:str, data:str, whole_night_indices:list, data_type:tf.DType, num_output_filtering:int=0, num_sleep_stage_history:int=0) -> list:
    interpreter = tf.lite.Interpreter(model_path=model_fp)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    total = 0
    output_filter = MovingAverage(num_output_filtering)
    if num_sleep_stage_history > 0: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history), dtype=data_type)

    for x, y in zip(data["signals_val"], data["sleep_stages_val"]):
        if (data_type == tf.int8):
            x = x / 2**15 * 2**7
        x = tf.cast(x=x, dtype=data_type)
        x = tf.reshape(x, [1, x.shape[0]]) # Prepend 1 to shape to make it a batch of 1

        if num_sleep_stage_history > 0:
            x = tf.concat([x[:,:-num_sleep_stage_history], historical_pred], axis=1) # Concatenate historical prediction to input
            if whole_night_indices[total].numpy()[0] == 1.0: historical_pred = tf.zeros(shape=(1, num_sleep_stage_history)) # Reset historical prediction at 0 (unknown) if at the start a new night

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        sleep_stage_pred = interpreter.get_tensor(output_details[0]['index'])
        sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)

        if whole_night_indices[total].numpy()[0] == 1.0: output_filter.reset() # Reset filter if starting a new night
        sleep_stage_pred = tf.cast(output_filter.filter(sleep_stage_pred), dtype=data_type) # Filter sleep stage

        if num_sleep_stage_history > 0: historical_pred = tf.concat([tf.expand_dims(sleep_stage_pred, axis=1), historical_pred[:, 0:num_sleep_stage_history-1]], axis=1)
        predictions.append(int(sleep_stage_pred[0].numpy()))
        total += 1

    return predictions

def folder_base_path() -> str:
    if socket.gethostname() == "claude-ryzen":              return f"/home/trobitaille/engsci-thesis/python_prototype/"
    elif socket.gethostname() == "MBP_Tristan":             return f"/Users/tristan/Desktop/engsci-thesis/python_prototype/"
    elif "cedar.computecanada.ca" in socket.gethostname():  return f"/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/"

def shuffle(signal:tf.Tensor, sleep_stages:tf.Tensor, random_seed:int) -> (tf.Tensor, tf.Tensor):
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

def main():
    print("Nothing to do in utilities main.")

if __name__ == "__main__":
    main()
