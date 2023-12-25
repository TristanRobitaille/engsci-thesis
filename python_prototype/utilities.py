import time
import glob
import socket
import argparse
import datetime

import numpy as np
import glob as glob
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Some utility functions used by models.
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
    
#--- FUNCTIONS ---#
def plot_1D_tensor(input_tensor: tf.Tensor):
    """
    Simply plots a given tensor and prints its min, max and dimensions.
    Plot blocks execution of script until the window is closed.
    """

    data = np.array(input_tensor)
    
    plt.plot(data)
    plt.title(f"min: {min(data)}, max: {max(data)}, shape: {data.shape}")
    plt.show(block=True)

    return

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
    if socket.gethostname() == "claude-ryzen":
        error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/trobitaille/engsci-thesis/python_prototype/error_logs/*")
    elif socket.gethostname() == "MBP_Tristan":
        error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/Users/tristan/Desktop/engsci-thesis/python_prototype/error_logs/*")
    elif "cedar.computecanada.ca" in socket.gethostname():
        error_log = find_txt_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/error_logs/*")

    print(f"Received error: {exception}")

    with open(error_log, 'w') as f:
        f.write(manual_description)
        f.write(f"Exception: {str(exception)}")
        f.write(additional_msg)
    exit()

def get_weight_distribution(model:tf.keras.Model):
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

def run_model(model_fp:str, data_fp:str):
    NUM_CLIPS_PER_FILE = 500 # Only valid for 256Hz 

    # Load saved model
    model = tf.keras.models.load_model(model_fp, custom_objects={"CustomSchedule": CustomSchedule})
    input_data_filepaths = glob.glob(data_fp + "/*.npy")
    print(f"Filepaths: {input_data_filepaths}")

    inference_times = np.empty((1))

    # Run inference
    print('---- INFERENCE ----')
    file_cnt = 0
    results_string = ""

    for filepath in input_data_filepaths:
        input_data = np.load(filepath) # Dimensions (# of clips, 1, clip_length). 1 indicates a batch_size of 1.
        input_data = input_data.astype(np.float32)
    
        for i in range(input_data.shape[0]):
            start = time.perf_counter()
            sleep_stage_pred = model(input_data[i], training=False)
            inference_time = time.perf_counter() - start
            sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)
            results_string += str(sleep_stage_pred[0].numpy()) + '\n'
            inference_times = np.append(inference_times, 1000*inference_time)

            print(f"Clip {NUM_CLIPS_PER_FILE*file_cnt + i} output: {sleep_stage_pred[0].numpy()} (inference time: {(inference_time * 1000):.2f}ms)")

        file_cnt += 1

    # Write results to file
    with open(data_fp + "/out_cpu.txt", 'w') as text_file:
        text_file.write(f"Model: {model_fp}\n")
        text_file.write(f"Mean time: {np.mean(inference_times[2:]):.3f}ms, std. dev.: {np.std(inference_times[2:]):.3f}ms\n")
        text_file.write(results_string)

    # Report stats
    print(f"Done. Mean time: {np.mean(inference_times[2:]):.3f}ms, std. dev.: {np.std(inference_times[2:]):.3f}ms")

def run_tflite_model(model_fp:str, data_fp:str):
    NUM_CLIPS_PER_FILE = 500 # Only valid for 256Hz 

    # Load saved model
    interpreter = tf.lite.Interpreter(model_path=model_fp)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data_filepaths = glob.glob(data_fp + "/*.npy")
    print(f"Filepaths: {input_data_filepaths}")

    inference_times = np.empty((1))

    # Run inference
    print('---- INFERENCE ----')
    file_cnt = 0
    results_string = ""

    for filepath in input_data_filepaths:
        input_data = np.load(filepath) # Dimensions (# of clips, 1, clip_length). 1 indicates a batch_size of 1.
        input_data = input_data.astype(np.float32)
    
        for i in range(input_data.shape[0]):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_data[i])
            inference_time = time.perf_counter() - start
            interpreter.invoke()
            sleep_stage_pred = interpreter.get_tensor(output_details[0]['index'])
            sleep_stage_pred = tf.argmax(sleep_stage_pred, axis=1)
            results_string += str(sleep_stage_pred[0].numpy()) + '\n'
            inference_times = np.append(inference_times, 1000*inference_time)

            print(f"Clip {NUM_CLIPS_PER_FILE*file_cnt + i} output: {sleep_stage_pred[0].numpy()} (inference time: {(inference_time * 1000):.2f}ms)")

        file_cnt += 1

    # Write results to file
    with open(data_fp + "/out_cpu.txt", 'w') as text_file:
        text_file.write(f"Model: {model_fp}\n")
        text_file.write(f"Mean time: {np.mean(inference_times[2:]):.3f}ms, std. dev.: {np.std(inference_times[2:]):.3f}ms\n")
        text_file.write(results_string)

    # Report stats
    print(f"Done. Mean time: {np.mean(inference_times[2:]):.3f}ms, std. dev.: {np.std(inference_times[2:]):.3f}ms")

def main():

    # run_model(model_fp="/home/trobitaille/engsci-thesis/python_prototype/results/2023-12-06_15-01-24_vision_best/2023-12-06_15-01-24_vision.tf", data_fp="/home/trobitaille/engsci-thesis/python_prototype/edgetpu_data")
    run_tflite_model(model_fp="/home/trobitaille/engsci-thesis/python_prototype/results/2023-12-25_14-42-11_vision/2023-12-25_14-42-11_vision_quant.tflite", data_fp="/home/trobitaille/engsci-thesis/python_prototype/edgetpu_data")
              
    return

if __name__ == "__main__":
    main()
