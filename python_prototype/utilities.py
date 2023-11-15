import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob as glob
import datetime
import argparse
import socket

"""
Some utility functions used by models.
"""

#--- CLASSES ---#
class ArgumentParserWithError(argparse.ArgumentParser):
    def error(self, message):
        raise Exception(message)

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

def find_file_name(candidate_file_name:str, directory:str) -> str:
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
        error_log = find_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/trobitaille/engsci-thesis/python_prototype/error_logs/*")
    elif socket.gethostname() == "MBP_Tristan":
        error_log = find_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/Users/tristan/Desktop/engsci-thesis/python_prototype/error_logs/*")
    elif "cedar.computecanada.ca" in socket.gethostname():
        error_log = find_file_name(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_error.txt", "/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/error_logs/*")

    print(f"Received error: {exception}")

    with open(error_log, 'w') as f:
        f.write(manual_description)
        f.write(f"Exception: {str(exception)}")
        f.write(additional_msg)
    exit()

def count_instances_per_class(input, num_classes) -> list:
    count = [0 for _ in range(num_classes)]
    for elem in input:
        count[int(elem)] += 1
    return count