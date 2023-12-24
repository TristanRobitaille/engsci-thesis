"""
Methods to extract data from EDF files and generate pseudo-random data.
"""
import time
start_time = time.time()

import os
import glob
import math
import json

from typing import List
from pyedflib import EdfReader
from argparse import ArgumentParser
from scipy.interpolate import interp1d

import numpy as np
import tensorflow as tf
import multiprocessing as mp

SLEEP_STAGE_RESOLUTION_SEC = 30.0 #Nominal (i.e. in EDF file) length or each clip/sleep stage annotation
NOMINAL_FREQUENCY_HZ = 256 #Sampling frequency for most channels
HISTORICAL_LOOKBACK_LENGTH = 0 #We save the last HISTORICAL_LOOKBACK_LENGTH sleep stages. Set to 0 to disable.
SLEEP_STAGE_ANNOTATONS_CHANNEL = 2 #Channel of sleep stages in annotations file
NUM_PSEUDO_RANDOM_CLIP_PER_SLEEP_STAGE = 5000 #Number of pseudo-random clips to generate for each sleep stage
MAX_VOLTAGE = 2**15 - 1
MIN_VOLTAGE = 0
NUM_SLEEP_STAGES = 5 #Excluding 'unknown'
ONE_HOT_OUTPUT = False #If true, sleep stages are exported as their one-hot classes tensor, else they are reported as a scalar

sleep_stage_annotation_to_int = { #Note: Stages 3 and 4 are combined and '0' is reserved for unknown
                                    "Sleep stage 1": 3,
                                    "Sleep stage 2": 2,
                                    "Sleep stage 3": 1,
                                    "Sleep stage 4": 1,
                                    "Sleep stage R": 4,
                                    "Sleep stage W": 5,
                                    "Sleep stage ?": 0}

def signals_processing(signals:List) -> List:
    """
    Can apply data processing before exporting data.
    Returns input tensor with processing applied.
    """
    # Shift up signal to remove negative values
    for i in range(len(signals)):
        signals[i] += 2**15

    return signals

def scalar_to_one_hot(input:int) -> List:
    """
    Returns the one-hot representation of the class number as a list of dimension NUM_SLEEP_STAGES
    """

    return [1 if i == input else 0 for i in range(NUM_SLEEP_STAGES+1)] #The +1 is to account for the 'unknown' class

def pseudo_random_clip(sleep_stage:int, clip_length_num_samples:int, max_min:tuple, stddev:float=2000) -> tf.Tensor:
    """
    Outputs a clip of random measurements whose mean is centred around the sleep stage.
    This is useful to validate trainability of a model as it should produce a very easily-trainable dataset.
    
    Output dimension: (clip_length_num_samples,)
    """

    clip_max, clip_min, sleep_max, sleep_min = max_min
    mean = (clip_max) * sleep_stage/sleep_max
    
    clip = np.random.normal(loc=mean, scale=stddev, size=int(clip_length_num_samples)).astype(np.int32)
    clip = np.round(clip)
    clip = np.clip(clip, a_min=clip_min, a_max=clip_max)

    return clip

def pseudo_random_dataset(num_clips_per_sleep_stage:int, clip_length_num_samples, data_type:tf.DType=tf.float32):
    """
    Returns a properly formatted dictionary with the pseudo random data, with equal number of each sleep stage.
    """

    output = {"sleep_stage":[], "pseudo_random":[]}

    for _ in range(num_clips_per_sleep_stage):
        for sleep_stage in range(1, NUM_SLEEP_STAGES+1):
            output["sleep_stage"].append(sleep_stage)

            new_clip = pseudo_random_clip(sleep_stage, clip_length_num_samples, max_min=(MAX_VOLTAGE, MIN_VOLTAGE, NUM_SLEEP_STAGES, 1))
            output["pseudo_random"].append(new_clip)

    # Convert to tensor
    output["pseudo_random"] = tf.convert_to_tensor(value=output["pseudo_random"], dtype=data_type)
    output["sleep_stage"] = tf.convert_to_tensor(value=output["sleep_stage"], dtype=data_type)
    output["sleep_stage"] = tf.reshape(tensor=output["sleep_stage"], shape=(output["sleep_stage"].shape[0], 1), name="sleep_stage")

    return output, 0 # Return code == 0

def prune_unknown(sleep_stages, signals):
    """
    Removes unknown sleep stages and corresponding list.
    """

    clips_to_remove = []
    for sleep_stage_number in range(len(sleep_stages)):
        if (sleep_stages[sleep_stage_number] == sleep_stage_annotation_to_int["Sleep stage ?"]): clips_to_remove.append(sleep_stage_number)
            
    clips_to_remove.sort(reverse=True)

    for clip_number in clips_to_remove:
        sleep_stages.pop(clip_number)
        for channel_signals in signals:
            channel_signals.pop(clip_number)

    return sleep_stages, signals

def resample_list(input:list, input_freq:int, target_freq:int) -> list:
    original_indices = np.arange(0, len(input))
    target_length = int(len(input) * (target_freq / input_freq))
    target_indices = np.linspace(0, len(input) - 1, target_length)

    interpolator = interp1d(original_indices, input, kind='linear', fill_value="extrapolate")
    resample = interpolator(target_indices)
    resample = resample.astype(int)

    return resample

def read_single_whole_night(args, psg_filepath:str, annotations_filepath:str, channels_to_read:List[str], historical_lookback_length=HISTORICAL_LOOKBACK_LENGTH, data_type:tf.DType=tf.float32) -> (tf.Tensor, tf.Tensor, List[str]):
    """
    Returns a dictionary of channels with signals cut into clips and a return code.

    Dimensions:
      -Dictionary value (one channel): (num_clips, num_samples_per_clip)
      -Sleep stages: (num_clips, 1)
    
    Assumptions:
        -Each channel is sampled at the same frequency, and is the same duration
        -Annotation resolution, is 30s
    """
    # Load data
    try: sleep_stages = EdfReader(annotations_filepath)
    except:
        print(f"Could not read file '{annotations_filepath}'! Skipping file.")
        return -1
    
    sleep_stages = list(sleep_stages.readAnnotations()[SLEEP_STAGE_ANNOTATONS_CHANNEL])
    sleep_stages = [sleep_stage_annotation_to_int[item] for item in sleep_stages]
    
    try: signal_reader = EdfReader(psg_filepath)
    except:
        print(f"Could not read file '{psg_filepath}'! Skipping file.")
        return -1
    
    signals = list()
    channels = list(signal_reader.getSignalLabels())

    total_raw_clips = min(math.floor(signal_reader.getFileDuration() / float(args.clip_length_s)), int(SLEEP_STAGE_RESOLUTION_SEC/float(args.clip_length_s))*len(sleep_stages))
    clip_duration_samples = int(float(args.clip_length_s) * NOMINAL_FREQUENCY_HZ)

    # Extract clips for each channel
    for channel in channels_to_read:
        if channel not in signal_reader.getSignalLabels():
            print(f"Channel '{channel}' not found in file '{psg_filepath}'! Skipping file.")
            return -1
        
        temp_clip = list()
        channel_number = channels.index(channel)

        for clip_number in range(total_raw_clips): #Split measurement list into clips of clip_length_s duration
            measurement = signal_reader.readSignal(channel_number, start=clip_duration_samples*clip_number, n=clip_duration_samples, digital=True)
            measurement = signals_processing(measurement)
            temp_clip.append(measurement)

        signals.append(temp_clip)

    # Duplicate sleep stages to account for stages being shorter than the nominal 30s
    sleep_stages = [item for item in sleep_stages for _ in range(int(SLEEP_STAGE_RESOLUTION_SEC/float(args.clip_length_s)))]
    sleep_stages = sleep_stages[0:total_raw_clips] #On some files, sleep stages are longer than PSG signals so we slice it. Assume that they line up at the lowest index.

    # Prune unknown sleep stages from all lists
    sleep_stages, signals = prune_unknown(sleep_stages, signals)

    # Generate historical lookback
    if HISTORICAL_LOOKBACK_LENGTH > 0: 
        sleep_stages_history = list()
        for i in range(len(sleep_stages)):
            sleep_stage_history = list()

            if i < historical_lookback_length:
                sleep_stage_history = [sleep_stage_annotation_to_int["Sleep stage ?"] for _ in range(historical_lookback_length-i)]
                sleep_stage_history = sleep_stages[0:i] + sleep_stage_history
            else:
                sleep_stage_history = [sleep_stages[i-j] for j in range(historical_lookback_length)]
            
            sleep_stages_history.append(sleep_stage_history)

    # Generate pseudo-random signal
    pseudo_random_list = list()
    for sleep_stage in sleep_stages:
        new_clip = pseudo_random_clip(sleep_stage, int(float(args.clip_length_s) * args.downsampling_freq_hz), max_min=(MAX_VOLTAGE, MIN_VOLTAGE, NUM_SLEEP_STAGES, 0))
        pseudo_random_list.append(new_clip)

    # Downsample
    if (args.downsampling_freq_hz != NOMINAL_FREQUENCY_HZ):
        for channel_index in range(len(signals)):
            for clip_index in range(len(signals[channel_index])):
                signals[channel_index][clip_index] = resample_list(input=signals[channel_index][clip_index], input_freq=NOMINAL_FREQUENCY_HZ, target_freq=args.downsampling_freq_hz)

    # Convert to tensors
    sleep_stages = tf.convert_to_tensor(sleep_stages, dtype=data_type, name="sleep_stage")
    pseudo_random_list = tf.convert_to_tensor(pseudo_random_list, dtype=data_type, name="pseudo_random")
    if HISTORICAL_LOOKBACK_LENGTH > 0: sleep_stages_history = tf.convert_to_tensor(sleep_stages_history, dtype=data_type, name=f"history_{historical_lookback_length}-steps")

    if ONE_HOT_OUTPUT: sleep_stages = tf.transpose(sleep_stages, perm=[0, 2, 1])
    else: sleep_stages = tf.expand_dims(sleep_stages, axis=1)

    for i in range(len(channels_to_read)):
        signals[i] = tf.convert_to_tensor(signals[i], dtype=data_type, name=channels_to_read[i])      

    signals = dict(zip(channels_to_read, signals)) #Convert to dictionary
    signals["pseudo_random"] = pseudo_random_list
    signals["sleep_stage"] = sleep_stages
    if HISTORICAL_LOOKBACK_LENGTH > 0: signals[f"history_{historical_lookback_length}-steps"] = sleep_stages_history

    return signals

def insert_into_all_night_dict(output_all_nights, output_one_night):
    for key, value in output_one_night.items():
        if output_all_nights[key] == None:
            output_all_nights[key] = output_one_night[key]
        else:
            output_all_nights[key] =tf.concat([output_all_nights[key], output_one_night[key]], axis=0)

    return output_all_nights

def process_dispatch(args, PSG_file_list, labels_file_list, channels_to_read, data_type, result_queue):
    try:
        if PSG_file_list == []: # No file to process
            print(f"Process ID {os.getpid()} received empty EDF file list ({PSG_file_list})!")
            result_queue.put(-1)
            return

        if HISTORICAL_LOOKBACK_LENGTH > 0: output_all_nights = {key: None for key in channels_to_read + ["sleep_stage"] + ["pseudo_random"] + [f"history_{HISTORICAL_LOOKBACK_LENGTH}-steps"] + ["new_night_marker"]}
        else: output_all_nights = {key: None for key in channels_to_read + ["sleep_stage"] + ["pseudo_random"] + ["new_night_marker"]}

        for PSG_file, labels_file in zip(PSG_file_list, labels_file_list):
            print(f"[{(time.time()-start_time):.2f}s] Process ID {os.getpid()} processing: {os.path.basename(PSG_file)} with {os.path.basename(labels_file)}")

            output_one_night = read_single_whole_night(args, psg_filepath=PSG_file, annotations_filepath=labels_file, channels_to_read=channels_to_read, data_type=data_type)
            if output_one_night == -1:
                print(f"Received failure code {output_one_night} from read_single_whole_night. Skipping night.")
                print(vars(args))
                continue #Skip this file

            # Mark file clip of this night to indicate new file in dataset
            new_file_marker = [False for _ in range(len(output_one_night["sleep_stage"]))]
            new_file_marker[0] = True
            output_one_night["new_night_marker"] = tf.convert_to_tensor(new_file_marker, dtype=data_type, name="new_night_marker")
            output_one_night["new_night_marker"] = tf.expand_dims(output_one_night["new_night_marker"], axis=1)

            output_all_nights = insert_into_all_night_dict(output_all_nights, output_one_night)

        result_queue.put(output_all_nights)
        print(f"[{(time.time()-start_time):.2f}s] Process ID {os.getpid()} completed reading its files.")

    except Exception as e:
        print(f"Error in child process {os.getpid()}: {e}")

def read_all_nights_from_directory(args, channels_to_read:List[str], data_type:tf.DType=tf.float32) -> (tf.Tensor, tf.Tensor, List[str]):
    """
    Calls read_single_whole_night() for all files in folder and returns a dictionary of all channels and sleep stages concatenated along with a success flag.

    Dimensions:
      -Dictionary value (one channel): (num_clips_total, num_samples_per_clip)
      -Sleep stages: (num_clips_total, 1)
    
    Assumptions:
        -All equivalent channels have the same labels across all nights
        -All PSG files have the same number of channels
    """

    # Get filenames (assume base of filenames for PSG and Labels match)
    PSG_file_list = glob.glob(os.path.join(args.directory_psg, "*PSG.edf"))
    if len(PSG_file_list) == 0:
        print(f"No valid PSG (*PSG.edf) files found in search directory ({args.directory_psg})!")
        return -1

    if (len(PSG_file_list) < args.num_files):
        print(f"Did not find the requested number of files ({args.num_files}). Will use {len(PSG_file_list)} files instead.")    
        args.num_files = len(PSG_file_list)

    labels_file_list = [f"{args.directory_labels}/{os.path.basename(psg_file).replace('PSG', 'Base')}" for psg_file in PSG_file_list]

    if args.enable_multiprocessing: # Use multiprocessing
        # Prepare for multiprocessing
        num_cpus = 24 # Leave 2 CPUs
        processes = []
        result_queue = mp.SimpleQueue()

        file_PSG_assignment, file_labels_assignment = [list() for _ in range(num_cpus)], [list() for _ in range(num_cpus)]
        for i in range(args.num_files):
            file_PSG_assignment[i%num_cpus].append(PSG_file_list[i])
            file_labels_assignment[i%num_cpus].append(labels_file_list[i])

        # Start processes
        for i in range(num_cpus):
            if not file_PSG_assignment[i] == []:
                process = mp.Process(target=process_dispatch, args=(args, file_PSG_assignment[i], file_labels_assignment[i], channels_to_read, data_type, result_queue))
                processes.append(process)
                process.start()

        # Collect results
        children_outputs = [result_queue.get() for _ in range(len(processes))] # Need to do this before .join() because processes don't exit until their data has been .get() from the queue

        # Wait for children to die
        for process in processes:
            process.join()

        # Concatenate nights from all processes
        if (HISTORICAL_LOOKBACK_LENGTH > 0): output_all_nights = {key: None for key in channels_to_read + ["sleep_stage"] + ["pseudo_random"] + [f"history_{HISTORICAL_LOOKBACK_LENGTH}-steps"] + ["new_night_marker"]}
        else: output_all_nights = {key: None for key in channels_to_read + ["sleep_stage"] + ["pseudo_random"] + ["new_night_marker"]}

        for output in children_outputs:
            if output == -1: continue # No files processed
            output_all_nights = insert_into_all_night_dict(output_all_nights, output)

    else: # No multiprocessing
        if HISTORICAL_LOOKBACK_LENGTH > 0: output_all_nights = {key: None for key in channels_to_read + ["sleep_stage"] + ["pseudo_random"] + ["new_night_marker"] + [f"history_{HISTORICAL_LOOKBACK_LENGTH}-steps"] + ["new_night_marker"]}
        else: output_all_nights = {key: None for key in channels_to_read + ["sleep_stage"] + ["pseudo_random"] + ["new_night_marker"] + ["new_night_marker"]}

        for PSG_file, labels_file in zip(PSG_file_list[0:args.num_files], labels_file_list[0:args.num_files]):
            print(f"[{(time.time()-start_time):.2f}s] Processing: {os.path.basename(PSG_file)} with {os.path.basename(labels_file)}")

            output_one_night = read_single_whole_night(args, psg_filepath=PSG_file, annotations_filepath=labels_file, channels_to_read=channels_to_read, data_type=data_type)
            if output_one_night == -1:
                print(f"""Received failure code {output_one_night} from read_single_whole_night. Skipping night.\n
                    PSG_file: {PSG_file}\n
                    sleep_stage_file: {labels_file}\n
                    channels_to_read: {channels_to_read}\n
                    clip_length_s: {args.clip_length_s}\n
                    data_type: {data_type}\n
                    multiprocessing: {args.enable_multiprocessing}""")
                continue #Skip this file

            # Indicate first clip of the night
            new_file_marker = [False for _ in range(len(output_one_night["sleep_stage"]))]
            new_file_marker[0] = True
            output_one_night["end_night_marker"] = tf.convert_to_tensor(new_file_marker, dtype=data_type, name="new_night_marker")
            output_one_night["end_night_marker"] = tf.expand_dims(output_one_night["new_night_marker"], axis=1)

            output_all_nights = insert_into_all_night_dict(output_all_nights, output_one_night)

    return output_all_nights, labels_file_list, 0

def parse_arguments():
    """"
    Parses command line arguments and return parser object
    """

    # Parser
    parser = ArgumentParser(description='Script to extract data from EDF files and export to a Tensorflow dataset.')
    parser.add_argument('--type', help='Type of generation: "EDF" (read PSG files and format their data) or "pseudo_random" (generate pseudo-random data for each sleep stage and equal number of sleep stages in dataset). Defaults to "EDF".', choices=["EDF", "pseudo_random"], default='EDF')
    parser.add_argument('--clip_length_s', help='Clip length (in sec). Must be one of 3.25, 5, 7.5, 10, 15, 30.', choices=["3.25", "7.5", "15", "30"], default=30, type=str)
    parser.add_argument('--directory_psg', help='Directory from which to fetch the PSG EDF files. Searches current workding directory by default.', default="")
    parser.add_argument('--directory_labels', help='Directory from which to fetch the PSG label files. Defaults to --directory_psg.', default="")
    parser.add_argument('--export_directory', help='Location to export dataset. Defaults to cwd.', default="")
    parser.add_argument('--num_files', help='Number of files to parse. Parsed in alphabetical order. Defaults to 5 files.', type=int, default=5)
    parser.add_argument('--enable_multiprocessing', help='Enables multiprocessing of data. Defaults to False if argument unused.', action='store_true')
    parser.add_argument('--downsampling_freq_hz', help='Desired sampling frequency (Hz) at which to downsample. Frequencies that are not an integer multiple of the original frequency result in \
                        interpolated samples.', type=int, default=NOMINAL_FREQUENCY_HZ)

    # Parse arguments
    args = parser.parse_args()
    if args.directory_psg == "": args.directory_psg = os.getcwd()
    if args.directory_labels == "": args.directory_labels = args.directory_psg
    if args.export_directory == "": args.export_directory = os.getcwd()
    print(f"[{(time.time()-start_time):.2f}s] Arguments loaded. Started dataset generation.")

    return args

def save_metadata_json(json_fp:str, args, channels_to_read:list):
    """
    Make and save .json file containing metadata about the dataset.
    """

    json_metadata = {
        "single_channel": (len(channels_to_read) == 1),
        "historical_lookback_length": HISTORICAL_LOOKBACK_LENGTH,
        "num_stages": NUM_SLEEP_STAGES,
        "one_hot_encoding": ONE_HOT_OUTPUT,
        "clip_length_s": args.clip_length_s,
        "sampling_freq_Hz": args.downsampling_freq_hz,
        "type": args.type
    }

    with open(json_fp, 'w') as json_file:
        json.dump(json_metadata, json_file, indent=4)

def main():
    # Parse arguments
    args = parse_arguments()

    # Generate data dictionary
    if args.type == "pseudo_random":
        print(f"Will generate {NUM_PSEUDO_RANDOM_CLIP_PER_SLEEP_STAGE} pseudo-random clips per sleep stage.")
        output, return_code = pseudo_random_dataset(num_clips_per_sleep_stage=NUM_PSEUDO_RANDOM_CLIP_PER_SLEEP_STAGE, clip_length_num_samples=NOMINAL_FREQUENCY_HZ*float(args.clip_length_s), data_type=tf.float32)

    elif args.type == "EDF":
        # Load data
        channels_to_read = ["EEG C4-LER", "EEG Fp1-LER", "EEG Cz-LER", "EEG Fz-LER", "EEG F4-LER"]

        print(f"Will read the following channels: {channels_to_read}")
        print(f"PSG file directory: {args.directory_psg}")
        print(f"Labels file directory: {args.directory_labels}")
        print(f"Output directory: {args.export_directory}")
        output, labels_file_list, return_code = read_all_nights_from_directory(args, channels_to_read=channels_to_read, data_type=tf.float32)

    # Create and save dataset
    if return_code == 0:
        ds = tf.data.Dataset.from_tensors(output)

        # Make dataset filepath
        if args.type == 'pseudo_random':
            ds_filepath = f"{args.export_directory}/Pseudo_Random_Tensorized_{NUM_SLEEP_STAGES}-stg_{args.clip_length_s}s"
        elif "filter" in os.path.basename(args.directory_psg):
            ds_filepath = f"{args.export_directory}/SS3_EDF_filtered_Tensorized_{NUM_SLEEP_STAGES}-stg_{args.clip_length_s}s"
        else:
            ds_filepath = f"{args.export_directory}/SS3_EDF_Tensorized_{NUM_SLEEP_STAGES}-stg_{args.clip_length_s}s"

        ds_filepath = ds_filepath + f'_{args.downsampling_freq_hz}Hz'
        if ONE_HOT_OUTPUT: ds_filepath = ds_filepath + '_one-hot'
        if HISTORICAL_LOOKBACK_LENGTH > 0: ds_filepath = ds_filepath + f'_history_{HISTORICAL_LOOKBACK_LENGTH}-steps'
        if len(channels_to_read) == 1: ds_filepath = ds_filepath + f"_{channels_to_read[0]}"
        if args.num_files == 1: ds_filepath = ds_filepath + f"_{os.path.basename(labels_file_list[0]).split(' ')[0]}"
        ds_filepath = ds_filepath.replace('.', '-')

        # Save metadata JSON
        save_metadata_json(json_fp=ds_filepath+".json", args=args, channels_to_read=channels_to_read)

        # Save dataset
        tf.data.Dataset.save(ds, compression=None, path=ds_filepath)

        print(f"[{(time.time()-start_time):.2f}s] Dataset saved at: {ds_filepath}. It contains {output['sleep_stage'].shape[0]} clips.")
    
    else:
        print(f"[{(time.time()-start_time):.2f}s] Could not generate dataset! Error return code: {return_code}. Nothing will be saved.")

if __name__ == "__main__":
    main()
