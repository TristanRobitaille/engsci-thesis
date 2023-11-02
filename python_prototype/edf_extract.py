"""
Methods to extract data from EDF files

Arguments:
    --clip_length_s: Clip length (in sec). Must be one of 3.25, 5, 7.5, 10, 15, 30. Default = 30s.
    --directory_psg: Directory from which to fetch EDF files. Searches current workding directory by default. Default = current directory.
    --directory_labels: Directory from which to fetch sleep stage files. Searches current workding directory by default. Default = current directory.
    --num_files: Number of files to parse. Parsed in alphabetical order. Defaults = 10 files.
    --equal_num_sleep_stages: If True, exports the same number of example tensors for each sleep stage to avoid bias in dataset. Defaults to False
"""

from pyedflib import EdfReader
from pkg_resources import get_distribution
from os import getcwd, path
from glob import glob
from tensorflow import Tensor, DType, float32, uint16
from tensorflow import convert_to_tensor, zeros, concat, reshape, expand_dims, cast, convert_to_tensor, transpose
from tensorflow import data
from typing import List
from argparse import ArgumentParser
from math import floor
from copy import copy

import matplotlib.pyplot as plt

SLEEP_STAGE_RESOLUTION_SEC = 30.0 #Nominal (i.e. in EDF file) length or each clip/sleep stage annotation
NOMINAL_FREQUENCY_HZ = 256 #Sampling frequency for most channels
HISTORICAL_LOOKBACK_LENGTH = 16 #We save the last HISTORICAL_LOOKBACK_LENGTH - 1 sleep stages  
SLEEP_STAGE_ANNOTATONS_CHANNEL = 2 #Channel of sleep stages in annotations file
NUM_SLEEP_STAGES = 5 #Excluding 'unknown'
ONE_HOT_OUTPUT = False #If true, sleep stages are exported as their one-hot classes tensor, else they are reported as a scalar
EQUAL_NUMBER_OF_SLEEP_STAGES = True #If true, we output the same number of each sleep stages (simply delete clips from stages s.t. that number of clips for each stage are equal)

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

def read_single_whole_night(psg_filepath:str, annotations_filepath:str, channels_to_read:List[str],
                            clip_duration_sec:float32=SLEEP_STAGE_RESOLUTION_SEC, historical_lookback_length=HISTORICAL_LOOKBACK_LENGTH,
                            equal_num_sleep_stages:bool=False, data_type:DType=float32) -> (Tensor, Tensor, List[str]):
    """
    Returns a tuple of tensors and list with each channel cut into clips: (signals_tensor, sleep_stage_tensor, channel_names_list) Measurements are converted into their digital ADC code, and cast to data_type.

    Dimensions:
        -signals_tensor: (num_clips, num_channels, num_samples_per_clip)
        -sleep_stage_tensor:
            [if ONE_HOT_OUTPUT] (num_clips, num_sleep_stages+1, historical_sample_length)
            [if not ONE_HOT_OUTPUT] (num_clips, 1, historical_sample_length)
        -channel_names_list: 1D list
    where num_clips = signal_recording_length//sleep_stage_resolution_sec and num_samples_per_clip = measurement_freq*sleep_stage_resolution_sec

    Assumptions:
        -Each channel is sampled at the same frequency, and is the same duration
        -Clip duration, and therefore annotation resolution, is 30s
    """

    sleep_stage_annotation_to_int = { #Note: Stages 3 and 4 are combined and '0' is reserved to be a padding mask
                                     "Sleep stage 1": 1,
                                     "Sleep stage 2": 2,
                                     "Sleep stage 3": 3,
                                     "Sleep stage 4": 3,
                                     "Sleep stage R": 4,
                                     "Sleep stage W": 5,
                                     "Sleep stage ?": -1}
    sleep_stage_unknown = -1

    # Load data
    sleep_stages = list(EdfReader(annotations_filepath).readAnnotations()[SLEEP_STAGE_ANNOTATONS_CHANNEL])
    sleep_stages = [sleep_stage_annotation_to_int[item] for item in sleep_stages]
    signal_reader = EdfReader(psg_filepath)
    signals = list()
    channels = list(signal_reader.getSignalLabels())

    total_raw_clips = min(floor(signal_reader.getFileDuration() / clip_duration_sec), int(SLEEP_STAGE_RESOLUTION_SEC/clip_duration_sec)*len(sleep_stages))
    clip_duration_samples = int(clip_duration_sec * NOMINAL_FREQUENCY_HZ)

    # Make list (channels) of list of clips for signals
    for channel in channels_to_read:
        if channel not in signal_reader.getSignalLabels():
            print(f"Channel '{channel}' not found in file '{psg_filepath}'! Aborting.")
            return -1
        
        temp_clip = list()
        channel_number = channels.index(channel)

        for clip_number in range(total_raw_clips): #Split measurement list into clips of clip_duration_sec duration
            measurement = signal_reader.readSignal(channel_number, start=clip_duration_samples*clip_number, n=clip_duration_samples, digital=True)
            measurement = signals_processing(measurement)
            temp_clip.append(measurement)

        signals.append(temp_clip)

    # Duplicate sleep stages to account for stages being shorter than the nominal 30s
    sleep_stages = [item for item in sleep_stages for _ in range(int(SLEEP_STAGE_RESOLUTION_SEC/clip_duration_sec))]
    sleep_stages = sleep_stages[0:total_raw_clips] #On some files, sleep stages are longer than PSG signals so we slice it. Assume that they line up at the lowest index.

    # Prune unknown sleep stages from all lists
    clips_to_remove = []
    for sleep_stage_number in range(len(sleep_stages)):
        if (sleep_stages[sleep_stage_number] == sleep_stage_unknown): clips_to_remove.append(sleep_stage_number)
            
    clips_to_remove.sort(reverse=True)

    for clip_number in clips_to_remove:
        sleep_stages.pop(clip_number)
        for channel_signals in signals:
            channel_signals.pop(clip_number)

    # Output equal number of sleep stages
    sleep_stage_count = [0, 0, 0, 0, 0, 0]
    if equal_num_sleep_stages:
        for sleep_stage_number in range(len(sleep_stages)):
            sleep_stage_count[sleep_stages[sleep_stage_number]] += 1
        
        minimum_clips_over_classes = min(sleep_stage_count[1:-1])

        for clip_number in range(len(sleep_stages)-1, -1, -1):
            if sleep_stage_count[sleep_stages[clip_number]] > minimum_clips_over_classes:
                sleep_stage_count[sleep_stages[clip_number]] -= 1
                for channel in range(len(channels_to_read)):
                    signals[channel].pop(clip_number)
                sleep_stages.pop(clip_number)

    # Create historical lookbacks
    sleep_stages_with_history = copy(sleep_stages)
    for clip_number in range(HISTORICAL_LOOKBACK_LENGTH, len(sleep_stages)):
        temp_lookback = list()
        for i in range(HISTORICAL_LOOKBACK_LENGTH):
            if ONE_HOT_OUTPUT: temp_lookback.append(scalar_to_one_hot(sleep_stages[clip_number-i]))
            else: temp_lookback.append(sleep_stages[clip_number-i])

        sleep_stages_with_history[clip_number] = temp_lookback

    # Remove historical lookback length from start of list to avoid having to pad
    sleep_stages_with_history = sleep_stages_with_history[HISTORICAL_LOOKBACK_LENGTH:-1]
    for channel_number in range(len(channels_to_read)): signals[channel_number] = signals[channel_number][HISTORICAL_LOOKBACK_LENGTH:-1]

    # Convert to tensors
    sleep_stages = convert_to_tensor(sleep_stages_with_history, dtype=data_type)
    if ONE_HOT_OUTPUT: sleep_stages = transpose(sleep_stages, perm=[0, 2, 1])
    else: sleep_stages = expand_dims(sleep_stages, axis=1)

    signals = convert_to_tensor(signals, dtype=data_type)
    signals = transpose(signals, perm=[1, 0, 2])

    return (signals, sleep_stages, channels_to_read)

def read_all_nights_from_directory(directory_psg:str, directory_labels:str, channels_to_read:List[str], num_files:int=-1, 
                                   clip_duration_sec:float32=SLEEP_STAGE_RESOLUTION_SEC, 
                                   equal_num_sleep_stages:bool=False, data_type:DType=float32) -> (Tensor, Tensor, List[str]):
    """
    Calls read_single_whole_night() for all files in folder and returns concatenated versions of each output tensor along with a success flag

    Dimensions:
        -signals_tensor: (total_num_clips, num_channels, num_samples_per_clip)
        -sleep_stage_tensor: (total_num_clips, historical_sample_length, num_sleep_stages+1)
        -channel_names_list: 1D list
    where total_num_clips = sum(num_clips for each night) and num_samples_per_clip = measurement_freq*SLEEP_STAGE_RESOLUTION_SEC

    Assumptions:
        -All equivalent channels have the same labels across all nights
        -All PSG files have the same number of channels
    """

    # Get filenames (assume base of filenames for PSG and Labels match)
    PSG_file_list = glob(path.join(directory_psg, "*PSG.edf"))
    labels_file_list = glob(path.join(directory_labels, "*Base.edf"))
    if len(PSG_file_list) == 0:
        print(f"No valid PSG (*PSG.edf) files found in search directory ({directory_psg})!")
        return 0, 0, 0, -1
    if len(labels_file_list) == 0:
        print(f"No valid sleep stages (*Base.edf) files found in search directory ({directory_labels})!")
        return 0, 0, 0, -1

    output_signals_tensor = zeros(shape=(0, 0, 0), dtype=data_type)
    output_sleep_stages = zeros(shape=(0, 0, 0), dtype=data_type)
    signal_channel_names = []

    file_cnt = 0
    for PSG_file in PSG_file_list:
        print(f"Processing: {PSG_file}")

        sleep_stage_file = labels_file_list[file_cnt]

        output = read_single_whole_night(PSG_file, sleep_stage_file, channels_to_read, clip_duration_sec, equal_num_sleep_stages=equal_num_sleep_stages, data_type=data_type)
        if output == -1:
            print(f"""Received failure code {output} from read_single_whole_night. Skipping night.\n
                PSG_file: {PSG_file}\n
                sleep_stage_file: {sleep_stage_file}\n
                channels_to_read: {channels_to_read}\n
                clip_duration_sec: {clip_duration_sec}\n
                equal_num_sleep_stages: {equal_num_sleep_stages}\n
                data_type: {data_type}""")
            continue
        else: new_signals_tensor, new_sleep_stages_tensor, new_signal_labels = output

        if output_signals_tensor.shape == (0, 0, 0): #Inherit num_channels, num_samples_per_clip and channel label from first tensor
            output_signals_tensor = reshape(output_signals_tensor, shape=(0, new_signals_tensor.shape[1], new_signals_tensor.shape[2]))
            output_sleep_stages =   reshape(output_signals_tensor, shape=(0, new_sleep_stages_tensor.shape[1], new_sleep_stages_tensor.shape[2]))
            signal_channel_names = new_signal_labels

        output_signals_tensor = concat([output_signals_tensor, new_signals_tensor], axis=0)
        output_sleep_stages =   concat([output_sleep_stages, new_sleep_stages_tensor], axis=0)

        if num_files != -1:
            file_cnt += 1
            if file_cnt == num_files:
                break

    return output_signals_tensor, output_sleep_stages, signal_channel_names, 0

def main():
    # Parser
    parser = ArgumentParser(description='Script to extract data from EDF files and export to a Tensorflow dataset.')
    parser.add_argument('--clip_length_s', help='Clip length (in sec). Must be one of 3.25, 5, 7.5, 10, 15, 30.', default='30')
    parser.add_argument('--directory_psg', help='Directory from which to fetch the PSG EDF files. Searches current workding directory by default.', default="")
    parser.add_argument('--directory_labels', help='Directory from which to fetch the PSG label files. Searches current workding directory by default.', default="")
    parser.add_argument('--num_files', help='Number of files to parse. Parsed in alphabetical order. Defaults to 10 files.', default=10)
    parser.add_argument('--equal_num_sleep_stages', help='If True, exports the same number of example tensors for each sleep stage to avoid bias in dataset. Defaults to False', default="False")
    args = parser.parse_args()

    if not float(args.clip_length_s) in [3.25, 7.5, 15, 30]:
        print(f"Requested clip length ({float(args.clip_length_s):.2f}) not one of [3.25, 5, 7.5, 10, 15, 30]. Exiting program.")
        return

    equal_num_sleep_stages = (args.equal_num_sleep_stages.lower() == "true") or (args.equal_num_sleep_stages.lower() == "1")

    # Load data
    channels_to_read = ["EEG Pz-LER", "EEG T6-LER", "EEG Fp1-LER", "EEG T3-LER", "EEG Cz-LER"]
    if args.directory_psg == "": directory_psg = getcwd()
    else: directory_psg = args.directory_psg
    if args.directory_labels == "": directory_labels = getcwd()
    else: directory_labels = args.directory_labels

    output_signals_tensor, output_sleep_stages, signal_channel_names, return_code = read_all_nights_from_directory(directory_psg, directory_labels, channels_to_read=channels_to_read, num_files=int(args.num_files), clip_duration_sec=float(args.clip_length_s),
                                                                                                                    equal_num_sleep_stages=equal_num_sleep_stages, data_type=uint16)

    # Create and save dataset
    if return_code == 0:
        ds = data.Dataset.from_tensors((output_signals_tensor, output_sleep_stages, signal_channel_names))
        if args.directory_psg == "/home/tristanr/projects/def-xilinliu/data/SS3_EDF":
            ds_filepath = f"home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data/SS3_EDF_Tensorized_{args.clip_length_s}s"
        elif args.directory_psg == "/home/tristanr/projects/def-xilinliu/data/SS3_filter_new":
            ds_filepath = f"home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data/SS3_EDF_filtered_Tensorized_{args.clip_length_s}s"
        if ONE_HOT_OUTPUT: ds_filepath = ds_filepath + '_one-hot'
        if equal_num_sleep_stages: ds_filepath = ds_filepath + '_equal-sleep-stages'
        ds_filepath = ds_filepath.replace('.', '-')

        if get_distribution("tensorflow").version == '2.8.0+computecanada': #Tensorflow 2.8.0 does not support .save()
            data.experimental.save(ds, compression=None, path=ds_filepath)
        else:
            data.Dataset.save(ds, compression=None, path=ds_filepath)

        print(f"Dataset saved at: {getcwd()}/{ds_filepath}")

if __name__ == "__main__":
    main()
