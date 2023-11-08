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
from tensorflow import Tensor, DType, float32
from tensorflow import convert_to_tensor, concat, expand_dims, convert_to_tensor, transpose, reshape
from tensorflow import data
from typing import List
from argparse import ArgumentParser
from math import floor

import tensorflow as tf
import numpy as np

SLEEP_STAGE_RESOLUTION_SEC = 30.0 #Nominal (i.e. in EDF file) length or each clip/sleep stage annotation
NOMINAL_FREQUENCY_HZ = 256 #Sampling frequency for most channels
HISTORICAL_LOOKBACK_LENGTH = 16 #We save the last HISTORICAL_LOOKBACK_LENGTH - 1 sleep stages  
SLEEP_STAGE_ANNOTATONS_CHANNEL = 2 #Channel of sleep stages in annotations file
NUM_PSEUDO_RANDOM_CLIP_PER_SLEEP_STAGE = 5000 #Number of pseudo-random clips to generate for each sleep stage
MAX_VOLTAGE = 2**15 - 1
MIN_VOLTAGE = 0
NUM_SLEEP_STAGES = 5 #Excluding 'unknown'
ONE_HOT_OUTPUT = False #If true, sleep stages are exported as their one-hot classes tensor, else they are reported as a scalar

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

def pseudo_random_dataset(num_clips_per_sleep_stage:int, clip_length_num_samples, data_type:DType=float32):
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
    output["pseudo_random"] = convert_to_tensor(value=output["pseudo_random"], dtype=data_type)
    output["sleep_stage"] = convert_to_tensor(value=output["sleep_stage"], dtype=data_type)
    output["sleep_stage"] = reshape(tensor=output["sleep_stage"], shape=(output["sleep_stage"].shape[0], 1), name="sleep_stage")

    return output, 0 # Return code == 0

def read_single_whole_night(psg_filepath:str, annotations_filepath:str, channels_to_read:List[str],
                            clip_duration_sec:float32=SLEEP_STAGE_RESOLUTION_SEC, historical_lookback_length=HISTORICAL_LOOKBACK_LENGTH,
                            equal_num_sleep_stages:bool=False, data_type:DType=float32) -> (Tensor, Tensor, List[str]):
    """
    Returns a dictionary of channels with signals cut into clips and a return code.

    Dimensions:
      -Dictionary value (one channel): (num_clips, num_samples_per_clip)
      -Sleep stages: (num_clips, 1)
    
    Assumptions:
        -Each channel is sampled at the same frequency, and is the same duration
        -Annotation resolution, is 30s
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
            print(f"Channel '{channel}' not found in file '{psg_filepath}'! Skipping file.")
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

    # Generate pseudo-random signal
    pseudo_random_list = list()
    for sleep_stage in sleep_stages:
        new_clip = pseudo_random_clip(sleep_stage, clip_duration_samples, max_min=(MAX_VOLTAGE, MIN_VOLTAGE, NUM_SLEEP_STAGES, 0))
        pseudo_random_list.append(new_clip)

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

    # Convert to tensors
    sleep_stages = convert_to_tensor(sleep_stages, dtype=data_type, name="sleep_stage")
    pseudo_random_list = convert_to_tensor(pseudo_random_list, dtype=data_type, name="pseudo_random")
    
    if ONE_HOT_OUTPUT: sleep_stages = transpose(sleep_stages, perm=[0, 2, 1])
    else: sleep_stages = expand_dims(sleep_stages, axis=1)

    for i in range(len(channels_to_read)):
        signals[i] = convert_to_tensor(signals[i], dtype=data_type, name=channels_to_read[i])      

    signals = dict(zip(channels_to_read, signals)) #Convert to dictionary
    signals["pseudo_random"] = pseudo_random_list
    signals["sleep_stage"] = sleep_stages

    return signals

def read_all_nights_from_directory(directory_psg:str, directory_labels:str, channels_to_read:List[str], num_files:int=-1, 
                                   clip_duration_sec:float32=SLEEP_STAGE_RESOLUTION_SEC, 
                                   equal_num_sleep_stages:bool=False, data_type:DType=float32) -> (Tensor, Tensor, List[str]):

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
    PSG_file_list = glob(path.join(directory_psg, "*PSG.edf"))
    labels_file_list = glob(path.join(directory_labels, "*Base.edf"))
    if len(PSG_file_list) == 0:
        print(f"No valid PSG (*PSG.edf) files found in search directory ({directory_psg})!")
        return -1
    if len(labels_file_list) == 0:
        print(f"No valid sleep stages (*Base.edf) files found in search directory ({directory_labels})!")
        return -1

    output = {key: None for key in channels_to_read + ["sleep_stage"] + ["pseudo_random"]}
    file_cnt = 0

    # Go through each file found in the input directory
    for PSG_file in PSG_file_list:
        print(f"Processing: {PSG_file}")

        sleep_stage_file = labels_file_list[file_cnt]

        output_one_night = read_single_whole_night(PSG_file, sleep_stage_file, channels_to_read, clip_duration_sec, equal_num_sleep_stages=equal_num_sleep_stages, data_type=data_type)
        if output == -1:
            continue #Skip this file

        for key, value in output_one_night.items():
            if output[key] == None:
                output[key] = output_one_night[key]
            else:
                output[key] = concat([output[key], output_one_night[key]], axis=0)

        if num_files != -1:
            file_cnt += 1
            if file_cnt == num_files: break

    return output, 0

def main():
    # Parser
    parser = ArgumentParser(description='Script to extract data from EDF files and export to a Tensorflow dataset.')
    parser.add_argument('--type', help='Type of generation: "EDF" (read PSG files and format their data) or "pseudo_random" (generate pseudo-random data for each sleep stage and equal number of sleep stages in dataset). Defaults to "EDF".', choices=["EDF", "pseudo_random"], default='EDF')
    parser.add_argument('--clip_length_s', help='Clip length (in sec). Must be one of 3.25, 5, 7.5, 10, 15, 30.', default=30, type=str)
    parser.add_argument('--directory_psg', help='Directory from which to fetch the PSG EDF files. Searches current workding directory by default.', default="")
    parser.add_argument('--directory_labels', help='Directory from which to fetch the PSG label files. Searches current workding directory by default.', default="")
    parser.add_argument('--num_files', help='Number of files to parse. Parsed in alphabetical order. Defaults to 10 files.', type=int, default=10)
    parser.add_argument('--equal_num_sleep_stages', help='If True, exports the same number of example tensors for each sleep stage to avoid bias in dataset. Defaults to False.', type=bool, default=False)
    parser.add_argument('--export_directory', help='Location to export dataset. Defaults to cwd.', default="")

    # Parse arguments
    args = parser.parse_args()

    if not float(args.clip_length_s) in [3.25, 7.5, 15, 30]:
        print(f"Requested clip length ({float(args.clip_length_s):.2f}) not one of [3.25, 5, 7.5, 10, 15, 30]. Exiting program.")
        return

    if args.directory_psg == "": args.directory_psg = getcwd()
    if args.directory_labels == "": args.directory_labels = getcwd()
    if args.export_directory == "": args.export_directory = getcwd()

    # Generate data dictionary
    if args.type == "pseudo_random":
        print(f"Will generate {NUM_PSEUDO_RANDOM_CLIP_PER_SLEEP_STAGE} pseudo-random clips per sleep stage.")
        output, return_code = pseudo_random_dataset(num_clips_per_sleep_stage=NUM_PSEUDO_RANDOM_CLIP_PER_SLEEP_STAGE, clip_length_num_samples=NOMINAL_FREQUENCY_HZ*float(args.clip_length_s), data_type=float32)

    elif args.type == "EDF":
        # Load data
        channels_to_read = ["EEG Pz-LER", "EEG T6-LER", "EEG Fp1-LER", "EEG T3-LER", "EEG Cz-LER"]
        print(f"Will read the following channels: {channels_to_read}")

        output, return_code = read_all_nights_from_directory(args.directory_psg, args.directory_labels, channels_to_read=channels_to_read, num_files=args.num_files, clip_duration_sec=float(args.clip_length_s),
                                                                                                                        equal_num_sleep_stages=args.equal_num_sleep_stages, data_type=float32)

    # Create and save dataset
    if return_code == 0:
        ds = data.Dataset.from_tensors(output)

        # Make dataset filepath
        if args.type == 'pseudo_random':
            ds_filepath = f"{args.export_directory}/Pseudo_Random_Tensorized_{args.clip_length_s}s"
        elif "filter" in path.basename(args.directory_psg):
            ds_filepath = f"{args.export_directory}/SS3_EDF_filtered_Tensorized_{args.clip_length_s}s"
        else:
            ds_filepath = f"{args.export_directory}/SS3_EDF_Tensorized_{args.clip_length_s}s"

        if ONE_HOT_OUTPUT: ds_filepath = ds_filepath + '_one-hot'
        if args.equal_num_sleep_stages: ds_filepath = ds_filepath + '_equal-sleep-stages'
        ds_filepath = ds_filepath.replace('.', '-')

        # Save dataset
        if get_distribution("tensorflow").version == '2.8.0+computecanada': #Tensorflow 2.8.0 does not support .save()
            data.experimental.save(ds, compression=None, path=ds_filepath)
        else:
            data.Dataset.save(ds, compression=None, path=ds_filepath)

        print(f"Dataset saved at: {ds_filepath}. It contains {output['sleep_stage'].shape[0]} clips.")
    
    else:
        print(f"Could not generate dataset! Error return code: {return_code}. Nothing will be saved.")

if __name__ == "__main__":
    main()
