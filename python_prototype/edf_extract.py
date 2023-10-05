"""
Methods to extract data from EDF files
"""

import pyedflib
import os
from tensorflow import Tensor, DType, float32, int16
from tensorflow import convert_to_tensor, zeros, concat, reshape, ones, expand_dims
from tensorflow import data
from typing import List

CLIP_DURATION_SEC = 30.0 #Length or each clip/sleep stage annotation

def read_single_whole_night(psg_filepath:str, annotations_filepath:str, channels_to_read:List[int], clip_duration_sec:float32=CLIP_DURATION_SEC, data_type:DType=float32) -> (Tensor, Tensor, List[str]):
    """
    Returns a tuple of tensors and list with each channel cut into clips: (signals_tensor, sleep_stage_tensor, channel_names_list) Measurements are converted into their digital ADC code, and cast to data_type.

    Dimensions:
        -signals_tensor: (num_clips, num_channels, num_samples_per_clip)
        -sleep_stage_tensor: (num_clips, 1, 1)
        -channel_names_list: 1D list
    where num_clips = signal_recording_length//sleep_stage_resolution_sec and num_samples_per_clip = measurement_freq*sleep_stage_resolution_sec

    Assumptions:
        -Each channel is sampled at the same frequency, and is the same duration
        -Clip duration, and therefore annotation resolution, is 30s
    """
    
    SLEEP_STAGE_ANNOTATONS_CHANNEL = 2 #Channel of sleep stages in annotations file

    # What's available:
    # {'Sleep stage 1', 'Sleep stage ?', 'Sleep stage R', 'Sleep stage W', 'Sleep stage 3', 'Sleep stage 2'}

    sleep_stage_annotation_to_int_unsure_if_correct = {"Sleep stage ?": 0, #TODO: Make this correct
                                     "Sleep stage 1": 1,
                                     "Sleep stage 2": 2,
                                     "Sleep stage 3": 3, 
                                     "Sleep stage R": 4,
                                     "Sleep stage W": 5}
    
    # Load data
    signal_reader = pyedflib.EdfReader(psg_filepath)
    sleep_stages  = pyedflib.EdfReader(annotations_filepath).readAnnotations()[SLEEP_STAGE_ANNOTATONS_CHANNEL]

    clip_duration_samples = int(clip_duration_sec * signal_reader.getSampleFrequency(0)) #Duration of a clip in number of samples
    recording_duration_s = signal_reader.getFileDuration()

    number_clips = min(int(recording_duration_s // clip_duration_sec), len(sleep_stages)) #Sometimes number of sleep stages in Base file and clips in PSG file don't match
    if number_clips == 0: raise Exception(f"Insufficient data in file ({psg_filepath})! File duration: {recording_duration_s}s.")

    # Construct tensors
    channel_names_list = []
    output_tensor = zeros(shape=(number_clips, 0, clip_duration_samples), dtype=data_type)
    
    # Measurements
    for channel_number in channels_to_read: #Iterate over each signal in measurement file
        if (signal_reader.getSampleFrequency(channel_number) != signal_reader.getSampleFrequency(0)): #Simply skip if channel frequency doesn't match others
            continue

        signal_tensor = zeros(shape=(0, clip_duration_samples), dtype=data_type)
        channel_names_list.append(signal_reader.getLabel(channel_number))

        #Iterate over each clip in the signal
        for clip_number in range(number_clips):
            clip_samples = signal_reader.readSignal(channel_number, start=clip_duration_samples*clip_number, n=clip_duration_samples, digital=True)
            clip_samples = reshape(convert_to_tensor(clip_samples, dtype=data_type), (1, clip_duration_samples))
            signal_tensor = concat([signal_tensor, clip_samples], axis=0) #Stack vertically

        signal_tensor = expand_dims(signal_tensor, axis=1)
        output_tensor = concat([output_tensor, signal_tensor], axis=1) #Stack horizontally with output tensor

    # Sleep stages
    sleep_stage_tensor = zeros(shape=(0,1), dtype=data_type)
    for clip_number in range(number_clips): #Iterate over each clip in the signal
        sleep_stage = sleep_stage_annotation_to_int_unsure_if_correct[sleep_stages[clip_number]]
        sleep_stage = sleep_stage * ones(shape=(1,1), dtype=data_type)
        sleep_stage_tensor = concat([sleep_stage_tensor, sleep_stage], axis=0) #Stack vertically

    sleep_stage_tensor = expand_dims(sleep_stage_tensor, axis=1)

    return (output_tensor, sleep_stage_tensor, channel_names_list)

def read_all_nights_from_directory(directory_filepath:str, channels_to_read:List[int], clip_duration_sec:float32=CLIP_DURATION_SEC, data_type:DType=float32) -> (Tensor, Tensor, List[str]):
    """
    Calls read_single_whole_night() for all files in folder and returns concatenated versions of each output tensor

    Dimensions:
        -signals_tensor: (total_num_clips, num_channels, num_samples_per_clip)
        -sleep_stage_tensor: (total_num_clips, 1, 1)
        -channel_names_list: 1D list
    where total_num_clips = sum(num_clips for each night) and num_samples_per_clip = measurement_freq*SLEEP_STAGE_RESOLUTION_SEC

    Assumptions:
        -PSG and sleep stages files share the same base filename, but PSG filesare appended with "PSG" and sleep stages files are appended with "Base"
        -All equivalent channels have the same labels across all nights
        -All PSG files have the same number of channels
    """

    PSG_file_list = [f for f in os.listdir(directory_filepath) if (os.path.isfile(os.path.join(directory_filepath, f)) and (f.find(" PSG.edf") != -1))]
    
    output_signals_tensor = zeros(shape=(0, 0, 0), dtype=data_type)
    output_sleep_stages = zeros(shape=(0, 0, 0), dtype=data_type)
    signal_channel_names = []

    for PSG_file in PSG_file_list:
        sleep_stage_file = PSG_file.replace("PSG", "Base")
        (new_signals_tensor, new_sleep_stages_tensor, new_signal_labels) = read_single_whole_night(directory_filepath+PSG_file, directory_filepath+sleep_stage_file, channels_to_read, clip_duration_sec, data_type=data_type)

        if output_signals_tensor.shape == (0, 0, 0): #Inherit num_channels, num_samples_per_clip and channel label from first tensor
            output_signals_tensor = reshape(output_signals_tensor, shape=(0, new_signals_tensor.shape[1], new_signals_tensor.shape[2]))
            output_sleep_stages =   reshape(output_signals_tensor, shape=(0, new_sleep_stages_tensor.shape[1], new_sleep_stages_tensor.shape[2]))
            signal_channel_names = new_signal_labels

        output_signals_tensor = concat([output_signals_tensor, new_signals_tensor], axis=0)
        output_sleep_stages =   concat([output_sleep_stages, new_sleep_stages_tensor], axis=0)

    return output_signals_tensor, output_sleep_stages, signal_channel_names

def main():
    channels_to_read=[0,1]
    output_signals_tensor, output_sleep_stages, signal_channel_names = read_all_nights_from_directory("python_prototype/", channels_to_read=channels_to_read, data_type=int16)

    # Create dataset
    ds = data.Dataset.from_tensors((output_signals_tensor, output_sleep_stages, signal_channel_names))
    data.Dataset.save(ds, compression=None, path="test_dataset")

    ds_open = data.Dataset.load(compression=None, path="test_dataset")
    print(ds_open)

if __name__ == "__main__":
    main()