"""
Methods to extract data from EDF files
"""

from pyedflib import EdfReader
from pkg_resources import get_distribution
from os import getcwd, path
from glob import glob
from pathlib import Path
from tensorflow import Tensor, DType, float32, uint16
from tensorflow import convert_to_tensor, zeros, concat, reshape, ones, expand_dims, add, reduce_min, reduce_max, cast
from tensorflow import data
from typing import List

CLIP_DURATION_SEC = 30.0 #Length or each clip/sleep stage annotation

def signals_processing(signals:Tensor) -> Tensor:
    """
    Can apply data processing before exporting data.
    Returns input tensor with processing applied.
    """
    # Shift up signal to remove negative values
    signals += cast(2**15, dtype=signals.dtype)
    return signals

def read_single_whole_night(psg_filepath:str, annotations_filepath:str, channels_to_read:List[str], clip_duration_sec:float32=CLIP_DURATION_SEC, data_type:DType=float32) -> (Tensor, Tensor, List[str]):
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

    sleep_stage_unknown = "Sleep stage ?"
    sleep_stage_annotation_to_int = { #Note: Stages 1 and 2 are combined
                                     "Sleep stage 1": 2,
                                     "Sleep stage 2": 2,
                                     "Sleep stage 3": 3,
                                     "Sleep stage R": 4,
                                     "Sleep stage W": 5}

    # Load data
    signal_reader = EdfReader(psg_filepath)
    sleep_stages  = EdfReader(annotations_filepath).readAnnotations()[SLEEP_STAGE_ANNOTATONS_CHANNEL]
    channel_labels = list(signal_reader.getSignalLabels())

    clip_duration_samples = int(clip_duration_sec * signal_reader.getSampleFrequency(0)) #Duration of a clip in number of samples
    recording_duration_s = signal_reader.getFileDuration()

    number_clips = min(int(recording_duration_s // clip_duration_sec), len(sleep_stages)) #Sometimes number of sleep stages in Base file and clips in PSG file don't match
    if number_clips == 0: raise Exception(f"Insufficient data in file ({psg_filepath})! File duration: {recording_duration_s}s.")

    valid_clips = list(range(number_clips)) #Clips are invalid when their sleep stage is unknown

    # Construct tensors
    channel_names_list = []
    output_tensor = zeros(shape=(1, 0, clip_duration_samples), dtype=data_type)

    # Sleep stages
    sleep_stage_tensor = zeros(shape=(0,1), dtype=data_type)
    for clip_number in range(number_clips): #Iterate over each clip in the signal
        if sleep_stages[clip_number] == sleep_stage_unknown:
            valid_clips.remove(clip_number)
            continue

        sleep_stage = sleep_stage_annotation_to_int[sleep_stages[clip_number]]
        sleep_stage = sleep_stage * ones(shape=(1,1), dtype=data_type)
        sleep_stage_tensor = concat([sleep_stage_tensor, sleep_stage], axis=0) #Stack vertically

    # Measurements
    first_channel = 1
    for channel_name in channels_to_read: #Iterate over each signal in measurement file
        if channel_name not in channel_labels:
            continue

        channel_number = channel_labels.index(channel_name)

        if (signal_reader.getSampleFrequency(channel_number) != signal_reader.getSampleFrequency(0)): #Simply skip if channel frequency doesn't match others
            continue

        signal_tensor = zeros(shape=(0, clip_duration_samples), dtype=data_type)
        channel_names_list.append(channel_name)

        #Iterate over each clip in the signal
        for clip_number in valid_clips:
            clip_samples = signal_reader.readSignal(channel_number, start=clip_duration_samples*clip_number, n=clip_duration_samples, digital=True)
            clip_samples = signals_processing(clip_samples)
            clip_samples = reshape(convert_to_tensor(clip_samples), (1, clip_duration_samples))
            clip_samples = cast(clip_samples, dtype=data_type)
            signal_tensor = concat([signal_tensor, clip_samples], axis=0) #Stack vertically

        signal_tensor = expand_dims(signal_tensor, axis=1)
        if first_channel:
            output_tensor = reshape(output_tensor, shape=(signal_tensor.shape[0], 0, signal_tensor.shape[2]))
            first_channel = 0
        output_tensor = concat([output_tensor, signal_tensor], axis=1) #Stack horizontally with output tensor

    # Processing
    sleep_stage_tensor = expand_dims(sleep_stage_tensor, axis=1)

    return (output_tensor, sleep_stage_tensor, channel_names_list)

def read_all_nights_from_directory(directory_filepath:str, channels_to_read:List[str], num_files:int=-1, clip_duration_sec:float32=CLIP_DURATION_SEC, data_type:DType=float32) -> (Tensor, Tensor, List[str]):
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
    PSG_file_list = glob(path.join(directory_filepath, "*PSG.edf"))

    output_signals_tensor = zeros(shape=(0, 0, 0), dtype=data_type)
    output_sleep_stages = zeros(shape=(0, 0, 0), dtype=data_type)
    signal_channel_names = []

    file_cnt = 0

    for PSG_file in PSG_file_list:
        print(f"Processing: {PSG_file}")

        sleep_stage_file = PSG_file.replace(" ", "\ ")
        sleep_stage_file = PSG_file.replace("PSG", "Base")

        (new_signals_tensor, new_sleep_stages_tensor, new_signal_labels) = read_single_whole_night(PSG_file, sleep_stage_file, channels_to_read, clip_duration_sec, data_type=data_type)

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

    return output_signals_tensor, output_sleep_stages, signal_channel_names

def main():
    channels_to_read = ["EEG F4-LER", "EOG Upper Vertic", "EMG Chin"]
    num_files = 10
    PSG_dir = "/home/tristanr/projects/def-xilinliu/data/SS3_EDF"

    output_signals_tensor, output_sleep_stages, signal_channel_names = read_all_nights_from_directory(PSG_dir, channels_to_read=channels_to_read, num_files=num_files, data_type=uint16)

    # Create and save dataset
    ds = data.Dataset.from_tensors((output_signals_tensor, output_sleep_stages, signal_channel_names))

    if get_distribution("tensorflow").version == '2.8.0+computecanada': #Tensorflow 2.8.0 does not support .save()
        data.experimental.save(ds, compression=None, path="SS3_EDF_Tensorized")
    else:
        data.Dataset.save(ds, compression=None, path="SS3_EDF_Tensorized")

    print(f"Dataset saved at: {getcwd() + 'SS3_EDF_Tensorized'}")

if __name__ == "__main__":
    main()
