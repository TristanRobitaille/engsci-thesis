"""
Methods to extract data from EDF files

Arguments:
    --clip_length_s: Clip length (in sec). Must be one of 3.25, 5, 7.5, 10, 15, 30. Default = 30s.
    --directory: Directory from which to fetch EDF files. Searches current workding directory by default. Default = current directory.
    --num_files: Number of files to parse. Parsed in alphabetical order. Defaults = 10 files.
"""

from pyedflib import EdfReader
from pkg_resources import get_distribution
from os import getcwd, path
from glob import glob
from tensorflow import Tensor, DType, float32, uint16
from tensorflow import convert_to_tensor, zeros, concat, reshape, expand_dims, cast, convert_to_tensor
from tensorflow import data
from typing import List
from argparse import ArgumentParser

SLEEP_STAGE_RESOLUTION_SEC = 30.0 #Nominal (i.e. in EDF file) length or each clip/sleep stage annotation
NOMINAL_FREQUENCY_HZ = 256 #Sampling frequency for most channels
HISTORICAL_LOOKBACK_LENGTH = 16 #We save the last HISTORICAL_LOOKBACK_LENGTH - 1 sleep stages  

def signals_processing(signals:Tensor) -> Tensor:
    """
    Can apply data processing before exporting data.
    Returns input tensor with processing applied.
    """
    # Shift up signal to remove negative values
    signals += cast(2**15, dtype=signals.dtype)
    return signals

def read_single_whole_night(psg_filepath:str, annotations_filepath:str, channels_to_read:List[str],
                            clip_duration_sec:float32=SLEEP_STAGE_RESOLUTION_SEC, historical_lookback_length=HISTORICAL_LOOKBACK_LENGTH,
                            data_type:DType=float32) -> (Tensor, Tensor, List[str]):
    """
    Returns a tuple of tensors and list with each channel cut into clips: (signals_tensor, sleep_stage_tensor, channel_names_list) Measurements are converted into their digital ADC code, and cast to data_type.

    Dimensions:
        -signals_tensor: (num_clips, num_channels, num_samples_per_clip)
        -sleep_stage_tensor: (num_clips, 1, historical_sample_length)
        -channel_names_list: 1D list
    where num_clips = signal_recording_length//sleep_stage_resolution_sec and num_samples_per_clip = measurement_freq*sleep_stage_resolution_sec

    Assumptions:
        -Each channel is sampled at the same frequency, and is the same duration
        -Clip duration, and therefore annotation resolution, is 30s
    """

    SLEEP_STAGE_ANNOTATONS_CHANNEL = 2 #Channel of sleep stages in annotations file

    sleep_stage_unknown = "Sleep stage ?"
    sleep_stage_annotation_to_int = { #Note: Stages 3 and 4 are combined and '0' is reserved to be a padding mask
                                     "Sleep stage 1": 1,
                                     "Sleep stage 2": 2,
                                     "Sleep stage 3": 3,
                                     "Sleep stage 4": 3,
                                     "Sleep stage R": 4,
                                     "Sleep stage W": 5,}

    # Load data
    ratio_sleep_stage_resolution_to_clip_length = int(SLEEP_STAGE_RESOLUTION_SEC/clip_duration_sec)
    signal_reader = EdfReader(psg_filepath)
    sleep_stages  = list(EdfReader(annotations_filepath).readAnnotations()[SLEEP_STAGE_ANNOTATONS_CHANNEL])
    channel_labels = list(signal_reader.getSignalLabels())

    clip_duration_samples = int(clip_duration_sec * signal_reader.getSampleFrequency(0)) #Duration of a clip in number of samples
    recording_duration_s = signal_reader.getFileDuration()

    number_clips = min(int(recording_duration_s // clip_duration_sec), int(len(sleep_stages)*ratio_sleep_stage_resolution_to_clip_length)) #Sometimes number of sleep stages in Base file and clips in PSG file don't match
    if number_clips == 0: raise Exception(f"Insufficient data in file ({psg_filepath})! File duration: {recording_duration_s}s.")

    valid_clips = list(range(number_clips)) #Clips are invalid when their sleep stage is unknown

    # Construct tensors
    channel_names_list = []
    output_tensor = zeros(shape=(1, 0, clip_duration_samples), dtype=data_type)

    # Prune unknown sleep stages and construct list of valid clip numbers
    for sleep_stage_number in range(len(sleep_stages)):
        if sleep_stages[sleep_stage_number] == sleep_stage_unknown:
            for i in range(ratio_sleep_stage_resolution_to_clip_length):
                valid_clips.remove(ratio_sleep_stage_resolution_to_clip_length*sleep_stage_number + i)
            continue
        
    sleep_stages = [sleep_stage_annotation_to_int[item] for item in sleep_stages if item != sleep_stage_unknown] #Convert to int and prune unknown values
    sleep_stages = [item for item in sleep_stages for _ in range(ratio_sleep_stage_resolution_to_clip_length)]
        
    # Measurements
    first_channel = 1
    sleep_stages_with_lookback = list()
    for channel_name in channels_to_read: #Iterate over each signal in measurement file
        if channel_name not in channel_labels:
            print(f"Requested channel ({channel_name}) not in available channels!")
            continue

        channel_number = channel_labels.index(channel_name)

        if (signal_reader.getSampleFrequency(channel_number) != NOMINAL_FREQUENCY_HZ): #Simply skip if channel frequency doesn't match others
            print(f"Sampling frequency ({signal_reader.getSampleFrequency(channel_number)}Hz) for channel '{channel_name}' does not match nominal frequency ({NOMINAL_FREQUENCY_HZ}Hz)!")
            continue

        signal_tensor = zeros(shape=(0, clip_duration_samples), dtype=data_type)
        channel_names_list.append(channel_name)

        #Iterate over each clip in the signal
        clip_counter = 0
        for clip_number in valid_clips:
            clip_samples = signal_reader.readSignal(channel_number, start=clip_duration_samples*clip_number, n=clip_duration_samples, digital=True)
            clip_samples = signals_processing(clip_samples)
            clip_samples = reshape(convert_to_tensor(clip_samples), (1, clip_duration_samples))
            clip_samples = cast(clip_samples, dtype=data_type)
            signal_tensor = concat([signal_tensor, clip_samples], axis=0) #Stack vertically

            if first_channel:
                #Sleep stages -> This gives a 2D list with shape (len(valid_clips), historical_lookback_length)
                local_sleep_stage_historical_lookback = []
                for i in range(0, historical_lookback_length): #Append historical lookback to sleep stage tensor
                    if i > clip_counter: local_sleep_stage_historical_lookback.append(0)
                    else: local_sleep_stage_historical_lookback.append(sleep_stages[clip_counter-i])

                sleep_stages_with_lookback.append(local_sleep_stage_historical_lookback)
                clip_counter += 1

                if (len(sleep_stages_with_lookback) > clip_counter):
                    print('here')

        signal_tensor = expand_dims(signal_tensor, axis=1)
        if first_channel:
            output_tensor = reshape(output_tensor, shape=(signal_tensor.shape[0], 0, signal_tensor.shape[2]))
            first_channel = 0
        output_tensor = concat([output_tensor, signal_tensor], axis=1) #Stack horizontally with output tensor

    sleep_stage_tensor = convert_to_tensor(sleep_stages_with_lookback, dtype=data_type)
    sleep_stage_tensor = expand_dims(sleep_stage_tensor, axis=1)

    return (output_tensor, sleep_stage_tensor, channel_names_list)

def read_all_nights_from_directory(directory_filepath:str, channels_to_read:List[str], num_files:int=-1, clip_duration_sec:float32=SLEEP_STAGE_RESOLUTION_SEC, data_type:DType=float32) -> (Tensor, Tensor, List[str], int):
    """
    Calls read_single_whole_night() for all files in folder and returns concatenated versions of each output tensor along with a success flag

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
    if len(PSG_file_list) == 0:
        print(f"No valid PSG (*PSG.edf) files found in search directory ({directory_filepath})!")
        return 0, 0, 0, -1

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

    return output_signals_tensor, output_sleep_stages, signal_channel_names, 0

def main():
    # Parser
    parser = ArgumentParser(description='Script to extract data from EDF files and export to a Tensorflow dataset.')
    parser.add_argument('--clip_length_s', help='Clip length (in sec). Must be one of 3.25, 5, 7.5, 10, 15, 30.', default='30')
    parser.add_argument('--directory', help='Directory from which to fetch EDF files. Searches current workding directory by default.', default="")
    parser.add_argument('--num_files', help='Number of files to parse. Parsed in alphabetical order. Defaults to 10 files.', default=10)
    args = parser.parse_args()

    if not float(args.clip_length_s) in [3.25, 7.5, 15, 30]:
        print(f"Requested clip length ({float(args.clip_length_s):.2f}) not one of [3.25, 5, 7.5, 10, 15, 30]. Exiting program.")
        return 

    # Load data
    channels_to_read = ["EEG Pz-LER", "EEG T6-LER", "EEG Fp1-LER", "EEG T3-LER", "EEG Cz-LER"]
    if args.directory == "": directory = getcwd()
    else: directory = args.directory

    output_signals_tensor, output_sleep_stages, signal_channel_names, return_code = read_all_nights_from_directory(directory, channels_to_read=channels_to_read, num_files=int(args.num_files), clip_duration_sec=float(args.clip_length_s), data_type=uint16)

    # Create and save dataset
    if return_code == 0:
        ds = data.Dataset.from_tensors((output_signals_tensor, output_sleep_stages, signal_channel_names))
        ds_filepath = f"/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/SS3_EDF_Tensorized_{args.clip_length_s}s"
        ds_filepath = ds_filepath.replace('.', '-')

        if get_distribution("tensorflow").version == '2.8.0+computecanada': #Tensorflow 2.8.0 does not support .save()
            data.experimental.save(ds, compression=None, path=ds_filepath)
        else:
            data.Dataset.save(ds, compression=None, path=ds_filepath)

        print(f"Dataset saved at: {getcwd()}/{ds_filepath}")

if __name__ == "__main__":
    main()
