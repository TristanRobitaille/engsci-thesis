#!/bin/bash

python3 /home/trobitaille/engsci-thesis/python_prototype/edf_extract.py \
--type=EDF \
--clip_length_s=30 \
--num_files=100 \
--sampling_freq_hz=128 \
--sleep_map_name=both_light_deep_combine \
--signal_processing_ops notch_60Hz 15b_offset 0_3Hz-100Hz_bandpass \
--directory_psg="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF" \
--directory_labels="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF" \
--export_directory="/mnt/data/tristan/engsci_thesis_python_prototype_data" \
--enable_multiprocessing