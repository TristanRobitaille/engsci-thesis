#!/bin/bash

python3 /home/trobitaille/engsci-thesis/python_prototype/edf_extract.py \
--type=EDF \
--clip_length_s=30 \
--num_files=100 \
--downsampling_freq_hz=256 \
--sleep_map_name=both_light_deep_combine \
--directory_psg="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF" \
--directory_labels="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF" \
--export_directory="/mnt/data/tristan/engsci_thesis_python_prototype_data"
--enable_multiprocessing \