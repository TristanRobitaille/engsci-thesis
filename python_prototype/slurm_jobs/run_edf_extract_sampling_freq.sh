#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j_sampling_freq.out  # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load apptainer/1.2.4
apptainer run engsci-thesis.sif python python_prototype/main_vision_transformer.py \
--type=EDF \
--clip_length_s=30 \
--num_files=100 \
--sleep_map_name=both_light_deep_combine \
--enable_multiprocessing \
--sampling_freq_hz=128 \
--signal_processing_ops notch_60Hz 15b_offset 0_3Hz-100Hz_bandpass \
--directory_psg=/home/tristanr/projects/def-xilinliu/data/SS3_EDF \
--directory_labels=/home/tristanr/projects/def-xilinliu/data/SS3_EDF \
--export_directory=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data
