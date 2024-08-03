#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=64000M        # memory per node
#SBATCH --time=00-00:20     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load apptainer/1.2.4
apptainer run engsci-thesis.sif python python_prototype/edf_extract.py \
--type=EDF \
--clip_length_s=30 \
--num_files=100 \
--sleep_map_name="no_combine" \
--enable_multiprocessing \
--sampling_freq_hz=128 \
--signal_processing_ops 15b_offset \
--directory_psg="/home/tristanr/projects/def-xilinliu/data/SS3_EDF" \
--directory_labels="/home/tristanr/projects/def-xilinliu/data/SS3_EDF" \
--export_directory="/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data"
