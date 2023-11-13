#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#!/bin/bash

module load cuda cudnn 
source ~/tensorflow/bin/activate

python /home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/edf_extract.py \
--type=EDF --clip_length_s=30 --num_files=100 --directory_psg=/home/tristanr/projects/def-xilinliu/data/SS3_EDF \
--directory_labels=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data --export_directory=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data

python /home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/edf_extract.py \
--type=EDF --clip_length_s=15 --num_files=100 --directory_psg=/home/tristanr/projects/def-xilinliu/data/SS3_EDF \
--directory_labels=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data --export_directory=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data

python /home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/edf_extract.py \
--type=EDF --clip_length_s=7.5 --num_files=100 --directory_psg=/home/tristanr/projects/def-xilinliu/data/SS3_EDF \
--directory_labels=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data --export_directory=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data