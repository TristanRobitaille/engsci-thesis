#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j_sampling_freq.out  # %N for node name, %j for jobID
#SBATCH --array=16,32,64,128,256,512,1024
#!/bin/bash

module load cuda cudnn
module load python/3
source ~/tensorflow/bin/activate

python3 /home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/edf_extract.py \
--type=EDF \
--clip_length_s=30 \
--num_files=100 \
--sleep_map_name=both_light_deep_combine \
--enable_multiprocessing \
--sampling_freq_hz=1024 \
--directory_psg=/home/tristanr/projects/def-xilinliu/data/SS3_EDF \
--directory_labels=/home/tristanr/projects/def-xilinliu/data/SS3_EDF \
--export_directory=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data