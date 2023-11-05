#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=00-16:59      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load cuda cudnn 
source ~/tensorflow/bin/activate
tensorboard --logdir=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/logs/fit --host 0.0.0.0 --load_fast false &

# python edf_extract.py --directory /home/tristanr/projects/def-xilinliu/data/SS3_EDF --clip_length_s 15 --num_files 25
# python edf_extract.py --directory /home/tristanr/projects/def-xilinliu/data/SS3_EDF --clip_length_s 30 --num_files 25

python main.py --clip_length_s 15 --voltage_embedding_depth $1 --num_training_clips 20000 --input_channel 'EEG Pz-LER' --input_dataset '/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/SS3_EDF_Tensorized_15s'
# python main.py --clip_length_s 30 --num_training_clips 20000 --input_channel 'EEG Pz-LER' --input_dataset '/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/SS3_EDF_Tensorized_30s'
