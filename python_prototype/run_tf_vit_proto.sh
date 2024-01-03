#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=00-2:59      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load cuda cudnn 
source ~/tensorflow/bin/activate
# tensorboard --logdir=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/logs/fit --host 0.0.0.0 --load_fast false &

# First set
python3 /home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/main_vision_transformer.py \
--batch_size=16 \
--learning_rate=1e-3 \
--patch_length_s=1 \
--num_epochs=1 \
--input_channel='EEG Cz-LER' \
--num_clips=115000 \
--embedding_depth=64 \
--num_layers=4 \
--num_heads=8 \
--mlp_dim=32 \
--mlp_head_num_dense=1 \
--historical_lookback_DNN_depth=32 \
--dropout_rate=0.3 \
--class_weights 1 1 1 0.65 1 \
--input_dataset="/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/data/SS3_EDF_Tensorized_both_light_deep_combine-stg_30-0s_256Hz" \
--dataset_resample_algo="ADASYN" \
--training_set_target_count 4600 4600 4600 4600 4600 \
--save_model \
--enable_dataset_resample_replacement \
--use_class_embedding \
--enable_positional_embedding \
--enable_input_rescale \
--k_fold_val_set=0 \
--num_out_filter=1 \
--k_fold_val_results_fp="/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/results/k_fold_val_results/val_1" \
--note="Number of encoder layers study"
# --output_edgetpu_data \