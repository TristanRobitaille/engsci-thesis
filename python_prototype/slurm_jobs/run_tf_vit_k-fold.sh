#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=00-10:59      # time (DD-HH:MM)
#SBATCH --output=%N-%j_k-fold.out  # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-31

# tensorboard --logdir=/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/logs/fit --host 0.0.0.0 --load_fast false &

module load apptainer/1.2.4
apptainer run engsci-thesis.sif python python_prototype/main_vision_transformer.py \
--batch_size=16 \
--learning_rate=1e-3 \
--patch_length=64 \
--num_epochs=100 \
--input_channel='EEG Cz-LER' \
--num_clips=115000 \
--embedding_depth=64 \
--num_layers=1 \
--num_heads=8 \
--mlp_dim=32 \
--mlp_head_num_dense=1 \
--historical_lookback_DNN_depth=32 \
--dropout_rate_percent=30 \
--class_weights 1 1 1 1 1 \
--input_dataset="/home/tristanr/projects/def-xilinliu/data/tristan_weihang/SS3_EDF_Tensorized_both_light_deep_combine-stg_30-0s_128Hz_15b_offset" \
--dataset_resample_algo="ADASYN" \
--training_set_target_count 4600 4600 4600 4600 4600 \
--save_model \
--enable_dataset_resample_replacement \
--use_class_embedding \
--enable_positional_embedding \
--enable_input_rescale \
--enc_2nd_res_conn_arg="inputs" \
--k_fold_val_set=$SLURM_ARRAY_TASK_ID \
--num_out_filter=3 \
--out_filter_type="pre_argmax" \
--filter_self_reset_threshold=-1 \
--k_fold_val_results_fp="/home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/python_prototype/results/k_fold_val_results/center_scale_ln_disabled" \
--num_runs=5 \
--note="" \
# --disable_ln_gamma_beta \
# --output_edgetpu_data \
# --export_plot \