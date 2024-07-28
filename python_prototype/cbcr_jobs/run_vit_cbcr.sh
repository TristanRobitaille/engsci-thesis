#!/bin/bash

python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_both_light_deep_combine-stg_30-0s_64Hz_notch_60Hz_15b_offset_0_3Hz-100Hz_bandpass" \
--k_fold_val_results_fp="/home/trobitaille/engsci-thesis/python_prototype/results/k_fold_val_results/val_1.csv" \
--batch_size=16 \
--learning_rate=1e-3 \
--patch_length=64 \
--num_epochs=100 \
--input_channel='EEG Cz-LER' \
--num_clips=150000 \
--embedding_depth=64 \
--num_layers=2 \
--num_heads=8 \
--mlp_dim=32 \
--historical_lookback_DNN_depth=32 \
--dropout_rate_percent=30 \
--mlp_head_num_dense=1 \
--class_weights 1 1 1 1 1 \
--dataset_resample_algo="ADASYN" \
--training_set_target_count 4600 4600 4600 4600 4600 \
--k_fold_val_set=0 \
--num_out_filter=3 \
--save_model \
--enable_dataset_resample_replacement \
--use_class_embedding \
--enable_positional_embedding \
--enable_input_rescale \
--optimizer="AdamW" \
--out_filter_type="pre_argmax" \
--filter_self_reset_threshold=-1 \
--num_runs=10 \
--note="" \

# --output_edgetpu_data \
