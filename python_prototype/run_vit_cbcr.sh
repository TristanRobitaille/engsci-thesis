#!/bin/bash

# First set
python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--clip_length_s=30 \
--dataset_resample_replacement=True \
--batch_size=16 \
--learning_rate=3e-3 \
--num_epochs=50 \
--input_channel='EEG Cz-LER' \
--num_clips=115000 \
--embedding_depth=64 \
--num_layers=8 \
--num_heads=8 \
--mlp_dim=32 \
--class_weights 1 1 1 1 1 1 \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_5-stg_30s" \
--dataset_resample_algo="RandomUnderSampler" \
--training_set_target_count 4100   4100   4100   4100   4100 \
# 30s dataset distribution: 4253,  26343, 6844,  9368,  5732
