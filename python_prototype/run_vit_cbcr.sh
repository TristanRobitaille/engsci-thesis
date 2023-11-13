#!/bin/bash

# First set
python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--clip_length_s=30 \
--dataset_resample_strategy=auto \
--dataset_resample_replacement=True \
--batch_size=16 \
--learning_rate=3e-3 \
--num_epochs=50 \
--input_channel='EEG F8-LER' \
--num_clips=115000 \
--embedding_depth=128 \
--num_layers=16 \
--num_heads=8 \
--mlp_dim=32 \
--class_weights 1 1 1 1 1 1 \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_5-stg_30s" &

python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--clip_length_s=30 \
--dataset_resample_strategy=auto \
--dataset_resample_replacement=True \
--batch_size=16 \
--learning_rate=3e-3 \
--num_epochs=50 \
--input_channel='EEG F3-LER' \
--num_clips=115000 \
--embedding_depth=128 \
--num_layers=16 \
--num_heads=8 \
--mlp_dim=32 \
--class_weights= 1 1 1 1 1 1 \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_5-stg_30s" &

python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--clip_length_s=30 \
--dataset_resample_strategy=auto \
--dataset_resample_replacement=True \
--batch_size=16 \
--learning_rate=3e-3 \
--num_epochs=50 \
--input_channel='EEG F4-LER' \
--num_clips=115000 \
--embedding_depth=128 \
--num_layers=16 \
--num_heads=8 \
--mlp_dim=32 \
--class_weights 1 1 1 1 1 1 \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_5-stg_30s" &

wait

# Second set
python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--clip_length_s=30 \
--dataset_resample_strategy=auto \
--dataset_resample_replacement=True \
--batch_size=16 \
--learning_rate=3e-3 \
--num_epochs=50 \
--input_channel='EEG T4-LER' \
--num_clips=115000 \
--embedding_depth=128 \
--num_layers=16 \
--num_heads=8 \
--mlp_dim=32 \
--class_weights 1 1 1 1 1 1 \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_5-stg_30s" &

python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--clip_length_s=30 \
--dataset_resample_strategy=auto \
--dataset_resample_replacement=True \
--batch_size=16 \
--learning_rate=3e-3 \
--num_epochs=50 \
--input_channel='EEG C3-LER' \
--num_clips=115000 \
--embedding_depth=128 \
--num_layers=16 \
--num_heads=8 \
--mlp_dim=32 \
--class_weights 1 1 1 1 1 1 \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_5-stg_30s" &

python3 /home/trobitaille/engsci-thesis/python_prototype/main_vision_transformer.py \
--clip_length_s=30 \
--dataset_resample_strategy=auto \
--dataset_resample_replacement=True \
--batch_size=16 \
--learning_rate=3e-3 \
--num_epochs=50 \
--input_channel='EEG C4-LER' \
--num_clips=115000 \
--embedding_depth=128 \
--num_layers=16 \
--num_heads=8 \
--mlp_dim=32 \
--class_weights 1 1 1 1 1 1 \
--input_dataset="/mnt/data/tristan/engsci_thesis_python_prototype_data/SS3_EDF_Tensorized_5-stg_30s" &