#!/bin/bash

# DPO Training Script for Single GPU using verl
# Preference pairs: 1390
# Oracle: jnk3

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting DPO training with 1390 preference pairs..."
echo "Oracle: jnk3"
echo "Using 1 GPU"

# Run DPO training with verl
python -m verl.trainer.fsdp_dpo_trainer \
    --config_path ./dpo_training_setup/dpo_training_1gpu/verl_config.json \
    --num_gpus 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --num_epochs 3 \
    --max_length 2048 \
    --beta 0.1 \
    --output_dir ./dpo_training_setup/dpo_training_1gpu/checkpoints \
    --logging_dir ./dpo_training_setup/dpo_training_1gpu/logs \
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 50 \
    --report_to tensorboard

echo "DPO training completed. Checkpoints saved to ./dpo_training_setup/dpo_training_1gpu/checkpoints"
