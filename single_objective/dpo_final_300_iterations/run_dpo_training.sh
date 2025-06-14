#!/bin/bash

# DPO Training Script for 2x H100 GPUs using verl
# Generated automatically by dpo_step.py

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install verl if not already installed
# pip install verl

# Run DPO training with verl
python -m verl.trainer.fsdp_dpo_trainer \
    --config_path ./dpo_final_300_iterations/verl_config.json \
    --num_gpus 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --num_epochs 3 \
    --max_length 2048 \
    --beta 0.1 \
    --output_dir ./dpo_final_300_iterations/checkpoints \
    --logging_dir ./dpo_final_300_iterations/logs \
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 50 \
    --report_to tensorboard

echo "DPO training completed. Checkpoints saved to ./dpo_final_300_iterations/checkpoints"
