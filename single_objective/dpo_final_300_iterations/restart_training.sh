#!/bin/bash

# DPO Training with Optimized Parameters
# This script restarts training with better learning rate and configuration

echo "=== Restarting DPO Training with Optimized Parameters ==="
echo "Key Changes:"
echo "  - Learning Rate: 1e-6 → 5e-5 (50x increase)"
echo "  - Better warmup and scheduler"
echo "  - Gradient clipping and weight decay"
echo "  - More frequent evaluation"
echo ""

# Stop any existing training
pkill -f train_dpo_trl.py

# Clean up previous checkpoints if desired (optional)
# rm -rf ./trl_checkpoints/*

# Start training with optimized parameters
python train_dpo_trl.py \
    --data_path ./preference_pairs.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./trl_checkpoints_optimized \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-05 \
    --max_length 512 \
    --beta 0.1 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --warmup_steps 100 \
    --save_steps 100 \
    --eval_steps 50

echo "Training completed!" 