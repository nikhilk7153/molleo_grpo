#!/bin/bash

# DPO Training Script for 4 H100 GPUs
# Optimized for distributed training with torchrun

echo "Starting DPO training on 4 H100 GPUs..."
echo "Make sure you have activated the molleo conda environment!"

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "molleo" ]]; then
    echo "Warning: molleo conda environment not activated!"
    echo "Run: conda activate molleo"
    exit 1
fi

# Check if preferences.json exists
if [ ! -f "preferences.json" ]; then
    echo "Error: preferences.json not found in current directory!"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Set CUDA environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

# Clear any existing checkpoints if requested
if [ "$1" == "--clean" ]; then
    echo "Cleaning previous checkpoints..."
    rm -rf ./checkpoints_4gpu
    rm -rf ./logs_4gpu
fi

# Create directories
mkdir -p ./checkpoints_4gpu
mkdir -p ./logs_4gpu

echo "Using preferences.json: $(wc -l preferences.json)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# Launch distributed training with torchrun
echo "Launching distributed DPO training..."
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    run_dpo_training_4gpu.py

echo "DPO training completed!"
echo "Checkpoints saved in: ./checkpoints_4gpu"
echo "Logs saved in: ./logs_4gpu" 