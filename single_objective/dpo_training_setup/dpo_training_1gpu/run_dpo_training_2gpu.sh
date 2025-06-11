#!/bin/bash

# DPO Training Script for 2 H100 GPUs (GPUs 2,3)
# Optimized for distributed training with torchrun while vLLM uses GPUs 0,1

echo "Starting DPO training on GPUs 2,3 (while vLLM runs on GPUs 0,1)..."
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

# Set CUDA environment variables for GPUs 2,3 only
export CUDA_VISIBLE_DEVICES=2,3
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

# Clear any existing checkpoints if requested
if [ "$1" == "--clean" ]; then
    echo "Cleaning previous checkpoints..."
    rm -rf ./checkpoints_2gpu
    rm -rf ./logs_2gpu
fi

# Create directories
mkdir -p ./checkpoints_2gpu
mkdir -p ./logs_2gpu

echo "Using preferences.json: $(wc -l preferences.json)"
echo "GPUs for DPO training: $CUDA_VISIBLE_DEVICES"
echo "Note: vLLM server should be running on GPUs 0,1"

# Launch distributed training with torchrun for 2 GPUs
echo "Launching distributed DPO training on 2 GPUs..."
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    run_dpo_training_2gpu.py

echo "DPO training completed!"
echo "Checkpoints saved in: ./checkpoints_2gpu"
echo "Logs saved in: ./logs_2gpu" 