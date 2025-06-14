#!/bin/bash
set -e

echo "Starting DPO Training with TRL"
echo "=============================="

# We're already in the DPO directory, no need to cd

# Check if filtered preference pairs file exists
if [ ! -f "preference_pairs_filtered_0.08.json" ]; then
    echo "Error: preference_pairs_filtered_0.08.json not found!"
    echo "Please run filter_preference_pairs.py first to create the filtered dataset."
    exit 1
fi

echo "Using filtered preference pairs (gap > 0.08)"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set environment variables for stability
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_FSDP=0
export ACCELERATE_USE_DEEPSPEED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "Starting training..."

# Set environment variables for proper training
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# Check if we should use distributed training
if [ -z "$1" ]; then
    echo "Usage: $0 [single|multi] [filtered|unfiltered]"
    echo "  single: Single GPU training"
    echo "  multi: Multi-GPU distributed training"
    echo "  filtered: Use filtered dataset (default)"
    echo "  unfiltered: Use original dataset"
    exit 1
fi

TRAINING_MODE=$1
DATASET_MODE=${2:-filtered}

# Set dataset path based on mode
if [ "$DATASET_MODE" = "unfiltered" ]; then
    DATA_PATH="./preference_pairs.json"
    OUTPUT_DIR="./trl_checkpoints"
else
    DATA_PATH="./preference_pairs_filtered_0.08.json"
    OUTPUT_DIR="./trl_checkpoints_filtered"
fi

echo "Training mode: $TRAINING_MODE"
echo "Dataset mode: $DATASET_MODE"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

# Common training arguments
COMMON_ARGS="
    --data_path $DATA_PATH \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir $OUTPUT_DIR \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_length 512 \
    --beta 0.1 \
    --use_4bit \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 25
"

if [ "$TRAINING_MODE" = "single" ]; then
    echo "Starting single GPU training..."
    python train_dpo_trl.py $COMMON_ARGS
elif [ "$TRAINING_MODE" = "multi" ]; then
    echo "Starting multi-GPU distributed training..."
    # Use accelerate launch for proper distributed training
    accelerate launch --config_file accelerate_config.yaml \
                     train_dpo_trl.py $COMMON_ARGS
else
    echo "Invalid training mode: $TRAINING_MODE"
    echo "Use 'single' or 'multi'"
    exit 1
fi

echo "DPO training completed!"

echo "Model checkpoints saved to: single_objective/dpo_final_300_iterations/trl_checkpoints_filtered"
echo "Logs available at: single_objective/dpo_final_300_iterations/trl_checkpoints_filtered/logs"
echo ""
echo "To monitor training progress:"
echo "tensorboard --logdir single_objective/dpo_final_300_iterations/trl_checkpoints_filtered/logs --port 6006" 