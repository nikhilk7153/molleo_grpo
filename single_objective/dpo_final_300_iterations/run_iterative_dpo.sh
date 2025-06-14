#!/bin/bash
set -e

echo "Starting Iterative DPO Training (Fine-tuning the already DPO-optimized model)"
echo "============================================================================="

# Check if we have the DPO-optimized model
if [ ! -d "dpo_optimized_model" ]; then
    echo "Error: dpo_optimized_model directory not found!"
    echo "Please make sure you have the DPO-optimized model from previous training."
    echo "Expected path: ./dpo_optimized_model/"
    exit 1
fi

# Check if we have new preference pairs from llm_blackbox_optimizer runs
if [ ! -f "preference_pairs.json" ]; then
    echo "Error: preference_pairs.json not found!"
    echo "This should contain the new trajectories from llm_blackbox_optimizer.py runs."
    echo "Make sure you've run the optimizer and it generated new preference data."
    exit 1
fi

echo "✅ Found DPO-optimized model: ./dpo_optimized_model/"
echo "✅ Found new preference pairs: ./preference_pairs.json"

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
echo "Starting iterative DPO training..."

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
    OUTPUT_DIR="./trl_checkpoints_iterative"
else
    # Filter the new preference pairs first
    echo "Filtering new preference pairs..."
    python filter_preference_pairs.py
    DATA_PATH="./preference_pairs_filtered_0.08.json"
    OUTPUT_DIR="./trl_checkpoints_iterative_filtered"
fi

echo "Training mode: $TRAINING_MODE"
echo "Dataset mode: $DATASET_MODE"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

# Use the DPO-optimized model as base instead of original Qwen
MODEL_PATH="./dpo_optimized_model"

echo "🎯 Using DPO-optimized model as base: $MODEL_PATH"

# Common training arguments - using the fine-tuned model as base
COMMON_ARGS="
    --data_path $DATA_PATH \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --max_length 512 \
    --beta 0.1 \
    --use_4bit \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --warmup_steps 50 \
    --save_steps 50 \
    --eval_steps 25
"

echo "🔧 Training Configuration:"
echo "  Base Model: $MODEL_PATH (DPO-optimized)"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: 2 (fewer since we're fine-tuning an already optimized model)"
echo "  Learning Rate: 1e-5 (lower since we're fine-tuning)"
echo "  LoRA Rank: 16 (smaller for fine-tuning)"

if [ "$TRAINING_MODE" = "single" ]; then
    echo "Starting single GPU iterative training..."
    python train_dpo_trl.py $COMMON_ARGS
elif [ "$TRAINING_MODE" = "multi" ]; then
    echo "Starting multi-GPU distributed iterative training..."
    # Use accelerate launch for proper distributed training
    accelerate launch --config_file accelerate_config.yaml \
                     train_dpo_trl.py $COMMON_ARGS
else
    echo "Invalid training mode: $TRAINING_MODE"
    echo "Use 'single' or 'multi'"
    exit 1
fi

echo "🎉 Iterative DPO training completed!"

echo "📁 Model checkpoints saved to: $OUTPUT_DIR"
echo "📊 Logs available at: $OUTPUT_DIR/logs"
echo ""
echo "🔄 Next steps:"
echo "1. Convert the new model: python convert_to_hf_model.py --input_dir $OUTPUT_DIR/final --output_dir ./dpo_optimized_model_v2"
echo "2. Start new server: Update start_dpo_server.sh to use ./dpo_optimized_model_v2"
echo "3. Run optimizer again: python llm_blackbox_optimizer.py --oracle jnk3 --max_iterations 300"
echo ""
echo "🔍 To monitor training progress:"
echo "tensorboard --logdir $OUTPUT_DIR/logs --port 6006" 