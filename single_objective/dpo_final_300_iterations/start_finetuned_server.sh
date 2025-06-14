#!/bin/bash

echo "🚀 Starting Fine-tuned DPO Model Server"
echo "======================================="

# Navigate to the script directory
cd "$(dirname "$0")"

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate molleo

# Check if port 8001 is available
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Port 8001 is already in use!"
    echo "Please stop the existing service or choose a different port."
    exit 1
fi

# Start the fine-tuned model server
echo "🔧 Starting vLLM server with LoRA adapter..."
echo "📍 Model: Qwen/Qwen2.5-7B-Instruct"
echo "🔗 Adapter: ./trl_checkpoints_optimized/final"
echo "🌐 Port: 8001"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules dpo_finetuned=./trl_checkpoints_optimized/final \
    --port 8001 \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --disable-log-requests \
    --trust-remote-code 