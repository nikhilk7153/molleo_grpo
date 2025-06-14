#!/bin/bash

echo "🚀 Starting DPO-Optimized Model Server"
echo "======================================"

# Navigate to the script directory
cd "$(dirname "$0")"

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate molleo

# Check if port 8000 is available (llm_blackbox_optimizer expects port 8000)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Port 8000 is already in use!"
    echo "Please stop the existing service first."
    echo "To stop existing vLLM servers: pkill -f vllm"
    exit 1
fi

# Check if the converted model exists
MODEL_PATH="./dpo_optimized_model"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ DPO optimized model not found at: $MODEL_PATH"
    echo "Please run convert_to_hf_model.py first to create the merged model."
    exit 1
fi

# Start the DPO-optimized model server
echo "🔧 Starting vLLM server with DPO-optimized merged model..."
echo "📍 Model: $MODEL_PATH"
echo "🌐 Port: 8000 (for llm_blackbox_optimizer.py compatibility)"
echo "🎯 Model name: Qwen/Qwen2.5-7B-Instruct (API compatibility)"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --disable-log-requests \
    --trust-remote-code \
    --dtype float16 