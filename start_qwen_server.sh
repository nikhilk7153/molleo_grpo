#!/bin/bash

# Start vLLM server for Qwen2.5-7B-Instruct
# This script starts the vLLM server that MolLEO will connect to

echo "Starting vLLM server for Qwen2.5-7B-Instruct..."
echo "Make sure you have activated the molleo conda environment first!"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "molleo" ]]; then
    echo "Warning: molleo conda environment not activated!"
    echo "Run: conda activate molleo"
    echo ""
fi

# Start the server
python single_objective/vllm_server.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 4096

echo "Server stopped." 