#!/bin/bash

echo "🧬 Running Fine-tuned LLM Black-box Optimization"
echo "==============================================="

# Navigate to the parent directory (single_objective)
cd "$(dirname "$0")/.."

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate molleo

# Check if fine-tuned server is running on port 8001
if ! curl -s http://localhost:8001/health > /dev/null; then
    echo "❌ Fine-tuned model server not detected on port 8001!"
    echo "Please start it first with: bash dpo_final_300_iterations/start_finetuned_server.sh"
    exit 1
fi

echo "✅ Fine-tuned server detected on port 8001"
echo ""

# Run the fine-tuned optimization
echo "🚀 Starting optimization with fine-tuned DPO model..."
echo "Parameters:"
echo "  - Oracle: jnk3"
echo "  - Initial pool: 500 molecules"
echo "  - Iterations: 15"
echo "  - Molecules per iteration: 15"
echo "  - Max oracle calls: 5000"
echo ""

python run_finetuned_optimizer.py \
    --oracle jnk3 \
    --n_init 500 \
    --m_per_iter 15 \
    --max_iterations 15 \
    --max_oracle_calls 5000 \
    --finetuned_port 8001 \
    --finetuned_model_name dpo_finetuned \
    --output_dir ./results_finetuned \
    --seed 42

echo ""
echo "🎉 Fine-tuned optimization completed!"
echo "📊 Results saved to: ./results_finetuned" 