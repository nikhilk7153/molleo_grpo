#!/usr/bin/env python
"""
Run DPO Training from Existing Preferences

This script takes an existing all_preferences.json file and creates
DPO training configuration for 1 GPU.
"""

import json
import os
import sys
from typing import Dict, Any, List

def create_single_gpu_dpo_config(preferences_file: str, 
                                output_dir: str,
                                oracle_name: str = "jnk3") -> Dict[str, Any]:
    """
    Create DPO training files optimized for 1 GPU.
    
    Args:
        preferences_file: Path to all_preferences.json
        output_dir: Directory to save training files
        oracle_name: Name of the oracle used
        
    Returns:
        Dictionary with file paths and statistics
    """
    
    # Load existing preferences
    with open(preferences_file, 'r') as f:
        preferences_data = json.load(f)
    
    if 'data' in preferences_data:
        all_preferences = preferences_data['data']
        total_iterations = preferences_data.get('stats', {}).get('iterations_completed', 0)
    else:
        # Assume the file contains the preferences directly
        all_preferences = preferences_data
        total_iterations = len(set([p.get('iteration', 1) for p in all_preferences]))
    
    if not all_preferences:
        return {"status": "error", "message": "No preference pairs found in file"}
        
    print(f"Loaded {len(all_preferences)} preference pairs from {preferences_file}")
    
    # Create output directory
    final_output_dir = os.path.join(output_dir, "dpo_training_1gpu")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Create final training dataset
    final_data = {
        "data": all_preferences,
        "dataset_info": {
            "total_pairs": len(all_preferences),
            "total_iterations": total_iterations,
            "oracle": oracle_name,
            "avg_reward_difference": sum([p["chosen_reward"] - p["rejected_reward"] for p in all_preferences]) / len(all_preferences)
        }
    }
    
    # Save final preference dataset
    final_file = os.path.join(final_output_dir, "preferences.json")
    with open(final_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    # Create verl training config for 1 GPU
    verl_config = {
        "model": {
            "path": "Qwen/Qwen2.5-7B-Instruct",
            "trust_remote_code": True
        },
        "training": {
            "algorithm": "dpo",
            "num_gpus": 1,  # Single GPU
            "per_device_train_batch_size": 8,  # Larger batch size for single GPU
            "gradient_accumulation_steps": 8,  # More accumulation steps
            "learning_rate": 5e-7,
            "num_epochs": 3,
            "max_length": 2048,
            "beta": 0.1
        },
        "data": {
            "train_files": [final_file],
            "eval_files": [final_file],
            "prompt_key": "prompt",
            "chosen_key": "chosen",
            "rejected_key": "rejected"
        },
        "output": {
            "save_dir": os.path.join(final_output_dir, "checkpoints"),
            "logging_dir": os.path.join(final_output_dir, "logs")
        }
    }
    
    config_file = os.path.join(final_output_dir, "verl_config.json")
    with open(config_file, 'w') as f:
        json.dump(verl_config, f, indent=2)
    
    # Create training script for 1 GPU
    training_script = f"""#!/bin/bash

# DPO Training Script for Single GPU using verl
# Preference pairs: {len(all_preferences)}
# Oracle: {oracle_name}

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting DPO training with {len(all_preferences)} preference pairs..."
echo "Oracle: {oracle_name}"
echo "Using 1 GPU"

# Run DPO training with verl
python -m verl.trainer.fsdp_dpo_trainer \\
    --config_path {config_file} \\
    --num_gpus 1 \\
    --per_device_train_batch_size 8 \\
    --gradient_accumulation_steps 8 \\
    --learning_rate 5e-7 \\
    --num_epochs 3 \\
    --max_length 2048 \\
    --beta 0.1 \\
    --output_dir {os.path.join(final_output_dir, "checkpoints")} \\
    --logging_dir {os.path.join(final_output_dir, "logs")} \\
    --save_steps 100 \\
    --eval_steps 100 \\
    --warmup_steps 50 \\
    --report_to tensorboard

echo "DPO training completed. Checkpoints saved to {os.path.join(final_output_dir, "checkpoints")}"
"""
    
    script_file = os.path.join(final_output_dir, "run_dpo_training.sh")
    with open(script_file, 'w') as f:
        f.write(training_script)
    os.chmod(script_file, 0o755)
    
    # Create a simple Python training script as alternative
    python_script = f"""#!/usr/bin/env python
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import torch

def main():
    # Load data
    with open('{final_file}', 'r') as f:
        data = json.load(f)
    
    preferences = data['data']
    
    # Create dataset
    dataset = Dataset.from_list(preferences)
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Training config
    training_args = DPOConfig(
        output_dir="{os.path.join(final_output_dir, "checkpoints")}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        num_train_epochs=3,
        max_length=2048,
        beta=0.1,
        save_steps=100,
        eval_steps=100,
        warmup_steps=50,
        logging_dir="{os.path.join(final_output_dir, "logs")}",
        report_to="tensorboard"
    )
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset.select(range(min(100, len(dataset)))),  # Small eval set
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    
    print("DPO training completed!")

if __name__ == "__main__":
    main()
"""
    
    python_script_file = os.path.join(final_output_dir, "run_dpo_training.py")
    with open(python_script_file, 'w') as f:
        f.write(python_script)
    os.chmod(python_script_file, 0o755)
    
    results = {
        "status": "success",
        "dataset_stats": {
            "total_pairs": len(all_preferences),
            "total_iterations": total_iterations,
            "oracle": oracle_name,
            "avg_reward_difference": final_data['dataset_info']['avg_reward_difference']
        },
        "files": {
            "preference_data": final_file,
            "verl_config": config_file,
            "bash_script": script_file,
            "python_script": python_script_file
        },
        "output_dir": final_output_dir
    }
    
    print(f"\n🎉 DPO Training Files Created for 1 GPU! 🎉")
    print(f"✅ Total preference pairs: {len(all_preferences)}")
    print(f"📊 From {total_iterations} iterations")
    print(f"💾 Files saved to: {final_output_dir}")
    print(f"🚀 To start training:")
    print(f"   Option 1 (verl): bash {script_file}")
    print(f"   Option 2 (transformers): python {python_script_file}")
    print(f"📈 Avg reward difference: {final_data['dataset_info']['avg_reward_difference']:.4f}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create DPO training from existing preferences')
    parser.add_argument('--preferences_file', type=str, required=True, 
                       help='Path to all_preferences.json file')
    parser.add_argument('--output_dir', type=str, default='./dpo_training_setup',
                       help='Output directory for training files')
    parser.add_argument('--oracle', type=str, default='jnk3',
                       help='Oracle name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.preferences_file):
        print(f"Error: Preferences file {args.preferences_file} not found!")
        sys.exit(1)
    
    results = create_single_gpu_dpo_config(
        args.preferences_file, 
        args.output_dir,
        args.oracle
    )
    
    if results["status"] == "success":
        print("\n✅ Ready for DPO training!")
    else:
        print(f"\n❌ Error: {results['message']}")

if __name__ == "__main__":
    main() 