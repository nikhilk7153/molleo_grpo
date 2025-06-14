#!/usr/bin/env python3
"""
DPO Training with TRL for Molecular Generation

This script trains a Qwen model using Direct Preference Optimization (DPO)
on molecular generation preference data.
"""

import os
# Set critical environment variables before any other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")

# Fix the specific error about None in ALL_PARALLEL_STYLES
# Ensure all FSDP and distributed training variables are set to proper string values
fsdp_vars = {
    "ACCELERATE_USE_FSDP": "false",
    "FSDP_CPU_RAM_EFFICIENT_LOADING": "false",
    "FSDP_AUTO_WRAP_POLICY": "NO_WRAP",
    "FSDP_BACKWARD_PREFETCH": "BACKWARD_PRE",
    "FSDP_FORWARD_PREFETCH": "false",
    "FSDP_STATE_DICT_TYPE": "FULL_STATE_DICT",
    "FSDP_OFFLOAD_PARAMS": "false",
    "FSDP_ACTIVATION_CHECKPOINTING": "false",
    "FSDP_SYNC_MODULE_STATES": "true",
    "FSDP_USE_ORIG_PARAMS": "false",
    "FSDP_TRANSFORMER_CLS_TO_WRAP": "",
    "FSDP_MIN_NUM_PARAMS": "0",
    "FSDP_SHARDING_STRATEGY": "FULL_SHARD"
}

for var, default_value in fsdp_vars.items():
    os.environ.setdefault(var, default_value)

# Additional environment variables for stability
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

import json
import torch
import warnings
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import argparse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_preference_data(data_path):
    """Load and format preference data for DPO training"""
    print(f"Loading preference data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data['data']:
        formatted_data.append({
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected']
        })
    
    print(f"Loaded {len(formatted_data)} preference pairs")
    return Dataset.from_list(formatted_data)


def setup_model_and_tokenizer(model_name, use_4bit=False):
    """Setup model and tokenizer with optional 4-bit quantization"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False  # Use slow tokenizer for better compatibility
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure model loading arguments
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "use_cache": False,  # Disable cache for training
    }
    
    # Configure quantization if requested
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        print("Using 4-bit quantization")
    else:
        # Only set device_map if not using quantization
        if torch.cuda.device_count() > 1:
            model_kwargs["device_map"] = "balanced"
        else:
            model_kwargs["device_map"] = "auto"
    
    try:
        # Try loading with specified configuration
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Failed to load model with auto device mapping: {e}")
        print("Trying without device_map...")
        
        # Remove device_map and try again
        if "device_map" in model_kwargs:
            del model_kwargs["device_map"]
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e2:
            print(f"Failed again: {e2}")
            print("Trying with minimal configuration...")
            
            # Minimal configuration as last resort
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Fall back to float16
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None,  # Explicitly disable device mapping
            )
    
    return model, tokenizer


def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """Setup LoRA for efficient fine-tuning"""
    print("Setting up LoRA configuration")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def main():
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    
    # Set additional environment variables at runtime
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description='DPO Training with TRL')
    parser.add_argument('--data_path', type=str, default='./preference_pairs.json',
                        help='Path to preference pairs JSON file')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Base model to fine-tune')
    parser.add_argument('--output_dir', type=str, default='./trl_checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Per device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-05,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta parameter')
    parser.add_argument('--use_4bit', action='store_true',
                        help='Use 4-bit quantization')
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--save_steps', type=int, default=200,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluation every N steps')
    
    args = parser.parse_args()
    
    # Only print on main process in distributed training
    if accelerator.is_main_process:
        print("="*50)
        print("DPO Training with TRL")
        print("="*50)
        print(f"Model: {args.model_name}")
        print(f"Data: {args.data_path}")
        print(f"Output: {args.output_dir}")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Beta: {args.beta}")
        print(f"Use 4-bit: {args.use_4bit}")
        print(f"Use LoRA: {args.use_lora}")
        print(f"Distributed training processes: {accelerator.num_processes}")
        print(f"Current process: {accelerator.process_index}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load preference data
    dataset = load_preference_data(args.data_path)
    
    # Split dataset (80% train, 20% eval)
    train_dataset = dataset.select(range(int(0.8 * len(dataset))))
    eval_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))
    
    if accelerator.is_main_process:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_4bit)
    
    # Setup LoRA if requested
    if args.use_lora:
        model = setup_lora(model, args.lora_r, args.lora_alpha)
    
    # Setup DPO configuration
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        warmup_steps=args.warmup_steps,
        warmup_ratio=0.1,  # Alternative warmup approach
        bf16=torch.cuda.is_bf16_supported(),  # Auto-detect bf16 support
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 as fallback
        remove_unused_columns=False,
        report_to=None,  # Disable reporting to avoid errors
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=False,  # Disable to prevent OOM at end
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,  # Reduce to 0 to avoid multiprocessing issues
        push_to_hub=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        lr_scheduler_type="cosine",  # Use cosine learning rate schedule
        weight_decay=0.01,  # Add weight decay for regularization
        max_grad_norm=1.0,  # Gradient clipping
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length,
        max_target_length=args.max_length,
        # DPO specific parameters for better training
        loss_type="sigmoid",  # More stable than default
        label_smoothing=0.0,  # Can help with overfitting
    )
    
    # Initialize DPO trainer with new API
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,

    )
    
    if accelerator.is_main_process:
        print("Starting DPO training...")
    
    # Create a simple loss tracker
    loss_history = []
    
    # Train the model
    result = dpo_trainer.train()
    
    # Print loss summary (only on main process)
    if accelerator.is_main_process and hasattr(dpo_trainer.state, 'log_history'):
        print("\n" + "="*50)
        print("TRAINING LOSS SUMMARY")
        print("="*50)
        for i, log_entry in enumerate(dpo_trainer.state.log_history):
            if 'train_loss' in log_entry:
                step = log_entry.get('step', i)
                loss = log_entry['train_loss']
                lr = log_entry.get('learning_rate', 'N/A')
                print(f"Step {step:3d}: Loss = {loss:.6f}, LR = {lr}")
                loss_history.append((step, loss))
    
    # Save the final model (only on main process)
    if accelerator.is_main_process:
        print("Saving final model...")
        dpo_trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        
        print("Training completed!")
        print(f"Model saved to: {args.output_dir}/final")
            
        # Save loss history to file
        if loss_history:
            import csv
            loss_file = f"{args.output_dir}/loss_history.csv"
            with open(loss_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Step', 'Loss'])
                writer.writerows(loss_history)
            print(f"Loss history saved to: {loss_file}")
            
            # Print final statistics
            if loss_history:
                initial_loss = loss_history[0][1]
                final_loss = loss_history[-1][1]
                improvement = initial_loss - final_loss
                print(f"Initial loss: {initial_loss:.6f}")
                print(f"Final loss: {final_loss:.6f}")
                print(f"Improvement: {improvement:.6f} ({improvement/initial_loss*100:.2f}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTips for troubleshooting:")
        print("1. Try using --use_4bit flag for memory efficiency")
        print("2. Reduce batch size with --batch_size 1")
        print("3. Reduce max_length with --max_length 256")
        print("4. Ensure model is available and accessible")
        exit(1) 