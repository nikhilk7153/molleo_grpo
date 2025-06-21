#!/usr/bin/env python3
"""
Multi-GPU SFT training script for pareto optimization trajectories
Optimized for 4x A100 GPUs (40GB each)
Uses pre-processed single-molecule training data
"""

import json
import torch
import argparse
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

def load_training_data(training_data_file, min_score_threshold=1.5):
    """Load pre-processed training data from JSON file"""
    print(f"Loading training data from {training_data_file}...")
    
    try:
        with open(training_data_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {training_data_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {training_data_file}")
        return []
    
    # Filter by score threshold and extract formatted text
    training_examples = []
    for example in data:
        if example.get('pareto_score', 0.0) >= min_score_threshold:
            training_examples.append({
                "text": example['text'],
                "pareto_score": example['pareto_score'],
                "iteration": example['iteration']
            })
    
    print(f"Loaded {len(training_examples)} training examples with pareto score >= {min_score_threshold}")
    
    if len(training_examples) == 0:
        print("No training examples found. Try lowering min_score_threshold.")
        return []
    
    # Show score distribution
    scores = [ex['pareto_score'] for ex in training_examples]
    print(f"Pareto score distribution:")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")
    print(f"  Examples from iterations: {len(set(ex['iteration'] for ex in training_examples))} unique iterations")
    
    return training_examples

def train_multi_gpu_sft(
    training_data_file,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./pareto_sft_4gpu_model",
    min_score_threshold=1.5,
    max_length=2048,
    per_device_batch_size=8,  # Higher batch size per GPU for A100
    learning_rate=2e-5,
    num_epochs=3,
    use_lora=True,
    lora_r=64,    # Larger LoRA rank for better capacity
    lora_alpha=128,
    gradient_checkpointing=True,
    use_deepspeed=False,
    validation_data_file=None  # Optional validation data for monitoring overfitting
):
    """
    Multi-GPU SFT training optimized for 4x A100 GPUs
    """
    
    print("="*80)
    print("MULTI-GPU PARETO OPTIMIZATION SFT TRAINING")
    print("="*80)
    print(f"GPUs detected: {torch.cuda.device_count()}")
    print(f"GPU memory per device: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Load training data
    training_examples = load_training_data(training_data_file, min_score_threshold)
    if not training_examples:
        print("No training data found. Exiting.")
        return
    
    print(f"\nLoading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimal settings for A100s
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # bfloat16 is optimal for A100s
        trust_remote_code=True,
        # Remove device_map for proper multi-GPU training
    )
    
    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Setup LoRA if requested
    if use_lora:
        print(f"Setting up LoRA configuration (r={lora_r}, alpha={lora_alpha})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,  # Lower dropout for better performance
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            modules_to_save=None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    def tokenize_function(examples):
        # Tokenize the text
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # Use max_length padding
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        # Set labels for causal language modeling
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Create dataset from formatted training data
    dataset = Dataset.from_list(training_examples)
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text", "pareto_score", "iteration"],
        num_proc=4  # Use multiple processes for faster tokenization
    )
    
    print(f"Training dataset size: {len(tokenized_dataset)}")
    
    # Load validation data if provided
    eval_dataset = None
    if validation_data_file and os.path.exists(validation_data_file):
        print(f"Loading validation data from {validation_data_file}...")
        val_examples = load_training_data(validation_data_file, min_score_threshold)
        print(f"Validation examples: {len(val_examples)}")
        
        val_dataset = Dataset.from_list(val_examples)
        eval_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "pareto_score", "iteration"],
            num_proc=4
        )
        print(f"Validation dataset size: {len(eval_dataset)}")
    else:
        print("No validation data provided - training without validation monitoring")
    
    # Calculate optimal batch sizes for 4 A100s
    world_size = torch.cuda.device_count()
    total_batch_size = per_device_batch_size * world_size
    gradient_accumulation_steps = max(1, 32 // total_batch_size)  # Target effective batch size of 32
    
    print(f"Training configuration:")
    print(f"  World size (GPUs): {world_size}")
    print(f"  Per-device batch size: {per_device_batch_size}")
    print(f"  Total batch size per step: {total_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {total_batch_size * gradient_accumulation_steps}")
    
    # Training arguments optimized for A100s
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # More gradual learning rate schedule
        warmup_steps=max(200, len(tokenized_dataset) // (total_batch_size * gradient_accumulation_steps) // 10),  # 10% of epoch
        learning_rate=min(learning_rate, 5e-5),  # Cap learning rate at 5e-5 to prevent overfitting
        lr_scheduler_type="cosine",  # Use cosine annealing for better convergence
        
        # A100-optimized settings
        bf16=True,  # Use bfloat16 for A100s
        tf32=True,  # Enable TF32 for better performance
        
        # Multi-GPU settings - simplified for stability
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,  # Disable multiprocessing to avoid device issues
        
        # Evaluation and logging
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50 if eval_dataset is not None else None,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        
        # Memory optimization
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
        
        # Performance - disable pin memory for device compatibility
        dataloader_pin_memory=False,
        
        # Disable wandb by default
        report_to=None,
        logging_dir=f"{output_dir}/logs",
        
        # DeepSpeed config if enabled
        deepspeed="deepspeed_config.json" if use_deepspeed else None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting multi-GPU SFT training on pareto trajectories...")
    print(f"Training will use {torch.cuda.device_count()} GPUs")
    
    trainer.train()
    
    # Save the model
    print("Saving trained model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training metadata
    training_info = {
        "base_model": model_name,
        "training_examples": len(training_examples),
        "min_score_threshold": min_score_threshold,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": total_batch_size * gradient_accumulation_steps,
        "gpus_used": torch.cuda.device_count(),
        "lora_config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "enabled": use_lora
        }
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("="*80)
    print("MULTI-GPU SFT TRAINING COMPLETE!")
    print(f"Model saved to: {output_dir}")
    print(f"Training examples: {len(training_examples)}")
    print(f"GPUs used: {torch.cuda.device_count()}")
    print("="*80)
    
    return output_dir

def create_deepspeed_config():
    """Create DeepSpeed configuration for ZeRO-2"""
    config = {
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "none"
            },
            "offload_param": {
                "device": "none"
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
    
    with open("deepspeed_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created deepspeed_config.json")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU SFT training for pareto optimization")
    parser.add_argument("--training_data_file", type=str, 
                       default="pareto_single_molecule_training_data.json",
                       help="Path to pre-processed training data JSON file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model to finetune")
    parser.add_argument("--output_dir", type=str, default="./pareto_sft_4gpu_model",
                       help="Output directory for trained model")
    parser.add_argument("--min_score_threshold", type=float, default=1.5,
                       help="Minimum pareto score for training examples")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--per_device_batch_size", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--validation_data_file", type=str, default=None,
                       help="Path to validation data JSON file for monitoring overfitting")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA training")
    parser.add_argument("--use_deepspeed", action="store_true",
                       help="Use DeepSpeed for training")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                       help="Disable gradient checkpointing")
    
    args = parser.parse_args()
    
    # Create DeepSpeed config if requested
    if args.use_deepspeed:
        create_deepspeed_config()
    
    # Run training
    train_multi_gpu_sft(
        training_data_file=args.training_data_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        min_score_threshold=args.min_score_threshold,
        max_length=args.max_length,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_deepspeed=args.use_deepspeed
    )

if __name__ == "__main__":
    main() 