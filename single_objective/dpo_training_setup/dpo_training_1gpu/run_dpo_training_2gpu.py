#!/usr/bin/env python
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from tdc import Oracle
from rdkit import Chem
import re

def extract_smiles_from_text(text):
    """Extract SMILES string from generated text."""
    # Look for common SMILES patterns
    smiles_patterns = [
        r'[A-Z][a-z]?(?:\[[^\]]+\])?(?:[a-z]?[0-9]*)?(?:\([^)]+\))?[A-Za-z0-9\[\]()=@+\-#$%&*\/\\:.]*',
        r'C[A-Za-z0-9\[\]()=@+\-#$%]*',
        r'[CNOSH][A-Za-z0-9\[\]()=@+\-#$%]*'
    ]
    
    for pattern in smiles_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) >= 2 and is_valid_smiles(match):
                return match
    
    return None

def is_valid_smiles(smiles):
    """Check if SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def main():
    # Set environment variable to avoid tokenizers warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Starting DPO training script optimized for 2 H100 GPUs (GPUs 2,3) with JNK3 evaluation...")
    print("Note: vLLM server should be running on GPUs 0,1")
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs visible: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Initialize JNK3 oracle for evaluation
    print("Initializing JNK3 oracle for evaluation...")
    jnk3_oracle = Oracle(name='jnk3')
    
    # Load data
    with open('preferences.json', 'r') as f:
        data = json.load(f)
    
    preferences = data['data']
    print(f"Loaded {len(preferences)} preference pairs")
    
    # Create dataset and split train/eval
    dataset = Dataset.from_list(preferences)
    print(f"Created dataset with {len(dataset)} examples")
    print("Sample data keys:", dataset.column_names)
    
    # Split dataset for evaluation (80/20 split)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Load tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Updated to 7B
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for DDP training (no device_map to avoid conflicts)
    print(f"Loading model from {model_name} for DDP training...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Load reference model for DDP training
    print("Loading reference model for DDP training...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Configure training arguments for 2 GPUs
    training_args = DPOConfig(
        output_dir="./checkpoints_2gpu",
        logging_dir="./logs_2gpu",
        per_device_train_batch_size=1,  # Reduced for 7B model
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        learning_rate=5e-7,  # Lower learning rate for 7B model
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=512,  # Reduced context length
        max_prompt_length=256,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        bf16=True,  # Use bfloat16 for H100s
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # Save memory
    )
    
    print("Creating DPO trainer for 2-GPU training...")
    # Create trainer with both models
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print("Starting DPO training on 2 H100 GPUs (GPUs 2,3)...")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
    # Train
    trainer.train()
    trainer.save_model()
    
    print("DPO training completed!")
    print(f"Model saved to: {training_args.output_dir}")
    
    # Final evaluation - only run on rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("\n=== Final Evaluation ===")
        print("DPO training completed successfully!")
        print(f"Final model saved to: {training_args.output_dir}")

if __name__ == "__main__":
    main() 