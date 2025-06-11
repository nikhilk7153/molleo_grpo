#!/usr/bin/env python
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

def main():
    print("Starting DPO training script...")
    
    # Load data
    with open('preferences.json', 'r') as f:
        data = json.load(f)
    
    preferences = data['data']
    print(f"Loaded {len(preferences)} preference pairs")
    
    # Create dataset
    dataset = Dataset.from_list(preferences)
    print(f"Created dataset with {len(dataset)} examples")
    print("Sample data keys:", dataset.column_names)
    
    # Load tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    print(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",  # Use auto device mapping for multi-GPU
    )
    
    # Load reference model on CPU to save GPU memory
    print("Loading reference model on CPU...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",  # Keep reference model on CPU
    )
    
    # Training config
    training_args = DPOConfig(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,  # Very small batch size
        gradient_accumulation_steps=16,  # High accumulation to compensate
        learning_rate=5e-7,
        num_train_epochs=1,  # Reduced for testing
        save_steps=200,
        logging_steps=20,
        warmup_steps=50,
        logging_dir="./logs",
        report_to="tensorboard",
        save_strategy="steps",
        logging_strategy="steps",
        remove_unused_columns=False,
    )
    
    print("Creating DPO trainer...")
    # Create trainer with reference model and correct parameter names
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting DPO training...")
    # Train
    trainer.train()
    trainer.save_model()
    
    print("DPO training completed!")

if __name__ == "__main__":
    main()
