#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for molecular design
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import argparse

def load_training_data(file_path):
    """Load training data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt(example):
    """Format the prompt and response for training"""
    prompt = example['prompt']
    response = example['response']
    
    # Use a consistent format
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    return {"text": formatted_text}

def train_sft_model(
    training_data_path,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./sft_model",
    max_length=1024,
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3,
    use_lora=True
):
    """
    Train a supervised fine-tuned model
    
    Args:
        training_data_path: Path to training data JSONL file
        model_name: Base model to fine-tune
        output_dir: Directory to save the trained model
        max_length: Maximum sequence length
        batch_size: Training batch size
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        use_lora: Whether to use LoRA for efficient fine-tuning
    """
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Setup LoRA if requested
    if use_lora:
        print("Setting up LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load and format training data
    print("Loading training data...")
    raw_data = load_training_data(training_data_path)
    formatted_data = [format_prompt(example) for example in raw_data]
    
    def tokenize_function(examples):
        # Tokenize the text
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        # Set labels for causal language modeling
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    print(f"Training dataset size: {len(tokenized_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to=None,  # Disable wandb logging
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
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Train SFT model for molecular design")
    parser.add_argument("--data", required=True, help="Path to training data JSONL file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--output", default="./sft_model", help="Output directory")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    
    args = parser.parse_args()
    
    train_sft_model(
        training_data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        use_lora=not args.no_lora
    )

if __name__ == "__main__":
    main() 