#!/usr/bin/env python3
"""
LoRA training script for pareto optimization trajectories
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

def load_pareto_data(pareto_file, min_score=1.5):
    """Load and extract training data from pareto results"""
    print(f"Loading pareto data from: {pareto_file}")
    
    with open(pareto_file, 'r') as f:
        data = json.load(f)
    
    training_examples = []
    
    if 'detailed_iterations' in data:
        for iteration in data['detailed_iterations']:
            prompt = iteration.get('prompt_sent_to_llm', '')
            generated_mols = iteration.get('generated_molecules_with_explanations', [])
            
            for mol_data in generated_mols:
                molecule = mol_data.get('molecule', '')
                explanation = mol_data.get('explanation', '')
                score = mol_data.get('pareto_score', 0.0)
                
                if molecule and explanation and score >= min_score:
                    response = f"{explanation}\n\nSMILES: {molecule}"
                    training_examples.append({
                        "prompt": prompt.strip(),
                        "response": response.strip()
                    })
    
    print(f"Extracted {len(training_examples)} training examples")
    return training_examples

def format_for_training(examples):
    """Format examples for Qwen chat format"""
    formatted = []
    for ex in examples:
        text = f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>"
        formatted.append({"text": text})
    return formatted

def train_lora_model(
    pareto_file,
    output_dir="./pareto_lora_model",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    min_score=1.5,
    r=16,
    alpha=32,
    dropout=0.1,
    max_length=2048,
    batch_size=4,
    epochs=3,
    lr=5e-5
):
    """Train LoRA model on pareto trajectories"""
    
    print("="*50)
    print("PARETO LoRA TRAINING")
    print("="*50)
    
    # Load data
    training_examples = load_pareto_data(pareto_file, min_score)
    if not training_examples:
        print("No training data found!")
        return
    
    # Load model and tokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Setup LoRA
    print(f"Setting up LoRA (r={r}, alpha={alpha}, dropout={dropout})")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("Preparing dataset...")
    formatted_data = format_for_training(training_examples)
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    print(f"Dataset size: {len(tokenized_dataset)}")
    
    # Training setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=None
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train
    print("Starting LoRA training...")
    trainer.train()
    
    # Save
    print(f"Saving LoRA model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        "base_model": model_name,
        "lora_config": {
            "r": r,
            "alpha": alpha,
            "dropout": dropout
        },
        "training_examples": len(training_examples),
        "min_score": min_score,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print("="*50)
    print("LoRA TRAINING COMPLETE!")
    print(f"Model saved to: {output_dir}")
    print(f"Training examples: {len(training_examples)}")
    print("="*50)
    
    return output_dir

def test_model(model_dir):
    """Quick test of the trained model"""
    print(f"Testing model from {model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    test_prompt = """You are a black-box reward optimizer for multi-objective molecular design. 

OBJECTIVES: Maximizing: jnk3, qed; Minimizing: sa

Here are 2 positive samples:
[CC(C)c1ccc(C(=O)N2CCN(c3ccc(F)cc3)CC2)cc1, 2.01]
[COc1ccc(CN2CCN(C(=O)c3ccc(C(C)C)cc3)CC2)cc1, 1.98]

Here are 2 negative samples:
[CCc1ccc(C(=O)N2CCCCC2)cc1, 1.85]
[Nc1ccc(C(=O)O)cc1, 1.86]

Please generate exactly 1 new molecular structure that should achieve higher combined rewards.

For the molecule, provide both an explanation and the molecule using this exact format:
<explanation>Your reasoning</explanation> + <molecule>SMILES_STRING</molecule>

Generate 1 molecule with explanation:"""

    inputs = tokenizer(f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + 300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assistant_response = response[assistant_start:].strip()
    
    print("="*40)
    print("TEST OUTPUT:")
    print("="*40)
    print(assistant_response)
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="LoRA training on pareto trajectories")
    parser.add_argument("--pareto_file", required=True, help="Path to optimization_results_pareto.json")
    parser.add_argument("--output", default="./pareto_lora_model", help="Output directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
    parser.add_argument("--min_score", type=float, default=1.5, help="Min pareto score")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--test", action="store_true", help="Test model after training")
    
    args = parser.parse_args()
    
    # Train LoRA model
    model_dir = train_lora_model(
        pareto_file=args.pareto_file,
        output_dir=args.output,
        model_name=args.model,
        min_score=args.min_score,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Test if requested
    if args.test:
        test_model(model_dir)

if __name__ == "__main__":
    main() 