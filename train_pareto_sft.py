#!/usr/bin/env python3
"""
Complete SFT training script for pareto optimization trajectories
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

def extract_training_data(pareto_results_file, min_score_threshold=1.5):
    """Extract training data from pareto optimization results"""
    print(f"Loading pareto optimization data from {pareto_results_file}...")
    
    try:
        with open(pareto_results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {pareto_results_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {pareto_results_file}")
        return []
    
    training_examples = []
    
    # Extract from detailed iterations
    if 'detailed_iterations' in data:
        print(f"Found {len(data['detailed_iterations'])} iterations")
        
        for iteration in data['detailed_iterations']:
            iteration_num = iteration.get('iteration', 0)
            prompt = iteration.get('prompt_sent_to_llm', '')
            
            # Get generated molecules with explanations
            generated_mols = iteration.get('generated_molecules_with_explanations', [])
            
            for mol_data in generated_mols:
                molecule = mol_data.get('molecule', '')
                explanation = mol_data.get('explanation', '')
                pareto_score = mol_data.get('pareto_score', 0.0)
                
                if molecule and explanation and pareto_score >= min_score_threshold:
                    # Format for SFT: prompt -> explanation + molecule
                    response = f"{explanation}\n\nSMILES: {molecule}"
                    
                    training_examples.append({
                        "prompt": prompt.strip(),
                        "response": response.strip(),
                        "iteration": iteration_num,
                        "pareto_score": pareto_score
                    })
    
    print(f"Extracted {len(training_examples)} training examples with pareto score >= {min_score_threshold}")
    
    if len(training_examples) == 0:
        print("No training examples found. Try lowering min_score_threshold.")
        return []
    
    # Show score distribution
    scores = [ex['pareto_score'] for ex in training_examples]
    print(f"Pareto score distribution:")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")
    print(f"  Examples from iterations: {sorted(set(ex['iteration'] for ex in training_examples))}")
    
    return training_examples

def format_conversation(example):
    """Format the conversation for training using Qwen chat format"""
    prompt = example['prompt']
    response = example['response']
    
    # Use Qwen's chat format
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    return {"text": formatted_text}

def train_sft_model(
    pareto_results_file,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./pareto_sft_model",
    min_score_threshold=1.5,
    max_length=2048,
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3,
    use_lora=True,
    lora_r=16,
    lora_alpha=32
):
    """
    Train SFT model on pareto optimization trajectories
    """
    
    print("="*60)
    print("PARETO OPTIMIZATION SFT TRAINING")
    print("="*60)
    
    # Extract training data
    training_examples = extract_training_data(pareto_results_file, min_score_threshold)
    if not training_examples:
        print("No training data found. Exiting.")
        return
    
    print(f"\nLoading model: {model_name}")
    
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
        print(f"Setting up LoRA configuration (r={lora_r}, alpha={lora_alpha})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Format training data
    print("Formatting training data...")
    formatted_data = [format_conversation(example) for example in training_examples]
    
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
        logging_dir=f"{output_dir}/logs",
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
    print("Starting SFT training on pareto trajectories...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training metadata
    metadata = {
        "base_model": model_name,
        "training_data_source": pareto_results_file,
        "min_score_threshold": min_score_threshold,
        "num_training_examples": len(training_examples),
        "score_distribution": {
            "min": float(min([ex['pareto_score'] for ex in training_examples])),
            "max": float(max([ex['pareto_score'] for ex in training_examples])),
            "mean": float(np.mean([ex['pareto_score'] for ex in training_examples]))
        },
        "training_params": {
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "use_lora": use_lora,
            "lora_r": lora_r if use_lora else None,
            "lora_alpha": lora_alpha if use_lora else None
        }
    }
    
    with open(f"{output_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("="*60)
    print("SFT TRAINING COMPLETED!")
    print("="*60)
    print(f"Model saved to: {output_dir}")
    print(f"Training examples used: {len(training_examples)}")
    print(f"Score threshold: {min_score_threshold}")
    print(f"Epochs: {num_epochs}")
    print(f"LoRA enabled: {use_lora}")
    
    return output_dir

def test_trained_model(model_dir):
    """Test the trained model with a sample prompt"""
    print(f"\nTesting trained model from {model_dir}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        
        # Sample prompt similar to pareto optimization
        test_prompt = """You are a black-box reward optimizer for multi-objective molecular design. 

OBJECTIVES: Maximizing: jnk3, qed; Minimizing: sa

Here are 3 positive samples:
[CC(C)c1ccc(C(=O)N2CCN(c3ccc(F)cc3)CC2)cc1, 2.0109]
[COc1ccc(CN2CCN(C(=O)c3ccc(C(C)C)cc3)CC2)cc1, 1.9884]
[Cc1ccc(C(=O)N2CCN(c3ccc(Cl)cc3)CC2)cc1, 1.9773]

Here are 2 negative samples:
[CCc1ccc(C(=O)N2CCCCC2)cc1, 1.8487]
[Nc1ccc(C(=O)O)cc1, 1.8592]

The combined reward balances multiple objectives. Higher rewards indicate better performance across the objective set.

Please generate exactly 2 new diverse molecular structures as SMILES strings that should achieve higher combined rewards than the positive samples. The new samples should be diversified and consider all objectives.

You can either make crossover and mutations based on the given molecules or just propose new molecules based on your knowledge.

For each molecule, provide both an explanation and the molecule using this exact format:
<explanation>Your reasoning for why this molecule should have higher multi-objective reward considering Maximizing: jnk3, qed; Minimizing: sa</explanation> + <molecule>SMILES_STRING</molecule>

Generate 2 molecules with their explanations. Do not output any other text:"""

        inputs = tokenizer(f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_length=inputs.input_ids.shape[1] + 512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        assistant_response = response[assistant_start:].strip()
        
        print("="*50)
        print("TEST GENERATION:")
        print("="*50)
        print(assistant_response)
        print("="*50)
        
    except Exception as e:
        print(f"Error testing model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train SFT model on pareto optimization trajectories")
    parser.add_argument("--pareto_results", required=True, help="Path to optimization_results_pareto.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--output", default="./pareto_sft_model", help="Output directory")
    parser.add_argument("--min_score", type=float, default=1.5, help="Minimum pareto score threshold")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--test", action="store_true", help="Test the model after training")
    
    args = parser.parse_args()
    
    # Train the model
    model_dir = train_sft_model(
        pareto_results_file=args.pareto_results,
        model_name=args.model,
        output_dir=args.output,
        min_score_threshold=args.min_score,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Test the model if requested
    if args.test:
        test_trained_model(model_dir)

if __name__ == "__main__":
    main() 