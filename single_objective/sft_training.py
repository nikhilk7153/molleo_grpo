#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Script for Molecular Generation
Trains on explanation + molecule pairs from evolutionary algorithm output

Usage:
    python sft_training.py --data_dir evolutionary_output_* --model_name gpt-4o-mini --output_dir ./sft_model
"""

import os
import json
import csv
import argparse
from typing import List, Dict, Tuple
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import DataLoader
import wandb
from datetime import datetime

class MolecularSFTDataset:
    def __init__(self, data_dirs: List[str], min_score_threshold: float = 0.1):
        """
        Initialize dataset from evolutionary algorithm output directories
        
        Args:
            data_dirs: List of output directories from evolutionary runs
            min_score_threshold: Minimum score threshold for including molecules
        """
        self.data_dirs = data_dirs
        self.min_score_threshold = min_score_threshold
        self.training_data = []
        
    def load_data(self):
        """Load and process data from all output directories"""
        print(f"Loading data from {len(self.data_dirs)} directories...")
        
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Directory {data_dir} not found, skipping...")
                continue
                
            # Load from JSON file (preferred)
            json_file = os.path.join(data_dir, "run_data.json")
            if os.path.exists(json_file):
                self._load_from_json(json_file)
            else:
                # Fallback to CSV
                csv_file = os.path.join(data_dir, "generations.csv")
                if os.path.exists(csv_file):
                    self._load_from_csv(csv_file)
                    
        print(f"Loaded {len(self.training_data)} training examples")
        return self.training_data
    
    def _load_from_json(self, json_file: str):
        """Load data from JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        oracle_name = data.get('oracle_name', 'unknown')
        
        for generation in data.get('generations', []):
            for molecule in generation.get('molecules', []):
                explanation = molecule.get('explanation', '').strip()
                smiles = molecule.get('smiles', '').strip()
                score = molecule.get('score', 0.0)
                
                # Filter by score and explanation quality (score used only for filtering, not training)
                if (score >= self.min_score_threshold and 
                    explanation and 
                    explanation != "No explanation available" and
                    explanation != "Baseline genetic operations (crossover + mutation)" and
                    smiles):
                    
                    self.training_data.append({
                        'oracle': oracle_name,
                        'smiles': smiles,
                        'explanation': explanation,
                        'generation': generation.get('generation', 0)
                        # Note: score excluded from training data - only used for filtering
                    })
    
    def _load_from_csv(self, csv_file: str):
        """Load data from CSV file"""
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            explanation = str(row.get('explanation', '')).strip()
            smiles = str(row.get('smiles', '')).strip()
            score = float(row.get('score', 0.0))
            
            if (score >= self.min_score_threshold and 
                explanation and 
                explanation != "No explanation available" and
                explanation != "Baseline genetic operations (crossover + mutation)" and
                explanation != "nan" and
                smiles):
                
                self.training_data.append({
                    'oracle': 'unknown',
                    'smiles': smiles,
                    'explanation': explanation,
                    'generation': int(row.get('generation', 0))
                    # Note: score excluded from training data - only used for filtering
                })

class MolecularSFTTrainer:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 output_dir: str = "./sft_model",
                 max_length: int = 512):
        """
        Initialize the SFT trainer
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save the fine-tuned model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
    def create_training_prompt(self, oracle: str, parent_molecules: str, explanation: str, target_molecule: str) -> str:
        """Create training prompt in the format expected by the model
        
        Note: This creates input-output pairs where:
        - Input: Oracle description + parent molecules + task instruction
        - Output: Explanation + target molecule
        - Score is NOT included in training (only used for data filtering)
        """
        prompt = f"""You are a helpful agent who can answer questions based on your molecule knowledge.

I have molecules and their {oracle} scores. The {oracle} score measures molecular activity against {oracle}.

{parent_molecules}

Please propose a new molecule that has a higher {oracle} score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.

Your output should follow the format: {{<<<Explanation>>>: $EXPLANATION, <<<Molecule>>>: \\box{{$Molecule}}}}. Here are the requirements:

1. $EXPLANATION should be your analysis.
2. The $Molecule should be the smiles of your proposed molecule.
3. The molecule should be valid.

<<<Explanation>>>: {explanation}, <<<Molecule>>>: \\box{{{target_molecule}}}"""
        
        return prompt
    
    def prepare_dataset(self, training_data: List[Dict]) -> Dataset:
        """Prepare dataset for training"""
        print("Preparing dataset...")
        
        formatted_data = []
        
        for item in training_data:
            # Create mock parent molecules (in real scenario, you'd track actual parents)
            parent_molecules = f"\n[MOCK_PARENT_1,0.05]\n[MOCK_PARENT_2,0.03]"
            
            prompt = self.create_training_prompt(
                oracle=item['oracle'],
                parent_molecules=parent_molecules,
                explanation=item['explanation'],
                target_molecule=item['smiles']
            )
            
            formatted_data.append({'text': prompt})
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    def train(self, 
              dataset: Dataset,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              logging_steps: int = 10,
              save_steps: int = 500,
              use_wandb: bool = False):
        """Train the model"""
        
        if use_wandb:
            wandb.init(
                project="molecular-sft",
                name=f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_name": self.model_name,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_length": self.max_length
                }
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        train_size = int(0.9 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if use_wandb else None,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training completed! Model saved to {self.output_dir}")
        
        if use_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on molecular explanations")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                       help="Directories containing evolutionary algorithm output")
    parser.add_argument("--model_name", default="microsoft/DialoGPT-small",
                       help="Base model to fine-tune")
    parser.add_argument("--output_dir", default="./sft_molecular_model",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--min_score", type=float, default=0.1,
                       help="Minimum score threshold for including molecules")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Load data
    dataset_loader = MolecularSFTDataset(args.data_dirs, args.min_score)
    training_data = dataset_loader.load_data()
    
    if len(training_data) == 0:
        print("No training data found! Make sure you have run the evolutionary algorithm with explanations.")
        return
    
    # Initialize trainer
    trainer = MolecularSFTTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length
    )
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(training_data)
    
    # Train
    trainer.train(
        dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Training examples used: {len(training_data)}")

if __name__ == "__main__":
    main() 