#!/usr/bin/env python3
"""
Create train/validation split from the pareto training data
"""

import json
import random
import argparse
from pathlib import Path

def create_train_val_split(input_file, output_prefix="pareto_training", val_ratio=0.1, seed=42):
    """
    Split training data into train and validation sets
    
    Args:
        input_file: Path to the training data JSON file
        output_prefix: Prefix for output files
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    
    print(f"Loading training data from {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file}")
        return
    
    if not isinstance(data, list):
        print(f"Error: Expected a list of training examples, got {type(data)}")
        return
    
    print(f"Total examples: {len(data)}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split sizes
    val_size = int(len(shuffled_data) * val_ratio)
    train_size = len(shuffled_data) - val_size
    
    # Split the data
    val_data = shuffled_data[:val_size]
    train_data = shuffled_data[val_size:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Save train set
    train_file = f"{output_prefix}_train.json"
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved training data to {train_file}")
    
    # Save validation set
    val_file = f"{output_prefix}_val.json"
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved validation data to {val_file}")
    
    # Print some statistics
    if val_data:
        train_scores = [ex.get('pareto_score', 0) for ex in train_data]
        val_scores = [ex.get('pareto_score', 0) for ex in val_data]
        
        print(f"\nPareto score statistics:")
        print(f"  Train - Min: {min(train_scores):.3f}, Max: {max(train_scores):.3f}, Mean: {sum(train_scores)/len(train_scores):.3f}")
        print(f"  Val   - Min: {min(val_scores):.3f}, Max: {max(val_scores):.3f}, Mean: {sum(val_scores)/len(val_scores):.3f}")
    
    return train_file, val_file

def main():
    parser = argparse.ArgumentParser(description="Create train/validation split for pareto training data")
    parser.add_argument("--input_file", type=str, 
                       default="pareto_single_molecule_training_data.json",
                       help="Path to input training data JSON file")
    parser.add_argument("--output_prefix", type=str, default="pareto_training",
                       help="Prefix for output train/val files")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Fraction of data to use for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_train_val_split(
        input_file=args.input_file,
        output_prefix=args.output_prefix,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 