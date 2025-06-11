#!/usr/bin/env python
"""
DPO Dataset Creation for LLM Black-box Optimizer - Optimized for 2x H100 GPUs

Creates datasets that can be used directly with verl's DPO training pipeline.
Inspired by the cell-o1 repository approach for reinforcement learning with verifiable rewards.

Usage:
    from dpo_step import create_dpo_dataset
    
    # After generating new molecules in the optimizer
    dataset = create_dpo_dataset(
        positive_samples, negative_samples, new_molecules, 
        oracle, output_dir='./dpo_data'
    )
"""

import os
import json
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import re

def sanitize_smiles(smiles: str) -> Optional[str]:
    """
    Sanitize and canonicalize SMILES string.
    Identical to the function in Qwen.py for consistency.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None

def extract_molecule_from_response(response: str) -> Optional[str]:
    """
    Extract molecule from model response using the same format as Qwen.py.
    Looks for molecules in \\box{SMILES} format.
    """
    # Extract from \box{SMILES} format (same as Qwen.py)
    pattern = r'\\box\{(.*?)\}'
    matches = re.findall(pattern, response)
    
    for match in matches:
        sanitized = sanitize_smiles(match.strip())
        if sanitized:
            return sanitized
    
    return None

def calculate_diversity_reward(molecules: List[str]) -> float:
    """
    Calculate diversity reward based on average Tanimoto similarity.
    Returns 1.0 - average_similarity as diversity measure.
    """
    if len(molecules) < 2:
        return 0.0
    
    valid_mols = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)
    
    if len(valid_mols) < 2:
        return 0.0
    
    # Calculate Morgan fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in valid_mols]
    
    # Calculate pairwise Tanimoto similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    if not similarities:
        return 0.0
    
    avg_similarity = np.mean(similarities)
    return 1.0 - avg_similarity  # Higher diversity = lower similarity

def calculate_combined_reward(new_molecules: List[str], 
                            positive_molecules: List[str],
                            oracle: Any,
                            max_improvement_weight: float = 0.5,
                            mean_improvement_weight: float = 0.3,
                            diversity_weight: float = 0.2) -> float:
    """
    Calculate the combined reward as specified in the requirements:
    1. new_reward_max - positive_reward_max (weight: 0.5)
    2. new_reward_mean - positive_reward_mean (weight: 0.3) 
    3. diversity of new_samples (weight: 0.2)
    """
    # Get rewards for all molecules
    new_rewards = []
    for smiles in new_molecules:
        if smiles:
            try:
                reward = oracle(smiles)
                new_rewards.append(reward)
            except:
                continue
    
    positive_rewards = []
    for smiles in positive_molecules:
        if smiles:
            try:
                reward = oracle(smiles)
                positive_rewards.append(reward)
            except:
                continue
    
    if not new_rewards or not positive_rewards:
        return 0.0
    
    # Component 1: Max improvement
    new_max = max(new_rewards)
    positive_max = max(positive_rewards)
    max_improvement = new_max - positive_max
    
    # Component 2: Mean improvement  
    new_mean = np.mean(new_rewards)
    positive_mean = np.mean(positive_rewards)
    mean_improvement = new_mean - positive_mean
    
    # Component 3: Diversity
    diversity_reward = calculate_diversity_reward(new_molecules)
    
    # Combined reward
    combined_reward = (
        max_improvement_weight * max_improvement +
        mean_improvement_weight * mean_improvement +
        diversity_weight * diversity_reward
    )
    
    return combined_reward

# SFT training prompts removed - focusing only on DPO preference pairs

def create_preference_pairs(molecules_with_rewards: List[Tuple[str, float]], 
                          num_pairs: int = 1000) -> List[Dict[str, Any]]:
    """
    Create preference pairs for DPO training.
    Higher reward molecules are preferred over lower reward ones.
    """
    if len(molecules_with_rewards) < 2:
        return []
    
    # Sort by reward
    sorted_molecules = sorted(molecules_with_rewards, key=lambda x: x[1], reverse=True)
    
    pairs = []
    
    for _ in range(num_pairs):
        # Sample two molecules with different rewards
        if len(sorted_molecules) < 2:
            break
            
        # Pick a high-reward molecule (preferred)
        high_idx = np.random.randint(0, min(len(sorted_molecules)//2, len(sorted_molecules)))
        high_smiles, high_reward = sorted_molecules[high_idx]
        
        # Pick a low-reward molecule (rejected)
        low_idx = np.random.randint(max(len(sorted_molecules)//2, 1), len(sorted_molecules))
        low_smiles, low_reward = sorted_molecules[low_idx]
        
        if high_reward <= low_reward:
            continue
        
        # Create the prompt
        prompt = """Generate a novel molecule that optimizes the given objective function.
Task: Generate a molecule with high reward.
Use the format: {{"<<<Explanation>>>": "Your reasoning here", "<<<Molecule>>>": "\\box{SMILES_HERE}"}}

Response:"""
        
        # Create preferred and rejected responses
        preferred_response = f'{{"<<<Explanation>>>": "I will generate a molecule that optimizes the objective function.", "<<<Molecule>>>": "\\box{{{high_smiles}}}"}}'
        rejected_response = f'{{"<<<Explanation>>>": "I will generate a molecule for the objective function.", "<<<Molecule>>>": "\\box{{{low_smiles}}}"}}'
        
        pairs.append({
            "prompt": prompt,
            "chosen": preferred_response,
            "rejected": rejected_response,
            "chosen_reward": high_reward,
            "rejected_reward": low_reward
        })
    
    return pairs

def create_dpo_dataset(positive_samples: List[str],
                      negative_samples: List[str], 
                      new_molecules: List[str],
                      oracle: Any,
                      output_dir: str = './dpo_data',
                      num_preference_pairs: int = 1000) -> Dict[str, Any]:
    """
    Create DPO dataset that can be used directly with verl.
    
    Args:
        positive_samples: List of high-reward SMILES
        negative_samples: List of low-reward SMILES  
        new_molecules: List of newly generated SMILES
        oracle: Oracle function to evaluate molecules
        output_dir: Directory to save dataset files
        num_preference_pairs: Number of preference pairs to create
        
    Returns:
        Dictionary with dataset statistics and file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all molecules and get their rewards
    all_molecules = positive_samples + negative_samples + new_molecules
    molecules_with_rewards = []
    
    for smiles in all_molecules:
        if smiles:
            sanitized = sanitize_smiles(smiles)
            if sanitized:
                try:
                    reward = oracle(sanitized)
                    molecules_with_rewards.append((sanitized, reward))
                except:
                    continue
    
    # Remove duplicates
    seen = set()
    unique_molecules = []
    for smiles, reward in molecules_with_rewards:
        if smiles not in seen:
            seen.add(smiles)
            unique_molecules.append((smiles, reward))
    
    molecules_with_rewards = unique_molecules
    
    print(f"Created dataset with {len(molecules_with_rewards)} unique molecules")
    
    # Create preference pairs for DPO (no SFT data)
    preference_pairs = create_preference_pairs(molecules_with_rewards, num_preference_pairs)
    
    # Calculate combined reward for the new molecules
    combined_reward = calculate_combined_reward(new_molecules, positive_samples, oracle)
    
    # Save only DPO preference data in JSON format compatible with verl
    preference_data = {
        "data": preference_pairs,
        "dataset_info": {
            "total_pairs": len(preference_pairs),
            "avg_reward_difference": np.mean([p["chosen_reward"] - p["rejected_reward"] for p in preference_pairs]) if preference_pairs else 0,
            "combined_reward": combined_reward
        }
    }
    
    # Save only preference pairs file
    preference_file = os.path.join(output_dir, "preference_pairs.json")
    
    with open(preference_file, 'w') as f:
        json.dump(preference_data, f, indent=2)
    
    # Create verl configuration for 2 H100s
    verl_config = {
        "model": {
            "path": "Qwen/Qwen2.5-7B-Instruct",  # or your preferred model
            "trust_remote_code": True
        },
        "training": {
            "algorithm": "dpo",
            "num_gpus": 2,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-7,
            "num_epochs": 3,
            "max_length": 2048,
            "beta": 0.1
        },
        "data": {
            "train_files": [preference_file],
            "eval_files": [preference_file],
            "prompt_key": "prompt",
            "chosen_key": "chosen", 
            "rejected_key": "rejected"
        },
        "output": {
            "save_dir": os.path.join(output_dir, "checkpoints"),
            "logging_dir": os.path.join(output_dir, "logs")
        }
    }
    
    config_file = os.path.join(output_dir, "verl_config.json")
    with open(config_file, 'w') as f:
        json.dump(verl_config, f, indent=2)
    
    # Create training script for verl
    training_script = f"""#!/bin/bash

# DPO Training Script for 2x H100 GPUs using verl
# Generated automatically by dpo_step.py

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install verl if not already installed
# pip install verl

# Run DPO training with verl
python -m verl.trainer.fsdp_dpo_trainer \\
    --config_path {config_file} \\
    --num_gpus 2 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 5e-7 \\
    --num_epochs 3 \\
    --max_length 2048 \\
    --beta 0.1 \\
    --output_dir {os.path.join(output_dir, "checkpoints")} \\
    --logging_dir {os.path.join(output_dir, "logs")} \\
    --save_steps 100 \\
    --eval_steps 100 \\
    --warmup_steps 50 \\
    --report_to tensorboard

echo "DPO training completed. Checkpoints saved to {os.path.join(output_dir, "checkpoints")}"
"""
    
    script_file = os.path.join(output_dir, "run_dpo_training.sh")
    with open(script_file, 'w') as f:
        f.write(training_script)
    
    os.chmod(script_file, 0o755)  # Make executable
    
    results = {
        "dataset_stats": {
            "total_molecules": len(molecules_with_rewards),
            "preference_pairs": len(preference_pairs),
            "combined_reward": combined_reward
        },
        "files": {
            "preference_data": preference_file,
            "verl_config": config_file,
            "training_script": script_file
        },
        "next_steps": [
            f"1. Review the generated preference pairs in {output_dir}",
            f"2. Run DPO training: bash {script_file}",
            "3. Monitor training progress in the logs directory",
            "4. Use the trained model for improved molecular generation"
        ]
    }
    
    print("\n" + "="*60)
    print("DPO Dataset Creation Complete!")
    print("="*60)
    print(f"📊 Generated {len(preference_pairs)} preference pairs (no SFT data)")
    print(f"🎯 Combined reward: {combined_reward:.3f}")
    print(f"💾 Files saved to: {output_dir}")
    print(f"🚀 To start training: bash {script_file}")
    print("="*60)
    
    return results

def add_dpo_step(llm_blackbox_optimizer_instance: Any,
                positive_samples: List[str], 
                negative_samples: List[str],
                new_molecules: List[str],
                oracle: Any) -> Dict[str, Any]:
    """
    Convenience function to add DPO step to existing optimizer.
    """
    return create_dpo_dataset(
        positive_samples=positive_samples,
        negative_samples=negative_samples, 
        new_molecules=new_molecules,
        oracle=oracle,
        output_dir='./dpo_results'
    )

def create_iteration_preferences(positive_samples: List[Tuple[str, float]], 
                               negative_samples: List[Tuple[str, float]], 
                               new_molecules: List[str],
                               oracle: Any,
                               oracle_name: str,
                               iteration: int) -> List[Dict[str, Any]]:
    """
    Create preference pairs for a single iteration in DPO training format.
    
    Args:
        positive_samples: List of (smiles, score) tuples for positive examples
        negative_samples: List of (smiles, score) tuples for negative examples  
        new_molecules: List of newly generated SMILES
        oracle: Oracle function to evaluate molecules
        oracle_name: Name of the oracle (e.g., 'jnk3')
        iteration: Current iteration number
        
    Returns:
        List of preference pairs in DPO format
    """
    # Combine all molecules from this iteration with their scores
    iteration_molecules = []
    
    # Add positive samples
    for smi, score in positive_samples:
        iteration_molecules.append((smi, score))
    
    # Add negative samples  
    for smi, score in negative_samples:
        iteration_molecules.append((smi, score))
        
    # Add new molecules (get their scores)
    for smi in new_molecules:
        sanitized = sanitize_smiles(smi)
        if sanitized:
            try:
                score = oracle.score_smi(smi) if hasattr(oracle, 'score_smi') else oracle(smi)
                iteration_molecules.append((sanitized, score))
            except:
                iteration_molecules.append((sanitized, 0.0))
    
    # Remove duplicates
    seen = set()
    unique_molecules = []
    for smi, score in iteration_molecules:
        if smi not in seen:
            seen.add(smi)
            unique_molecules.append((smi, score))
    
    if len(unique_molecules) < 2:
        return []
        
    # Sort by score (highest first)
    sorted_molecules = sorted(unique_molecules, key=lambda x: x[1], reverse=True)
    
    # Create preference pairs for this iteration
    pairs = []
    num_pairs = min(50, len(sorted_molecules) * 2)  # Create reasonable number of pairs
    
    for _ in range(num_pairs):
        if len(sorted_molecules) < 2:
            break
            
        # Pick a high-reward molecule (preferred) - from top 30%
        high_idx = np.random.randint(0, max(1, len(sorted_molecules) * 3 // 10))
        high_smiles, high_reward = sorted_molecules[high_idx]
        
        # Pick a low-reward molecule (rejected) - from bottom 50%
        low_idx = np.random.randint(max(1, len(sorted_molecules) // 2), len(sorted_molecules))
        low_smiles, low_reward = sorted_molecules[low_idx]
        
        if high_reward <= low_reward:
            continue
            
        # Create the prompt
        prompt = f"""Generate a molecule optimized for {oracle_name}. Analyze the following examples and generate a better performing molecule.

Task: Generate a molecule with high {oracle_name} score.
Use the format: {{"<<<Explanation>>>": "Your reasoning here", "<<<Molecule>>>": "\\box{{SMILES_HERE}}"}}

Response:"""
        
        # Create preferred and rejected responses
        preferred_response = f'{{"<<<Explanation>>>": "Based on molecular analysis, this structure should have high {oracle_name} activity.", "<<<Molecule>>>": "\\box{{{high_smiles}}}"}}'
        rejected_response = f'{{"<<<Explanation>>>": "This molecule has lower predicted {oracle_name} activity.", "<<<Molecule>>>": "\\box{{{low_smiles}}}"}}'
        
        pairs.append({
            "prompt": prompt,
            "chosen": preferred_response,
            "rejected": rejected_response,
            "chosen_reward": float(high_reward),
            "rejected_reward": float(low_reward),
            "iteration": iteration,
            "oracle": oracle_name
        })
    
    return pairs

def save_iteration_preferences(preferences: List[Dict[str, Any]], 
                             iteration: int, 
                             output_dir: str,
                             oracle_name: str,
                             all_preferences: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Save preference pairs for this iteration and return updated cumulative list.
    
    Args:
        preferences: Preference pairs for this iteration
        iteration: Current iteration number
        output_dir: Base output directory
        oracle_name: Name of the oracle
        all_preferences: Cumulative list of all preferences (will be updated)
        
    Returns:
        Updated cumulative list of all preferences
    """
    # Initialize cumulative list if not provided
    if all_preferences is None:
        all_preferences = []
        
    # Add to cumulative list (even if empty - we still want to track the iteration)
    if preferences:
        all_preferences.extend(preferences)
    
    # Create iteration-specific directory (always, even if no preferences)
    iter_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    
    # Save iteration preferences (or metadata even if empty)
    iter_data = {
        "iteration": iteration,
        "oracle": oracle_name,
        "data": preferences,
        "stats": {
            "num_pairs": len(preferences),
            "avg_reward_diff": np.mean([p["chosen_reward"] - p["rejected_reward"] for p in preferences]) if preferences else 0,
            "has_preferences": len(preferences) > 0,
            "reason_for_no_preferences": "Insufficient unique molecules (< 2)" if not preferences else None
        }
    }
    
    iter_file = os.path.join(iter_dir, "preferences.json")
    with open(iter_file, 'w') as f:
        json.dump(iter_data, f, indent=2)
    
    # Save cumulative preferences (all iterations so far)
    cumulative_data = {
        "total_iterations": iteration,
        "oracle": oracle_name,
        "data": all_preferences,
        "stats": {
            "total_pairs": len(all_preferences),
            "avg_reward_diff": np.mean([p["chosen_reward"] - p["rejected_reward"] for p in all_preferences]) if all_preferences else 0,
            "iterations_completed": iteration
        }
    }
    
    cumulative_file = os.path.join(output_dir, "all_preferences.json")
    with open(cumulative_file, 'w') as f:
        json.dump(cumulative_data, f, indent=2)
    
    if preferences:
        print(f"  💾 Saved {len(preferences)} preference pairs for iteration {iteration}")
    else:
        print(f"  ⚠️  Iteration {iteration}: No preference pairs created (insufficient unique molecules)")
    print(f"  📊 Total preference pairs collected: {len(all_preferences)}")
    
    return all_preferences

def create_final_dpo_training_files(all_preferences: List[Dict[str, Any]],
                                  output_dir: str,
                                  oracle_name: str,
                                  total_iterations: int) -> Dict[str, Any]:
    """
    Create final DPO training files with all accumulated preferences.
    
    Args:
        all_preferences: All preference pairs collected across iterations
        output_dir: Base output directory  
        oracle_name: Name of the oracle
        total_iterations: Total number of iterations completed
        
    Returns:
        Dictionary with file paths and statistics
    """
    if not all_preferences:
        return {"status": "no_preferences", "message": "No preference pairs collected for DPO training"}
        
    final_output_dir = os.path.join(output_dir, "final_dpo_training")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Create final training dataset
    final_data = {
        "data": all_preferences,
        "dataset_info": {
            "total_pairs": len(all_preferences),
            "total_iterations": total_iterations,
            "oracle": oracle_name,
            "avg_reward_difference": np.mean([p["chosen_reward"] - p["rejected_reward"] for p in all_preferences])
        }
    }
    
    # Save final preference dataset
    final_file = os.path.join(final_output_dir, "final_preferences.json")
    with open(final_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    # Create verl training config
    verl_config = {
        "model": {
            "path": "Qwen/Qwen2.5-7B-Instruct",
            "trust_remote_code": True
        },
        "training": {
            "algorithm": "dpo",
            "num_gpus": 2,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
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
    
    # Create training script
    training_script = f"""#!/bin/bash

# Final DPO Training Script for Molecular Optimization
# Generated from {total_iterations} iterations of blackbox optimization

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting DPO training with {len(all_preferences)} preference pairs..."
echo "Oracle: {oracle_name}"
echo "Iterations completed: {total_iterations}"

# Run DPO training with verl
python -m verl.trainer.fsdp_dpo_trainer \\
    --config_path {config_file} \\
    --num_gpus 2 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
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
            "training_script": script_file
        },
        "output_dir": final_output_dir
    }
    
    print(f"\n🎉 FINAL DPO Training Files Created! 🎉")
    print(f"✅ Total preference pairs: {len(all_preferences)}")
    print(f"📊 From {total_iterations} iterations")
    print(f"💾 Files saved to: {final_output_dir}")
    print(f"🚀 To start DPO training: bash {script_file}")
    print(f"📈 Avg reward difference: {final_data['dataset_info']['avg_reward_difference']:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("DPO Dataset Creation Module")
    print("This module creates datasets for verl DPO training")
    print("Use create_dpo_dataset() to generate training data")