#!/usr/bin/env python
"""
Example Usage of DPO Dataset Creation for 2x H100 Setup

This script demonstrates how to create DPO datasets from molecular optimization results
and prepare them for training with verl on 2 H100 GPUs.

Usage:
    python example_dpo_usage.py
"""

import os
import sys
sys.path.append('.')

from dpo_step import create_dpo_dataset, calculate_combined_reward
from main.molleo import Oracle

def main():
    print("=" * 60)
    print("DPO Dataset Creation Example - 2x H100 Setup")
    print("=" * 60)
    
    # Example molecular optimization results
    # These would come from your actual optimization run
    positive_samples = [
        "CCO",           # Ethanol - simple, high QED
        "CC(C)O",        # Isopropanol  
        "CCCCO",         # Butanol
        "CC(C)(C)O",     # tert-Butanol
        "CCCCCO",        # Pentanol
        "CC(O)C",        # Acetone (with OH)
        "CCC(C)O",       # sec-Butanol
        "CCCO",          # Propanol
        "C(C(C)O)O",     # Propanediol
        "CC(C)CO"        # Isobutanol
    ]
    
    negative_samples = [
        "C",             # Methane - too simple
        "CC",            # Ethane
        "CCC",           # Propane  
        "CCCC",          # Butane
        "CCCCC",         # Pentane
        "CCCCCC",        # Hexane
        "C1CCCCC1",      # Cyclohexane
        "CCCCCCC",       # Heptane
        "CCCCCCCC",      # Octane
        "CCCCCCCCC"      # Nonane
    ]
    
    new_molecules = [
        "CCCO",          # Propanol - newly generated
        "CC(C)CO",       # Isobutanol variant
        "CCC(O)C",       # Butanone with OH
        "CCOC",          # Diethyl ether variant
        "CC(O)CC",       # 2-Butanol
        "CCCOC",         # Propyl methyl ether
        "CC(C)OC",       # Isopropyl methyl ether
        "CCCC(C)O"       # 2-Methyl-1-butanol
    ]
    
    print(f"✅ Loaded {len(positive_samples)} positive samples")
    print(f"✅ Loaded {len(negative_samples)} negative samples") 
    print(f"✅ Loaded {len(new_molecules)} new molecules")
    
    # Initialize oracle (you can change this to your target property)
    print("\n🔬 Initializing Oracle for QED (Drug-likeness)...")
    oracle = Oracle(prop='qed')
    
    # Calculate combined reward to show the system works
    print("\n📊 Calculating Combined Reward...")
    combined_reward = calculate_combined_reward(
        new_molecules=new_molecules,
        positive_molecules=positive_samples,
        oracle=oracle,
        max_improvement_weight=0.5,
        mean_improvement_weight=0.3, 
        diversity_weight=0.2
    )
    print(f"Combined Reward: {combined_reward:.4f}")
    
    # Create DPO dataset
    print("\n🏗️  Creating DPO Dataset...")
    output_dir = './dpo_example_data'
    
    results = create_dpo_dataset(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        new_molecules=new_molecules,
        oracle=oracle,
        output_dir=output_dir,
        num_preference_pairs=500  # Smaller for example
    )
    
    print("\n📈 Dataset Statistics:")
    stats = results['dataset_stats']
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n📁 Files Created:")
    for name, path in results['files'].items():
        print(f"  {name}: {path}")
    
    print(f"\n🚀 Next Steps:")
    for i, step in enumerate(results['next_steps'], 1):
        print(f"  {i}. {step}")
    
    # Show example of generated data
    print(f"\n🔍 Example Preference Pair:")
    import json
    with open(results['files']['preference_data'], 'r') as f:
        pref_data = json.load(f)
    
    if pref_data['data']:
        example = pref_data['data'][0]
        print(f"  Prompt: {example['prompt'][:100]}...")
        print(f"  Chosen (reward {example['chosen_reward']:.3f}): {example['chosen'][:100]}...")
        print(f"  Rejected (reward {example['rejected_reward']:.3f}): {example['rejected'][:100]}...")
    
    # Show verl config
    print(f"\n⚙️  Generated verl Config for 2x H100:")
    with open(results['files']['verl_config'], 'r') as f:
        config = json.load(f)
    
    print(f"  Model: {config['model']['path']}")
    print(f"  GPUs: {config['training']['num_gpus']}")
    print(f"  Batch Size per Device: {config['training']['per_device_train_batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    
    print(f"\n✨ Ready for DPO Training!")
    print(f"Run: cd {output_dir} && bash run_dpo_training.sh")
    print("=" * 60)

def demo_custom_rewards():
    """Demonstrate custom reward weighting"""
    print("\n🎛️  Custom Reward Weighting Demo")
    print("-" * 40)
    
    # Example molecules
    new_mols = ["CCO", "CCCO", "CC(C)O"]
    pos_mols = ["CCO", "CCC(C)O"] 
    oracle = Oracle(prop='qed')
    
    # Different weighting schemes
    configs = [
        {"max": 0.7, "mean": 0.2, "div": 0.1, "name": "Max-focused"},
        {"max": 0.3, "mean": 0.6, "div": 0.1, "name": "Mean-focused"}, 
        {"max": 0.2, "mean": 0.2, "div": 0.6, "name": "Diversity-focused"},
        {"max": 0.5, "mean": 0.3, "div": 0.2, "name": "Balanced"}
    ]
    
    for config in configs:
        reward = calculate_combined_reward(
            new_molecules=new_mols,
            positive_molecules=pos_mols,
            oracle=oracle,
            max_improvement_weight=config["max"],
            mean_improvement_weight=config["mean"],
            diversity_weight=config["div"]
        )
        print(f"  {config['name']:15} (max:{config['max']}, mean:{config['mean']}, div:{config['div']}): {reward:.4f}")

if __name__ == "__main__":
    main()
    demo_custom_rewards()
    
    print(f"\n💡 Tips:")
    print(f"  - Adjust batch sizes in verl_config.json for different model sizes")
    print(f"  - Monitor GPU usage with: nvidia-smi -l 1")
    print(f"  - Use tensorboard for training visualization")
    print(f"  - Scale num_preference_pairs for larger datasets") 