#!/usr/bin/env python3
"""
Filter preference pairs to keep only those with reward gap > 0.08
"""

import json
import os

def filter_preference_pairs(input_file, output_file, min_gap=0.08):
    """
    Filter preference pairs to keep only those with sufficient reward gap.
    
    Args:
        input_file: Path to input preference_pairs.json
        output_file: Path to output filtered file
        min_gap: Minimum gap between chosen and rejected rewards
    """
    
    print(f"Loading preference pairs from: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    original_count = len(data['data'])
    print(f"Original preference pairs: {original_count}")
    
    # Filter pairs based on reward gap
    filtered_pairs = []
    gaps = []
    
    for pair in data['data']:
        chosen_reward = pair['chosen_reward']
        rejected_reward = pair['rejected_reward']
        gap = chosen_reward - rejected_reward
        gaps.append(gap)
        
        if gap > min_gap:
            filtered_pairs.append(pair)
    
    filtered_count = len(filtered_pairs)
    print(f"Filtered preference pairs (gap > {min_gap}): {filtered_count}")
    print(f"Filtering ratio: {filtered_count/original_count:.3f}")
    
    # Statistics
    print(f"\nGap statistics:")
    print(f"  Min gap: {min(gaps):.4f}")
    print(f"  Max gap: {max(gaps):.4f}")
    print(f"  Mean gap: {sum(gaps)/len(gaps):.4f}")
    
    # Count gaps by ranges
    gap_ranges = [
        (0.0, 0.02),
        (0.02, 0.05), 
        (0.05, 0.08),
        (0.08, 0.1),
        (0.1, 0.2),
        (0.2, float('inf'))
    ]
    
    print(f"\nGap distribution:")
    for min_g, max_g in gap_ranges:
        count = sum(1 for g in gaps if min_g <= g < max_g)
        print(f"  {min_g:.2f} - {max_g:.2f}: {count} pairs ({count/len(gaps)*100:.1f}%)")
    
    # Create filtered dataset
    filtered_data = {
        'data': filtered_pairs,
        'metadata': {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'min_gap_threshold': min_gap,
            'filtering_ratio': filtered_count/original_count
        }
    }
    
    # Save filtered data
    print(f"\nSaving filtered data to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"✅ Filtering complete!")
    print(f"📊 Kept {filtered_count}/{original_count} pairs ({filtered_count/original_count*100:.1f}%)")
    
    return filtered_data

if __name__ == "__main__":
    input_file = "preference_pairs.json"
    output_file = "preference_pairs_filtered_0.08.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        exit(1)
    
    # Filter with gap > 0.08
    filtered_data = filter_preference_pairs(input_file, output_file, min_gap=0.08) 