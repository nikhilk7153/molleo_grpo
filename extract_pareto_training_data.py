#!/usr/bin/env python3
"""
Extract training data from pareto optimization trajectories for SFT
"""

import json
import sys
import os
from pathlib import Path

def extract_sft_data_from_pareto(input_file, output_file, min_score_threshold=1.5):
    """
    Extract training data from pareto optimization results for SFT
    
    Args:
        input_file: Path to optimization_results_pareto.json
        output_file: Path to output training data
        min_score_threshold: Minimum pareto score to consider as positive example
    """
    
    print(f"Loading pareto optimization data from {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file}")
        return
    
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
        return
    
    # Show score distribution
    scores = [ex['pareto_score'] for ex in training_examples]
    print(f"Pareto score distribution:")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")
    
    # Save training data in jsonl format (common for SFT)
    with open(output_file, 'w') as f:
        for example in training_examples:
            # Only save prompt and response for training
            training_record = {
                "prompt": example["prompt"],
                "response": example["response"]
            }
            f.write(json.dumps(training_record) + '\n')
    
    print(f"Saved training data to {output_file}")
    
    # Also save in alpaca format for convenience
    alpaca_file = output_file.replace('.jsonl', '_alpaca.json')
    alpaca_format = []
    for example in training_examples:
        alpaca_format.append({
            "instruction": example["prompt"],
            "input": "",
            "output": example["response"]
        })
    
    with open(alpaca_file, 'w') as f:
        json.dump(alpaca_format, f, indent=2)
    
    print(f"Also saved in Alpaca format to {alpaca_file}")
    
    # Save detailed version with metadata
    detailed_file = output_file.replace('.jsonl', '_detailed.json')
    with open(detailed_file, 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"Saved detailed version with metadata to {detailed_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_pareto_training_data.py <input_json> <output_file> [min_score_threshold]")
        print("\nExample:")
        print("  python extract_pareto_training_data.py multi_objective/main/molleo_multi_pareto/results_pareto/optimization_results_pareto.json pareto_training_data.jsonl 1.5")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_score = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    
    extract_sft_data_from_pareto(input_file, output_file, min_score)

if __name__ == "__main__":
    main() 