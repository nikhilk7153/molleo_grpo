#!/usr/bin/env python3
"""
Extract individual molecule training examples from pareto optimization results
Modifies the prompt to ask for 1 molecule instead of 10 for better SFT training
"""

import json
import argparse
import re

def extract_single_molecule_training_data(pareto_results_file, min_score_threshold=1.5, num_examples_to_show=3):
    """Extract individual molecule training examples from pareto optimization results"""
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
            original_prompt = iteration.get('prompt_sent_to_llm', '')
            
            # Get generated molecules with explanations
            generated_mols = iteration.get('generated_molecules_with_explanations', [])
            
            for mol_data in generated_mols:
                molecule = mol_data.get('molecule', '')
                explanation = mol_data.get('explanation', '')
                pareto_score = mol_data.get('pareto_score', 0.0)
                
                if molecule and explanation and pareto_score >= min_score_threshold:
                    # Modify the prompt to ask for 1 molecule instead of 10
                    modified_prompt = modify_prompt_for_single_molecule(original_prompt)
                    
                    # Format the response as explanation + SMILES
                    response = f"<explanation>{explanation}</explanation> + <molecule>{molecule}</molecule>"
                    
                    training_examples.append({
                        "prompt": modified_prompt.strip(),
                        "response": response.strip(),
                        "iteration": iteration_num,
                        "pareto_score": pareto_score,
                        "molecule": molecule,
                        "explanation": explanation
                    })
    
    print(f"\nExtracted {len(training_examples)} individual molecule training examples with pareto score >= {min_score_threshold}")
    
    if len(training_examples) == 0:
        print("No training examples found. Try lowering min_score_threshold.")
        return []
    
    # Show score distribution
    scores = [ex['pareto_score'] for ex in training_examples]
    print(f"\nPareto score distribution:")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")
    print(f"  Examples from iterations: {len(set(ex['iteration'] for ex in training_examples))} unique iterations")
    
    # Sort by pareto score (highest first) and show examples
    training_examples.sort(key=lambda x: x['pareto_score'], reverse=True)
    
    print(f"\n" + "="*100)
    print(f"SHOWING TOP {min(num_examples_to_show, len(training_examples))} SINGLE-MOLECULE TRAINING EXAMPLES")
    print("="*100)
    
    for i, example in enumerate(training_examples[:num_examples_to_show]):
        print(f"\n{'─'*80}")
        print(f"EXAMPLE {i+1}/{min(num_examples_to_show, len(training_examples))}")
        print(f"Iteration: {example['iteration']}")
        print(f"Pareto Score: {example['pareto_score']:.4f}")
        print(f"{'─'*80}")
        
        print(f"\n🔸 MODIFIED PROMPT (asks for 1 molecule):")
        print(f"{example['prompt']}")
        
        print(f"\n🔸 GROUND TRUTH RESPONSE:")
        print(f"{example['response']}")
        
        print(f"\n🔸 MOLECULE: {example['molecule']}")
        print(f"\n🔸 EXPLANATION: {example['explanation']}")
        
        # Show the formatted conversation as it will appear in training
        formatted_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        print(f"\n🔸 FORMATTED FOR TRAINING (Qwen Chat Format):")
        print(f"{formatted_text}")
        
        if i < min(num_examples_to_show, len(training_examples)) - 1:
            print(f"\n{'='*100}")
    
    print(f"\n{'='*100}")
    print(f"SUMMARY:")
    print(f"  Total single-molecule training examples: {len(training_examples)}")
    print(f"  Score threshold: {min_score_threshold}")
    print(f"  Average prompt length: {sum(len(ex['prompt']) for ex in training_examples) / len(training_examples):.1f} chars")
    print(f"  Average response length: {sum(len(ex['response']) for ex in training_examples) / len(training_examples):.1f} chars")
    avg_formatted_len = sum(len(f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>") for ex in training_examples) / len(training_examples)
    print(f"  Average formatted length: {avg_formatted_len:.1f} chars")
    print("="*100)
    
    return training_examples

def modify_prompt_for_single_molecule(original_prompt):
    """Modify the prompt to ask for 1 molecule instead of 10"""
    
    # Replace "exactly 10 new diverse molecular structures" with "1 new molecular structure"
    modified_prompt = re.sub(
        r"Please generate exactly 10 new diverse molecular structures",
        "Please generate 1 new molecular structure",
        original_prompt
    )
    
    # Replace "The new samples should be diversified" with "The new sample should"
    modified_prompt = re.sub(
        r"The new samples should be diversified and consider all objectives",
        "The new sample should consider all objectives and achieve higher combined rewards",
        modified_prompt
    )
    
    # Replace "Generate 10 molecules with their explanations" with "Generate 1 molecule with its explanation"
    modified_prompt = re.sub(
        r"Generate 10 molecules with their explanations\. Do not output any other text:",
        "Generate 1 molecule with its explanation. Do not output any other text:",
        modified_prompt
    )
    
    # Replace any remaining references to multiple molecules
    modified_prompt = re.sub(r"molecules", "molecule", modified_prompt)
    modified_prompt = re.sub(r"structures", "structure", modified_prompt)
    
    return modified_prompt

def save_training_data(training_examples, output_file):
    """Save training examples to a JSON file for use in SFT training"""
    formatted_examples = []
    
    for example in training_examples:
        formatted_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        formatted_examples.append({
            "text": formatted_text,
            "pareto_score": example['pareto_score'],
            "iteration": example['iteration']
        })
    
    with open(output_file, 'w') as f:
        json.dump(formatted_examples, f, indent=2)
    
    print(f"Saved {len(formatted_examples)} training examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract single-molecule training data from pareto optimization results")
    parser.add_argument("--pareto_results_file", type=str, 
                       default="multi_objective/main/molleo_multi_pareto/results_pareto/optimization_results_pareto.json",
                       help="Path to pareto optimization results JSON file")
    parser.add_argument("--min_score_threshold", type=float, default=1.5,
                       help="Minimum pareto score for training examples")
    parser.add_argument("--num_examples_to_show", type=int, default=3,
                       help="Number of examples to display")
    parser.add_argument("--save_training_data", type=str, default=None,
                       help="Save training data to JSON file")
    
    args = parser.parse_args()
    
    training_examples = extract_single_molecule_training_data(
        pareto_results_file=args.pareto_results_file,
        min_score_threshold=args.min_score_threshold,
        num_examples_to_show=args.num_examples_to_show
    )
    
    if args.save_training_data and training_examples:
        save_training_data(training_examples, args.save_training_data)

if __name__ == "__main__":
    main() 