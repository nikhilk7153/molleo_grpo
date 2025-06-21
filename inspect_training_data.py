#!/usr/bin/env python3
"""
Script to inspect the training data extracted from pareto optimization results
"""

import json
import argparse

def extract_and_display_training_data(pareto_results_file, min_score_threshold=1.5, num_examples=5):
    """Extract and display training data from pareto optimization results"""
    print(f"Loading pareto optimization data from {pareto_results_file}...")
    
    try:
        with open(pareto_results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {pareto_results_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {pareto_results_file}")
        return
    
    training_examples = []
    
    # Extract from detailed iterations - use full LLM responses
    if 'detailed_iterations' in data:
        print(f"Found {len(data['detailed_iterations'])} iterations")
        
        for iteration in data['detailed_iterations']:
            iteration_num = iteration.get('iteration', 0)
            prompt = iteration.get('prompt_sent_to_llm', '')
            llm_response = iteration.get('llm_raw_response', '')
            
            # Only include examples where we have both prompt and response
            if prompt and llm_response:
                # Calculate average pareto score for this iteration
                generated_mols = iteration.get('generated_molecules_with_explanations', [])
                if generated_mols:
                    avg_pareto_score = sum(mol.get('pareto_score', 0.0) for mol in generated_mols) / len(generated_mols)
                    max_pareto_score = max(mol.get('pareto_score', 0.0) for mol in generated_mols)
                    
                    # Only include if the iteration has good molecules
                    if max_pareto_score >= min_score_threshold:
                        training_examples.append({
                            "prompt": prompt.strip(),
                            "response": llm_response.strip(),
                            "iteration": iteration_num,
                            "avg_pareto_score": avg_pareto_score,
                            "max_pareto_score": max_pareto_score,
                            "num_molecules": len(generated_mols)
                        })
    
    print(f"\nExtracted {len(training_examples)} training examples with max pareto score >= {min_score_threshold}")
    
    if len(training_examples) == 0:
        print("No training examples found. Try lowering min_score_threshold.")
        return
    
    # Show score distribution
    avg_scores = [ex['avg_pareto_score'] for ex in training_examples]
    max_scores = [ex['max_pareto_score'] for ex in training_examples]
    print(f"\nPareto score distribution:")
    print(f"  Average scores - Min: {min(avg_scores):.4f}, Max: {max(avg_scores):.4f}, Mean: {sum(avg_scores)/len(avg_scores):.4f}")
    print(f"  Max scores - Min: {min(max_scores):.4f}, Max: {max(max_scores):.4f}, Mean: {sum(max_scores)/len(max_scores):.4f}")
    print(f"  Examples from iterations: {sorted(set(ex['iteration'] for ex in training_examples))[:20]}...")
    
    # Sort by max pareto score (highest first) and show examples
    training_examples.sort(key=lambda x: x['max_pareto_score'], reverse=True)
    
    print(f"\n" + "="*100)
    print(f"SHOWING TOP {min(num_examples, len(training_examples))} TRAINING EXAMPLES")
    print("="*100)
    
    for i, example in enumerate(training_examples[:num_examples]):
        print(f"\n{'─'*80}")
        print(f"EXAMPLE {i+1}/{min(num_examples, len(training_examples))}")
        print(f"Iteration: {example['iteration']}")
        print(f"Max Pareto Score: {example['max_pareto_score']:.4f}")
        print(f"Avg Pareto Score: {example['avg_pareto_score']:.4f}")
        print(f"Number of molecules generated: {example['num_molecules']}")
        print(f"{'─'*80}")
        
        print(f"\n🔸 PROMPT:")
        print(f"{example['prompt']}")
        
        print(f"\n🔸 GROUND TRUTH RESPONSE (Full LLM Output with ~10 molecules):")
        print(f"{example['response']}")
        
        # Show the formatted conversation as it will appear in training
        formatted_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        print(f"\n🔸 FORMATTED FOR TRAINING (Qwen Chat Format):")
        print(f"{formatted_text}")
        
        if i < min(num_examples, len(training_examples)) - 1:
            print(f"\n{'='*100}")
    
    print(f"\n{'='*100}")
    print(f"SUMMARY:")
    print(f"  Total training examples: {len(training_examples)}")
    print(f"  Score threshold: {min_score_threshold}")
    print(f"  Average prompt length: {sum(len(ex['prompt']) for ex in training_examples) / len(training_examples):.1f} chars")
    print(f"  Average response length: {sum(len(ex['response']) for ex in training_examples) / len(training_examples):.1f} chars")
    avg_formatted_len = sum(len(f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>") for ex in training_examples) / len(training_examples)
    print(f"  Average formatted length: {avg_formatted_len:.1f} chars")
    print(f"  Total molecules in dataset: {sum(ex['num_molecules'] for ex in training_examples)}")
    print("="*100)

def main():
    parser = argparse.ArgumentParser(description="Inspect training data from pareto optimization results")
    parser.add_argument("--pareto_results_file", type=str, 
                       default="multi_objective/main/molleo_multi_pareto/results_pareto/optimization_results_pareto.json",
                       help="Path to pareto optimization results JSON file")
    parser.add_argument("--min_score_threshold", type=float, default=1.5,
                       help="Minimum pareto score for training examples")
    parser.add_argument("--num_examples", type=int, default=2,
                       help="Number of examples to display")
    
    args = parser.parse_args()
    
    extract_and_display_training_data(
        pareto_results_file=args.pareto_results_file,
        min_score_threshold=args.min_score_threshold,
        num_examples=args.num_examples
    )

if __name__ == "__main__":
    main() 