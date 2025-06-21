#!/usr/bin/env python
"""
LLM Black-box Optimizer V2

Uses Qwen-2.5-7B-Instruct via OpenRouter API.

Follows the specified algorithm:
1. Randomly generate n samples to initialize a candidate pool
2. For each iteration:
   - Sample 2m samples from pool using exp(±r(sample)) weighting
   - Score samples with reward oracle
   - Prompt LLM with specific format
   - Generate m new diverse samples
   - Calculate combined reward and do RL training
"""

import argparse
import numpy as np
import random
import os
import sys
import json
from typing import List, Tuple, Dict, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from openai import OpenAI
import re

# Add path for MolLEO imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from main.optimizer import Oracle as MolleoOracle
from main.molleo.GPT4 import sanitize_smiles

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "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")
)

class LLMBlackBoxOptimizerV2:
    def __init__(self, args):
        self.args = args
        self.oracle = MolleoOracle(args=args)
        self.candidate_pool = []  # List of (smiles, score) tuples
        self.model_name = "qwen/qwen-2.5-7b-instruct"
        
        print("🎯 Using Qwen-2.5-7B-Instruct via OpenRouter for molecular generation")
        
        # Test OpenRouter API connection
        if not self._test_openrouter_connection():
            print("❌ Cannot connect to OpenRouter API!")
            print("   Please check your OPENROUTER_API_KEY environment variable")
            raise ConnectionError("OpenRouter API not accessible")
        else:
            print("✅ OpenRouter API connection verified")
        
        # Load dataset for random sampling
        self.all_smiles = self._load_zinc_dataset()
        
        # Track optimization history
        self.iteration_results = []
        self.best_scores_history = []
        
    def _test_openrouter_connection(self) -> bool:
        """Test if the OpenRouter API is accessible"""
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10,
                temperature=0.1,
                extra_headers={
                    "HTTP-Referer": "https://github.com/molleo-optimizer",
                    "X-Title": "MolLEO Optimizer",
                }
            )
            return True
        except Exception as e:
            print(f"OpenRouter API test failed: {e}")
            return False
    
    def _load_zinc_dataset(self) -> List[str]:
        """Load ZINC dataset for random sampling"""
        zinc_path = "data/zinc.tab"
        
        if os.path.exists(zinc_path):
            print(f"Loading ZINC dataset from: {zinc_path}")
            with open(zinc_path, 'r') as f:
                lines = f.readlines()
                all_smiles = []
                for line in lines[1:]:  # Skip header
                    smi = line.strip().strip('"')
                    if smi and smi != 'nan':
                        all_smiles.append(smi)
                print(f"Loaded {len(all_smiles)} ZINC molecules")
                return all_smiles
        
        # Fallback to TDC
        print("ZINC file not found locally, using TDC MolGen...")
        from tdc.generation import MolGen
        data = MolGen(name='ZINC')
        all_smiles = data.get_data()['smiles'].tolist()
        print(f"Loaded {len(all_smiles)} ZINC molecules from TDC")
        return all_smiles
    
    def initialize_pool_random(self, n_samples: int):
        """Initialize candidate pool with n randomly sampled molecules"""
        print(f"Initializing candidate pool with {n_samples} random molecules...")
        
        # Initialize oracle evaluator
        from tdc import Oracle as TDCOracle
        tdc_oracle = TDCOracle(name=self.args.oracle)
        self.oracle.assign_evaluator(tdc_oracle)
        
        # Randomly sample molecules
        random_molecules = random.sample(self.all_smiles, min(n_samples, len(self.all_smiles)))
        
        # Score them
        print(f"Scoring {len(random_molecules)} random molecules...")
        for i, smi in enumerate(random_molecules):
            score = self.oracle.score_smi(smi)
            self.candidate_pool.append((smi, score))
            
            if (i + 1) % 100 == 0:
                print(f"  Scored {i + 1}/{len(random_molecules)} molecules")
        
        # Sort by score (best first)
        self.candidate_pool.sort(key=lambda x: x[1], reverse=True)
        
        scores = [score for _, score in self.candidate_pool]
        print(f"Random initialization complete:")
        print(f"  Pool size: {len(self.candidate_pool)}")
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Worst score: {min(scores):.4f}")
        print(f"  Mean score: {np.mean(scores):.4f}")
        print(f"  Best molecule: {self.candidate_pool[0][0]}")
    
    def sample_with_exponential_weighting(self, m: int) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Sample 2m samples from pool using exponential weighting:
        - Positive samples: weighted by exp(+r(sample))
        - Negative samples: weighted by exp(-r(sample))
        """
        if len(self.candidate_pool) < 2 * m:
            print(f"Warning: Pool size ({len(self.candidate_pool)}) < 2m ({2*m}), using all available molecules")
            half = len(self.candidate_pool) // 2
            return self.candidate_pool[:half], self.candidate_pool[half:]
        
        molecules = [smi for smi, _ in self.candidate_pool]
        scores = np.array([score for _, score in self.candidate_pool])
        
        # Normalize scores to prevent overflow in exp
        scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        
        # Positive sampling: higher weights for higher scores
        positive_weights = np.exp(scores_normalized)
        positive_weights = positive_weights / np.sum(positive_weights)
        
        # Negative sampling: higher weights for lower scores  
        negative_weights = np.exp(-scores_normalized)
        negative_weights = negative_weights / np.sum(negative_weights)
        
        # Sample without replacement
        positive_indices = np.random.choice(
            len(self.candidate_pool), 
            size=m, 
            replace=False, 
            p=positive_weights
        )
        
        # For negative sampling, exclude already selected positive indices
        remaining_indices = [i for i in range(len(self.candidate_pool)) if i not in positive_indices]
        remaining_weights = negative_weights[remaining_indices]
        remaining_weights = remaining_weights / np.sum(remaining_weights)
        
        negative_indices = np.random.choice(
            remaining_indices,
            size=min(m, len(remaining_indices)),
            replace=False,
            p=remaining_weights
        )
        
        positive_samples = [self.candidate_pool[i] for i in positive_indices]
        negative_samples = [self.candidate_pool[i] for i in negative_indices]
        
        pos_scores = [score for _, score in positive_samples]
        neg_scores = [score for _, score in negative_samples]
        
        print(f"Exponential weighted sampling:")
        print(f"  Positive samples: {len(positive_samples)}, score range: {min(pos_scores):.4f} - {max(pos_scores):.4f}")
        print(f"  Negative samples: {len(negative_samples)}, score range: {min(neg_scores):.4f} - {max(neg_scores):.4f}")
        
        return positive_samples, negative_samples
    
    def query_llm_with_specific_prompt(self, positive_samples: List[Tuple[str, float]], 
                                     negative_samples: List[Tuple[str, float]], 
                                     m: int) -> str:
        """Query LLM with the specific prompt format requested"""
        
        # Extract molecules and rewards
        positive_molecules = [smi for smi, _ in positive_samples]
        positive_rewards = [score for _, score in positive_samples]
        negative_molecules = [smi for smi, _ in negative_samples]
        negative_rewards = [score for _, score in negative_samples]
        
                # Create the specific prompt format
        prompt = f"""You are a black-box reward optimizer. Here are {len(positive_samples)} positive samples {positive_molecules} with corresponding reward {positive_rewards} and {len(negative_samples)} negative samples {negative_molecules} with corresponding reward {negative_rewards}. Please analyze the results, and output {m} new samples with rewards better than the positive samples. The new samples should be diversified.

        Please generate exactly {m} new diverse molecular structures as SMILES strings that should achieve higher rewards than the positive samples.

        For each molecule, provide both an explanation and the molecule using this exact format:
        <explanation>Your reasoning for why this molecule should have higher reward</explanation> + <molecule>SMILES_STRING</molecule>

        Generate {m} molecules:"""

        print(f"\nPrompting LLM with specific format:")
        print(f"  Positive samples: {len(positive_samples)}")
        print(f"  Negative samples: {len(negative_samples)}")
        print(f"  Requesting: {m} new molecules")
        
        # Query the LLM
        messages = [
            {"role": "system", "content": "You are a molecular design expert specializing in black-box optimization. Analyze the given examples and generate improved molecules."},
            {"role": "user", "content": prompt}
        ]
        
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=2048,
                    temperature=0.7,
                    messages=messages,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/molleo-optimizer",
                        "X-Title": "MolLEO Optimizer",
                    }
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenRouter API call failed (attempt {retry + 1}/3): {e}")
                if retry == 2:
                    raise e
        
        return None
    
    def extract_molecules_from_response(self, response: str) -> List[Tuple[str, str]]:
        """Extract SMILES strings and explanations from LLM response"""
        molecules_with_explanations = []
        
        # Look for the specific format: <explanation>...</explanation> + <molecule>...</molecule>
        pattern = r'<explanation>(.*?)</explanation>\s*\+\s*<molecule>(.*?)</molecule>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for explanation, molecule in matches:
            explanation = explanation.strip()
            molecule = molecule.strip()
            
            # Validate SMILES
            sanitized = sanitize_smiles(molecule)
            if sanitized:
                molecules_with_explanations.append((sanitized, explanation))
        
        # If the specific format isn't found, try fallback patterns
        if not molecules_with_explanations:
            print("  Specific format not found, trying fallback extraction...")
            
            # Try to find any molecule tags
            molecule_matches = re.findall(r'<molecule>(.*?)</molecule>', response, re.IGNORECASE)
            explanation_matches = re.findall(r'<explanation>(.*?)</explanation>', response, re.IGNORECASE)
            
            # Pair them up if we have both
            for i, molecule in enumerate(molecule_matches):
                molecule = molecule.strip()
                sanitized = sanitize_smiles(molecule)
                if sanitized:
                    explanation = explanation_matches[i].strip() if i < len(explanation_matches) else "No explanation provided"
                    molecules_with_explanations.append((sanitized, explanation))
        
        # Remove duplicates while preserving order
        unique_molecules_with_explanations = []
        seen = set()
        for mol, exp in molecules_with_explanations:
            if mol not in seen:
                unique_molecules_with_explanations.append((mol, exp))
                seen.add(mol)
        
        print(f"Extracted {len(unique_molecules_with_explanations)} valid unique molecules with explanations from LLM response")
        return unique_molecules_with_explanations
    
    def calculate_tanimoto_distance(self, smi1: str, smi2: str) -> float:
        """Calculate Tanimoto distance between two SMILES"""
        try:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            
            if mol1 is None or mol2 is None:
                return 1.0
            
            fp1 = GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            
            tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            return 1.0 - tanimoto_sim
            
        except Exception:
            return 1.0
    
    def calculate_diversity_score(self, molecules: List[str]) -> float:
        """Calculate diversity score for a set of molecules"""
        if len(molecules) <= 1:
            return 0.0
        
        distances = []
        for i in range(len(molecules)):
            for j in range(i + 1, len(molecules)):
                distance = self.calculate_tanimoto_distance(molecules[i], molecules[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_combined_reward(self, new_molecules: List[str], new_scores: List[float],
                                positive_samples: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Calculate combined reward with three weighted metrics:
        1. new_reward_max - positive_reward_max: Best new molecule vs best positive sample from prompt (weight: 0.2)
        2. new_reward_mean - positive_reward_mean: Mean new molecules vs mean positive samples from prompt (weight: 0.5)
        3. diversity of new_samples: Tanimoto distance-based diversity score (weight: 0.3)
        
        Combined = 0.2*reward_1 + 0.5*reward_2 + 0.3*reward_3
        
        Args:
            new_molecules: Newly generated molecules from LLM
            new_scores: Scores of newly generated molecules  
            positive_samples: The same {m} positive samples that were used in the LLM prompt
        """
        if not new_scores or not positive_samples:
            return {"reward_1": 0.0, "reward_2": 0.0, "reward_3": 0.0, "combined_reward": 0.0}
        
        positive_scores = [score for _, score in positive_samples]
        
        # Metric 1: Best new molecule score - Best positive sample score (from prompt)
        reward_1 = max(new_scores) - max(positive_scores)
        
        # Metric 2: Mean new molecules score - Mean positive samples score (from prompt)
        reward_2 = np.mean(new_scores) - np.mean(positive_scores)
        
        # Metric 3: diversity of new_samples
        reward_3 = self.calculate_diversity_score(new_molecules)
        
        # Combined reward with higher weights on mean and diversity
        combined_reward = 0.2 * reward_1 + 0.5 * reward_2 + 0.3 * reward_3
        
        return {
            "reward_1": reward_1,
            "reward_2": reward_2, 
            "reward_3": reward_3,
            "combined_reward": combined_reward
        }
    
    def optimize(self, n_init: int = 120, m: int = 5, max_iterations: int = 20):
        """Run the optimization following the specified algorithm"""
        
        print("="*80)
        print("LLM BLACK-BOX OPTIMIZER V2")
        print("="*80)
        
        # Step 1: Randomly generate n samples to initialize candidate pool
        self.initialize_pool_random(n_init)
        
        # Optimization loop
        for iteration in range(max_iterations):
            print(f"\n{'='*20} ITERATION {iteration + 1}/{max_iterations} {'='*20}")
            
            # Check stopping condition
            if self.oracle.finish:
                print("Oracle budget exhausted, stopping optimization")
                break
            
            # Step 2: Sample 2m samples from pool using exponential weighting
            positive_samples, negative_samples = self.sample_with_exponential_weighting(m)
            
            # Step 3: Prompt LLM with specific format
            response = self.query_llm_with_specific_prompt(positive_samples, negative_samples, m)
            
            if not response:
                print("Failed to get LLM response, skipping iteration")
                continue
            
            # Extract molecules with explanations from response
            molecules_with_explanations = self.extract_molecules_from_response(response)
            
            if not molecules_with_explanations:
                print("No valid molecules generated, skipping iteration")
                continue
            
            # Step 4: Calculate rewards of proposed samples
            print(f"\nScoring {len(molecules_with_explanations)} new molecules...")
            new_scores = []
            valid_molecules = []
            valid_explanations = []
            
            for i, (smi, explanation) in enumerate(molecules_with_explanations):
                # First check if molecule is chemically valid
                if sanitize_smiles(smi) is None:
                    # Invalid molecule - assign score of 0
                    score = 0.0
                    new_scores.append(score)
                    valid_molecules.append(smi)
                    valid_explanations.append(explanation)
                    print(f"  {i+1}. {smi}: 0.0000 (invalid chemistry)")
                    print(f"      💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
                else:
                    try:
                        score = self.oracle.score_smi(smi)
                        new_scores.append(score)
                        valid_molecules.append(smi)
                        valid_explanations.append(explanation)
                        print(f"  {i+1}. {smi}: {score:.4f}")
                        print(f"      💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
                    except Exception as e:
                        # Oracle failed - assign score of 0
                        score = 0.0
                        new_scores.append(score)
                        valid_molecules.append(smi)
                        valid_explanations.append(explanation)
                        print(f"  {i+1}. {smi}: 0.0000 (oracle error: {str(e)[:50]})")
                        print(f"      💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
                
                if self.oracle.finish:
                    break
            
            if not new_scores:
                print("No molecules could be scored, skipping iteration")
                continue
            
            # Step 5: Calculate combined reward and do RL training
            reward_metrics = self.calculate_combined_reward(valid_molecules, new_scores, positive_samples)
            
            print(f"\n🎯 Combined Reward Calculation:")
            print(f"  Reward 1 (max improvement): {reward_metrics['reward_1']:.4f} (weight: 0.2)")
            print(f"  Reward 2 (mean improvement): {reward_metrics['reward_2']:.4f} (weight: 0.5)")
            print(f"  Reward 3 (diversity): {reward_metrics['reward_3']:.4f} (weight: 0.3)")
            print(f"  Combined Reward: {reward_metrics['combined_reward']:.4f}")
            
            # Add new molecules to candidate pool
            for smi, score in zip(valid_molecules, new_scores):
                self.candidate_pool.append((smi, score))
            
            # Sort pool and keep best molecules
            self.candidate_pool.sort(key=lambda x: x[1], reverse=True)
            
            # Keep pool size manageable
            max_pool_size = 5000
            if len(self.candidate_pool) > max_pool_size:
                self.candidate_pool = self.candidate_pool[:max_pool_size]
            
            # Track best score
            current_best = self.candidate_pool[0][1]
            self.best_scores_history.append(current_best)
            
            # Print iteration summary
            print(f"\nIteration {iteration + 1} Summary:")
            print(f"  Generated molecules: {len(valid_molecules)}")
            print(f"  Best new score: {max(new_scores):.4f}")
            print(f"  Mean new score: {np.mean(new_scores):.4f}")
            print(f"  Current best overall: {current_best:.4f}")
            print(f"  Pool size: {len(self.candidate_pool)}")
            print(f"  Oracle calls used: {len(self.oracle)}")
            
            # Save iteration results
            iteration_data = {
                'iteration': iteration + 1,
                'positive_samples': [(smi, score) for smi, score in positive_samples],
                'negative_samples': [(smi, score) for smi, score in negative_samples],
                'generated_molecules': valid_molecules,
                'generated_explanations': valid_explanations,
                'generated_scores': new_scores,
                'reward_metrics': reward_metrics,
                'best_score_overall': current_best,
                'oracle_calls': len(self.oracle)
            }
            self.iteration_results.append(iteration_data)
            
            # Check for improvement
            positive_scores = [score for _, score in positive_samples]
            if max(new_scores) > max(positive_scores):
                print(f"  ✅ Generated molecule better than best positive sample!")
            
        # Final results
        print(f"\n{'='*20} OPTIMIZATION COMPLETE {'='*20}")
        print(f"Final pool size: {len(self.candidate_pool)}")
        print(f"Total oracle calls: {len(self.oracle)}")
        print(f"Best molecule found: {self.candidate_pool[0][0]}")
        print(f"Best score: {self.candidate_pool[0][1]:.4f}")
        
        # Save final results
        self._save_results()
        
        return self.candidate_pool
    
    def _save_results(self):
        """Save optimization results to JSON file"""
        results = {
            'optimization_info': {
                'oracle': self.args.oracle,
                'seed': self.args.seed,
                'algorithm': 'LLM_BlackBox_V2',
                'total_iterations': len(self.iteration_results)
            },
            'final_results': {
                'best_molecule': self.candidate_pool[0][0] if self.candidate_pool else None,
                'best_score': self.candidate_pool[0][1] if self.candidate_pool else 0.0,
                'total_oracle_calls': len(self.oracle),
                'pool_size': len(self.candidate_pool)
            },
            'best_scores_history': self.best_scores_history,
            'iteration_details': self.iteration_results,
            'top_molecules': [
                {'smiles': smi, 'score': score} 
                for smi, score in self.candidate_pool[:50]
            ]
        }
        
        output_file = os.path.join(self.args.output_dir, 'optimization_results_v2.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='LLM Black-box Optimizer V2')
    parser.add_argument('--oracle', type=str, default='jnk3', help='Oracle to optimize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_init', type=int, default=120, help='Initial pool size (random)')
    parser.add_argument('--m', type=int, default=5, help='Number of positive/negative samples per iteration')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations')
    parser.add_argument('--max_oracle_calls', type=int, default=20000, help='Maximum oracle calls')
    parser.add_argument('--output_dir', type=str, default='./results_blackbox_v2', help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add required attributes for MolLEO compatibility
    args.oracles = [args.oracle]
    args.mol_lm = 'GPT4'
    args.freq_log = 100
    args.n_jobs = 1
    args.log_results = True
    args.smi_file = None
    
    print(f"Starting LLM Black-box Optimization V2:")
    print(f"  Oracle: {args.oracle}")
    print(f"  Seed: {args.seed}")
    print(f"  Initial pool size (random): {args.n_init}")
    print(f"  Samples per iteration (m): {args.m}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Max oracle calls: {args.max_oracle_calls}")
    print(f"  Using Qwen-2.5-7B-Instruct via OpenRouter")
    
    # Create optimizer
    optimizer = LLMBlackBoxOptimizerV2(args)
    
    # Run optimization
    final_pool = optimizer.optimize(
        n_init=args.n_init,
        m=args.m,
        max_iterations=args.max_iterations
    )
    
    print("Optimization completed!")

if __name__ == "__main__":
    main() 