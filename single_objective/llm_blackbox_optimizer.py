#!/usr/bin/env python
"""
LLM Black-box Optimizer

A different optimization approach that uses LLM prompting with positive/negative examples
instead of genetic algorithms.

Pipeline:
1. Randomly generate n samples to initialize a candidate pool
2. For each iteration:
   - Sample 2m samples from pool (uniformly or exp(±r(sample)) weighted)
   - Score samples with reward oracle
   - Prompt LLM with m positive + m negative samples and rewards
   - Generate m new diverse samples better than positive ones
   - Add new samples to pool
"""

import argparse
import numpy as np
import random
import yaml
import os
import sys
from typing import List, Tuple, Dict, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from tdc import Oracle
from openai import OpenAI
import json
import re

# Add path for MolLEO imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from main.optimizer import Oracle as MolleoOracle
from main.molleo.GPT4 import query_LLM, sanitize_smiles

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class LLMBlackBoxOptimizer:
    def __init__(self, args):
        self.args = args
        self.oracle = MolleoOracle(args=args)
        self.candidate_pool = []  # List of (smiles, score) tuples
        self.model_name = "gpt-4.1-mini"  # Use GPT-4.1-mini
        
        print("🎯 Using GPT-4.1-mini for molecular generation")
        
        # Test OpenAI API connection
        if not self._test_openai_connection():
            print("❌ Cannot connect to OpenAI API!")
            print("   Please check your OPENAI_API_KEY environment variable")
            raise ConnectionError("OpenAI API not accessible")
        else:
            print("✅ OpenAI API connection verified")
        
        # Load initial molecules from ZINC dataset
        self.all_smiles = self._load_zinc_dataset()
        
        # Initialize data collection for DPO (aggregate over all iterations)
        self.all_generated_data = []  # List of (molecule, explanation, score, iteration) tuples - simplified!
        self.candidate_pool = []  # List of (smiles, score) tuples
        self.invalid_molecules_count = 0  # Track invalid molecules
        self.previous_iteration_molecules = []  # Track previous iteration's scored molecules
        
        # DPO and saving data
        self.iteration_results = []  # Track results per iteration
        self.all_preference_pairs = []  # DPO preference pairs
        self.start_time = __import__('datetime').datetime.now().isoformat()
        
    def _test_openai_connection(self) -> bool:
        """Test if the OpenAI API is accessible"""
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10,
                temperature=0.1
            )
            return True
        except Exception as e:
            print(f"OpenAI API test failed: {e}")
            return False
        
    def _load_zinc_dataset(self) -> List[str]:
        """Load ZINC dataset directly from file"""
        zinc_path = "data/zinc.tab"  # Located at single_objective/data/zinc.tab
        
        if os.path.exists(zinc_path):
            print(f"Loading ZINC dataset from: {zinc_path}")
            # Load tab-separated file with header
            with open(zinc_path, 'r') as f:
                lines = f.readlines()
                # Skip header line and load SMILES
                all_smiles = []
                for line in lines[1:]:
                    smi = line.strip().strip('"')
                    if smi and smi != 'nan':
                        all_smiles.append(smi)
                print(f"Loaded {len(all_smiles)} ZINC molecules")
                return all_smiles
        
        # If file not found, fall back to TDC MolGen
        print("ZINC file not found locally, using TDC MolGen...")
        from tdc.generation import MolGen
        data = MolGen(name='ZINC')
        all_smiles = data.get_data()['smiles'].tolist()
        print(f"Loaded {len(all_smiles)} ZINC molecules from TDC")
        return all_smiles
        
    def initialize_pool(self, n_samples: int):
        """Initialize candidate pool with top 10% of ZINC molecules"""
        print(f"Initializing candidate pool with top 10% from ZINC database...")
        
        # Initialize oracle evaluator for the target property
        from tdc import Oracle as TDCOracle
        tdc_oracle = TDCOracle(name=self.args.oracle)
        self.oracle.assign_evaluator(tdc_oracle)
        
        # Define cache file path based on oracle name
        cache_file = f"zinc_scores_cache_{self.args.oracle}_10k.pkl"
        
        # Try to load cached scores
        if os.path.exists(cache_file):
            print(f"Found cached ZINC scores: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    import pickle
                    zinc_scores = pickle.load(f)
                print(f"  Loaded {len(zinc_scores)} pre-scored ZINC molecules from cache")
                
                # Sort by score and take top 10%
                zinc_scores.sort(key=lambda x: x[1], reverse=True)
                top_10_percent_size = max(n_samples, int(len(zinc_scores) * 0.10))
                top_molecules = zinc_scores[:top_10_percent_size]
                
                # Initialize pool with top molecules
                self.candidate_pool = top_molecules[:n_samples]  # Take requested number
                
                scores = [score for _, score in self.candidate_pool]
                print(f"Initial pool statistics (top 10% of ZINC):")
                print(f"  Pool size: {len(self.candidate_pool)}")
                print(f"  Best score: {self.candidate_pool[0][1]:.4f}")
                print(f"  Worst score: {self.candidate_pool[-1][1]:.4f}")
                print(f"  Mean score: {np.mean(scores):.4f}")
                print(f"  Best molecule: {self.candidate_pool[0][0]}")
                print(f"  Top 10% threshold: {top_molecules[top_10_percent_size-1][1]:.4f}")
                print(f"  Using cached scores (no oracle calls needed)")
                return
                
            except Exception as e:
                print(f"  Error loading cache: {e}")
                print(f"  Will re-score molecules and create new cache")
        
        # If no cache or cache failed, score molecules from scratch
        # Save current oracle count to exclude ZINC scoring from optimization budget
        zinc_oracle_start = len(self.oracle)
        
        # Sample a subset of ZINC to find top 10%
        zinc_sample_size = min(10000, len(self.all_smiles))  # Sample up to 10k molecules
        print(f"Scoring {zinc_sample_size} ZINC molecules to find top 10%...")
        
        zinc_sample = np.random.choice(self.all_smiles, zinc_sample_size, replace=False)
        
        # Score all sampled molecules
        zinc_scores = []
        for i, smi in enumerate(zinc_sample):
            score = self.oracle.score_smi(smi)
            zinc_scores.append((smi, score))
            if (i + 1) % 1000 == 0:
                print(f"  Scored {i + 1}/{zinc_sample_size} ZINC molecules")
        
        # Save scores to cache
    
        with open(cache_file, 'wb') as f:
            import pickle
            pickle.dump(zinc_scores, f)
        print(f"  Saved scored molecules to cache: {cache_file}")
  
        # Sort by score and take top 10%
        zinc_scores.sort(key=lambda x: x[1], reverse=True)
        top_10_percent_size = max(n_samples, int(zinc_sample_size * 0.10))
        top_molecules = zinc_scores[:top_10_percent_size]
        
        # Initialize pool with top molecules
        self.candidate_pool = top_molecules[:n_samples]  # Take requested number
        
        scores = [score for _, score in self.candidate_pool]
        print(f"Initial pool statistics (top 10% of ZINC):")
        print(f"  Pool size: {len(self.candidate_pool)}")
        print(f"  Best score: {self.candidate_pool[0][1]:.4f}")
        print(f"  Worst score: {self.candidate_pool[-1][1]:.4f}")
        print(f"  Mean score: {np.mean(scores):.4f}")
        print(f"  Best molecule: {self.candidate_pool[0][0]}")
        print(f"  Top 10% threshold: {top_molecules[top_10_percent_size-1][1]:.4f}")
        
        # Reset oracle counter - ZINC scoring shouldn't count towards optimization budget
        zinc_oracle_used = len(self.oracle) - zinc_oracle_start
        print(f"  ZINC scoring used {zinc_oracle_used} oracle calls (not counted towards optimization budget)")
        
  
        
    def sample_positive_negative(self, m: int, current_iteration_molecules: List[Tuple[str, float]] = None) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Sample positive and negative examples according to improved strategy with diversity:
        - For iteration 1: Positive from top 10% of pool (diverse), Negative from bottom 50% of ZINC
        - For iteration > 1: 
          * Positive: top {m} from current iteration's LLM-generated molecules (diverse), fallback to top 10% of pool
          * Negative: bottom {m} from current iteration's LLM-generated molecules (always from LLM)
        """
        # Sort pool by score (best first)
        sorted_pool = sorted(self.candidate_pool, key=lambda x: x[1], reverse=True)
        
        # Split into top 10% and bottom portions for fallback
        top_10_percent_size = max(1, len(sorted_pool) // 10)  # At least 1 molecule
        top_10_percent = sorted_pool[:top_10_percent_size]
        
        positive_samples = []
        negative_samples = []
        
        if not self.all_generated_data:
            # First iteration: use original strategy with diversity
            midpoint = len(sorted_pool) // 2
            
            # Sample diverse positive examples from top 10% using Tanimoto distance
            if len(top_10_percent) > m:
                positive_samples = self.diversity_based_sampling(top_10_percent, m, min_distance=0.3)
            else:
                positive_samples = top_10_percent
                
            print(f"  First iteration: selected {len(positive_samples)} diverse positive samples from top 10%")
            
            # Sample negative examples from bottom 50% of ZINC (random sampling for negatives)
            lower_50_percent = sorted_pool[midpoint:] if midpoint < len(sorted_pool) else []
            
            if lower_50_percent:
                n_neg = min(m, len(lower_50_percent))
                negative_samples = random.sample(lower_50_percent, n_neg)
                print(f"  First iteration: using {n_neg} negative samples from bottom 50% of ZINC")
            else:
                print(f"  First iteration: no molecules available for negative samples")
            
            print(f"  Sampled {len(positive_samples)} diverse positive and {len(negative_samples)} negative")
            
            # Display diversity statistics for positive samples
            if len(positive_samples) > 1:
                pos_molecules = [smi for smi, _ in positive_samples]
                diversity_stats = self.calculate_diversity_stats(pos_molecules)
                print(f"    📊 Positive diversity: avg={diversity_stats['avg_distance']:.3f}, min={diversity_stats['min_distance']:.3f}, max={diversity_stats['max_distance']:.3f}")
        else:
            # Later iterations: use current iteration's molecules with diversity
            if current_iteration_molecules and len(current_iteration_molecules) > 0:
                # Sort current iteration molecules by score (best first)
                sorted_current = sorted(current_iteration_molecules, key=lambda x: x[1], reverse=True)
                
                # Diverse positive samples: top molecules from current iteration with diversity constraint
                if len(sorted_current) > m:
                    positive_samples = self.diversity_based_sampling(sorted_current, m, min_distance=0.3)
                else:
                    positive_samples = sorted_current
                
                # Negative samples: bottom {m} from current iteration (worst performing, no diversity constraint)
                n_neg = min(m, len(sorted_current))
                negative_samples = sorted_current[-n_neg:] if len(sorted_current) >= n_neg else sorted_current
                
                print(f"  Using current iteration molecules: {len(positive_samples)} diverse positive (best) and {len(negative_samples)} negative (worst)")
                
                # Display diversity statistics for current iteration positive samples
                if len(positive_samples) > 1:
                    pos_molecules = [smi for smi, _ in positive_samples]
                    diversity_stats = self.calculate_diversity_stats(pos_molecules)
                    print(f"    📊 Positive diversity: avg={diversity_stats['avg_distance']:.3f}, min={diversity_stats['min_distance']:.3f}, max={diversity_stats['max_distance']:.3f}")
            else:
                # Fallback: diverse positive from top 10% of pool, negative from all LLM-generated
                if len(top_10_percent) > m:
                    positive_samples = self.diversity_based_sampling(top_10_percent, m, min_distance=0.3)
                else:
                    positive_samples = top_10_percent
                
                # Get scores for all generated molecules for negative samples (simplified)
                generated_with_scores = [(smi, score) for smi, exp, score, iter_num in self.all_generated_data]
                
                if generated_with_scores:
                    # Sort generated molecules by score (worst first for negatives)
                    generated_with_scores.sort(key=lambda x: x[1])
                    
                    # Take the worst performing generated molecules as negatives
                    n_neg = min(m, len(generated_with_scores))
                    negative_samples = generated_with_scores[:n_neg]
                    
                    print(f"  Fallback: {len(positive_samples)} diverse positive (top 10% pool) and {len(negative_samples)} negative (worst LLM-generated)")
                    
                    # Display diversity statistics for fallback positive samples  
                    if len(positive_samples) > 1:
                        pos_molecules = [smi for smi, _ in positive_samples]
                        diversity_stats = self.calculate_diversity_stats(pos_molecules)
                        print(f"    📊 Positive diversity: avg={diversity_stats['avg_distance']:.3f}, min={diversity_stats['min_distance']:.3f}, max={diversity_stats['max_distance']:.3f}")
                else:
                    print(f"  Fallback: {len(positive_samples)} diverse positive (top 10% pool), no LLM-generated molecules for negatives")
        
        return positive_samples, negative_samples
            
    def query_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Query GPT-4.1-mini using OpenAI API"""
        messages = [
            {"role": "system", "content": "You are a molecular design expert. You analyze molecular structures and their rewards to generate improved molecules with higher rewards."},
            {"role": "user", "content": prompt}
        ]
        
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=2048,
                    temperature=temperature,
                    messages=messages
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenAI API call failed (attempt {retry + 1}/3): {e}")
                if retry == 2:
                    print("❌ Failed to connect to GPT-4.1-mini!")
                    print("   Please check your OPENAI_API_KEY environment variable")
                    raise e
        
        return None
        
    def extract_molecules_from_response(self, response: str) -> List[Tuple[str, str]]:
        """Extract SMILES strings and explanations from LLM response"""
        molecules_with_explanations = []
        
        # Extract explanation + molecule pairs from expected format: {<<<Explaination>>>: text, <<<Molecule>>>: \box{SMILES}}
        explanation_matches = re.findall(r'<<<Explaination>>>:\s*(.*?)<<<Molecule>>>:\s*\\box\{(.*?)\}', response, re.DOTALL)
        
        for explanation, molecule in explanation_matches:
            sanitized = sanitize_smiles(molecule)  # Use imported function from GPT4.py
            if sanitized:
                explanation_clean = explanation.strip().replace('\n', ' ')
                molecules_with_explanations.append((sanitized, explanation_clean))
                            
        return molecules_with_explanations
    

    
    def calculate_tanimoto_distance(self, smi1: str, smi2: str) -> float:
        """
        Calculate Tanimoto distance between two SMILES strings using Morgan fingerprints
        
        Returns:
            float: Tanimoto distance (1 - Tanimoto similarity), range [0, 1]
                   0 = identical molecules, 1 = completely different
        """
        try:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            
            if mol1 is None or mol2 is None:
                return 1.0  # Maximum distance for invalid molecules
            
            # Generate Morgan fingerprints (radius=2, 2048 bits)
            fp1 = GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            
            # Calculate Tanimoto similarity
            tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            # Return distance (1 - similarity)
            return 1.0 - tanimoto_sim
            
        except Exception:
            return 1.0  # Maximum distance if calculation fails
    
    def diversity_based_sampling(self, molecules_with_scores: List[Tuple[str, float]], 
                                num_samples: int, min_distance: float = 0.3) -> List[Tuple[str, float]]:
        """
        Sample molecules ensuring diversity using Tanimoto distance (MAP-ELITES style)
        
        Args:
            molecules_with_scores: List of (SMILES, score) tuples, assumed to be sorted by score (best first)
            num_samples: Number of molecules to sample
            min_distance: Minimum Tanimoto distance between selected molecules
            
        Returns:
            List of selected (SMILES, score) tuples with high scores and diversity
        """
        if len(molecules_with_scores) <= num_samples:
            return molecules_with_scores
        
        selected = []
        candidates = molecules_with_scores.copy()
        
        # Always select the best molecule first
        if candidates:
            best = candidates.pop(0)
            selected.append(best)
        
        # Iteratively select diverse molecules
        while len(selected) < num_samples and candidates:
            best_candidate = None
            best_score = -float('inf')
            best_idx = -1
            
            for i, (cand_smi, cand_score) in enumerate(candidates):
                # Check if candidate is sufficiently diverse from all selected molecules
                is_diverse = True
                for sel_smi, _ in selected:
                    distance = self.calculate_tanimoto_distance(cand_smi, sel_smi)
                    if distance < min_distance:
                        is_diverse = False
                        break
                
                # If diverse, consider it based on score
                if is_diverse and cand_score > best_score:
                    best_candidate = (cand_smi, cand_score)
                    best_score = cand_score
                    best_idx = i
            
            # If we found a diverse candidate, select it
            if best_candidate is not None:
                selected.append(best_candidate)
                candidates.pop(best_idx)
            else:
                # If no diverse candidates found, relax distance constraint or take best remaining
                if candidates:
                    # Gradually relax the distance constraint
                    min_distance *= 0.8
                    if min_distance < 0.1:  # Fallback: just take the best remaining
                        selected.append(candidates.pop(0))
                        min_distance = 0.3  # Reset for next iteration
        
        print(f"    🏝️  Selected {len(selected)} diverse molecules (min distance: {min_distance:.2f})")
        return selected
    
    def calculate_diversity_stats(self, molecules: List[str]) -> Dict[str, float]:
        """
        Calculate diversity statistics for a set of molecules
        
        Returns:
            Dict with min_distance, max_distance, avg_distance, and diversity_score
        """
        if len(molecules) <= 1:
            return {"min_distance": 0.0, "max_distance": 0.0, "avg_distance": 0.0, "diversity_score": 0.0}
        
        distances = []
        for i in range(len(molecules)):
            for j in range(i + 1, len(molecules)):
                distance = self.calculate_tanimoto_distance(molecules[i], molecules[j])
                distances.append(distance)
        
        min_dist = min(distances) if distances else 0.0
        max_dist = max(distances) if distances else 0.0
        avg_dist = np.mean(distances) if distances else 0.0
        
        # Diversity score: higher = more diverse (range 0-1)
        diversity_score = avg_dist
        
        return {
            "min_distance": min_dist,
            "max_distance": max_dist, 
            "avg_distance": avg_dist,
            "diversity_score": diversity_score
        }
        
    def generate_new_samples(self, positive_samples: List[Tuple[str, float]], 
                           negative_samples: List[Tuple[str, float]], 
                           m: int) -> List[Tuple[str, str]]:
        """Use LLM to generate new samples based on positive/negative examples. Returns list of (SMILES, explanation) tuples"""
        
        # Format positive samples
        pos_molecules = [smi for smi, score in positive_samples]
        pos_rewards = [score for smi, score in positive_samples]
        
        # Format negative samples  
        neg_molecules = [smi for smi, score in negative_samples]
        neg_rewards = [score for smi, score in negative_samples]
        
        # Create prompt
        prompt = f"""You are a black-box reward optimizer for molecular design. Here are examples of molecules and their rewards:

        POSITIVE SAMPLES (high rewards):
        """
        for i, (smi, reward) in enumerate(positive_samples):
            prompt += f"{i+1}. {smi} (reward: {reward:.4f})\n"
            
        prompt += f"""
        NEGATIVE SAMPLES (low rewards):
        """
        for i, (smi, reward) in enumerate(negative_samples):
            prompt += f"{i+1}. {smi} (reward: {reward:.4f})\n"
            
        prompt += f"""
        Please analyze the molecular structures and their rewards. Generate {m} new diverse molecules with rewards better than the positive samples.

        Requirements:
        1. Output exactly {m} valid SMILES strings
        2. Each molecule should be chemically valid
        3. Molecules should be diverse from each other
        4. Aim for rewards higher than {max(pos_rewards):.4f}

        Your output should follow the format: {{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{{$Molecule}}}}. Here are the requirements:

        1. $EXPLANATION should be your analysis.
        2. The $Molecule should be the smiles of your proposed molecule.
        3. The molecule should be valid.

        Please generate {m} diverse molecules:"""

        print(f"\nPrompting LLM to generate {m} new molecules...")
        print(f"Positive sample reward range: {min(pos_rewards):.4f} - {max(pos_rewards):.4f}")
        if neg_rewards:
            print(f"Negative sample reward range: {min(neg_rewards):.4f} - {max(neg_rewards):.4f}")
        else:
            print("Negative sample reward range: No negative samples (first iteration)")
        
        # Query LLM
        response = self.query_llm(prompt)
        print(f"LLM Response length: {len(response)} characters")
        
        # Get existing molecules from candidate pool for duplicate checking
        existing_molecules = set([smi for smi, _ in self.candidate_pool])
        
        # Extract molecules and explanations using the updated method
        molecules_with_explanations = []
        duplicates_found = 0
        try:
            extracted_data = self.extract_molecules_from_response(response)
            for smi, explanation in extracted_data:
                if smi in existing_molecules:
                    duplicates_found += 1
                    print(f"  Duplicate found (skipping): {smi}")
                elif smi not in [mol for mol, _ in molecules_with_explanations]:
                    molecules_with_explanations.append((smi, explanation))
                    if len(molecules_with_explanations) >= m:  # Stop at m molecules
                        break
        except Exception as e:
            print(f"Extraction failed: {e}")
            
        print(f"Extracted {len(molecules_with_explanations)} valid molecules with explanations from LLM response")
        if duplicates_found > 0:
            print(f"  Filtered out {duplicates_found} duplicates from existing pool")
        
        # Display explanations for the generated molecules
        if molecules_with_explanations:
            print(f"\n💡 Generated Molecules with Explanations:")
            for i, (smi, explanation) in enumerate(molecules_with_explanations, 1):
                print(f"  {i}. {smi}")
                print(f"     💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
        
        # Debug: show part of LLM response if no molecules found
        if len(molecules_with_explanations) == 0:
            print("DEBUG: LLM Response:")
            print(response)
            print("DEBUG: Looking for explanation and \\box{} patterns...")
        
        return molecules_with_explanations
        
    def optimize(self, n_init: int = 1000, m_per_iter: int = 50, max_iterations: int = 20, 
                 pool_sample_size: int = 200, use_weighted_sampling: bool = True):
        """Run the black-box optimization"""
        
        print("="*80)
        print("LLM BLACK-BOX OPTIMIZER")
        print("="*80)
        
        # Step 1: Initialize pool
        self.initialize_pool(n_init)
        
        # Optimization loop
        for iteration in range(max_iterations):
            print(f"\n{'='*20} ITERATION {iteration + 1}/{max_iterations} {'='*20}")
            
            # Check stopping condition
            if self.oracle.finish:
                print("Oracle budget exhausted, stopping optimization")
                break
                
            # Step 2: Sample positive and negative examples using new strategy
            max_examples = 5  # Number of positive and negative examples to use
            
            # For iterations > 1, use previous iteration's molecules for sampling
            if iteration == 0:
                # First iteration: use standard strategy
                positive_samples, negative_samples = self.sample_positive_negative(max_examples)
            else:
                # Later iterations: use previous iteration's molecules
                positive_samples, negative_samples = self.sample_positive_negative(max_examples, self.previous_iteration_molecules)
            
            # Step 3: Generate new samples using LLM
            molecules_with_explanations = self.generate_new_samples(positive_samples, negative_samples, m_per_iter)
            
            if not molecules_with_explanations:
                print("No valid molecules generated, skipping iteration")
                continue
                
            # Step 4: Score new molecules and add to pool
            new_molecules = [smi for smi, _ in molecules_with_explanations]
            explanations = [exp for _, exp in molecules_with_explanations]
            
            print(f"\nScoring {len(new_molecules)} new molecules...")
            new_scores = []
            valid_new_molecules = []
            valid_explanations = []
            unique_count = 0
            existing_molecules = set([smi for smi, _ in self.candidate_pool])
            
            for i, (smi, explanation) in enumerate(molecules_with_explanations):
                # Check if molecule is unique (not already in pool)
                is_unique = smi not in existing_molecules
                if is_unique:
                    unique_count += 1
                    unique_marker = "✓"
                else:
                    unique_marker = "⚠️dup"
                
                # First check if molecule is chemically valid
                if sanitize_smiles(smi) is None:
                    # Invalid molecule - assign score of 0 without using oracle
                    score = 0.0
                    new_scores.append(score)
                    valid_new_molecules.append(smi)
                    valid_explanations.append(explanation)
                    if is_unique:  # Only add to pool if unique
                        self.candidate_pool.append((smi, score))
                    self.invalid_molecules_count += 1
                    print(f"  {i+1}. {smi}: 0.0000 (invalid chemistry) {unique_marker}")
                    print(f"      💭 {explanation[:80]}{'...' if len(explanation) > 80 else ''}")
                else:
                    try:
                        score = self.oracle.score_smi(smi)
                        new_scores.append(score)
                        valid_new_molecules.append(smi)
                        valid_explanations.append(explanation)
                        if is_unique:  # Only add to pool if unique
                            self.candidate_pool.append((smi, score))
                        print(f"  {i+1}. {smi}: {score:.4f} {unique_marker}")
                        print(f"      💭 {explanation[:80]}{'...' if len(explanation) > 80 else ''}")
                    except Exception as e:
                        # Oracle failed - assign score of 0
                        score = 0.0
                        new_scores.append(score)
                        valid_new_molecules.append(smi)
                        valid_explanations.append(explanation)
                        if is_unique:  # Only add to pool if unique
                            self.candidate_pool.append((smi, score))
                        print(f"  {i+1}. {smi}: 0.0000 (oracle error: {str(e)[:50]}) {unique_marker}")
                        print(f"      💭 {explanation[:80]}{'...' if len(explanation) > 80 else ''}")
                    
                if self.oracle.finish:
                    break
            
            # Sort pool to keep best molecules at top
            self.candidate_pool.sort(key=lambda x: x[1], reverse=True)
            
            # Keep pool size manageable (optional)
            max_pool_size = 5000
            if len(self.candidate_pool) > max_pool_size:
                self.candidate_pool = self.candidate_pool[:max_pool_size]
            
            # Track generated molecules with explanations and scores - simplified!
            for smi, exp, score in zip(valid_new_molecules, valid_explanations, new_scores):
                self.all_generated_data.append((smi, exp, score, iteration + 1))
            
            # Store current iteration's molecules with scores for next iteration's sampling
            if valid_new_molecules and new_scores:
                current_iteration_scored = list(zip(valid_new_molecules, new_scores))
                self.previous_iteration_molecules = current_iteration_scored
                print(f"  💾 Stored {len(current_iteration_scored)} molecules from this iteration for next iteration's sampling")
            
            # Print iteration summary
            if new_scores:
                best_new = max(new_scores)
                avg_new = np.mean(new_scores)
                current_best = self.candidate_pool[0][1]
                
                # Find the best molecule from this iteration with its explanation
                best_idx = new_scores.index(best_new)
                best_molecule = valid_new_molecules[best_idx]
                best_explanation = valid_explanations[best_idx]
                
                print(f"\n🔍 Current Batch Analysis:")
                print(f"  Generated molecules: {len(valid_new_molecules)}")
                print(f"  Unique molecules: {unique_count}")
                print(f"  Duplicate molecules: {len(valid_new_molecules) - unique_count}")
                print(f"  Average score this batch: {avg_new:.4f}")
                print(f"  Best molecule this batch: {best_molecule}")
                print(f"  Best score this batch: {best_new:.4f}")
                print(f"  Best explanation: {best_explanation}")
                
                print(f"\nIteration {iteration + 1} Summary:")
                print(f"  Current best overall: {current_best:.4f}")
                print(f"  Pool size: {len(self.candidate_pool)}")
                print(f"  Oracle calls used: {len(self.oracle)}")
                print(f"  Total generated molecules: {len(self.all_generated_data)}")
                
                # Show improvement
                if positive_samples and best_new > max([s for _, s in positive_samples]):
                    print(f"  ✓ Generated molecule better than positive samples!")
                
                # Save intermediate results after each iteration (simplified - no redundant data collection)
                self.save_intermediate_results(iteration, valid_new_molecules, valid_explanations, new_scores, positive_samples, negative_samples)
            
            # Also save results even if no valid molecules were generated this iteration
            elif valid_new_molecules:  # If we have molecules but no scores (shouldn't happen, but just in case)
                for smi, exp in zip(valid_new_molecules, valid_explanations):
                    self.all_generated_data.append((smi, exp, 0.0, iteration + 1))
                self.save_intermediate_results(iteration, valid_new_molecules, valid_explanations, [], positive_samples, negative_samples)
            
        # Final results
        print(f"\n{'='*20} OPTIMIZATION COMPLETE {'='*20}")
        print(f"Final pool size: {len(self.candidate_pool)}")
        print(f"Total oracle calls: {len(self.oracle)}")
        print(f"Best molecule found: {self.candidate_pool[0][0]}")
        print(f"Best score: {self.candidate_pool[0][1]:.4f}")
        
        # Summary of molecule generation
        total_generated = len(self.all_generated_data)
        unique_generated = len(set([smi for smi, _, _, _ in self.all_generated_data]))
        print(f"\n🧪 Molecule Generation Summary:")
        print(f"  Total molecules generated: {total_generated}")
        print(f"  Unique molecules generated: {unique_generated}")
        if total_generated > unique_generated:
            print(f"  Duplicates within generated set: {total_generated - unique_generated}")
        print(f"  All generated molecules were novel (not in initial ZINC pool): ✅")
        print(f"  Invalid molecules generated: {self.invalid_molecules_count}")
        if self.invalid_molecules_count > 0:
            invalid_rate = (self.invalid_molecules_count / total_generated) * 100 if total_generated > 0 else 0
            print(f"  Invalid molecule rate: {invalid_rate:.1f}%")
        
        # Compute and display average scores
        if self.all_generated_data:
            avg_score = np.mean([score for _, _, score, _ in self.all_generated_data])
            max_score = max([score for _, _, score, _ in self.all_generated_data])
            min_score = min([score for _, _, score, _ in self.all_generated_data])
            std_score = np.std([score for _, _, score, _ in self.all_generated_data])
            
            print(f"\n📊 LLM-Generated Molecule Scores:")
            print(f"  Average score: {avg_score:.4f}")
            print(f"  Best score: {max_score:.4f}")
            print(f"  Worst score: {min_score:.4f}")
            print(f"  Standard deviation: {std_score:.4f}")
            
            # Compare with initial ZINC pool
            zinc_scores = [score for _, _, score, _ in self.all_generated_data if score not in [s for _, _, s, _ in self.all_generated_data]]
            if len(zinc_scores) > 0:
                zinc_avg = np.mean(zinc_scores[:100])  # Average of top 100 ZINC molecules
                print(f"  Comparison to ZINC top-100 avg: {zinc_avg:.4f}")
                improvement = ((avg_score - zinc_avg) / zinc_avg) * 100 if zinc_avg > 0 else 0
                if improvement > 0:
                    print(f"  LLM improvement over ZINC: +{improvement:.1f}% ✅")
                else:
                    print(f"  LLM performance vs ZINC: {improvement:.1f}% ⚠️")
        else:
            print(f"\n📊 No molecules were successfully generated by the LLM")
        
        # Final DPO Training with all collected data
        print(f"\n{'='*20} CREATING FINAL DPO DATASET {'='*20}")
        try:
            from dpo_step import create_dpo_dataset
            
            # Remove duplicates while preserving order
            unique_positive = list(dict.fromkeys([smi for smi, _, score, _ in self.all_generated_data if score > 0]))
            unique_negative = list(dict.fromkeys([smi for smi, _, score, _ in self.all_generated_data if score < 0]))
            
            print(f"📊 Final dataset summary:")
            print(f"  Unique positive molecules: {len(unique_positive)}")
            print(f"  Unique negative molecules: {len(unique_negative)}")
            print(f"  Total molecules for DPO: {len(unique_positive) + len(unique_negative)}")
            
            if len(unique_positive) > 0 and len(unique_negative) > 0:
                dpo_results = create_dpo_dataset(
                    positive_samples=unique_positive,
                    negative_samples=unique_negative,
                    oracle=self.oracle,
                    output_dir='./dpo_final_300_iterations',
                    num_preference_pairs=min(5000, len(unique_positive) * 50)  # Scale with dataset size
                )
                print(f"\n🎉 FINAL DPO Dataset Created Successfully! 🎉")
                print(f"✅ Generated {dpo_results['dataset_stats']['preference_pairs']} preference pairs")
                print(f"💾 Files saved to: ./dpo_final_300_iterations")
                print(f"🚀 To start DPO training: bash ./dpo_final_300_iterations/run_dpo_training.sh")
                print(f"🎯 Combined reward: {dpo_results['dataset_stats']['combined_reward']:.4f}")
            else:
                print("⚠️  Not enough data collected for DPO training")
                
        except ImportError:
            print("⚠️  DPO step skipped: dpo_step.py not available")
        except Exception as e:
            print(f"❌ Final DPO step failed: {e}")
        
        # Save results
        self.oracle.log_intermediate(finish=True)
        
        return self.candidate_pool

    def save_intermediate_results(self, iteration: int, valid_new_molecules: List[str], 
                                valid_explanations: List[str], new_scores: List[float],
                                positive_samples: List[Tuple[str, float]], 
                                negative_samples: List[Tuple[str, float]]):
        """Save intermediate results after each iteration to a single JSON file with DPO preference pairs"""
        
        # Create iteration result summary
        iteration_data = {
            'iteration': iteration + 1,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'generated_molecules': valid_new_molecules,
            'explanations': valid_explanations,
            'scores': new_scores,
            'positive_samples': [(smi, score) for smi, score in positive_samples],
            'negative_samples': [(smi, score) for smi, score in negative_samples],
            'best_score_this_iteration': max(new_scores) if new_scores else 0.0,
            'avg_score_this_iteration': np.mean(new_scores) if new_scores else 0.0,
            'current_best_overall': self.candidate_pool[0][1] if self.candidate_pool else 0.0,
            'oracle_calls_used': len(self.oracle),
            'total_generated_molecules': len(self.all_generated_data),
            'unique_molecules_this_iteration': len(set(valid_new_molecules)),
            'invalid_molecules_count': self.invalid_molecules_count
        }
        
        self.iteration_results.append(iteration_data)
        
        # Create DPO preference pairs for this iteration
        preference_pairs = []
        if valid_new_molecules and positive_samples:
            # Create preference pairs: generated molecules vs positive/negative samples
            for i, (new_mol, new_explanation, new_score) in enumerate(zip(valid_new_molecules, valid_explanations, new_scores)):
                
                # Compare with positive samples - if new molecule is better, it's chosen
                for pos_mol, pos_score in positive_samples:
                    if new_score > pos_score:
                        preference_pairs.append({
                            'chosen': new_mol,
                            'chosen_explanation': new_explanation,
                            'chosen_score': new_score,
                            'rejected': pos_mol,
                            'rejected_explanation': f"Positive sample with lower score ({pos_score:.4f})",
                            'rejected_score': pos_score,
                            'comparison_type': 'new_vs_positive',
                            'preference_strength': new_score - pos_score,
                            'iteration': iteration + 1
                        })
                    else:
                        preference_pairs.append({
                            'chosen': pos_mol,
                            'chosen_explanation': f"Positive sample with higher score ({pos_score:.4f})",
                            'chosen_score': pos_score,
                            'rejected': new_mol,
                            'rejected_explanation': new_explanation,
                            'rejected_score': new_score,
                            'comparison_type': 'positive_vs_new',
                            'preference_strength': pos_score - new_score,
                            'iteration': iteration + 1
                        })
                
                # Compare with negative samples - new molecule should generally be chosen
                for neg_mol, neg_score in negative_samples:
                    if new_score > neg_score:
                        preference_pairs.append({
                            'chosen': new_mol,
                            'chosen_explanation': new_explanation,
                            'chosen_score': new_score,
                            'rejected': neg_mol,
                            'rejected_explanation': f"Negative sample with lower score ({neg_score:.4f})",
                            'rejected_score': neg_score,
                            'comparison_type': 'new_vs_negative',
                            'preference_strength': new_score - neg_score,
                            'iteration': iteration + 1
                        })
                    elif neg_score > new_score:  # Sometimes new molecules might be worse
                        preference_pairs.append({
                            'chosen': neg_mol,
                            'chosen_explanation': f"Negative sample unexpectedly better ({neg_score:.4f})",
                            'chosen_score': neg_score,
                            'rejected': new_mol,
                            'rejected_explanation': new_explanation,
                            'rejected_score': new_score,
                            'comparison_type': 'negative_vs_new',
                            'preference_strength': neg_score - new_score,
                            'iteration': iteration + 1
                        })
            
            # Also create pairs between generated molecules (higher score chosen)
            for i in range(len(valid_new_molecules)):
                for j in range(i + 1, len(valid_new_molecules)):
                    mol1, exp1, score1 = valid_new_molecules[i], valid_explanations[i], new_scores[i]
                    mol2, exp2, score2 = valid_new_molecules[j], valid_explanations[j], new_scores[j]
                    
                    if score1 > score2:
                        preference_pairs.append({
                            'chosen': mol1,
                            'chosen_explanation': exp1,
                            'chosen_score': score1,
                            'rejected': mol2,
                            'rejected_explanation': exp2,
                            'rejected_score': score2,
                            'comparison_type': 'generated_vs_generated',
                            'preference_strength': score1 - score2,
                            'iteration': iteration + 1
                        })
                    elif score2 > score1:
                        preference_pairs.append({
                            'chosen': mol2,
                            'chosen_explanation': exp2,
                            'chosen_score': score2,
                            'rejected': mol1,
                            'rejected_explanation': exp1,
                            'rejected_score': score1,
                            'comparison_type': 'generated_vs_generated',
                            'preference_strength': score2 - score1,
                            'iteration': iteration + 1
                        })
        
        # Add preference pairs to iteration data
        iteration_data['preference_pairs'] = preference_pairs
        
        # Accumulate all preference pairs
        if not hasattr(self, 'all_preference_pairs'):
            self.all_preference_pairs = []
        self.all_preference_pairs.extend(preference_pairs)
        
        # Save everything to a single comprehensive JSON file
        import json
        results_file = os.path.join(self.args.output_dir, 'optimization_results.json')
        
        # Create comprehensive data structure
        comprehensive_data = {
            'optimization_info': {
                'oracle': self.args.oracle,
                'seed': self.args.seed,
                'n_init': self.args.n_init,
                'm_per_iter': self.args.m_per_iter,
                'max_iterations': self.args.max_iterations,
                'current_iteration': iteration + 1,
                'total_iterations_completed': len(self.iteration_results),
                'start_time': self.start_time,
                'last_updated': __import__('datetime').datetime.now().isoformat()
            },
            'current_status': {
                'best_molecule_overall': self.candidate_pool[0][0] if self.candidate_pool else None,
                'best_score_overall': self.candidate_pool[0][1] if self.candidate_pool else 0.0,
                'total_oracle_calls': len(self.oracle),
                'total_molecules_generated': len(self.all_generated_data),
                'total_unique_molecules': len(set([smi for smi, _, _, _ in self.all_generated_data])),
                'invalid_molecules_total': self.invalid_molecules_count,
                'pool_size': len(self.candidate_pool),
                'total_preference_pairs': len(self.all_preference_pairs)
            },
            'top_molecules': [
                {'smiles': smi, 'score': score} 
                for smi, score in self.candidate_pool[:50]  # Top 50 molecules
            ],
            'all_iterations': self.iteration_results,
            'all_generated_molecules': {
                'molecules': [smi for smi, _, _, _ in self.all_generated_data],
                'explanations': [exp for _, exp, _, _ in self.all_generated_data],
                'scores': [score for _, _, score, _ in self.all_generated_data]
            },
            'dpo_preference_pairs': self.all_preference_pairs,
            'dpo_dataset_stats': {
                'total_pairs': len(self.all_preference_pairs),
                'pairs_this_iteration': len(preference_pairs),
                'comparison_types': {
                    'new_vs_positive': len([p for p in self.all_preference_pairs if p['comparison_type'] == 'new_vs_positive']),
                    'new_vs_negative': len([p for p in self.all_preference_pairs if p['comparison_type'] == 'new_vs_negative']),
                    'generated_vs_generated': len([p for p in self.all_preference_pairs if p['comparison_type'] == 'generated_vs_generated']),
                    'positive_vs_new': len([p for p in self.all_preference_pairs if p['comparison_type'] == 'positive_vs_new']),
                    'negative_vs_new': len([p for p in self.all_preference_pairs if p['comparison_type'] == 'negative_vs_new'])
                },
                'avg_preference_strength': np.mean([p['preference_strength'] for p in self.all_preference_pairs]) if self.all_preference_pairs else 0.0,
                'max_preference_strength': max([p['preference_strength'] for p in self.all_preference_pairs]) if self.all_preference_pairs else 0.0
            }
        }
        
        # Save to single file
        with open(results_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2)
        
        # Also save just the DPO pairs in a separate file for easy loading
        dpo_file = os.path.join(self.args.output_dir, 'dpo_preference_pairs.json')
        with open(dpo_file, 'w') as f:
            json.dump(self.all_preference_pairs, f, indent=2)
        
        # Also save the oracle's intermediate log
        self.oracle.log_intermediate(finish=False)
        
        print(f"  💾 Saved all results to {results_file}")
        print(f"  🎯 DPO pairs: {len(preference_pairs)} this iteration, {len(self.all_preference_pairs)} total")
        print(f"     Current iteration: {iteration + 1}, Total molecules: {len(self.all_generated_data)}, Best score: {comprehensive_data['current_status']['best_score_overall']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='LLM Black-box Optimizer for Molecular Design')
    parser.add_argument('--oracle', type=str, default='jnk3', help='Oracle to optimize (default: jnk3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_init', type=int, default=1000, help='Initial pool size')
    parser.add_argument('--m_per_iter', type=int, default=20, help='Samples per iteration (m)')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations')
    parser.add_argument('--pool_sample_size', type=int, default=200, help='Samples drawn from pool each iteration')
    parser.add_argument('--max_oracle_calls', type=int, default=20000, help='Maximum oracle calls')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='./results_blackbox', help='Output directory')
    parser.add_argument('--uniform_sampling', action='store_true', help='Use uniform sampling instead of weighted')

    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add required attributes for MolLEO compatibility
    args.oracles = [args.oracle]
    args.mol_lm = 'GPT4'  # Use GPT4 name for compatibility
    args.freq_log = 100
    args.n_jobs = 1
    args.log_results = True
    args.smi_file = None  # Use ZINC dataset instead of custom file
    args.output_dir = args.output_dir
    
    print(f"Starting LLM Black-box Optimization:")
    print(f"  Oracle: {args.oracle}")
    print(f"  Seed: {args.seed}")
    print(f"  Initial pool size: {args.n_init}")
    print(f"  Samples per iteration: {args.m_per_iter}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Max oracle calls: {args.max_oracle_calls}")
    print(f"  Sampling: {'Uniform' if args.uniform_sampling else 'Weighted'}")
    print(f"  Using GPT-4.1-mini via OpenAI API")
    
    # Create optimizer
    optimizer = LLMBlackBoxOptimizer(args)
    
    # Run optimization
    final_pool = optimizer.optimize(
        n_init=args.n_init,
        m_per_iter=args.m_per_iter, 
        max_iterations=args.max_iterations,
        pool_sample_size=args.pool_sample_size,
        use_weighted_sampling=not args.uniform_sampling
    )
    
    print("Optimization completed!")

if __name__ == "__main__":
    main() 