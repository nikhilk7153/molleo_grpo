#!/usr/bin/env python
"""
LLM Black-box Optimizer for Multi-Objective Pareto Optimization

Uses Qwen-2.5-7B-Instruct via OpenRouter API for multi-objective molecular optimization.

Follows the specified algorithm but adapted for Pareto optimization:
1. Randomly generate n samples to initialize a candidate pool
2. For each iteration:
   - Sample 2m samples from pool using exp(±r(sample)) weighting
   - Score samples with multi-objective oracle
   - Prompt LLM with specific format including multi-objective context
   - Generate m new diverse samples
   - Select Pareto front and calculate combined reward
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

# Add path for multi-objective imports
current_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(current_dir)
multi_objective_dir = os.path.dirname(main_dir)
sys.path.append(main_dir)
sys.path.append(multi_objective_dir)

from pareto_optimizer import Oracle
from GPT4 import sanitize_smiles
from pymoo.indicators.hv import HV

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class LLMBlackBoxOptimizerPareto:
    def __init__(self, args):
        self.args = args
        self.oracle = Oracle(args=args)
        self.candidate_pool = []  # List of (smiles, score) tuples
        self.model_name = "gpt-4.1-mini"
        
        print("🎯 Using GPT-4.1-mini for multi-objective Pareto optimization")
        
        # Set up persistent caching
        self.cache_file = os.path.join(args.output_dir, 'molecule_cache.json')
        self._load_molecule_cache()
        
        # Test OpenAI API connection
        if not self._test_openai_connection():
            print("❌ Cannot connect to OpenAI API!")
            print("   Please check your OPENAI_API_KEY environment variable")
            raise ConnectionError("OpenAI API not accessible")
        else:
            print("✅ OpenAI API connection verified")
        
        # Load dataset for random sampling
        self.all_smiles = self._load_zinc_dataset()
        
        # Track optimization history
        self.iteration_results = []
        self.best_scores_history = []
        self.pareto_front_history = []
        self.hypervolume_history = []
        
        # Initialize hypervolume indicator
        self.hv_indicator = None
        
    def _load_molecule_cache(self):
        """Load previously computed molecule scores from cache file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Convert cache data back to oracle buffer format
                for smi, cache_entry in cache_data.items():
                    score = cache_entry['score']
                    call_number = cache_entry.get('call_number', len(self.oracle.mol_buffer) + 1)
                    self.oracle.mol_buffer[smi] = [float(score), call_number]
                
                print(f"📁 Loaded {len(cache_data)} cached molecule scores from {self.cache_file}")
                
            except Exception as e:
                print(f"⚠️  Failed to load cache file: {e}")
                print("   Starting with empty cache")
        else:
            print(f"📁 No cache file found at {self.cache_file}, starting with empty cache")
    
    def _save_molecule_cache(self):
        """Save current molecule scores to cache file for future use"""
        try:
            # Convert oracle buffer to JSON-serializable format
            cache_data = {}
            for smi, buffer_entry in self.oracle.mol_buffer.items():
                cache_data[smi] = {
                    'score': buffer_entry[0],
                    'call_number': buffer_entry[1]
                }
            
            # Also include scores from storing_buffer if it exists
            if hasattr(self.oracle, 'storing_buffer') and self.oracle.storing_buffer:
                for smi, buffer_entry in self.oracle.storing_buffer.items():
                    if smi not in cache_data:  # Don't overwrite more recent scores
                        cache_data[smi] = {
                            'score': buffer_entry[0],
                            'call_number': buffer_entry[1]
                        }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            print(f"💾 Saved {len(cache_data)} molecule scores to cache: {self.cache_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to save cache file: {e}")
        
    def _get_individual_scores(self, smi: str) -> List[float]:
        """Get individual objective scores for a molecule"""
        scores = []
        
        # Get scores for maximization objectives
        for eva in self.oracle.max_evaluator:
            scores.append(eva(smi))
        
        # Get scores for minimization objectives (raw values, not inverted)
        for eva in self.oracle.min_evaluator:
            scores.append(eva(smi))
            
        return scores
    
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
        
        # Initialize oracle evaluator for multi-objective
        self.oracle.assign_evaluator(self.args)
        
        # Randomly sample molecules
        random_molecules = random.sample(self.all_smiles, min(n_samples, len(self.all_smiles)))
        
        # Score them using oracle batch scoring like in run.py
        print(f"Scoring {len(random_molecules)} random molecules...")
        scores = self.oracle([smi for smi in random_molecules])
        
        for i, (smi, score) in enumerate(zip(random_molecules, scores)):
            self.candidate_pool.append((smi, score))
            
            if (i + 1) % 50 == 0:
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
        
        # Print multi-objective setup
        print(f"  Maximizing objectives: {self.args.max_obj}")
        print(f"  Minimizing objectives: {self.args.min_obj}")
        
        # Save cache after initialization
        self._save_molecule_cache()
    
    def sample_with_uniform_halves(self, m: int) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Sample m total samples from pool using uniform sampling:
        - Positive samples: m//2 from top 50%
        - Negative samples: m//2 from bottom 50%
        """
        m_pos = m // 2
        m_neg = m // 2
        
        if len(self.candidate_pool) < m:
            print(f"Warning: Pool size ({len(self.candidate_pool)}) < m ({m}), using all available molecules")
            half = len(self.candidate_pool) // 2
            return self.candidate_pool[:half], self.candidate_pool[half:]
        
        # Split pool into top 50% and bottom 50%
        pool_size = len(self.candidate_pool)
        half_size = pool_size // 2
        
        top_half = self.candidate_pool[:half_size]  # Best 50% scoring molecules
        bottom_half = self.candidate_pool[half_size:]  # Worst 50% scoring molecules
        
        # Uniform sampling from each half
        positive_samples = random.sample(top_half, min(m_pos, len(top_half)))
        negative_samples = random.sample(bottom_half, min(m_neg, len(bottom_half)))
        
        pos_scores = [score for _, score in positive_samples]
        neg_scores = [score for _, score in negative_samples]
        
        print(f"Uniform half sampling (using combined Pareto scores):")
        print(f"  Positive samples: {len(positive_samples)} from top {len(top_half)} (top 50%), combined score range: {min(pos_scores):.4f} - {max(pos_scores):.4f}")
        print(f"  Negative samples: {len(negative_samples)} from bottom {len(bottom_half)} (bottom 50%), combined score range: {min(neg_scores):.4f} - {max(neg_scores):.4f}")
        
        return positive_samples, negative_samples
    
    def query_llm_with_specific_prompt(self, positive_samples: List[Tuple[str, float]], 
                                     negative_samples: List[Tuple[str, float]], 
                                     m: int) -> Tuple[str, str]:
        """Query LLM with the specific prompt format adapted for multi-objective optimization"""
        
        # Extract molecules and combined scores (following GPT4.py approach exactly)
        positive_molecules = [smi for smi, _ in positive_samples]
        positive_scores = [score for _, score in positive_samples]
        negative_molecules = [smi for smi, _ in negative_samples]
        negative_scores = [score for _, score in negative_samples]
        
        # Create multi-objective context
        objectives_info = f"Maximizing: {', '.join(self.args.max_obj)}; Minimizing: {', '.join(self.args.min_obj)}"
        
        # Format molecules with scores like GPT4.py: [molecule, score]
        mol_tuple_positive = ''
        for i, (smi, score) in enumerate(positive_samples):
            tu = f'\n[{smi}, {score:.4f}]'
            mol_tuple_positive += tu
            
        mol_tuple_negative = ''
        for i, (smi, score) in enumerate(negative_samples):
            tu = f'\n[{smi}, {score:.4f}]'
            mol_tuple_negative += tu
        
        # Create the specific prompt format for multi-objective (following GPT4.py exactly)
        prompt = f"""You are a black-box reward optimizer for multi-objective molecular design. 

OBJECTIVES: {objectives_info}

Here are {len(positive_samples)} positive samples:{mol_tuple_positive}

Here are {len(negative_samples)} negative samples:{mol_tuple_negative}

The combined reward balances multiple objectives. Higher rewards indicate better performance across the objective set.

Please generate exactly {m} new diverse molecular structures as SMILES strings that should achieve higher combined rewards than the positive samples. The new samples should be diversified and consider all objectives.

You can either make crossover and mutations based on the given molecules or just propose new molecules based on your knowledge.

For each molecule, provide both an explanation and the molecule using this exact format:
<explanation>Your reasoning for why this molecule should have higher multi-objective reward considering {objectives_info}</explanation> + <molecule>SMILES_STRING</molecule>

Generate {m} molecules with their explanations. Do not output any other text:"""

        print(f"\nPrompting LLM with multi-objective format:")
        print(f"  Objectives: {objectives_info}")
        print(f"  Positive samples: {len(positive_samples)} (sampled using combined Pareto scores)")
        print(f"  Negative samples: {len(negative_samples)} (sampled using combined Pareto scores)")
        print(f"  Prompt shows combined scores (like GPT4.py)")
        print(f"  Requesting: {m} new molecules")
        print(f"\n--- PROMPT SENT TO LLM ---")
        print(prompt)
        print(f"--- END PROMPT ---\n")
        
        # Query the LLM
        messages = [
            {"role": "system", "content": "You are a molecular design expert specializing in multi-objective black-box optimization. Analyze the given examples and generate improved molecules that balance multiple objectives."},
            {"role": "user", "content": prompt}
        ]
        
        for retry in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=2048,
                    temperature=0.7,
                    messages=messages
                )
                return prompt, response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenAI API call failed (attempt {retry + 1}/3): {e}")
                if retry == 2:
                    raise e
        
        return prompt, None
    
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
        Calculate combined reward with three weighted metrics for multi-objective:
        1. new_reward_max - positive_reward_max: Best new molecule vs best positive sample from prompt (weight: 0.2)
        2. new_reward_mean - positive_reward_mean: Mean new molecules vs mean positive samples from prompt (weight: 0.5)
        3. diversity of new_samples: Tanimoto distance-based diversity score (weight: 0.3)
        
        Combined = 0.2*reward_1 + 0.5*reward_2 + 0.3*reward_3
        
        Args:
            new_molecules: Newly generated molecules from LLM
            new_scores: Multi-objective combined scores of newly generated molecules  
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
    
    def apply_pareto_selection(self, molecules_with_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Apply Pareto front selection to the molecules"""
        if not molecules_with_scores:
            return []
        
        molecules = [smi for smi, _ in molecules_with_scores]
        pareto_molecules = self.oracle.select_pareto_front(molecules)
        
        # Create new list with only Pareto front molecules
        pareto_with_scores = []
        for pareto_mol in pareto_molecules:
            pareto_smi = Chem.MolToSmiles(pareto_mol)
            # Find the score for this molecule
            for smi, score in molecules_with_scores:
                if smi == pareto_smi:
                    pareto_with_scores.append((smi, score))
                    break
        
        print(f"  Pareto selection: {len(molecules_with_scores)} → {len(pareto_with_scores)} molecules")
        return pareto_with_scores
    
    def calculate_hypervolume(self, pareto_molecules: List[str]) -> float:
        """Calculate hypervolume of the current Pareto front using normalized objectives"""
        if not pareto_molecules:
            return 0.0
        
        # Get individual objective scores for Pareto front molecules
        pareto_objectives = []
        for smi in pareto_molecules:
            individual_scores = self._get_individual_scores(smi)
            
            # Normalize objectives to [0,1] range and convert to minimization format
            # For jnk3 and qed: already in [0,1], convert to minimization by (1 - score)
            # For SA: normalize to [0,1] by inverting and scaling: (1 - (sa-1)/9)
            normalized_minimization = []
            
            # Process maximization objectives (jnk3, qed)
            for i, obj in enumerate(self.args.max_obj):
                score = individual_scores[i]
                # Convert to minimization: 1 - score (since score is already in [0,1])
                normalized_minimization.append(1.0 - score)
            
            # Process minimization objectives (SA)
            for i, obj in enumerate(self.args.min_obj):
                sa_score = individual_scores[len(self.args.max_obj) + i]
                if obj == 'sa':
                    # SA is typically in range [1, 10], normalize to [0,1] minimization format
                    # Higher SA is worse, so we want it as is after normalization
                    normalized_sa = (sa_score - 1.0) / 9.0  # Normalize to [0,1]
                    normalized_minimization.append(min(max(normalized_sa, 0.0), 1.0))  # Clamp to [0,1]
                else:
                    # Other minimization objectives, assume already normalized
                    normalized_minimization.append(sa_score)
            
            pareto_objectives.append(normalized_minimization)
        
        pareto_objectives = np.array(pareto_objectives)
        
        # Use fixed reference point for consistent hypervolume calculation
        # Since all objectives are normalized to [0,1] minimization, reference point is (1.1, 1.1, ...)
        if self.hv_indicator is None:
            n_objectives = len(self.args.max_obj) + len(self.args.min_obj)
            ref_point = np.ones(n_objectives) * 1.1  # Slightly above maximum possible value
            self.hv_indicator = HV(ref_point=ref_point)
        
        try:
            hv_value = self.hv_indicator(pareto_objectives)
            return float(hv_value)
        except Exception as e:
            print(f"Warning: Hypervolume calculation failed: {e}")
            return 0.0
    
    def optimize(self, n_init: int = 120, m: int = 5, max_iterations: int = 20):
        """Run the multi-objective Pareto optimization following the specified algorithm"""
        
        print("="*80)
        print("LLM BLACK-BOX OPTIMIZER - MULTI-OBJECTIVE PARETO")
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
            
            # Step 2: Sample 2m samples from pool using uniform sampling from halves
            positive_samples, negative_samples = self.sample_with_uniform_halves(m)
            
            # Step 3: Prompt LLM with specific format
            prompt_text, response = self.query_llm_with_specific_prompt(positive_samples, negative_samples, m)
            
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
            
            # Batch score all molecules like in run.py
            valid_molecules = [smi for smi, _ in molecules_with_explanations]
            valid_explanations = [exp for _, exp in molecules_with_explanations]
            
            # Score all at once using oracle batch method
            try:
                new_scores = self.oracle([smi for smi in valid_molecules])
            except Exception as e:
                print(f"Batch scoring failed: {e}, falling back to individual scoring")
                new_scores = []
                for smi in valid_molecules:
                    try:
                        score = self.oracle.score_smi(smi)
                        new_scores.append(score)
                    except:
                        new_scores.append(0.0)
            
            # Print results
            for i, (smi, score, explanation) in enumerate(zip(valid_molecules, new_scores, valid_explanations)):
                print(f"  {i+1}. {smi}: {score:.4f}")
                print(f"      💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
                
                if self.oracle.finish:
                    break
            
            if not new_scores:
                print("No molecules could be scored, skipping iteration")
                continue
            
            # Step 5: Calculate combined reward
            reward_metrics = self.calculate_combined_reward(valid_molecules, new_scores, positive_samples)
            
            print(f"\n🎯 Combined Reward Calculation:")
            print(f"  Reward 1 (max improvement): {reward_metrics['reward_1']:.4f} (weight: 0.2)")
            print(f"  Reward 2 (mean improvement): {reward_metrics['reward_2']:.4f} (weight: 0.5)")
            print(f"  Reward 3 (diversity): {reward_metrics['reward_3']:.4f} (weight: 0.3)")
            print(f"  Combined Reward: {reward_metrics['combined_reward']:.4f}")
            
            # Add ALL new molecules to candidate pool (before Pareto selection)
            print(f"\n📥 Adding {len(valid_molecules)} new molecules to candidate pool...")
            print(f"  Candidate pool size before: {len(self.candidate_pool)}")
            for smi, score in zip(valid_molecules, new_scores):
                self.candidate_pool.append((smi, score))
            print(f"  Candidate pool size after adding new molecules: {len(self.candidate_pool)}")
            
            # Apply Pareto front selection to entire pool
            print(f"\n🔍 Applying Pareto Front Selection to entire pool:")
            self.candidate_pool = self.apply_pareto_selection(self.candidate_pool)
            
            # Sort pool by combined score
            self.candidate_pool.sort(key=lambda x: x[1], reverse=True)
            
            # Track best score and Pareto front
            current_best = self.candidate_pool[0][1] if self.candidate_pool else 0.0
            self.best_scores_history.append(current_best)
            
            # Store Pareto front for this iteration
            pareto_front_mols = [smi for smi, _ in self.candidate_pool]
            self.pareto_front_history.append(pareto_front_mols)
            
            # Calculate hypervolume
            current_hypervolume = self.calculate_hypervolume(pareto_front_mols)
            self.hypervolume_history.append(current_hypervolume)
            
            # Print iteration summary
            print(f"\nIteration {iteration + 1} Summary:")
            print(f"  Generated molecules: {len(valid_molecules)}")
            print(f"  Best new score: {max(new_scores):.4f}")
            print(f"  Mean new score: {np.mean(new_scores):.4f}")
            print(f"  Current best overall: {current_best:.4f}")
            print(f"  Pareto front size: {len(self.candidate_pool)}")
            print(f"  Hypervolume: {current_hypervolume:.6f}")
            print(f"  Oracle calls used: {len(self.oracle)}")
            
            # Save iteration results with prompt and response
            iteration_data = {
                'iteration': iteration + 1,
                'prompt_sent_to_llm': prompt_text,
                'llm_raw_response': response,
                'generated_molecules_with_explanations': [
                    {'molecule': mol, 'explanation': exp, 'pareto_score': score} 
                    for mol, exp, score in zip(valid_molecules, valid_explanations, new_scores)
                ],
                'current_pareto_front': [
                    {'molecule': smi, 'pareto_score': score} 
                    for smi, score in self.candidate_pool
                ],
                'pareto_front_size': len(self.candidate_pool),
                'oracle_calls': len(self.oracle)
            }
            self.iteration_results.append(iteration_data)
            
            # Save results after each iteration
            self._save_results_incremental(iteration + 1)
            
            # Save molecule cache after each iteration
            self._save_molecule_cache()
            
            # Check for improvement
            positive_scores = [score for _, score in positive_samples]
            if max(new_scores) > max(positive_scores):
                print(f"  ✅ Generated molecule better than best positive sample!")
            
        # Final results
        print(f"\n{'='*20} OPTIMIZATION COMPLETE {'='*20}")
        print(f"Final Pareto front size: {len(self.candidate_pool)}")
        print(f"Total oracle calls: {len(self.oracle)}")
        print(f"Best molecule found: {self.candidate_pool[0][0] if self.candidate_pool else 'None'}")
        best_score = self.candidate_pool[0][1] if self.candidate_pool else 0.0
        print(f"Best score: {best_score:.4f}")
        final_hypervolume = self.hypervolume_history[-1] if self.hypervolume_history else 0.0
        print(f"Final hypervolume: {final_hypervolume:.6f}")
        print(f"Multi-objective setup: Max {self.args.max_obj}, Min {self.args.min_obj}")
        
        # Save final results
        self._save_results()
        
        # Save final molecule cache
        self._save_molecule_cache()
        
        return self.candidate_pool
    
    def _save_results_incremental(self, current_iteration: int):
        """Save optimization results incrementally after each iteration"""
        results = {
            'optimization_info': {
                'max_obj': self.args.max_obj,
                'min_obj': self.args.min_obj,
                'seed': self.args.seed,
                'algorithm': 'LLM_BlackBox_Pareto',
                'current_iteration': current_iteration,
                'last_updated': f"After iteration {current_iteration}"
            },
            'current_results': {
                'total_oracle_calls': len(self.oracle),
                'pareto_front_size': len(self.candidate_pool)
            },
            'pareto_front': [
                {'molecule': smi, 'pareto_score': score} 
                for smi, score in self.candidate_pool
            ],
            'detailed_iterations': self.iteration_results
        }
        
        output_file = os.path.join(self.args.output_dir, 'optimization_results_pareto.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved after iteration {current_iteration} to: {output_file}")
        print(f"   📊 Saved {len(self.iteration_results)} iteration details with prompts, molecules, explanations, and pareto scores")
        print(f"   🧬 Current Pareto front: {len(self.candidate_pool)} molecules with pareto scores")
    
    def _save_results(self):
        """Save final optimization results to JSON file"""
        results = {
            'optimization_info': {
                'max_obj': self.args.max_obj,
                'min_obj': self.args.min_obj,
                'seed': self.args.seed,
                'algorithm': 'LLM_BlackBox_Pareto',
                'total_iterations': len(self.iteration_results),
                'status': 'COMPLETED'
            },
            'final_results': {
                'total_oracle_calls': len(self.oracle),
                'pareto_front_size': len(self.candidate_pool)
            },
            'pareto_front': [
                {'molecule': smi, 'pareto_score': score} 
                for smi, score in self.candidate_pool
            ],
            'detailed_iterations': self.iteration_results
        }
        
        output_file = os.path.join(self.args.output_dir, 'optimization_results_pareto.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Final results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='LLM Black-box Optimizer for Multi-Objective Pareto Optimization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_init', type=int, default=120, help='Initial pool size (random)')
    parser.add_argument('--m', type=int, default=5, help='Number of positive/negative samples per iteration')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations')
    parser.add_argument('--max_oracle_calls', type=int, default=20000, help='Maximum oracle calls')
    parser.add_argument('--freq_log', type=int, default=100, help='Frequency of logging')
    parser.add_argument('--output_dir', type=str, default='./results_pareto', help='Output directory')
    
    # Multi-objective specific arguments
    parser.add_argument('--max_obj', nargs="+", default=["jnk3", "qed"], help='Objectives to maximize')
    parser.add_argument('--min_obj', nargs="+", default=["sa"], help='Objectives to minimize')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add required attributes for compatibility
    args.oracles = args.max_obj + args.min_obj
    args.mol_lm = 'GPT4'
    args.n_jobs = 1
    args.log_results = True
    args.smi_file = None
    
    print(f"Starting LLM Black-box Multi-Objective Pareto Optimization:")
    print(f"  Maximizing objectives: {args.max_obj}")
    print(f"  Minimizing objectives: {args.min_obj}")
    print(f"  Seed: {args.seed}")
    print(f"  Initial pool size (random): {args.n_init}")
    print(f"  Samples per iteration (m): {args.m}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Max oracle calls: {args.max_oracle_calls}")
    print(f"  Using GPT-4.1-mini via OpenAI")
    
    # Create optimizer
    optimizer = LLMBlackBoxOptimizerPareto(args)
    
    # Run optimization
    final_pareto_front = optimizer.optimize(
        n_init=args.n_init,
        m=args.m,
        max_iterations=args.max_iterations
    )
    
    print("Multi-objective Pareto optimization completed!")

if __name__ == "__main__":
    main() 