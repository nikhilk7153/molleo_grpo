#!/usr/bin/env python
"""
LLM Black-box Optimizer V2 - GPT-4.1-mini Version

Uses GPT-4.1-mini via OpenAI API.

Follows the specified algorithm:
1. Randomly generate n samples to initialize a candidate pool
2. For each iteration:
   - Sample 2m samples from pool using weighted sampling
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
import re

# Add path for MolLEO imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from main.optimizer import Oracle as MolleoOracle
from main.molleo.GPT4 import sanitize_smiles

# OpenAI API configuration
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class LLMBlackBoxOptimizerV2GPT4:
    def __init__(self, args):
        self.args = args
        self.oracle = MolleoOracle(args=args)
        self.candidate_pool = []  # List of (smiles, score) tuples
        self.model_name = "gpt-4.1-mini"
        
        print("🎯 Using GPT-4.1-mini via OpenAI API for molecular generation")
        
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
        self.cumulative_reward = 0.0  # Track cumulative reward across iterations
        
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
    
    def sample_with_weighted_sampling(self, m: int) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Sample 2m samples from pool using weighted sampling:
        - Positive samples: weighted sampling using scores directly (higher scores = higher probability)
        - Negative samples: 1/2 invalid molecules (score=0) + 1/2 valid low-scoring molecules (inverse weights)
        - Uses proper weighted sampling and ensures positive samples score higher than valid negative samples
        """
        if len(self.candidate_pool) < 2 * m:
            print(f"Warning: Pool size ({len(self.candidate_pool)}) < 2m ({2*m}), using all available molecules")
            half = len(self.candidate_pool) // 2
            return self.candidate_pool[:half], self.candidate_pool[half:]
        
        molecules = [smi for smi, _ in self.candidate_pool]
        scores = np.array([score for _, score in self.candidate_pool])
        
        # Step 1: Sample positive samples using weighted sampling favoring high scores
        # Use scores directly as weights (higher scores = higher probability)
        positive_weights = scores + 1e-8  # Add small epsilon to avoid zero weights
        positive_weights = positive_weights / np.sum(positive_weights)
        
        positive_indices = np.random.choice(
            len(self.candidate_pool), 
            size=m, 
            replace=False, 
            p=positive_weights
        )
        positive_samples = [self.candidate_pool[i] for i in positive_indices]
        
        # Step 2: Sample negative samples - 1/2 invalid molecules + 1/2 valid low-scoring molecules
        remaining_indices = [i for i in range(len(self.candidate_pool)) if i not in positive_indices]
        
        # Separate invalid (score=0) and valid molecules from remaining pool
        invalid_indices = [i for i in remaining_indices if self.candidate_pool[i][1] == 0.0]
        valid_remaining_indices = [i for i in remaining_indices if self.candidate_pool[i][1] > 0.0]
        
        negative_samples = []
        
        # Sample m/2 invalid molecules (if available)
        target_invalid = m // 2
        if len(invalid_indices) >= target_invalid:
            # Randomly sample from invalid molecules
            sampled_invalid_indices = np.random.choice(
                invalid_indices, 
                size=target_invalid, 
                replace=False
            )
            negative_samples.extend([self.candidate_pool[i] for i in sampled_invalid_indices])
            used_invalid = target_invalid
        else:
            # Use all available invalid molecules
            negative_samples.extend([self.candidate_pool[i] for i in invalid_indices])
            used_invalid = len(invalid_indices)
        
        # Sample remaining slots from valid low-scoring molecules
        remaining_slots = m - used_invalid
        
        if remaining_slots > 0 and len(valid_remaining_indices) > 0:
            # Weighted sampling favoring lower scores among valid molecules
            valid_remaining_scores = scores[valid_remaining_indices]
            
            # Use inverse scores as weights (lower scores = higher probability)
            max_score = np.max(valid_remaining_scores)
            negative_weights = (max_score - valid_remaining_scores) + 1e-8  # Invert scores and add epsilon
            negative_weights = negative_weights / np.sum(negative_weights)
            
            sample_size = min(remaining_slots, len(valid_remaining_indices))
            negative_indices_in_valid = np.random.choice(
                len(valid_remaining_indices),
                size=sample_size,
                replace=False,
                p=negative_weights
            )
            valid_negative_indices = [valid_remaining_indices[i] for i in negative_indices_in_valid]
            negative_samples.extend([self.candidate_pool[i] for i in valid_negative_indices])
        
        pos_scores = [score for _, score in positive_samples]
        neg_scores = [score for _, score in negative_samples]
        
        # Step 3: Verify and potentially adjust to ensure positive > negative
        if pos_scores and neg_scores:
            min_positive = min(pos_scores)
            max_negative = max(neg_scores)
            
            if min_positive <= max_negative:
                print(f"  ⚠️  Adjusting samples: some negative samples ({max_negative:.4f}) scored higher than positive samples ({min_positive:.4f})")
                
                # Alternative approach: Sample 2m molecules and split by score
                all_samples_2m = positive_samples + negative_samples
                all_samples_2m.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
                
                positive_samples = all_samples_2m[:m]  # Top m
                negative_samples = all_samples_2m[m:2*m]  # Bottom m
                
                pos_scores = [score for _, score in positive_samples]
                neg_scores = [score for _, score in negative_samples]
                
                print(f"  ✅ Adjusted - Positive samples now guaranteed to be higher scoring")
        
        # Count invalid vs valid in negative samples
        invalid_neg_count = sum(1 for _, score in negative_samples if score == 0.0)
        valid_neg_count = len(negative_samples) - invalid_neg_count
        
        print(f"Weighted sampling:")
        print(f"  Positive samples: {len(positive_samples)}, score range: {min(pos_scores):.4f} - {max(pos_scores):.4f}")
        print(f"  Negative samples: {len(negative_samples)} total ({invalid_neg_count} invalid + {valid_neg_count} valid low-scoring)")
        if neg_scores:
            print(f"    Negative score range: {min(neg_scores):.4f} - {max(neg_scores):.4f}")
        
        # Final verification
        if pos_scores and neg_scores:
            min_positive = min(pos_scores)
            valid_neg_scores = [score for score in neg_scores if score > 0.0]  # Exclude invalid molecules (score=0)
            
            if valid_neg_scores:
                max_valid_negative = max(valid_neg_scores)
                if min_positive > max_valid_negative:
                    print(f"  ✅ Positive samples ({min_positive:.4f}+) all score higher than valid negative samples (≤{max_valid_negative:.4f})")
                else:
                    print(f"  ⚠️  Warning: Some valid negative samples ({max_valid_negative:.4f}) score higher than positive samples ({min_positive:.4f})")
            else:
                print(f"  ✅ All negative samples are invalid molecules (score=0), positive samples are all valid")
        
        return positive_samples, negative_samples
    
    def query_llm_with_specific_prompt(self, positive_samples: List[Tuple[str, float]], 
                                     negative_samples: List[Tuple[str, float]], 
                                     m: int) -> Tuple[str, str]:
        """Query LLM with the specific prompt format requested"""
        
        # Extract molecules and rewards
        positive_molecules = [smi for smi, _ in positive_samples]
        positive_rewards = [score for _, score in positive_samples]
        negative_molecules = [smi for smi, _ in negative_samples]
        negative_rewards = [score for _, score in negative_samples]
        
        # Create the specific prompt format with molecules paired with their JNK3 scores
        positive_pairs = [f"{smi} (JNK3: {score:.3f})" for smi, score in positive_samples]
        negative_pairs = [f"{smi} (JNK3: {score:.3f})" for smi, score in negative_samples]
        
        prompt = f"""You are a black-box reward optimizer. Here are {len(positive_samples)} positive samples with high JNK3 inhibition scores and {len(negative_samples)} negative samples with low/zero JNK3 scores.

POSITIVE SAMPLES (high JNK3 inhibition):
{chr(10).join([f"  • {pair}" for pair in positive_pairs])}

NEGATIVE SAMPLES (low/zero JNK3 inhibition):
{chr(10).join([f"  • {pair}" for pair in negative_pairs])}

Please analyze the results, and output {m} new samples with JNK3 scores better than the positive samples. The new samples should be diversified.

Please generate exactly {m} new diverse molecular structures as SMILES strings that should achieve higher JNK3 scores than the positive samples. Having diverse molecules contributes to your reward score, so avoid making minr tweaks to the top molecules from the positive sameple.


For each molecule, provide both an explanation and the molecule using this exact format:
<explanation>Your reasoning for why this molecule should have higher JNK3 score</explanation> + <molecule>SMILES_STRING</molecule>

Generate {m} diverse molecules:"""

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
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048
                )
                return prompt, response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenAI API call failed (attempt {retry + 1}/3): {e}")
                if retry == 2:
                    raise e
        
        return prompt, ""  # Return prompt and empty string instead of None
    
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
    
    def calculate_combined_reward_updated(self, all_molecules: List[str], all_scores: List[float],
                                        valid_molecules_only: List[str], positive_samples: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Calculate combined reward with three weighted metrics:
        1. new_reward_max - positive_reward_max: Best new molecule vs best positive sample from prompt (weight: 0.2)
        2. new_reward_mean - positive_reward_mean: Mean new molecules vs mean positive samples from prompt (weight: 0.5)
        3. diversity of valid_samples: Tanimoto distance-based diversity score (weight: 0.3)
        
        Combined = 0.2*reward_1 + 0.5*reward_2 + 0.3*reward_3
        
        Args:
            all_molecules: All newly generated molecules from LLM (valid + invalid)
            all_scores: Scores of all newly generated molecules (0 for invalid)
            valid_molecules_only: Only chemically valid molecules for diversity calculation
            positive_samples: The same {m} positive samples that were used in the LLM prompt
        """
        if not all_scores or not positive_samples:
            return {"reward_1": 0.0, "reward_2": 0.0, "reward_3": 0.0, "combined_reward": 0.0}
        
        positive_scores = [score for _, score in positive_samples]
        
        # Metric 1: Best new molecule score - Best positive sample score (from prompt)
        reward_1 = max(all_scores) - max(positive_scores)
        
        # Metric 2: Mean new molecules score - Mean positive samples score (from prompt)
        reward_2 = np.mean(all_scores) - np.mean(positive_scores)
        
        # Metric 3: diversity of valid_samples only (using Tanimoto distance)
        reward_3 = self.calculate_diversity_score(valid_molecules_only)
        
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
        print("LLM BLACK-BOX OPTIMIZER V2 - GPT-4.1-mini")
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
            
            # Step 2: Sample 2m samples from pool using weighted sampling
            positive_samples, negative_samples = self.sample_with_weighted_sampling(m)
            
            # Show the samples that will be used in the prompt
            print(f"\n📊 SAMPLES FOR LLM PROMPT:")
            print("=" * 60)
            print(f"🟢 POSITIVE SAMPLES (High-scoring examples):")
            for i, (smi, score) in enumerate(positive_samples):
                print(f"   {i+1}. {smi} (Score: {score:.4f})")
            
            print(f"\n🔴 NEGATIVE SAMPLES (Low-scoring/invalid examples):")
            for i, (smi, score) in enumerate(negative_samples):
                status = "invalid" if score == 0.0 else "low-scoring"
                print(f"   {i+1}. {smi} (Score: {score:.4f}, {status})")
            print("=" * 60)
            
            # Step 3: Prompt LLM with specific format
            prompt_sent, response = self.query_llm_with_specific_prompt(positive_samples, negative_samples, m)
            
            print(f"\n📝 PROMPT SENT TO LLM:")
            print("=" * 60)
            print(prompt_sent)
            print("=" * 60)
            
            if not response:
                print("Failed to get LLM response, skipping iteration")
                continue
            
            print(f"\n🤖 LLM RESPONSE:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
            # Extract molecules with explanations from response
            molecules_with_explanations = self.extract_molecules_from_response(response)
            
            if not molecules_with_explanations:
                print("No valid molecules generated, skipping iteration")
                continue
            
            # Step 4: Calculate rewards of proposed samples
            print(f"\nScoring {len(molecules_with_explanations)} new molecules...")
            all_molecules = []  # All molecules (valid + invalid)
            all_scores = []     # All scores (0 for invalid)
            all_explanations = []  # All explanations
            valid_molecules_only = []  # Only chemically valid molecules for diversity calculation
            valid_scores_only = []     # Only scores of valid molecules
            molecule_validity = []     # Track which molecules are valid
            
            for i, (smi, explanation) in enumerate(molecules_with_explanations):
                all_molecules.append(smi)
                all_explanations.append(explanation)
                
                # First check if molecule is chemically valid
                sanitized_smi = sanitize_smiles(smi)
                if sanitized_smi is None:
                    # Invalid molecule - assign score of 0
                    score = 0.0
                    all_scores.append(score)
                    molecule_validity.append(False)
                    print(f"  {i+1}. {smi}: 0.0000 (invalid chemistry)")
                    print(f"      💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
                else:
                    try:
                        score = self.oracle.score_smi(smi)
                        all_scores.append(score)
                        valid_molecules_only.append(smi)
                        valid_scores_only.append(score)
                        molecule_validity.append(True)
                        print(f"  {i+1}. {smi}: {score:.4f}")
                        print(f"      💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
                    except Exception as e:
                        # Oracle failed - assign score of 0
                        score = 0.0
                        all_scores.append(score)
                        molecule_validity.append(False)
                        print(f"  {i+1}. {smi}: 0.0000 (oracle error: {str(e)[:50]})")
                        print(f"      💭 {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
                
                if self.oracle.finish:
                    break
            
            if not all_scores:
                print("No molecules could be scored, skipping iteration")
                continue
            
            print(f"\nGenerated {len(all_molecules)} total molecules:")
            print(f"  Valid molecules: {len(valid_molecules_only)}")
            print(f"  Invalid molecules: {len(all_molecules) - len(valid_molecules_only)}")
            
            # Print detailed results for each molecule
            print(f"\n🔬 GENERATED MOLECULES WITH EXPLANATIONS & REWARDS:")
            print("=" * 80)
            for i, (mol, score, exp, is_valid) in enumerate(zip(all_molecules, all_scores, all_explanations, molecule_validity)):
                status = "✅ VALID" if is_valid else "❌ INVALID"
                print(f"\n🧪 Molecule {i+1}: {status}")
                print(f"   SMILES: {mol}")
                print(f"   JNK3 Score: {score:.4f}")
                print(f"   Explanation: {exp}")
                print("-" * 80)
            print()
            
            # Step 5: Calculate combined reward and do RL training
            # Use only valid molecules for diversity calculation
            reward_metrics = self.calculate_combined_reward_updated(
                all_molecules, all_scores, valid_molecules_only, positive_samples
            )
            
            # Update cumulative reward
            self.cumulative_reward += reward_metrics['combined_reward']
            
            print(f"\n🎯 Combined Reward Calculation:")
            print(f"  Reward 1 (max improvement): {reward_metrics['reward_1']:.4f} (weight: 0.2)")
            print(f"  Reward 2 (mean improvement): {reward_metrics['reward_2']:.4f} (weight: 0.5)")
            print(f"  Reward 3 (diversity): {reward_metrics['reward_3']:.4f} (weight: 0.3)")
            print(f"  Combined Reward: {reward_metrics['combined_reward']:.4f}")
            print(f"  Cumulative Reward: {self.cumulative_reward:.4f}")
            
            # Add new molecules to candidate pool
            for smi, score in zip(all_molecules, all_scores):
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
            print(f"  Generated molecules: {len(all_molecules)}")
            print(f"  Best new score: {max(all_scores):.4f}")
            print(f"  Mean new score: {np.mean(all_scores):.4f}")
            print(f"  Current best overall: {current_best:.4f}")
            print(f"  Pool size: {len(self.candidate_pool)}")
            print(f"  Oracle calls used: {len(self.oracle)}")
            
            # Save iteration results with all requested data
            iteration_data = {
                'iteration': iteration + 1,
                'prompt_sent_to_llm': prompt_sent,  # 1) Prompt sent to LLM
                'positive_samples': [(smi, score) for smi, score in positive_samples],
                'negative_samples': [(smi, score) for smi, score in negative_samples],
                'generated_molecules': all_molecules,  # 2) All generated molecules (valid + invalid)
                'jnk3_scores': all_scores,  # 2) JNK3 score for each molecule (0 for invalid)
                'generated_explanations': all_explanations,  # 2) Generated explanation for each molecule
                'molecule_validity': molecule_validity,  # Track which molecules are valid
                'valid_molecules_only': valid_molecules_only,  # Only valid molecules
                'valid_scores_only': valid_scores_only,  # Only scores of valid molecules
                'reward_metrics': reward_metrics,  # 2) Reward given by the model
                'combined_reward_this_iteration': reward_metrics['combined_reward'],
                'cumulative_reward': self.cumulative_reward,  # 3) Cumulative reward
                'best_score_overall': current_best,
                'oracle_calls': len(self.oracle),
                # Detailed molecule data for easy access (includes both valid and invalid)
                'molecule_details': [
                    {
                        'smiles': mol,
                        'jnk3_score': score,
                        'explanation': exp,
                        'is_valid': is_valid
                    }
                    for mol, score, exp, is_valid in zip(all_molecules, all_scores, all_explanations, molecule_validity)
                ]
            }
            self.iteration_results.append(iteration_data)
            
            # Save results after each iteration to avoid data loss
            self._save_results()
            print(f"  💾 Results saved after iteration {iteration + 1}")
            
            # Check for improvement
            positive_scores = [score for _, score in positive_samples]
            if max(all_scores) > max(positive_scores):
                print(f"  ✅ Generated molecule better than best positive sample!")
            
        # Final results
        print(f"\n{'='*20} OPTIMIZATION COMPLETE {'='*20}")
        print(f"Final pool size: {len(self.candidate_pool)}")
        print(f"Total oracle calls: {len(self.oracle)}")
        print(f"Best molecule found: {self.candidate_pool[0][0]}")
        print(f"Best score: {self.candidate_pool[0][1]:.4f}")
        
        # Save final results (final update)
        self._save_results()
        print(f"💾 Final results saved!")
        
        return self.candidate_pool
    
    def _save_results(self):
        """Save optimization results to JSON file
        
        Saves all requested data:
        1) Prompt sent to LLM for each iteration
        2) Generated molecules with JNK3 scores and explanations
        3) Cumulative reward across all iterations
        """
        results = {
            'optimization_info': {
                'oracle': self.args.oracle,
                'seed': self.args.seed,
                'algorithm': 'LLM_BlackBox_V2_GPT4',
                'model': self.model_name,
                'total_iterations': len(self.iteration_results)
            },
            'final_results': {
                'best_molecule': self.candidate_pool[0][0] if self.candidate_pool else None,
                'best_score': self.candidate_pool[0][1] if self.candidate_pool else 0.0,
                'total_oracle_calls': len(self.oracle),
                'pool_size': len(self.candidate_pool),
                'final_cumulative_reward': self.cumulative_reward
            },
            'best_scores_history': self.best_scores_history,
            # THIS IS THE KEY SECTION - Contains all prompts, molecules, explanations, and rewards
            'iteration_details': self.iteration_results,  # Each iteration has: prompt_sent_to_llm, generated_molecules, generated_explanations, jnk3_scores, reward_metrics, molecule_details
            'top_molecules': [
                {'smiles': smi, 'score': score} 
                for smi, score in self.candidate_pool[:50]
            ],
            # Summary of what's saved for each iteration:
            'data_description': {
                'iteration_details_contains': [
                    'prompt_sent_to_llm - Exact prompt sent to LLM',
                    'generated_molecules - All molecules (valid + invalid)', 
                    'jnk3_scores - JNK3 scores for each molecule (0 for invalid)',
                    'generated_explanations - LLM explanations for each molecule',
                    'molecule_validity - Boolean list indicating valid/invalid molecules',
                    'valid_molecules_only - Only chemically valid molecules',
                    'reward_metrics - All reward components (reward_1, reward_2, reward_3, combined)',
                    'cumulative_reward - Running total of rewards',
                    'molecule_details - Structured data with SMILES, scores, explanations, validity'
                ]
            }
        }
        
        # Use custom output filename if provided, otherwise use default
        if hasattr(self.args, 'output_file') and self.args.output_file:
            output_file = os.path.join(self.args.output_dir, self.args.output_file)
        else:
            output_file = os.path.join(self.args.output_dir, 'optimization_results_v2_gpt4.json')
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='LLM Black-box Optimizer V2 - GPT-4.1-mini')
    parser.add_argument('--oracle', type=str, default='jnk3', help='Oracle to optimize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_init', type=int, default=120, help='Initial pool size (random)')
    parser.add_argument('--m', type=int, default=5, help='Number of positive/negative samples per iteration')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations')
    parser.add_argument('--max_oracle_calls', type=int, default=20000, help='Maximum oracle calls')
    parser.add_argument('--output_dir', type=str, default='./results_blackbox_v2_gpt4', help='Output directory')
    parser.add_argument('--output_file', type=str, default=None, help='Custom output JSON filename (default: optimization_results_v2_gpt4.json)')
    
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
    
    print(f"Starting LLM Black-box Optimization V2 with GPT-4.1-mini:")
    print(f"  Oracle: {args.oracle}")
    print(f"  Seed: {args.seed}")
    print(f"  Initial pool size (random): {args.n_init}")
    print(f"  Samples per iteration (m): {args.m}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Max oracle calls: {args.max_oracle_calls}")
    print(f"  Using GPT-4.1-mini via OpenAI API")
    
    # Create optimizer
    optimizer = LLMBlackBoxOptimizerV2GPT4(args)
    
    # Run optimization
    final_pool = optimizer.optimize(
        n_init=args.n_init,
        m=args.m,
        max_iterations=args.max_iterations
    )
    
    print("Optimization completed!")

if __name__ == "__main__":
    main() 