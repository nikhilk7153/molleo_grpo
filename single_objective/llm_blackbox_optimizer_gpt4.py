#!/usr/bin/env python
"""
LLM Black-box Optimizer using GPT-4.1-mini

A different optimization approach that uses GPT-4.1-mini prompting with positive/negative examples
instead of genetic algorithms.

Pipeline:
1. Randomly generate n samples to initialize a candidate pool
2. For each iteration:
   - Sample 2m samples from pool (uniformly or exp(±r(sample)) weighted)
   - Score samples with reward oracle
   - Prompt GPT-4.1-mini with m positive + m negative samples and rewards
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
from tdc import Oracle
from openai import OpenAI
import re
import csv
import json
from datetime import datetime

# Add path for MolLEO imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from main.optimizer import Oracle as MolleoOracle

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMBlackBoxOptimizerGPT4:
    def __init__(self, args):
        self.args = args
        self.oracle = MolleoOracle(args=args)
        self.candidate_pool = []
        self.model_name = "gpt-4.1-mini"
        
        print("🎯 Using GPT-4.1-mini for molecular generation")
        
        self.setup_output_directory()
        self.load_zinc_dataset()
        self.initialize_data_storage()
        
        # Initialize adaptive sampling parameters
        self.score_threshold = 0.25  # Configurable threshold for triggering island sampling
        self.score_crisis = False
        self.island_crisis = False
        self.last_island_scores = {}
        
        
    def setup_output_directory(self):
        """Setup output directory and initialize data storage files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        oracle_name = getattr(self.args, 'oracle', 'unknown')
        seed = getattr(self.args, 'seed', 1)
        
        self.output_dir = f"gpt4_blackbox_output_{oracle_name}_seed{seed}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        
        # Setup JSON file for prompts and responses
        self.prompts_responses_file = os.path.join(self.output_dir, "prompts_and_responses.json")
        self.log_file = os.path.join(self.output_dir, "optimization_log.txt")
        
        # Initialize empty JSON file
        self.prompts_responses_data = []
    
    def load_zinc_dataset(self):
        """Load ZINC dataset from available sources"""
        print("Loading ZINC dataset...")
   
        args_copy = argparse.Namespace(**vars(self.args))
        args_copy.mol_lm = None
        from main.molleo.run import GB_GA_Optimizer
        temp_optimizer = GB_GA_Optimizer(args_copy)
        self.all_smiles = temp_optimizer.all_smiles
        print(f"  Loaded {len(self.all_smiles)} molecules via fallback")
            
    
    def initialize_data_storage(self):
        """Initialize essential data collection structures"""
        self.all_generated_molecules = []
        self.all_generated_scores = []
        self.invalid_molecules_count = 0
        self.best_molecules_per_iteration = []
        
    def log_to_file(self, message):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def save_data(self, iteration, prompt=None, response=None, molecules=None, scores=None, 
                  explanations=None, **kwargs):
        """Save prompts, responses, and detailed metrics to JSON file"""
        if prompt and response:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate metrics if molecules and scores are provided
            metrics = {}
            molecules_with_details = []
            
            if molecules and scores:
                # Calculate mean reward
                metrics['mean_reward'] = float(np.mean(scores))
                metrics['best_reward'] = float(max(scores))
                metrics['worst_reward'] = float(min(scores))
                metrics['std_reward'] = float(np.std(scores))
                
                # Calculate diversity
                metrics['diversity'] = self.calculate_diversity(molecules)
                
                # Count valid vs invalid molecules
                valid_count = sum(1 for mol in molecules if self.sanitize_smiles(mol) is not None)
                metrics['total_molecules'] = len(molecules)
                metrics['valid_molecules'] = valid_count
                metrics['invalid_molecules'] = len(molecules) - valid_count
                
                # Store molecules with their rewards and explanations
                for i, (mol, score) in enumerate(zip(molecules, scores)):
                    mol_entry = {
                        'molecule': mol,
                        'reward': float(score),
                        'rank': i + 1,
                        'explanation': explanations.get(mol, "No explanation provided") if explanations else "No explanation provided",
                        'is_valid': self.sanitize_smiles(mol) is not None
                    }
                    molecules_with_details.append(mol_entry)
            
            entry = {
                'iteration': iteration,
                'timestamp': timestamp,
                'prompt': prompt,
                'response': response,
                'molecules_generated': molecules if molecules else [],
                'molecules_with_details': molecules_with_details,
                'metrics': metrics
            }
            
            self.prompts_responses_data.append(entry)
            
            # Save to file after each iteration
            try:
                with open(self.prompts_responses_file, 'w') as f:
                    json.dump(self.prompts_responses_data, f, indent=2)
                print(f"💾 Saved prompt/response/metrics for iteration {iteration}")
                if metrics:
                    print(f"    📊 Mean reward: {metrics['mean_reward']:.4f}, Best: {metrics['best_reward']:.4f}, Diversity: {metrics['diversity']:.4f}")
                    print(f"    🧪 Valid: {metrics['valid_molecules']}/{metrics['total_molecules']} molecules")
            except Exception as e:
                print(f"⚠️ Error saving prompt/response data: {e}")
        
    def initialize_pool(self, n_samples: int):
        """Initialize candidate pool with top 10% of ZINC molecules"""
        print(f"Initializing candidate pool with top 10% from ZINC database...")
        
        from tdc import Oracle as TDCOracle
        tdc_oracle = TDCOracle(name=self.args.oracle)
        self.oracle.assign_evaluator(tdc_oracle)
        
        cache_file = f"zinc_scores_cache_{self.args.oracle}_10k.pkl"
        
        # Try to load cached scores
        if os.path.exists(cache_file):
            print(f"Found cached ZINC scores: {cache_file}")
         
            with open(cache_file, 'rb') as f:
                import pickle
                zinc_scores = pickle.load(f)
            print(f"  Loaded {len(zinc_scores)} pre-scored ZINC molecules from cache")
            
            zinc_scores.sort(key=lambda x: x[1], reverse=True)
            top_10_percent_size = max(n_samples, int(len(zinc_scores) * 0.10))
            top_molecules = zinc_scores[:top_10_percent_size]
            self.candidate_pool = top_molecules[:n_samples]
            
            self._print_pool_stats("cached scores")
            return
                
        
        # Score molecules from scratch
        zinc_oracle_start = len(self.oracle)
        zinc_sample_size = min(5000, len(self.all_smiles))
        print(f"Scoring {zinc_sample_size} ZINC molecules to find top 10%...")
        
        zinc_sample = np.random.choice(self.all_smiles, zinc_sample_size, replace=False)
        
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
   
        
        zinc_scores.sort(key=lambda x: x[1], reverse=True)
        top_10_percent_size = max(n_samples, int(zinc_sample_size * 0.10))
        top_molecules = zinc_scores[:top_10_percent_size]
        self.candidate_pool = top_molecules[:n_samples]
        
        self._print_pool_stats("newly scored")
        
        # Reset oracle counter to exclude ZINC scoring from optimization budget
        zinc_oracle_used = len(self.oracle) - zinc_oracle_start
        print(f"  ZINC scoring used {zinc_oracle_used} oracle calls (excluded from optimization budget)")
        
        if hasattr(self.oracle, 'mol_buffer'):
            self.oracle.zinc_calls = zinc_oracle_used
            optimization_buffer = self.oracle.mol_buffer[zinc_oracle_start:] if len(self.oracle.mol_buffer) > zinc_oracle_start else []
            self.oracle.mol_buffer = optimization_buffer
            print(f"  Oracle counter reset to {len(self.oracle)}")
    
    def _print_pool_stats(self, source):
        """Print candidate pool statistics"""
        scores = [score for _, score in self.candidate_pool]
        print(f"Initial pool statistics (top 10% of ZINC, {source}):")
        print(f"  Pool size: {len(self.candidate_pool)}")
        print(f"  Best score: {self.candidate_pool[0][1]:.4f}")
        print(f"  Worst score: {self.candidate_pool[-1][1]:.4f}")
        print(f"  Mean score: {np.mean(scores):.4f}")
        print(f"  Best molecule: {self.candidate_pool[0][0]}")
        
    def sample_positive_negative(self, m: int) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Sample positive and negative examples with island-based diversity strategy"""
        
        if not self.all_generated_molecules:
            # First iteration: use diverse high-scoring ZINC molecules from top 5%
            sorted_pool = sorted(self.candidate_pool, key=lambda x: x[1], reverse=True)
            top_5_percent_zinc = sorted_pool[:max(20, int(len(sorted_pool) * 0.05))]
            # Cluster into structural islands and sample from each
            positive_samples = self.get_diverse_molecules(top_5_percent_zinc, use_clustering=True)
            print(f"  First iteration: {len(positive_samples)} Tanimoto-diverse structural islands from top 5% (n={len(top_5_percent_zinc)})")
            print(f"  Selected islands represent distinct chemical scaffolds (randomized):")
            for i, (mol, score) in enumerate(positive_samples):
                print(f"    Structural Island {i+1}: {mol[:30]}... (score: {score:.4f})")
            
            # Store source information for prompt generation (first iteration only has structural islands)
            self.current_best_performers = []
            self.current_structural_islands = positive_samples
            
        else:
            # Later iterations: best performers + top 20 from pool
            generated_with_scores = []
            for smi in self.all_generated_molecules:
                for pool_smi, pool_score in self.candidate_pool:
                    if pool_smi == smi:
                        generated_with_scores.append((pool_smi, pool_score))
                        break
            
            if len(generated_with_scores) < 3:
                positive_samples = generated_with_scores
            else:
                positive_samples = []
                
                # Best performing molecules (take all top 20 generated)
                sorted_generated = sorted(generated_with_scores, key=lambda x: x[1], reverse=True)
                best_performers = sorted_generated[:min(20, len(sorted_generated))]  # Take all top 20
                positive_samples.extend(best_performers)
                
                # Top 20 molecules from entire candidate pool
                sorted_candidates = sorted(self.candidate_pool, key=lambda x: x[1], reverse=True)
                top_20_pool = sorted_candidates[:20]
                
                print(f"  Best performers: {len(best_performers)} top generated molecules")
                for i, (mol, score) in enumerate(best_performers):
                    print(f"    Best Generated {i+1}: {mol[:30]}... (score: {score:.4f})")
                print(f"  Top pool molecules: {len(top_20_pool)} top molecules from candidate pool")
                for i, (mol, score) in enumerate(top_20_pool):
                    print(f"    Top Pool {i+1}: {mol[:30]}... (score: {score:.4f})")
                
                # Store metadata to track source
                best_molecules_set = set(mol for mol, _ in best_performers)
                unique_pool_molecules = [(mol, score) for mol, score in top_20_pool 
                                         if mol not in best_molecules_set]
                
                # Store source information for prompt generation
                self.current_best_performers = best_performers
                self.current_structural_islands = unique_pool_molecules
                
                positive_samples.extend(unique_pool_molecules)
                
                if len(positive_samples) > 6:
                    positive_samples = positive_samples[:6]
                
        # Sample bottom 5 molecules from the last iteration (out of 10 generated)
        negative_samples = []
        if hasattr(self, 'last_iteration_molecules_with_scores') and len(self.last_iteration_molecules_with_scores) >= 5:
            # Sort by score (lowest to highest) and take bottom 5 molecules
            sorted_last_iteration = sorted(self.last_iteration_molecules_with_scores, key=lambda x: x[1])
            # Always take exactly the worst 5 molecules (since 10 are always generated)
            negative_samples = sorted_last_iteration[:5]
            print(f"  Negative sampling: Bottom 5 molecules from last iteration (out of {len(self.last_iteration_molecules_with_scores)} generated)")
            for i, (mol, score) in enumerate(negative_samples):
                status = "INVALID" if score == 0.0 else "VALID"
                print(f"    Last-iter-bottom {i+1}: {mol[:30]}... (score: {score:.4f}) - {status}")
                
        return positive_samples, negative_samples
    
    def get_diverse_molecules(self, molecules: List[Tuple[str, float]], max_molecules: int = 5, use_clustering: bool = False) -> List[Tuple[str, float]]:
        """Select diverse molecules using structural clustering"""
        if len(molecules) <= max_molecules:
            return molecules
            
        try:
            from rdkit import DataStructs
            from rdkit.Chem import rdMolDescriptors
            
            fps = []
            valid_molecules = []
            
            for smi, score in molecules:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                        fps.append(fp)
                        valid_molecules.append((smi, score))
                except:
                    continue
            
            if len(valid_molecules) <= max_molecules:
                return valid_molecules
                
            # Calculate similarity matrix
            n_mols = len(fps)
            similarity_matrix = np.zeros((n_mols, n_mols))
            
            for i in range(n_mols):
                for j in range(i+1, n_mols):
                    similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
            
            if use_clustering:
                # Cluster-based island sampling from top 20% of each cluster
                try:
                    from sklearn.cluster import AgglomerativeClustering
                    distance_matrix = 1.0 - similarity_matrix  # Convert to distance
                    
                    n_clusters = min(max(2, len(valid_molecules) // 3), 8)  # 2-8 clusters
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
                    cluster_labels = clustering.fit_predict(distance_matrix)
                    
                    # Group by cluster and sample 3 molecules from each island
                    islands = {}
                    for i, label in enumerate(cluster_labels):
                        if label not in islands:
                            islands[label] = []
                        islands[label].append(valid_molecules[i])
                    
                    print(f"  Identified {len(islands)} structural islands via Tanimoto clustering:")
                    sampled_molecules = []
                    for island_id, island_molecules in islands.items():
                        island_sorted = sorted(island_molecules, key=lambda x: x[1], reverse=True)
                        
                        # For islands with >= 10 molecules: take top 1% but ensure at least 1 molecule
                        # For islands with < 10 molecules: take all molecules
                        if len(island_sorted) >= 10:
                            top_1_percent_count = max(1, int(len(island_sorted) * 0.01))
                            top_performers = island_sorted[:top_1_percent_count]
                            num_to_sample = min(3, len(top_performers))
                        else:
                            top_performers = island_sorted
                            num_to_sample = len(top_performers)
                        
                        sampled_from_island = random.sample(top_performers, num_to_sample)
                        sampled_molecules.extend(sampled_from_island)
                        
                        top_1_percent_actual = int(len(island_sorted) * 0.01) if len(island_sorted) >= 100 else len(island_sorted)
                        print(f"    Island {island_id+1} ({len(island_molecules)} molecules, top 1%={top_1_percent_actual}, sampled {num_to_sample}):")
                        for j, (mol, score) in enumerate(sampled_from_island):
                            print(f"      {j+1}. {mol}... (score: {score:.4f})")
                    
                    return sampled_molecules
                except:
                    print("    Clustering failed, using greedy selection")
            
            # Greedy diverse selection with randomized starting point
            selected_indices = []
            sorted_indices = sorted(range(len(valid_molecules)), 
                                  key=lambda i: valid_molecules[i][1], reverse=True)
            # Randomly select starting molecule from top 5% to add diversity
            top_5_percent_count = max(1, int(len(sorted_indices) * 0.05))
            random_start = random.choice(sorted_indices[:top_5_percent_count])
            selected_indices.append(random_start)
            
            for _ in range(max_molecules - 1):
                if len(selected_indices) >= len(valid_molecules):
                    break
                    
                best_candidate = None
                best_diversity_score = -1
                
                for candidate_idx in sorted_indices:
                    if candidate_idx in selected_indices:
                        continue
                    
                    min_similarity = min(similarity_matrix[candidate_idx][selected_idx] 
                                       for selected_idx in selected_indices)
                    
                    molecule_score = valid_molecules[candidate_idx][1]
                    diversity_score = (1.0 - min_similarity) * 0.7 + molecule_score * 0.3
                    
                    if diversity_score > best_diversity_score:
                        best_diversity_score = diversity_score
                        best_candidate = candidate_idx
                
                if best_candidate is not None:
                    selected_indices.append(best_candidate)
            
            return [valid_molecules[i] for i in selected_indices]
            
        except Exception as e:
            print(f"    Diversity selection failed ({e}), using score-based selection")
            sorted_molecules = sorted(molecules, key=lambda x: x[1], reverse=True)
            return sorted_molecules[:max_molecules]
            
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
                    raise e
        
        return None
        
    def extract_molecules_from_response(self, response: str) -> Tuple[List[str], Dict[str, str]]:
        """Extract SMILES and explanations from GPT-4 response format - robust parsing"""
        molecules = []
        explanations = {}
        
        print("\n📋 EXTRACTING MOLECULES FROM RESPONSE:")
        print("="*80)
        
        import re
        
        # Find all explanation+molecule pairs directly
        # Pattern matches: <<<Explanation>>>: [text] <<<Molecule>>>: SMILES
        pattern = r'<<<Explanation>>>:\s*(.*?)<<<Molecule>>>:\s*([^\s\}]+)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        print(f"Found {len(matches)} explanation+molecule pairs")
        
        for i, (explanation, smiles) in enumerate(matches, 1):
            print(f"\n--- Processing Pair {i} ---")
            
            # Clean up explanation - remove trailing commas and whitespace
            explanation = explanation.strip()
            explanation = re.sub(r',\s*$', '', explanation)
            smiles = smiles.strip()
            
            print(f"  📝 Raw explanation: {explanation[:80]}{'...' if len(explanation) > 80 else ''}")
            print(f"  🧪 Raw SMILES: {smiles}")
            
            # Always include the molecule (valid or invalid)
            if smiles not in molecules:  # Avoid duplicates by raw SMILES
                molecules.append(smiles)  # Keep original SMILES (valid or invalid)
                explanations[smiles] = explanation
                
                # Check if it's valid for display purposes
                sanitized_smiles = self.sanitize_smiles(smiles)
                if sanitized_smiles:
                    print(f"  ✅ VALID: {smiles}")
                else:
                    print(f"  ❌ INVALID: {smiles} (will get score 0.0)")
            else:
                print(f"  ⚠️ DUPLICATE: {smiles} (skipped)")
        
        # Final summary
        print(f"\n✅ EXTRACTION COMPLETE:")
        print(f"   Total molecules extracted: {len(molecules)}")
        for i, mol in enumerate(molecules, 1):
            print(f"   {i}. {mol}")
            print(f"      Explanation: {explanations[mol][:100]}{'...' if len(explanations[mol]) > 100 else ''}")
        
            print("="*80)
                        
        return molecules, explanations

    
    def sanitize_smiles(self, smi: str) -> Optional[str]:
        """Return canonical SMILES representation"""
        if smi == '':
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            return None
        
    def generate_new_samples(self, positive_samples: List[Tuple[str, float]], 
                           negative_samples: List[Tuple[str, float]], 
                           m: int) -> Tuple[List[str], Dict[str, str], str, str]:
        """Use GPT-4.1-mini to generate new samples based on examples"""
        
        pos_molecules = [smi for smi, score in positive_samples]
        pos_rewards = [score for smi, score in positive_samples]
        neg_molecules = [smi for smi, score in negative_samples] if negative_samples else []
        neg_rewards = [score for smi, score in negative_samples] if negative_samples else []
        
        # Get top 20 molecules from entire candidate pool
        sorted_pool = sorted(self.candidate_pool, key=lambda x: x[1], reverse=True)
        top_20_pool = sorted_pool[:20]
        
        # Get diversity islands with adaptive sampling based on top 20 diversity and score thresholds
        diversity_islands = []
        if len(self.candidate_pool) > 50:
            # Check if top 20 molecules have low diversity (<0.80)
            top_20_molecules = [mol for mol, _ in sorted_pool[:20]]
            top_20_diversity = self.calculate_diversity(top_20_molecules)
            
            # Also check if top molecules have significantly low scores (below threshold)
            top_20_scores = [score for _, score in sorted_pool[:20]]
            score_threshold = 0.25  # Configurable threshold for "significantly low" scores
            low_score_molecules = [score for score in top_20_scores if score < score_threshold]
            score_crisis = len(low_score_molecules) >= 10  # More than half are below threshold
            
            # Check if any island has scores significantly below the threshold
            island_crisis = False
            if hasattr(self, 'last_island_scores'):
                for island_name, island_best_score in self.last_island_scores.items():
                    if island_best_score < score_threshold * 0.8:  # 20% below threshold
                        island_crisis = True
                        break
            
                    if top_20_diversity < 0.80 or score_crisis or island_crisis:
                        # Store crisis flags for context generation
                        self.score_crisis = score_crisis
                        self.island_crisis = island_crisis
                    
                    crisis_reasons = []
                    if top_20_diversity < 0.80:
                        crisis_reasons.append(f"low diversity ({top_20_diversity:.3f})")
                    if score_crisis:
                        crisis_reasons.append(f"score crisis ({len(low_score_molecules)}/20 below {score_threshold:.3f})")
                    if island_crisis:
                        crisis_reasons.append("island performance below threshold")
                        
                    print(f"  🚨 Triggering island sampling due to: {', '.join(crisis_reasons)}")
                    if score_crisis:
                        print(f"  📉 Score range: {max(top_20_scores):.3f} (best) to {min(top_20_scores):.3f} (worst)")
                    
                    print(f"  🔄 Activating island sampling to find better molecular regions")
                
                # Get islands from broader range and sample differently
                broader_pool = sorted_pool[:max(100, int(len(sorted_pool) * 0.50))]
                diversity_islands = self.get_diverse_molecules(broader_pool, max_molecules=12, use_clustering=True)
                
                # Replace top 20 with random samples from top 1% of each island
                top_20_pool = self.sample_from_island_top_5_percent(diversity_islands)
                print(f"  🎲 Generated new diverse top 20 from island sampling")
                
                # Track island performance for future crisis detection
                self.last_island_scores = {}
                molecules_per_island = 3
                num_islands = len(diversity_islands) // molecules_per_island + (1 if len(diversity_islands) % molecules_per_island > 0 else 0)
                for island_idx in range(num_islands):
                    start_idx = island_idx * molecules_per_island
                    end_idx = min(start_idx + molecules_per_island, len(diversity_islands))
                    island_molecules = diversity_islands[start_idx:end_idx]
                    island_scores = [score for _, score in island_molecules]
                    self.last_island_scores[f"island_{island_idx+1}"] = max(island_scores) if island_scores else 0.0
            else:
                print(f"  ✅ Top 20 molecules sufficiently diverse ({top_20_diversity:.3f}) and high-scoring")
                # Sample from top 5% of candidate pool for diversity analysis only
                top_5_percent = sorted_pool[:max(20, int(len(sorted_pool) * 0.05))]
                diversity_islands = self.get_diverse_molecules(top_5_percent, max_molecules=8, use_clustering=True)
        
            best_score = max(pos_rewards)
        
        # Adaptive context based on diversity detection
        # Use the actual top_20_pool (which may have been replaced with island samples)
        current_top_20_molecules = [mol for mol, _ in top_20_pool]
        current_top_20_diversity = self.calculate_diversity(current_top_20_molecules)
        
        # Determine if island sampling was triggered
        island_sampling_active = (top_20_diversity < 0.80 or 
                                 (hasattr(self, 'score_crisis') and self.score_crisis) or 
                                 (hasattr(self, 'island_crisis') and self.island_crisis))
        
        if island_sampling_active:
            iteration_context = f"You have access to {len(diversity_islands)} diverse structural islands representing elite performers from different chemical regions (diversity: {current_top_20_diversity:.3f})."
            optimization_goal = f"Generate {m} HIGHLY DIVERSE molecules that explore different chemical space from these elite island performers. Focus on NOVELTY and STRUCTURAL DIVERSITY while targeting high activity."
        else:
            iteration_context = f"You have access to the top 20 highest-scoring molecules and {len(diversity_islands)} diverse structural islands for analysis (diversity: {current_top_20_diversity:.3f})."
            optimization_goal = f"Generate {m} structurally diverse molecules that surpass the current best molecules."
        
        prompt = f"""You are an expert molecular design AI. {iteration_context}

"""
        
        # Show diversity islands for structural analysis
        import random
        if diversity_islands:
            prompt += f"DIVERSE STRUCTURAL ISLANDS (Top 1% from each island, randomly ordered):\n"
            prompt += f"These represent structurally diverse chemical scaffolds from the candidate pool.\n\n"
                
            # Group molecules back into islands and scramble each island
            molecules_per_island = 3
            num_islands = len(diversity_islands) // molecules_per_island + (1 if len(diversity_islands) % molecules_per_island > 0 else 0)
            
            for island_idx in range(num_islands):
                start_idx = island_idx * molecules_per_island
                end_idx = min(start_idx + molecules_per_island, len(diversity_islands))
                island_molecules = diversity_islands[start_idx:end_idx]
                
                # Randomly scramble molecules within each island
                scrambled_island = random.sample(island_molecules, len(island_molecules))
                
                prompt += f"ISLAND {island_idx + 1}:\n"
                for j, (smi, reward) in enumerate(scrambled_island):
                    prompt += f"  {j+1}. {smi} (reward: {reward:.4f})\n"
            
        if negative_samples:
            prompt += f"\nBOTTOM {len(negative_samples)} MOLECULES FROM LAST ITERATION (learn what to improve):\n"
            for i, (smi, reward) in enumerate(negative_samples):
                # Check if molecule is invalid (score 0.0 and fails sanitization)
                is_invalid = (reward == 0.0 and self.sanitize_smiles(smi) is None)
                status_text = " [INVALID SMILES]" if is_invalid else ""
                prompt += f"{i+1}. {smi} (reward: {reward:.4f}){status_text}\n"
        
        target_score = max(pos_rewards)
        prompt += f"""

TASK: {optimization_goal}



DESIGN STRATEGY:
Analyze the successful molecules above to understand what chemical features and structural patterns contribute to high scores.



YOUR APPROACH:
1. **Pattern Recognition**: Study what makes the top performers successful
2. **Feature Analysis**: Identify key functional groups, scaffolds, and molecular properties
3. **Diverse Design**: Generate structurally diverse molecules, not just variations of one approach. You can use the diversity islands for inspiration for the variety of molecules you can generate. Sometimes the top molecules may not lead to the best in the long run, so try to generate a variety of molecules that are structurally diverse and have high scores. Don't just try to make minor tweaks to the top molecules. Look at the diversity islands for inspiration.
4. **Strategic Combination**: Consider combining successful features from different molecules
5. **Balanced Exploration**: Balance learning from proven patterns with exploring new chemical space
6. **Crossover & Mutation**: You can perform crossover operations between any of the molecules shown above, or mutate them, or combine them with other molecules from your knowledge. 



MOLECULAR GENERATION OPTIONS:
- You can crossover/mutate any of the existing molecules shown above
- You can combine features from multiple molecules in the examples
- You can also design completely new molecules based on the patterns you observe
- You can use molecules from your training knowledge as crossover partners if beneficial
- The goal is to create better molecules using any approach that works. However, make them diverse enough so that the top molecules have diversity and are not just variations of the top 1-2 molecules.


Your output should follow the format: {{<<<Explanation>>>: $EXPLANATION, <<<Molecule>>>: $MOLECULE}}. Here are the requirements:

1. $EXPLANATION should explain your design rationale, which successful molecules inspired you, and why this approach should score higher.
2. The $MOLECULE should be the SMILES of your proposed molecule.
3. Ensure structural diversity across all your proposed molecules.

Generate {m} structurally diverse molecules:"""

        # Print the full prompt for transparency
        print(f"\n🔥 PROMPTING GPT-4.1-mini to generate {m} molecules (target > {max(pos_rewards):.4f})")
        print("="*100)
        print("FULL PROMPT:")
        print("="*100)
        print(prompt)
        print("="*100)
        
        response = self.query_llm(prompt)
        
        # Print the full response for transparency
        print("\n🤖 GPT-4.1-mini RESPONSE:")
        print("="*100)
        print(response)
        print("="*100)
    
        extracted_molecules, extracted_explanations = self.extract_molecules_from_response(response)
            
        # Return ALL extracted molecules (valid and invalid) for comprehensive negative sampling
        all_extracted = []
        all_explanations = {}
            
        for smi in extracted_molecules:
            if smi not in all_extracted:  # Avoid duplicates
                all_extracted.append(smi)  # Keep original SMILES (valid or invalid)
                all_explanations[smi] = extracted_explanations.get(smi, "No explanation provided")
                
            if len(all_extracted) >= m:
                    break
        
        return all_extracted, all_explanations, prompt, response
    
    def calculate_average_similarity(self, molecules: List[str]) -> float:
        """Calculate average pairwise Tanimoto similarity of molecules"""
        if len(molecules) < 2:
            return 0.0
            
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors
        
        fps = []
        for smi in molecules:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.append(fp)
        
        if len(fps) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(similarity)
        
        return np.mean(similarities)

    def sample_from_island_top_5_percent(self, diversity_islands: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Sample from top 1% of each island and randomly select 20 molecules"""
        import random
        
        # Group molecules into islands (assuming 3 molecules per island from get_diverse_molecules)
        molecules_per_island = 3
        num_islands = len(diversity_islands) // molecules_per_island + (1 if len(diversity_islands) % molecules_per_island > 0 else 0)
        
        # Collect top 1% from each island
        all_top_1_percent = []
        
        for island_idx in range(num_islands):
            start_idx = island_idx * molecules_per_island
            end_idx = min(start_idx + molecules_per_island, len(diversity_islands))
            island_molecules = diversity_islands[start_idx:end_idx]
            
            # For each island, find similar molecules in candidate pool and get top 5%
            island_representatives = [mol for mol, _ in island_molecules]
            similar_molecules = self.find_similar_molecules_in_pool(island_representatives)
            
            # Take top 1% of similar molecules from this island
            if similar_molecules:
                top_1_percent_count = max(1, int(len(similar_molecules) * 0.01))
                top_1_percent = similar_molecules[:top_1_percent_count]
                all_top_1_percent.extend(top_1_percent)
        
        # If we don't have enough molecules, add some from the original diversity islands
        if len(all_top_1_percent) < 20:
            remaining = [mol for mol in diversity_islands if mol not in all_top_1_percent]
            all_top_1_percent.extend(remaining)
        
        # Randomly sample 20 molecules from all the top 1% collections
        if len(all_top_1_percent) >= 20:
            sampled_molecules = random.sample(all_top_1_percent, 20)
        else:
            sampled_molecules = all_top_1_percent
        
        return sampled_molecules
    
    def find_similar_molecules_in_pool(self, representative_molecules: List[str]) -> List[Tuple[str, float]]:
        """Find molecules in candidate pool that are similar to the representatives"""
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors
        
        # Get fingerprints for representatives
        rep_fps = []
        for smi in representative_molecules:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                rep_fps.append(fp)
        
        if not rep_fps:
            return []
        
        # Find similar molecules in candidate pool
        similar_molecules = []
        for pool_smi, pool_score in self.candidate_pool:
            pool_mol = Chem.MolFromSmiles(pool_smi)
            if pool_mol is not None:
                pool_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(pool_mol, 2, nBits=1024)
                
                # Check similarity to any representative
                max_similarity = 0.0
                for rep_fp in rep_fps:
                    similarity = DataStructs.TanimotoSimilarity(pool_fp, rep_fp)
                    max_similarity = max(max_similarity, similarity)
                
                # If similarity > 0.3, consider it part of this island's chemical space
                if max_similarity > 0.3:
                    similar_molecules.append((pool_smi, pool_score))
        
        # Sort by score (highest first)
        similar_molecules.sort(key=lambda x: x[1], reverse=True)
        return similar_molecules

    def calculate_diversity(self, molecules: List[str]) -> float:
        """Calculate Tanimoto diversity of molecules"""
        return max(0.0, 1.0 - self.calculate_average_similarity(molecules))
            
    
    
    def calculate_combined_reward(self, new_molecules: List[str], new_scores: List[float], 
                                positive_samples: List[Tuple[str, float]]) -> float:
        """Calculate combined RL reward"""
        if not new_scores or not positive_samples:
            return 0.0
            
        positive_scores = [score for _, score in positive_samples]
        
        max_improvement = max(new_scores) - max(positive_scores)
        mean_improvement = np.mean(new_scores) - np.mean(positive_scores)
        diversity = self.calculate_diversity(new_molecules)
        
        combined_reward = (2.0 * max_improvement + 
                          1.0 * mean_improvement + 
                          0.5 * diversity)
        
        return combined_reward
        
    def optimize(self, n_init: int = 1000, m_per_iter: int = 50, max_iterations: int = 20, 
                 pool_sample_size: int = 200, use_weighted_sampling: bool = True):
        """Run the black-box optimization"""
        
        print("="*80)
        print("LLM BLACK-BOX OPTIMIZER (GPT-4.1-mini)")
        print("="*80)
        
        self.initialize_pool(n_init)
        
        for iteration in range(max_iterations):
            print(f"\n{'='*20} ITERATION {iteration + 1}/{max_iterations} {'='*20}")
            
            if self.oracle.finish:
                print("Oracle budget exhausted, stopping optimization")
                break
                
            # Sample examples
            positive_samples, negative_samples = self.sample_positive_negative(5)
            
            # Generate new samples (now includes both valid and invalid molecules)
            all_extracted_molecules, all_explanations, prompt, response = self.generate_new_samples(
                positive_samples, negative_samples, m_per_iter)
            
            
            
            if not all_extracted_molecules:
                print("❌ NO MOLECULES EXTRACTED FROM RESPONSE - SKIPPING ITERATION")
                continue
                
                            # Score all extracted molecules (valid and invalid)
            print(f"\nScoring {len(all_extracted_molecules)} extracted molecules...")
            new_scores = []
            valid_new_molecules = []
            iteration_molecules_with_scores = []  # Track ALL molecules for negative sampling
            unique_count = 0
            existing_molecules = set([smi for smi, _ in self.candidate_pool])
            
            for i, smi in enumerate(all_extracted_molecules):
                is_unique = smi not in existing_molecules
                if is_unique:
                    unique_count += 1
                
                sanitized_smi = self.sanitize_smiles(smi)
                if sanitized_smi is None:
                    # Invalid molecule - give score 0
                    print(f"  {i+1}. {smi}: INVALID - score: 0.0000")
                    score = 0.0
                    new_scores.append(score)
                    valid_new_molecules.append(smi)  # Keep original invalid SMILES for tracking
                    iteration_molecules_with_scores.append((smi, score))  # Track for negative sampling
                    if is_unique:
                        self.candidate_pool.append((smi, score))  # Add invalid SMILES to pool with score 0
                    self.invalid_molecules_count += 1
                    continue
                
                # Valid molecule - score it
                if is_unique:
                    score = self.oracle.score_smi(sanitized_smi)
                    self.candidate_pool.append((sanitized_smi, score))
                else:
                    # Find existing score
                    score = next((s for m, s in self.candidate_pool if m == sanitized_smi), 
                               self.oracle.score_smi(sanitized_smi))
                
                new_scores.append(score)
                valid_new_molecules.append(sanitized_smi)
                iteration_molecules_with_scores.append((sanitized_smi, score))  # Track for negative sampling
                explanation_preview = all_explanations.get(smi, "No explanation")[:80] + "..." if len(all_explanations.get(smi, "")) > 80 else all_explanations.get(smi, "No explanation")
                print(f"  {i+1}. {sanitized_smi}: {score:.4f} {'✓' if is_unique else '⚠️'}")
                print(f"      💡 {explanation_preview}")
                
                if self.oracle.finish:
                    break
            
            # Sort pool and limit size
            self.candidate_pool.sort(key=lambda x: x[1], reverse=True)
            if len(self.candidate_pool) > 5000:
                self.candidate_pool = self.candidate_pool[:5000]
            
            # Track data
            self.all_generated_molecules.extend(valid_new_molecules)
            self.all_generated_scores.extend(new_scores)
            
            # Track last iteration molecules with scores for negative sampling (including invalid ones)
            self.last_iteration_molecules_with_scores = iteration_molecules_with_scores
            
            # Save data with detailed metrics and explanations
            self.save_data(
                iteration + 1, 
                prompt=prompt,
                response=response,
                molecules=all_extracted_molecules,  # Save all molecules (valid and invalid)
                scores=new_scores,
                explanations=all_explanations
            )
            
            # Print summary
            if new_scores:
                best_new = max(new_scores)
                avg_new = np.mean(new_scores)
                current_best = self.candidate_pool[0][1]
                
                print(f"\nIteration {iteration + 1} Summary:")
                print(f"  Generated: {len(valid_new_molecules)} molecules")
                print(f"  Unique: {unique_count} molecules")
                print(f"  Best this iteration: {best_new:.4f}")
                print(f"  Average this iteration: {avg_new:.4f}")
                print(f"  Current best overall: {current_best:.4f}")
                print(f"  Oracle calls used: {len(self.oracle)}")
                
                # Track best molecule per iteration
                best_idx = new_scores.index(best_new)
                best_molecule = valid_new_molecules[best_idx]
                # Find original SMILES for explanation lookup
                original_smi = next((smi for smi in all_extracted_molecules if self.sanitize_smiles(smi) == best_molecule), best_molecule)
                best_explanation = all_explanations.get(original_smi, "No explanation")
                self.best_molecules_per_iteration.append((iteration + 1, best_molecule, best_new, best_explanation))
            
        # Final results
        print(f"\n{'='*20} OPTIMIZATION COMPLETE {'='*20}")
        print(f"Final pool size: {len(self.candidate_pool)}")
        print(f"Total oracle calls: {len(self.oracle)}")
        print(f"Best molecule found: {self.candidate_pool[0][0]}")
        print(f"Best score: {self.candidate_pool[0][1]:.4f}")
        
        if self.all_generated_scores:
            print(f"\nGenerated Molecule Statistics:")
            print(f"  Total generated: {len(self.all_generated_molecules)}")
            print(f"  Average score: {np.mean(self.all_generated_scores):.4f}")
            print(f"  Best score: {max(self.all_generated_scores):.4f}")
            print(f"  Invalid molecules: {self.invalid_molecules_count}")
        
        # Save final results
        self.save_final_results()
        self.oracle.log_intermediate(finish=True)
        
        return self.candidate_pool
        
    def save_final_results(self):
        """Save comprehensive final results"""
        final_results = {
            'optimization_summary': {
                'oracle_name': self.args.oracle,
                'model_used': self.model_name,
                'total_oracle_calls': len(self.oracle),
                'total_molecules_generated': len(self.all_generated_molecules),
                'unique_molecules_generated': len(set(self.all_generated_molecules)),
                'invalid_molecules_count': self.invalid_molecules_count,
                'final_pool_size': len(self.candidate_pool),
                'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'best_molecules': {
                'overall_best': {
                    'smiles': self.candidate_pool[0][0] if self.candidate_pool else None,
                    'score': self.candidate_pool[0][1] if self.candidate_pool else None
                },
                'top_10_molecules': [
                    {'rank': i+1, 'smiles': mol, 'score': score}
                    for i, (mol, score) in enumerate(self.candidate_pool[:10])
                ]
            }
        }
        
        final_results_file = os.path.join(self.output_dir, 'final_optimization_results.json')
       
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
            print(f"💾 Final results saved to: {final_results_file}")
      

def main():
    parser = argparse.ArgumentParser(description='LLM Black-box Optimizer using GPT-4.1-mini')
    parser.add_argument('--oracle', type=str, default='jnk3', help='Oracle to optimize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_init', type=int, default=500, help='Initial pool size')
    parser.add_argument('--m_per_iter', type=int, default=20, help='Samples per iteration')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations')
    parser.add_argument('--pool_sample_size', type=int, default=200, help='Pool sample size')
    parser.add_argument('--max_oracle_calls', type=int, default=20000, help='Maximum oracle calls')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='./results_blackbox_gpt4', help='Output directory')
    parser.add_argument('--uniform_sampling', action='store_true', help='Use uniform sampling')
    
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
    args.output_dir = args.output_dir
    
    print(f"Starting LLM Black-box Optimization:")
    print(f"  Oracle: {args.oracle}")
    print(f"  Model: GPT-4.1-mini")
    print(f"  Initial pool: {args.n_init}")
    print(f"  Samples per iteration: {args.m_per_iter}")
    print(f"  Max iterations: {args.max_iterations}")
    
    # Create and run optimizer
    optimizer = LLMBlackBoxOptimizerGPT4(args)
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