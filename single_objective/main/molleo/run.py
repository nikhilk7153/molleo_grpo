from __future__ import print_function

import random
from typing import List
import json
import csv
import os
from datetime import datetime

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import main.molleo.crossover as co, main.molleo.mutate as mu
from main.optimizer import BaseOptimizer

from main.molleo.mol_lm import MolCLIP
from main.molleo.biot5 import BioT5
from main.molleo.GPT4 import GPT4
from main.molleo.Qwen import Qwen
from .utils import get_fp_scores
from .network import create_and_train_network, obtain_model_pred

MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    all_tuples = list(zip(population_scores, population_mol))
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
    
    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    
    return mating_tuples


def reproduce(mating_tuples, mutation_rate, mol_lm=None, net=None):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent = []
    parent.append(random.choice(mating_tuples))
    parent.append(random.choice(mating_tuples))

    parent_mol = [t[1] for t in parent]
    new_child = co.crossover(parent_mol[0], parent_mol[1])
    new_child_mutation = None
    if new_child is not None:
        new_child_mutation = mu.mutate(new_child, mutation_rate, mol_lm)
    return new_child, new_child_mutation

def get_best_mol(population_scores, population_mol):
    top_mol = population_mol[np.argmax(population_scores)]
    top_smi = Chem.MolToSmiles(top_mol)
    return top_smi

class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "molleo"

        self.mol_lm = None
        if args.mol_lm == "GPT-4":
            self.mol_lm = GPT4()
        elif args.mol_lm == "BioT5":
            self.mol_lm = BioT5()
        elif args.mol_lm == "Qwen":
            self.mol_lm = Qwen()

        self.args = args
        lm_name = "baseline"
        if args.mol_lm != None:
            lm_name = args.mol_lm
            self.mol_lm.task = self.args.oracles
            
        # Initialize output logging
        self.output_dir = None
        self.csv_file = None
        self.json_file = None
        self.log_file = None
        self.run_data = {}
        
        # Initialize explanation tracking
        self.molecule_explanations = {}  # SMILES -> explanation mapping

    def setup_output_logging(self, oracle_name, seed):
        """Setup output files for logging evolutionary algorithm progress"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mol_lm_name = self.args.mol_lm if self.args.mol_lm else "baseline"
        
        # Create output directory
        self.output_dir = f"evolutionary_output_{oracle_name}_{mol_lm_name}_seed{seed}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup CSV file for generation-by-generation data
        self.csv_file = os.path.join(self.output_dir, "generations.csv")
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'molecule_rank', 'smiles', 'score', 
                'mean_score', 'max_score', 'min_score', 'std_score',
                'total_oracle_calls', 'is_offspring', 'parent_info', 'explanation'
            ])
        
        # Setup JSON file for detailed run data
        self.json_file = os.path.join(self.output_dir, "run_data.json")
        
        # Setup log file for console output
        self.log_file = os.path.join(self.output_dir, "console_output.log")
        
        # Initialize run data
        self.run_data = {
            'oracle_name': oracle_name,
            'mol_lm': mol_lm_name,
            'seed': seed,
            'start_time': timestamp,
            'config': {},
            'generations': [],
            'final_results': {}
        }
        
        print(f"Output will be saved to: {self.output_dir}")

    def log_to_file(self, message):
        """Log message to both console and file"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def save_generation_data(self, generation, population_mol, population_scores, offspring_info=None):
        """Save generation data to CSV and JSON"""
        if not self.csv_file:
            return
            
        # Calculate statistics
        mean_score = np.mean(population_scores)
        max_score = np.max(population_scores)
        min_score = np.min(population_scores)
        std_score = np.std(population_scores)
        
        # Save to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, (mol, score) in enumerate(zip(population_mol, population_scores)):
                smiles = Chem.MolToSmiles(mol)
                is_offspring = offspring_info and i in offspring_info.get('offspring_indices', [])
                parent_info = offspring_info.get('parent_info', {}).get(i, '') if offspring_info else ''
                
                # Get explanation from SMILES lookup
                explanation = self.molecule_explanations.get(smiles, "")
                
                writer.writerow([
                    generation, i+1, smiles, score,
                    mean_score, max_score, min_score, std_score,
                    len(self.oracle), is_offspring, parent_info, explanation
                ])
        
        # Save to JSON
        generation_data = {
            'generation': generation,
            'population_size': len(population_scores),
            'statistics': {
                'mean': float(mean_score),
                'max': float(max_score),
                'min': float(min_score),
                'std': float(std_score)
            },
            'molecules': [],
            'total_oracle_calls': len(self.oracle),
            'offspring_info': offspring_info
        }
        
        # Add molecules with explanations
        for i, (mol, score) in enumerate(zip(population_mol, population_scores)):
            smiles = Chem.MolToSmiles(mol)
            is_offspring = offspring_info and i in offspring_info.get('offspring_indices', [])
            
            # Get explanation from SMILES lookup
            explanation = self.molecule_explanations.get(smiles, "")
            
            molecule_data = {
                'rank': i+1,
                'smiles': smiles,
                'score': float(score),
                'is_offspring': is_offspring,
                'explanation': explanation
            }
            generation_data['molecules'].append(molecule_data)
        
        self.run_data['generations'].append(generation_data)
        
        # Save JSON file
        with open(self.json_file, 'w') as f:
            json.dump(self.run_data, f, indent=2)

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        
        # Setup output logging
        oracle_name = oracle.name if hasattr(oracle, 'name') else 'unknown'
        seed = getattr(self.args, 'current_seed', 0)  # Will be set by main run loop
        self.setup_output_logging(oracle_name, seed)
        
        # Save config
        self.run_data['config'] = config

        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, config["population_size"])

        # select initial population
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
        
        # Log and save initial population
        message = f"\n=== INITIAL POPULATION ==="
        self.log_to_file(message)
        message = f"Population size: {len(population_scores)}"
        self.log_to_file(message)
        message = f"Initial scores - Mean: {np.mean(population_scores):.4f}, Max: {np.max(population_scores):.4f}, Min: {np.min(population_scores):.4f}"
        self.log_to_file(message)
        message = f"Best initial molecule: {Chem.MolToSmiles(population_mol[np.argmax(population_scores)])} (score: {np.max(population_scores):.4f})"
        self.log_to_file(message)
        
        # Save initial population data
        self.save_generation_data(0, population_mol, population_scores)

        patience = 0
        generation = 0

        while True:
            generation += 1
            message = f"\n=== GENERATION {generation} ==="
            self.log_to_file(message)

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            # new_population
            mating_tuples = make_mating_pool(population_mol, population_scores, config["population_size"])
            
            fp_scores = []
            offspring_mol_temp = []
            offspring_info = {'offspring_indices': [], 'parent_info': {}, 'explanations': []}
            offspring_mol = []  # Initialize offspring_mol for all cases
            
            if self.args.mol_lm == "GPT-4" or self.args.mol_lm == "Qwen":
                message = f"Generating {config['offspring_size']} offspring using {self.args.mol_lm}..."
                self.log_to_file(message)
                offspring_smiles = []
                offspring_explanations = []
                for i in range(config["offspring_size"]):
                    message = f"\n  --- Offspring {i+1} ---"
                    self.log_to_file(message)
                    result = self.mol_lm.edit(mating_tuples, config["mutation_rate"])
                    
                    # Handle both old format (just molecule) and new format (molecule + explanation)
                    if isinstance(result, tuple) and len(result) == 2:
                        new_mol, explanation = result
                        offspring_explanations.append(explanation)
                        # Log the explanation
                        message = f"  LLM Explanation: {explanation}"
                        self.log_to_file(message)
                    else:
                        new_mol = result
                        explanation = "No explanation available"
                        offspring_explanations.append(explanation)
                    
                    offspring_mol.append(new_mol)
                    if new_mol is not None:
                        smi = Chem.MolToSmiles(new_mol)
                        offspring_smiles.append(smi)
                        message = f"  Generated SMILES: {smi}"
                        self.log_to_file(message)
                        # Get immediate score for this molecule
                        try:
                            score = self.oracle([smi])[0]
                            message = f"  Score: {score:.4f}"
                            self.log_to_file(message)
                        except Exception as e:
                            message = f"  Score: Error - {e}"
                            self.log_to_file(message)
                    else:
                        message = f"  Failed to generate valid molecule"
                        self.log_to_file(message)
                
                # Filter out None molecules and corresponding explanations
                valid_offspring = []
                valid_explanations = []
                for mol, exp in zip(offspring_mol, offspring_explanations):
                    if mol is not None:
                        valid_offspring.append(mol)
                        valid_explanations.append(exp)
                        # Store explanation by SMILES for persistent tracking
                        smiles = Chem.MolToSmiles(mol)
                        self.molecule_explanations[smiles] = exp
                
                offspring_mol = valid_offspring
                offspring_info['explanations'] = valid_explanations
                message = f"\nGenerated {len(offspring_mol)} valid offspring molecules"
                self.log_to_file(message)
                 
            elif self.args.mol_lm == "BioT5":
                top_smi = get_best_mol(population_scores, population_mol) 

                offspring_mol = [reproduce(mating_tuples, config["mutation_rate"]) for _ in range(config["offspring_size"])]
                offspring_mol = [item[0] for item in offspring_mol]
                editted_smi = []
                for m in offspring_mol:
                    if m != None:
                        editted_smi.append(Chem.MolToSmiles(m))
                ii = 0
                idxs = np.argsort(population_scores)[::-1]
                while len(editted_smi) < self.args.bin_size:
                    if ii == len(idxs):
                        self.log_to_file("exiting while loop before filling up bin..........")
                        break
                    m = population_mol[idxs[ii]]
                    editted_mol = self.mol_lm.edit([m])[0]

                    if editted_mol != None:
                        s = Chem.MolToSmiles(editted_mol)
                        if s != None:
                            self.log_to_file("adding editted molecule!!!")
                            editted_smi.append(s)
                    ii += 1
                sim = get_fp_scores(editted_smi, top_smi)
                self.log_to_file(f"fp_scores_to_top {sim}")
                sorted_idx = np.argsort(np.squeeze(sim))[::-1][:config["offspring_size"]]
                self.log_to_file(f"top 70 {sorted_idx}")
                editted_smi = np.array(editted_smi)[sorted_idx].tolist()
                offspring_mol = [Chem.MolFromSmiles(s) for s in editted_smi]
                
                # Store explanations for BioT5 generated molecules
                biot5_explanation = "BioT5 molecular editing based on top-performing molecules"
                for mol in offspring_mol:
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        self.molecule_explanations[smiles] = biot5_explanation
                
                self.log_to_file(f"len offspring_mol {len(offspring_mol)}")
            
            else:
                # Baseline case - no molecular LM, use standard genetic operations
                message = f"Generating {config['offspring_size']} offspring using baseline genetic operations..."
                self.log_to_file(message)
                offspring_results = [reproduce(mating_tuples, config["mutation_rate"]) for _ in range(config["offspring_size"])]
                offspring_mol = [item[0] for item in offspring_results if item[0] is not None]
                
                # Store explanations by SMILES for baseline
                baseline_explanation = "Baseline genetic operations (crossover + mutation)"
                for mol in offspring_mol:
                    smiles = Chem.MolToSmiles(mol)
                    self.molecule_explanations[smiles] = baseline_explanation
                
                offspring_info['explanations'] = [baseline_explanation] * len(offspring_mol)
                message = f"Generated {len(offspring_mol)} valid offspring molecules using crossover and mutation"
                self.log_to_file(message)

            # Track offspring indices for logging
            old_population_size = len(population_mol)
            offspring_info['offspring_indices'] = list(range(old_population_size, old_population_size + len(offspring_mol)))

            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # stats
            old_scores = population_scores
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            
            # Log scores for new offspring
            if len(population_scores) > len(old_scores):
                new_scores = population_scores[len(old_scores):]
                message = f"\nNew offspring scores:"
                self.log_to_file(message)
                for i, score in enumerate(new_scores):
                    if i < len(offspring_mol):
                        smi = Chem.MolToSmiles(population_mol[len(old_scores) + i])
                        message = f"  {smi}: {score:.4f}"
                        self.log_to_file(message)
            
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            
            # Log generation summary
            message = f"\n--- Generation {generation} Summary ---"
            self.log_to_file(message)
            message = f"Population size: {len(population_scores)}"
            self.log_to_file(message)
            message = f"Score statistics - Mean: {np.mean(population_scores):.4f}, Max: {np.max(population_scores):.4f}, Min: {np.min(population_scores):.4f}, Std: {np.std(population_scores):.4f}"
            self.log_to_file(message)
            message = f"Best molecule this generation: {Chem.MolToSmiles(population_mol[0])} (score: {population_scores[0]:.4f})"
            self.log_to_file(message)
            message = f"Total oracle calls so far: {len(self.oracle)}"
            self.log_to_file(message)
            
            # Show top 5 molecules
            message = f"Top 5 molecules:"
            self.log_to_file(message)
            for i in range(min(5, len(population_mol))):
                smi = Chem.MolToSmiles(population_mol[i])
                message = f"  {i+1}. {smi}: {population_scores[i]:.4f}"
                self.log_to_file(message)

            # Save generation data
            self.save_generation_data(generation, population_mol, population_scores, offspring_info)

            ### early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        message = 'convergence criteria met, abort ...... '
                        self.log_to_file(message)
                        break
                else:
                    patience = 0

                old_score = new_score
                
            if self.finish:
                break
        
        # Save final results
        self.run_data['final_results'] = {
            'total_generations': generation,
            'total_oracle_calls': len(self.oracle),
            'final_best_score': float(population_scores[0]),
            'final_best_molecule': Chem.MolToSmiles(population_mol[0]),
            'convergence_reason': 'patience_exceeded' if patience >= self.args.patience else 'max_calls_reached',
            'end_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Final save
        with open(self.json_file, 'w') as f:
            json.dump(self.run_data, f, indent=2)
        
        # Create summary file
        summary_file = os.path.join(self.output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Evolutionary Algorithm Run Summary\n")
            f.write(f"==================================\n\n")
            f.write(f"Oracle: {self.run_data['oracle_name']}\n")
            f.write(f"Model: {self.run_data['mol_lm']}\n")
            f.write(f"Seed: {self.run_data['seed']}\n")
            f.write(f"Start Time: {self.run_data['start_time']}\n")
            f.write(f"End Time: {self.run_data['final_results']['end_time']}\n")
            f.write(f"Total Generations: {self.run_data['final_results']['total_generations']}\n")
            f.write(f"Total Oracle Calls: {self.run_data['final_results']['total_oracle_calls']}\n")
            f.write(f"Final Best Score: {self.run_data['final_results']['final_best_score']:.6f}\n")
            f.write(f"Final Best Molecule: {self.run_data['final_results']['final_best_molecule']}\n")
            f.write(f"Convergence Reason: {self.run_data['final_results']['convergence_reason']}\n")
        
        message = f"\n=== RUN COMPLETED ==="
        self.log_to_file(message)
        message = f"All output saved to: {self.output_dir}"
        self.log_to_file(message)
        message = f"Files created:"
        self.log_to_file(message)
        message = f"  - generations.csv: Generation-by-generation data"
        self.log_to_file(message)
        message = f"  - run_data.json: Complete run data in JSON format"
        self.log_to_file(message)
        message = f"  - console_output.log: All console output"
        self.log_to_file(message)
        message = f"  - summary.txt: Run summary"
        self.log_to_file(message)

