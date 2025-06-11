from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import main.molleo_multi_pareto.crossover as co, main.molleo_multi_pareto.mutate as mu
from main.pareto_optimizer import BaseOptimizer

#from main.graph_ga.mol_lm import MolCLIP
from main.molleo_multi_pareto.biot5 import BioT5
from main.molleo_multi_pareto.GPT4 import GPT4
from main.molleo_multi_pareto.Qwen import Qwen
from utils import get_fp_scores
from network import create_and_train_network, obtain_model_pred

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
            self.mol_lm.task = self.args.task_mode

    def _optimize(self, config):

        self.oracle.assign_evaluator(self.args)

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
        
        # Print initial population scores
        print(f"\n=== INITIAL POPULATION ===")
        print(f"Population size: {len(population_scores)}")
        print(f"Initial scores - Mean: {np.mean(population_scores):.4f}, Max: {np.max(population_scores):.4f}, Min: {np.min(population_scores):.4f}")
        print(f"Best initial molecule: {Chem.MolToSmiles(population_mol[np.argmax(population_scores)])} (score: {np.max(population_scores):.4f})")

        patience = 0
        generation = 0

        while True:
            generation += 1
            print(f"\n=== GENERATION {generation} ===");

            if len(self.oracle) > 1:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
            else:
                old_score = 0

            # new_population
            mating_tuples = make_mating_pool(population_mol, population_scores, config["population_size"])

            
            fp_scores = []
            offspring_mol_temp = []
            if self.args.mol_lm == 'GPT-4' or self.args.mol_lm == 'Qwen':
                print(f"Generating {config['offspring_size']} offspring using {self.args.mol_lm}...")
                offspring_mol = []
                offspring_smiles = []
                for i in range(config["offspring_size"]):
                    print(f"\n  --- Offspring {i+1} ---")
                    new_mol = self.mol_lm.edit(mating_tuples, config["mutation_rate"])
                    offspring_mol.append(new_mol)
                    if new_mol is not None:
                        smi = Chem.MolToSmiles(new_mol)
                        offspring_smiles.append(smi)
                        print(f"  Generated SMILES: {smi}")
                        # Get immediate score for this molecule
                        try:
                            score = self.oracle([smi])[0]
                            print(f"  Score: {score:.4f}")
                        except Exception as e:
                            print(f"  Score: Error - {e}")
                    else:
                        print(f"  Failed to generate valid molecule")
                
                # Filter out None molecules
                offspring_mol = [mol for mol in offspring_mol if mol is not None]
                print(f"\nGenerated {len(offspring_mol)} valid offspring molecules")
            elif self.args.mol_lm == 'BioT5':
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
                        print("exiting while loop before filling up bin..........")
                        break
                    m = population_mol[idxs[ii]]
                    editted_mol = self.mol_lm.edit([m])[0]

                    if editted_mol != None:
                        s = Chem.MolToSmiles(editted_mol)
                        if s != None:
                            print("adding editted molecule!!!")
                            editted_smi.append(s)
                    ii += 1
                sim = get_fp_scores(editted_smi, top_smi)
                print("fp_scores_to_top", sim)
                sorted_idx = np.argsort(np.squeeze(sim))[::-1][:config["offspring_size"]]
                print("top 70", sorted_idx)
                editted_smi = np.array(editted_smi)[sorted_idx].tolist()
                offspring_mol = [Chem.MolFromSmiles(s) for s in editted_smi]
                print("len offspring_mol", len(offspring_mol))

            


            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)
            #Pareto optimal set
            self.oracle.clean_buffer()
            population_mol = self.oracle.select_pareto_front([Chem.MolToSmiles(mol) for mol in population_mol])
            # stats
            old_scores = population_scores
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            
            # Print scores for new offspring (before Pareto filtering)
            if len(population_scores) > len(old_scores):
                new_scores = population_scores[len(old_scores):]
                print(f"\nNew offspring scores (before Pareto filtering):")
                for i, score in enumerate(new_scores):
                    if i < len(offspring_mol):
                        smi = Chem.MolToSmiles(population_mol[len(old_scores) + i])
                        print(f"  {smi}: {score:.4f}")
            
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            
            # Print generation summary
            print(f"\n--- Generation {generation} Summary (Pareto Optimal) ---")
            print(f"Pareto front size: {len(population_scores)}")
            print(f"Score statistics - Mean: {np.mean(population_scores):.4f}, Max: {np.max(population_scores):.4f}, Min: {np.min(population_scores):.4f}, Std: {np.std(population_scores):.4f}")
            print(f"Best molecule this generation: {Chem.MolToSmiles(population_mol[0])} (score: {population_scores[0]:.4f})")
            print(f"Total oracle calls so far: {len(self.oracle)}")
            
            # Show all Pareto optimal molecules (since this is typically a smaller set)
            print(f"All Pareto optimal molecules:")
            for i in range(len(population_mol)):
                smi = Chem.MolToSmiles(population_mol[i])
                print(f"  {i+1}. {smi}: {population_scores[i]:.4f}")


            ### early stopping
            if len(self.oracle) > 1:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score
                
            if self.finish:
                break

