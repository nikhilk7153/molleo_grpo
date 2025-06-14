from __future__ import print_function
import os

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 

def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='molleo')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mol_lm', type=str, default=None, choices=[None, "BioT5", "MoleculeSTM", "GPT-4", "Qwen"])
    parser.add_argument('--bin_size', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--oracles', nargs="+", default=["qed"]) ###
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_dir', default="./results")
    args = parser.parse_args()


    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(f"Running method: {args.method}")
    print(f"Using molecular LM: {args.mol_lm if args.mol_lm else 'None (baseline)'}")
    print(f"Optimizing oracles: {args.oracles}")
    print(f"Seeds: {args.seed}")
    print(f"Output saving: ENABLED - All generations, explanations, and results will be saved")
    print("="*70)

    # Add method name here when adding new ones
    from main.molleo.run import GB_GA_Optimizer as Optimizer

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main
    
    for oracle_name in args.oracles:
        print(f"\n{'='*70}")
        print(f"OPTIMIZING ORACLE: {oracle_name.upper()}")
        print(f"{'='*70}")

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
        oracle = Oracle(name = oracle_name)
        optimizer = Optimizer(args=args)
        print(config_default)

        for seed in args.seed:
            print(f"\n{'-'*50}")
            print(f"RUNNING SEED: {seed}")
            print(f"{'-'*50}")
            # Set current seed in args for output logging
            args.current_seed = seed
            optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
            print(f"Completed seed {seed}")

        print(f"\nCompleted oracle {oracle_name}")

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    
    print(f"\n{'='*70}")
    print(f"ALL RUNS COMPLETED")
    print(f"{'='*70}")
    print('---- The whole process takes %.2f hours ----' % (hours))
    print()
    print("OUTPUT FILES CREATED:")
    print("Each run creates a directory with the pattern:")
    print("  evolutionary_output_{oracle}_{model}_seed{seed}_{timestamp}/")
    print()
    print("Each directory contains:")
    print("  - generations.csv: Generation-by-generation data with all molecules and scores")
    print("  - run_data.json: Complete run data in JSON format")
    print("  - console_output.log: All console output including LLM explanations")
    print("  - summary.txt: Run summary with final results")
    print()
    print("You can analyze the results using the CSV files or JSON data.")

if __name__ == "__main__":
    main()

