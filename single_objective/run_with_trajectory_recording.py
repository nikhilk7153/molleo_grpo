from __future__ import print_function
import os

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 
from molleo_trajectory_recorder import MolLEOTrajectoryRecorder

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
    
    # Trajectory recording arguments
    parser.add_argument('--record_trajectories', action='store_true', help='Enable trajectory recording')
    parser.add_argument('--trajectory_output_dir', type=str, default='./trajectory_data', help='Directory to save trajectory data')
    
    args = parser.parse_args()

    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones

    from main.molleo.run import GB_GA_Optimizer as Optimizer

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main
    
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
        oracle = Oracle(name = oracle_name)
        
        # Initialize optimizer with trajectory recording if enabled
        optimizer = Optimizer(args=args)
        print(config_default)

        for seed in args.seed:
            print('seed', seed)
            
            # Initialize trajectory recorder if enabled
            trajectory_recorder = None
            if args.record_trajectories:
                trajectory_recorder = MolLEOTrajectoryRecorder(
                    output_dir=args.trajectory_output_dir,
                    oracle_name=oracle_name,
                    seed=seed
                )
                print(f"Trajectory recording enabled. Output dir: {args.trajectory_output_dir}")
                
                # Patch the optimizer to record trajectories
                optimizer.trajectory_recorder = trajectory_recorder
            
            optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
            
            # Save trajectories if recording was enabled
            if trajectory_recorder:
                trajectory_recorder.save_trajectories()

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main() 