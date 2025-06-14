#!/usr/bin/env python3
"""
Test DPO-Optimized Model for Molecule Generation

This script uses the DPO-trained model to generate molecules and evaluates them
with the jnk3 oracle to see if the training improved generation quality.
"""

import torch
import sys
import os
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdkit import Chem
import re
from typing import List, Optional
import argparse
from tqdm import tqdm

# Add path for MolLEO imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from main.optimizer import Oracle as MolleoOracle

class DPOMoleculeGenerator:
    def __init__(self, model_path: str, oracle_name: str = 'jnk3'):
        """
        Initialize the DPO molecule generator
        
        Args:
            model_path: Path to the DPO-optimized model
            oracle_name: Name of the oracle to use for evaluation
        """
        self.model_path = model_path
        self.oracle_name = oracle_name
        
        print(f"🚀 Loading DPO-optimized model from: {model_path}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model parameters: {self.model.num_parameters():,}")
        
        # Setup oracle for evaluation
        self.setup_oracle()
        
    def setup_oracle(self):
        """Setup the oracle for molecule evaluation"""
        print(f"🔬 Setting up {self.oracle_name} oracle...")
        
        # Create args object for MolLEO compatibility
        class Args:
            def __init__(self):
                self.oracles = [self.oracle_name]
                self.output_dir = './test_results'
                self.freq_log = 100
                self.n_jobs = 1
                self.log_results = False
                self.max_oracle_calls = 10000
        
        args = Args()
        self.oracle = MolleoOracle(args=args)
        
        # Initialize oracle evaluator
        from tdc import Oracle as TDCOracle
        tdc_oracle = TDCOracle(name=self.oracle_name)
        self.oracle.assign_evaluator(tdc_oracle)
        
        print(f"✅ Oracle setup complete")
        
    def sanitize_smiles(self, smi: str) -> Optional[str]:
        """
        Canonicalize and validate SMILES string
        
        Args:
            smi: SMILES string to validate
            
        Returns:
            Canonical SMILES or None if invalid
        """
        if not smi or smi == '':
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            if mol is None:
                return None
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            return None
            
    def extract_molecules_from_response(self, response: str) -> List[str]:
        """Extract SMILES strings from model response"""
        molecules = []
        
        # Pattern 1: \box{SMILES} format
        box_matches = re.findall(r'\\box\{(.*?)\}', response)
        for match in box_matches:
            sanitized = self.sanitize_smiles(match)
            if sanitized and sanitized not in molecules:
                molecules.append(sanitized)
        
        # Pattern 2: SMILES: followed by molecule
        if not molecules:
            smiles_matches = re.findall(r'SMILES:\s*([A-Za-z0-9\[\]()=@#+\-\.\\/:]+)', response)
            for match in smiles_matches:
                sanitized = self.sanitize_smiles(match.strip())
                if sanitized and sanitized not in molecules:
                    molecules.append(sanitized)
        
        # Pattern 3: Look for chemistry-like strings
        if not molecules:
            # Find potential SMILES patterns (strings with common chemistry characters)
            potential_smiles = re.findall(r'\b[A-Za-z0-9\[\]()=@#+\-\.]{8,}\b', response)
            for match in potential_smiles:
                sanitized = self.sanitize_smiles(match)
                if sanitized and sanitized not in molecules:
                    molecules.append(sanitized)
                    if len(molecules) >= 5:  # Don't get too many from fallback
                        break
                        
        return molecules
    
    def generate_molecules(self, prompt: str, num_molecules: int = 5, temperature: float = 0.8, max_length: int = 200) -> List[str]:
        """
        Generate molecules using the DPO-optimized model
        
        Args:
            prompt: Generation prompt
            num_molecules: Number of molecules to generate
            temperature: Sampling temperature
            max_length: Maximum generation length
            
        Returns:
            List of valid SMILES strings
        """
        all_molecules = []
        attempts = 0
        max_attempts = num_molecules * 3  # Try up to 3x to get enough molecules
        
        while len(all_molecules) < num_molecules and attempts < max_attempts:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()  # Remove prompt from response
            
            # Extract molecules
            molecules = self.extract_molecules_from_response(response)
            
            # Add unique valid molecules
            for mol in molecules:
                if mol not in all_molecules:
                    all_molecules.append(mol)
                    if len(all_molecules) >= num_molecules:
                        break
            
            attempts += 1
            
        return all_molecules[:num_molecules]
    
    def evaluate_molecules(self, molecules: List[str]) -> List[float]:
        """
        Evaluate molecules with the oracle
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            List of scores
        """
        scores = []
        for mol in molecules:
            try:
                score = self.oracle.score_smi(mol)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Failed to score {mol}: {e}")
                scores.append(0.0)
        return scores
    
    def run_generation_test(self, num_batches: int = 10, molecules_per_batch: int = 10):
        """
        Run a comprehensive molecule generation test
        
        Args:
            num_batches: Number of generation batches
            molecules_per_batch: Molecules to generate per batch
        """
        print(f"\n🧪 STARTING MOLECULE GENERATION TEST")
        print(f"📊 Batches: {num_batches}, Molecules per batch: {molecules_per_batch}")
        print(f"🎯 Target: {self.oracle_name} optimization")
        print("="*60)
        
        all_molecules = []
        all_scores = []
        valid_molecules = 0
        total_generated = 0
        
        # Different prompts to test
        prompts = [
            "Generate a drug-like molecule with high JNK3 inhibition activity: \\box{",
            "Design a molecule that strongly inhibits JNK3 kinase: \\box{", 
            "Create a pharmaceutical compound for JNK3 targeting: \\box{",
            "Generate a potent JNK3 inhibitor molecule: \\box{",
            "Design a bioactive molecule with excellent JNK3 binding: \\box{"
        ]
        
        for batch in tqdm(range(num_batches), desc="Generating molecules"):
            # Rotate through prompts
            prompt = prompts[batch % len(prompts)]
            
            print(f"\n📝 Batch {batch + 1}/{num_batches}")
            print(f"Prompt: {prompt}")
            
            # Generate molecules
            molecules = self.generate_molecules(
                prompt=prompt,
                num_molecules=molecules_per_batch,
                temperature=0.8
            )
            
            total_generated += len(molecules)
            valid_molecules += len(molecules)
            
            print(f"✅ Generated {len(molecules)} valid molecules")
            
            if molecules:
                # Evaluate molecules
                scores = self.evaluate_molecules(molecules)
                
                # Track results
                all_molecules.extend(molecules)
                all_scores.extend(scores)
                
                # Print batch results
                best_score = max(scores) if scores else 0
                avg_score = np.mean(scores) if scores else 0
                
                print(f"📊 Batch scores - Best: {best_score:.4f}, Average: {avg_score:.4f}")
                
                # Show best molecule from this batch
                if scores:
                    best_idx = np.argmax(scores)
                    print(f"🏆 Best molecule: {molecules[best_idx]} (score: {scores[best_idx]:.4f})")
            else:
                print("❌ No valid molecules generated in this batch")
        
        # Final results analysis
        print(f"\n{'='*60}")
        print(f"🎉 GENERATION TEST COMPLETE!")
        print(f"{'='*60}")
        
        if all_scores:
            print(f"📊 OVERALL RESULTS:")
            print(f"  Total molecules generated: {len(all_molecules)}")
            print(f"  Valid molecule rate: {valid_molecules/total_generated*100:.1f}% ({valid_molecules}/{total_generated})")
            print(f"  Best score: {max(all_scores):.4f}")
            print(f"  Average score: {np.mean(all_scores):.4f}")
            print(f"  Score std: {np.std(all_scores):.4f}")
            print(f"  Molecules > 0.5: {sum(1 for s in all_scores if s > 0.5)}")
            print(f"  Molecules > 0.7: {sum(1 for s in all_scores if s > 0.7)}")
            print(f"  Molecules > 0.8: {sum(1 for s in all_scores if s > 0.8)}")
            
            # Find and display top molecules
            top_indices = np.argsort(all_scores)[-5:][::-1]
            print(f"\n🏆 TOP 5 MOLECULES:")
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. {all_molecules[idx]} (score: {all_scores[idx]:.4f})")
            
            # Save results
            results = {
                'oracle': self.oracle_name,
                'model_path': self.model_path,
                'total_molecules': len(all_molecules),
                'molecules': all_molecules,
                'scores': all_scores,
                'best_score': max(all_scores),
                'avg_score': np.mean(all_scores),
                'top_molecules': [(all_molecules[idx], all_scores[idx]) for idx in top_indices],
                'summary': {
                    'valid_rate': valid_molecules/total_generated,
                    'high_score_count': sum(1 for s in all_scores if s > 0.7),
                    'score_distribution': {
                        '>0.5': sum(1 for s in all_scores if s > 0.5),
                        '>0.6': sum(1 for s in all_scores if s > 0.6),
                        '>0.7': sum(1 for s in all_scores if s > 0.7),
                        '>0.8': sum(1 for s in all_scores if s > 0.8),
                        '>0.9': sum(1 for s in all_scores if s > 0.9),
                    }
                }
            }
            
            # Save to file
            output_file = f"dpo_generation_results_{self.oracle_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
            
        else:
            print("❌ No molecules were successfully generated and evaluated")
            
        print(f"🔬 Oracle calls used: {len(self.oracle)}")
        
        return all_molecules, all_scores

def main():
    parser = argparse.ArgumentParser(description='Test DPO-optimized model for molecule generation')
    parser.add_argument('--model_path', type=str, default='./dpo_optimized_model',
                        help='Path to DPO-optimized model')
    parser.add_argument('--oracle', type=str, default='jnk3',
                        help='Oracle to use for evaluation')
    parser.add_argument('--num_batches', type=int, default=10,
                        help='Number of generation batches')
    parser.add_argument('--molecules_per_batch', type=int, default=10,
                        help='Molecules per batch')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("Make sure you have converted the LoRA adapter to a full model first!")
        return 1
    
    try:
        # Create generator and run test
        generator = DPOMoleculeGenerator(args.model_path, args.oracle)
        molecules, scores = generator.run_generation_test(args.num_batches, args.molecules_per_batch)
        
        print(f"\n🎉 Test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 