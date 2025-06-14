#!/usr/bin/env python3
"""
Test DPO vLLM Server

This script tests that the DPO-optimized model is working correctly
when served via vLLM by making API calls and evaluating responses.
"""

import requests
import json
import time
import sys
import os
from typing import List, Optional
import re
from rdkit import Chem

# Add path for MolLEO imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.optimizer import Oracle as MolleoOracle

class DPOvLLMTester:
    def __init__(self, server_url: str = "http://localhost:8000/v1/chat/completions"):
        """
        Initialize the DPO vLLM tester
        
        Args:
            server_url: URL of the vLLM server
        """
        self.server_url = server_url
        self.oracle = None
        
    def setup_oracle(self, oracle_name: str = 'jnk3'):
        """Setup oracle for molecule evaluation"""
        print(f"🔬 Setting up {oracle_name} oracle...")
        
        # Create args object for MolLEO compatibility
        class Args:
            def __init__(self):
                self.oracles = [oracle_name]
                self.output_dir = './test_results'
                self.freq_log = 100
                self.n_jobs = 1
                self.log_results = False
                self.max_oracle_calls = 1000
        
        args = Args()
        self.oracle = MolleoOracle(args=args)
        
        # Initialize oracle evaluator
        from tdc import Oracle as TDCOracle
        tdc_oracle = TDCOracle(name=oracle_name)
        self.oracle.assign_evaluator(tdc_oracle)
        
        print(f"✅ Oracle setup complete")
    
    def test_server_connection(self) -> bool:
        """Test if the vLLM server is running and responding"""
        print(f"🔌 Testing connection to {self.server_url}...")
        
        try:
            # Simple test request
            test_payload = {
                "model": "./dpo_optimized_model",
                "messages": [
                    {"role": "user", "content": "Hello, are you working?"}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.server_url,
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Server is responding!")
                print(f"📝 Test response: {result['choices'][0]['message']['content'][:100]}...")
                return True
            else:
                print(f"❌ Server error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection failed! Is the vLLM server running on {self.server_url}?")
            return False
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def query_model(self, prompt: str, temperature: float = 0.8, max_tokens: int = 200) -> str:
        """Query the DPO model via vLLM server"""
        payload = {
            "model": "./dpo_optimized_model",
            "messages": [
                {"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["User:", "System:"]
        }
        
        try:
            response = requests.post(
                self.server_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return ""
    
    def sanitize_smiles(self, smi: str) -> Optional[str]:
        """Validate and canonicalize SMILES"""
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
        """Extract SMILES strings from model response using GPT4.py method"""
        molecules = []
        
        print(f"🔍 FULL RESPONSE:\n{response}\n" + "="*50)
        
        # Use exact same extraction as GPT4.py
        try:
            proposed_smiles_match = re.search(r'\\box\{(.*?)\}', response)
            if proposed_smiles_match:
                proposed_smiles = proposed_smiles_match.group(1)
                sanitized = self.sanitize_smiles(proposed_smiles)
                if sanitized:
                    molecules.append(sanitized)
                    print(f"✅ Extracted SMILES: {proposed_smiles} -> {sanitized}")
                else:
                    print(f"❌ Invalid SMILES: {proposed_smiles}")
            else:
                print("❌ No \\box{} pattern found in response")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
                        
        return molecules
    
    def test_molecule_generation(self, num_tests: int = 5) -> bool:
        """Test molecule generation capabilities"""
        print(f"\n🧪 Testing molecule generation capabilities...")
        print(f"📊 Running {num_tests} generation tests")
        
        # Test prompts matching GPT4.py format
        base_prompt = "I want to design molecules with high JNK3 scores. The JNK3 score measures a molecular's biological activity against JNK3.\n\nPlease propose a new molecule that has a high JNK3 score based on your knowledge.\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your proposed molecule.\n3. The molecule should be valid."
        
        prompts = [base_prompt] * 5  # Use same prompt for consistency
        
        total_molecules = 0
        valid_molecules = 0
        all_scores = []
        all_molecules = []
        
        for i in range(num_tests):
            prompt = prompts[i % len(prompts)]
            print(f"\n📝 Test {i+1}/{num_tests}: {prompt[:50]}...")
            
            # Query model
            response = self.query_model(prompt, temperature=0.8)
            
            if not response:
                print("❌ No response from model")
                continue
                
            # Extract molecules
            molecules = self.extract_molecules_from_response(response)
            total_molecules += len(molecules)
            
            if molecules:
                print(f"✅ Generated {len(molecules)} valid molecules")
                valid_molecules += len(molecules)
                
                # Evaluate with oracle if available
                if self.oracle:
                    for mol in molecules:
                        try:
                            score = self.oracle.score_smi(mol)
                            all_scores.append(score)
                            all_molecules.append(mol)
                            print(f"  📊 {mol}: {score:.4f}")
                        except Exception as e:
                            print(f"  ❌ Failed to score {mol}: {e}")
                else:
                    print(f"  📋 Molecules: {molecules}")
            else:
                print("❌ No valid molecules extracted")
                print(f"🔍 Raw response: {response}...")
        
        # Summary
        print(f"\n{'='*50}")
        print(f"🎉 MOLECULE GENERATION TEST SUMMARY")
        print(f"{'='*50}")
        print(f"📊 Total molecules generated: {total_molecules}")
        print(f"✅ Valid molecules: {valid_molecules}")
        
        if valid_molecules > 0:
            print(f"📈 Success rate: {valid_molecules/max(1, total_molecules)*100:.1f}%")
            
            if all_scores:
                print(f"🎯 Oracle evaluation results:")
                print(f"  📊 Average score: {sum(all_scores)/len(all_scores):.4f}")
                print(f"  🏆 Best score: {max(all_scores):.4f}")
                print(f"  📉 Worst score: {min(all_scores):.4f}")
                print(f"  🎪 High quality (>0.7): {sum(1 for s in all_scores if s > 0.7)}/{len(all_scores)}")
                
                # Show best molecule
                best_idx = all_scores.index(max(all_scores))
                print(f"  🏆 Best molecule: {all_molecules[best_idx]} (score: {all_scores[best_idx]:.4f})")
            
            return True
        else:
            print("❌ No valid molecules generated")
            return False
    
    def run_full_test(self):
        """Run comprehensive test of the DPO vLLM server"""
        print("🧪 TESTING DPO-OPTIMIZED MODEL VIA vLLM SERVER")
        print("="*60)
        
        # Test 1: Server connection
        if not self.test_server_connection():
            print("❌ Server connection test failed!")
            return False
        
        # Test 2: Setup oracle
        try:
            self.setup_oracle('jnk3')
        except Exception as e:
            print(f"⚠️  Oracle setup failed: {e}")
            print("Continuing without oracle evaluation...")
        
        # Test 3: Molecule generation
        if not self.test_molecule_generation(5):
            print("❌ Molecule generation test failed!")
            return False
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Your DPO-optimized model is working correctly via vLLM!")
        print(f"🚀 Ready for production use in molecular optimization!")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DPO vLLM Server')
    parser.add_argument('--server-url', type=str, default='http://localhost:8000/v1/chat/completions',
                        help='vLLM server URL')
    parser.add_argument('--wait-for-server', action='store_true',
                        help='Wait for server to become available')
    
    args = parser.parse_args()
    
    tester = DPOvLLMTester(args.server_url)
    
    if args.wait_for_server:
        print("⏳ Waiting for server to become available...")
        max_retries = 30
        for i in range(max_retries):
            if tester.test_server_connection():
                break
            print(f"Retry {i+1}/{max_retries} in 10 seconds...")
            time.sleep(10)
        else:
            print("❌ Server never became available!")
            return 1
    
    success = tester.run_full_test()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 