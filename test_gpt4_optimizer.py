#!/usr/bin/env python
"""
Test script for GPT-4 LLM Black-box Optimizer
"""

import os
import sys
sys.path.append('single_objective')

from llm_blackbox_optimizer import LLMBlackBoxOptimizer
import argparse

def test_optimizer():
    """Test the GPT-4 optimizer with minimal parameters"""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("   Please set it: export OPENAI_API_KEY='your-api-key'")
        return False
    
    # Create minimal args for testing
    args = argparse.Namespace()
    args.oracle = 'jnk3'
    args.seed = 42
    args.n_init = 100  # Small for testing
    args.m_per_iter = 5  # Small for testing
    args.max_iterations = 2  # Just 2 iterations for testing
    args.pool_sample_size = 50
    args.max_oracle_calls = 1000
    args.patience = 5
    args.output_dir = './test_results_gpt4'
    args.uniform_sampling = False
    
    try:
        print("🧪 Testing GPT-4 LLM Black-box Optimizer...")
        optimizer = LLMBlackBoxOptimizer(args)
        print("✅ Optimizer initialized successfully")
        
        # Test connection
        print("🔗 Testing OpenAI API connection...")
        if optimizer._test_openai_connection():
            print("✅ OpenAI API connection successful")
        else:
            print("❌ OpenAI API connection failed")
            return False
        
        # Test ZINC dataset loading
        print("📊 Testing ZINC dataset loading...")
        print(f"✅ Loaded {len(optimizer.all_smiles)} ZINC molecules")
        
        print("🎯 All tests passed! Ready to run optimization.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_optimizer()
    if success:
        print("\n🚀 You can now run the optimizer with:")
        print("cd single_objective && python llm_blackbox_optimizer.py --oracle jnk3 --max_iterations 5 --m_per_iter 10")
    else:
        print("\n❌ Please fix the issues above before running the optimizer") 