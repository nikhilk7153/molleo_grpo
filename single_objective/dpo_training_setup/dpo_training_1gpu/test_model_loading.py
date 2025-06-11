#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_loading():
    print("Testing model loading for DDP compatibility...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    try:
        # Test tokenizer loading
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")
        
        # Test model loading (same as in DPO script)
        print(f"Loading model from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✓ Main model loaded successfully")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Test reference model loading
        print(f"Loading reference model from {model_name}...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✓ Reference model loaded successfully")
        print(f"Reference model device: {next(ref_model.parameters()).device}")
        print(f"Reference model dtype: {next(ref_model.parameters()).dtype}")
        
        # Test moving to GPU
        if torch.cuda.is_available():
            print("Moving models to GPU for testing...")
            model = model.to('cuda:0')
            print(f"✓ Main model moved to: {next(model.parameters()).device}")
            
        print("✓ All model loading tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test completed successfully!")
        print("The DPO training script should now work properly.")
    else:
        print("\n💥 Model loading test failed.")
        print("Please check the error messages above.") 