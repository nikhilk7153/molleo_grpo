#!/usr/bin/env python3
"""
Convert Iteratively Trained DPO Model to HuggingFace Format

This script converts the iteratively trained LoRA model back to a full HuggingFace model
that can be served with vLLM.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

def convert_iterative_model(input_dir, output_dir, base_model_path="./dpo_optimized_model"):
    """
    Convert iteratively trained LoRA model to full HuggingFace model
    
    Args:
        input_dir: Directory containing the LoRA checkpoint (e.g., ./trl_checkpoints_iterative_filtered/final)
        output_dir: Directory to save the merged model (e.g., ./dpo_optimized_model_v2)
        base_model_path: Path to the base DPO-optimized model
    """
    
    print("🔄 Converting Iteratively Trained DPO Model")
    print("=" * 50)
    print(f"Input LoRA checkpoint: {input_dir}")
    print(f"Base model: {base_model_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📥 Loading base model and tokenizer...")
    
    # Load the base model (DPO-optimized model)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    print("🔗 Loading and merging LoRA weights...")
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, input_dir)
    
    # Merge LoRA weights into the base model
    merged_model = model.merge_and_unload()
    
    print("💾 Saving merged model...")
    
    # Save the merged model
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Copy any additional files from base model
    for file in ["config.json", "generation_config.json", "tokenizer_config.json"]:
        src_path = os.path.join(base_model_path, file)
        dst_path = os.path.join(output_dir, file)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            print(f"📋 Copied {file}")
    
    print("\n✅ Model conversion completed!")
    print(f"📁 Merged model saved to: {output_dir}")
    
    # Test the model
    print("\n🧪 Testing model generation...")
    try:
        # Test generation
        test_prompt = "Generate a molecule with high JNK3 inhibition activity:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = merged_model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test generation successful!")
        print(f"Input: {test_prompt}")
        print(f"Output: {generated_text[len(test_prompt):].strip()}")
        
    except Exception as e:
        print(f"⚠️ Test generation failed: {e}")
        print("Model conversion completed but generation test failed.")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Convert iteratively trained DPO model to HuggingFace format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the LoRA checkpoint (e.g., ./trl_checkpoints_iterative_filtered/final)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the merged model (e.g., ./dpo_optimized_model_v2)')
    parser.add_argument('--base_model_path', type=str, default='./dpo_optimized_model',
                        help='Path to the base DPO-optimized model (default: ./dpo_optimized_model)')
    
    args = parser.parse_args()
    
    try:
        output_path = convert_iterative_model(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            base_model_path=args.base_model_path
        )
        
        print("\n🎉 SUCCESS! Iterative model conversion completed!")
        print("\n🚀 Next steps:")
        print(f"1. Update your server script to use: {output_path}")
        print("2. Restart the vLLM server with the new model")
        print("3. Run the optimizer again to see improved performance")
        print("\nExample server command:")
        print(f"python -m vllm.entrypoints.openai.api_server --model {output_path} --port 8000 --tensor-parallel-size 4")
        
    except Exception as e:
        print(f"\n❌ ERROR: Model conversion failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main() 