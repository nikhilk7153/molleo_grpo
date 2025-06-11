#!/usr/bin/env python
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from tdc import Oracle
from rdkit import Chem
import re

def evaluate_molecules_with_jnk3(model, tokenizer, oracle, num_samples=50):
    """Generate molecules and evaluate them with JNK3 oracle."""
    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model_for_generation = model.module
    else:
        model_for_generation = model
    
    model_for_generation.eval()
    
    # Sample prompts from molecular optimization tasks
    prompts = [
        "Generate a molecule with high JNK3 activity:",
        "Design a drug-like molecule that inhibits JNK3:",
        "Create a molecule with improved JNK3 binding:",
        "Propose a molecule with better JNK3 activity:",
        "Generate a molecule that targets JNK3:"
    ]
    
    generated_molecules = []
    valid_molecules = []
    jnk3_scores = []
    
    # Get device from model - ensure it's GPU for DDP training
    try:
        device = next(model_for_generation.parameters()).device
        # Ensure we're using GPU, not CPU
        if device.type == 'cpu':
            # In DDP training, use local rank GPU
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
    except:
        # Default to GPU for DDP training
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate molecule
        with torch.no_grad():
            try:
                # Check for NaN in model parameters before generation
                has_nan = any(torch.isnan(p).any() for p in model_for_generation.parameters())
                if has_nan:
                    print(f"⚠️  Model has NaN parameters, skipping generation for sample {i}")
                    generated_molecules.append("NaN Error")
                    jnk3_scores.append(0.0)
                    continue
                
                outputs = model_for_generation.generate(
                    **inputs,
                    max_new_tokens=100,  # Reduced from 2048 to prevent memory issues
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,  # Add nucleus sampling for stability
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode generated text
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_molecules.append(generated_text)
                
                # Extract SMILES from generated text
                smiles = extract_smiles_from_text(generated_text)
                
                if smiles and is_valid_smiles(smiles):
                    valid_molecules.append(smiles)
                    try:
                        score = oracle([smiles])[0]
                        jnk3_scores.append(score)
                    except:
                        jnk3_scores.append(0.0)
                else:
                    jnk3_scores.append(0.0)
            except Exception as e:
                print(f"Generation error: {e}")
                generated_molecules.append("Error")
                jnk3_scores.append(0.0)
    
    model_for_generation.train()
    
    # Calculate metrics
    validity_rate = len(valid_molecules) / num_samples
    mean_jnk3_score = np.mean(jnk3_scores) if jnk3_scores else 0.0
    max_jnk3_score = np.max(jnk3_scores) if jnk3_scores else 0.0
    
    return {
        "validity_rate": validity_rate,
        "mean_jnk3_score": mean_jnk3_score,
        "max_jnk3_score": max_jnk3_score,
        "num_valid_molecules": len(valid_molecules),
        "generated_molecules": generated_molecules[:10],  # Store first 10 for inspection
        "valid_molecules": valid_molecules[:10]
    }

def extract_smiles_from_text(text):
    """Extract SMILES string from generated text."""
    # Look for common SMILES patterns
    smiles_patterns = [
        r'[A-Z][a-z]?(?:\[[^\]]+\])?(?:[a-z]?[0-9]*)?(?:\([^)]+\))?[A-Za-z0-9\[\]()=@+\-#$%&*\/\\:.]*',
        r'C[A-Za-z0-9\[\]()=@+\-#$%]*',
        r'[CNOSH][A-Za-z0-9\[\]()=@+\-#$%]*'
    ]
    
    for pattern in smiles_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Use minimum length of 2 to filter out single characters but allow simple molecules
            if len(match) >= 2 and is_valid_smiles(match):
                return match
    
    return None

def is_valid_smiles(smiles):
    """Check if SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def main():
    # Set environment variable to avoid tokenizers warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Starting DPO training script optimized for 4 H100 GPUs with JNK3 evaluation...")
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Initialize JNK3 oracle for evaluation
    print("Initializing JNK3 oracle for evaluation...")
    jnk3_oracle = Oracle(name='jnk3')
    
    # Load data
    with open('preferences.json', 'r') as f:
        data = json.load(f)
    
    preferences = data['data']
    print(f"Loaded {len(preferences)} preference pairs")
    
    # Create dataset and split train/eval
    dataset = Dataset.from_list(preferences)
    print(f"Created dataset with {len(dataset)} examples")
    print("Sample data keys:", dataset.column_names)
    
    # Split dataset for evaluation (80/20 split)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Load tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for DDP training (no device_map to avoid conflicts)
    print(f"Loading model from {model_name} for DDP training...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # Don't use device_map="auto" for DDP - let trainer handle device placement
    )
    
    # Load reference model for DDP training
    print("Loading reference model for DDP training...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # Don't use device_map="auto" for DDP - let trainer handle device placement
    )
    
    # Explicitly ensure models start on GPU for DDP training
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"Moving models to GPU {local_rank} for DDP training...")
        model = model.to(f'cuda:{local_rank}')
        ref_model = ref_model.to(f'cuda:{local_rank}')
        print(f"✓ Models moved to cuda:{local_rank}")
    
    # Optimized training config for 4 H100s with stability fixes
    training_args = DPOConfig(
        output_dir="./checkpoints_4gpu",
        per_device_train_batch_size=4,   # Reduced further for stability
        per_device_eval_batch_size=2,    # Smaller batch size for evaluation
        gradient_accumulation_steps=6,   # Maintain similar effective batch size
        learning_rate=1e-7,              # Much lower learning rate to prevent NaN
        num_train_epochs=3,              # Full training epochs
        save_steps=100,
        logging_steps=10,
        warmup_steps=50,                 # Reduced warmup steps
        logging_dir="./logs_4gpu",
        report_to="tensorboard",
        save_strategy="steps",
        logging_strategy="steps",
        remove_unused_columns=False,
        dataloader_num_workers=2,        # Reduced for DDP stability
        bf16=True,                       # Use bfloat16 for H100 stability
        gradient_checkpointing=True,     # Save memory with checkpointing
        ddp_find_unused_parameters=False, # Optimize DDP
        save_total_limit=3,              # Limit saved checkpoints
        max_grad_norm=1.0,               # Add gradient clipping to prevent NaN
        # DDP specific settings
        ddp_backend="nccl",              # Use NCCL for multi-GPU communication
        dataloader_pin_memory=True,      # Pin memory for faster GPU transfer
    )
    
    # Custom callback for JNK3 evaluation
    class JNK3EvaluationCallback(TrainerCallback):
        def __init__(self, oracle, tokenizer, eval_every_steps=100):
            self.oracle = oracle
            self.tokenizer = tokenizer
            self.evaluation_results = []
            self.eval_every_steps = eval_every_steps
            self.last_eval_step = 0
        
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            # Only run JNK3 evaluation on rank 0 to avoid multi-GPU issues
            if (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
                return
            
            # Run JNK3 evaluation every eval_every_steps
            if state.global_step - self.last_eval_step >= self.eval_every_steps and model is not None:
                self.last_eval_step = state.global_step
                
                print(f"\n=== Running JNK3 Evaluation at Step {state.global_step} ===")
                eval_results = evaluate_molecules_with_jnk3(model, self.tokenizer, self.oracle, num_samples=20)
                
                print(f"Validity Rate: {eval_results['validity_rate']:.3f}")
                print(f"Mean JNK3 Score: {eval_results['mean_jnk3_score']:.4f}")
                print(f"Max JNK3 Score: {eval_results['max_jnk3_score']:.4f}")
                print(f"Valid Molecules: {eval_results['num_valid_molecules']}/20")
                
                if eval_results['valid_molecules']:
                    print("Sample valid molecules:")
                    for i, mol in enumerate(eval_results['valid_molecules'][:3]):
                        print(f"  {i+1}. {mol}")
                
                self.evaluation_results.append({
                    'step': state.global_step,
                    **eval_results
                })
                
                # Log to tensorboard
                if logs is not None:
                    logs.update({
                        'eval_jnk3_validity_rate': eval_results['validity_rate'],
                        'eval_jnk3_mean_score': eval_results['mean_jnk3_score'],
                        'eval_jnk3_max_score': eval_results['max_jnk3_score'],
                    })
                
                print("=== JNK3 Evaluation Complete ===\n")
    
    print("Creating DPO trainer with multi-GPU optimization and JNK3 evaluation...")
    # Create trainer with both models on GPU
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Add JNK3 evaluation callback
    jnk3_callback = JNK3EvaluationCallback(jnk3_oracle, tokenizer)
    trainer.add_callback(jnk3_callback)
    
    print("Starting DPO training on 4 H100 GPUs with JNK3 evaluation...")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"JNK3 evaluation will run every 100 steps")
    print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
    # Show device information for models
    if hasattr(model, 'device'):
        print(f"Main model device: {model.device}")
    else:
        try:
            print(f"Main model device: {next(model.parameters()).device}")
        except:
            print("Main model device: Unknown")
    
    if hasattr(ref_model, 'device'):
        print(f"Reference model device: {ref_model.device}")
    else:
        try:
            print(f"Reference model device: {next(ref_model.parameters()).device}")
        except:
            print("Reference model device: Unknown")
    
    # Train
    trainer.train()
    trainer.save_model()
    
    # Save JNK3 evaluation results
    if jnk3_callback.evaluation_results:
        eval_results_file = os.path.join(training_args.output_dir, "jnk3_evaluation_results.json")
        with open(eval_results_file, 'w') as f:
            json.dump(jnk3_callback.evaluation_results, f, indent=2)
        print(f"JNK3 evaluation results saved to: {eval_results_file}")
    
    print("DPO training completed!")
    print(f"Model saved to: {training_args.output_dir}")
    
    # Final evaluation - only run on rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        try:
            print("\n=== Final JNK3 Evaluation ===")
            final_eval = evaluate_molecules_with_jnk3(model, tokenizer, jnk3_oracle, num_samples=50)
            print(f"Final Validity Rate: {final_eval['validity_rate']:.3f}")
            print(f"Final Mean JNK3 Score: {final_eval['mean_jnk3_score']:.4f}")
            print(f"Final Max JNK3 Score: {final_eval['max_jnk3_score']:.4f}")
            
            if final_eval['valid_molecules']:
                print("Best generated molecules:")
                for i, mol in enumerate(final_eval['valid_molecules'][:5]):
                    print(f"  {i+1}. {mol}")
            
            # Save final evaluation
            final_eval_file = os.path.join(training_args.output_dir, "final_jnk3_evaluation.json")
            with open(final_eval_file, 'w') as f:
                json.dump(final_eval, f, indent=2)
            print(f"Final evaluation saved to: {final_eval_file}")
        except Exception as e:
            print(f"⚠️  Final evaluation failed: {e}")
            print("This might be due to NaN values in the model from training instability.")
            print("The model was saved successfully and training completed.")

if __name__ == "__main__":
    main() 