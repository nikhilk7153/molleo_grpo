#!/usr/bin/env python
"""
vllm_server.py

Helper script to start a VLLM server with Qwen-2.5-7b-instruct model.
This provides an API endpoint that can be used with the molecule generation scripts.
"""

import argparse
import os
import subprocess
import sys

def check_vllm_installed():
    """Check if VLLM is installed and install if needed"""
    try:
        import vllm
        print("VLLM is already installed.")
        return True
    except ImportError:
        print("VLLM is not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm"])
            print("VLLM installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install VLLM. Please install manually with 'pip install vllm'.")
            return False

def setup_huggingface_token(token):
    """Set up HuggingFace token for authentication"""
    if token:
        print("Setting up HuggingFace token...")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        
        # Also configure the token using huggingface_hub CLI
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "huggingface_hub"
            ])
            import huggingface_hub
            huggingface_hub.login(token=token)
            print("HuggingFace authentication configured successfully.")
            return True
        except Exception as e:
            print(f"Warning: Failed to configure huggingface_hub: {e}")
            print("Authentication might still work through environment variable.")
            return True
    else:
        print("No HuggingFace token provided. Continuing without authentication.")
        return True

def list_available_models():
    """List some available models that can be used"""
    print("\nSome available models that might work with vLLM:")
    models = [
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2", 
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen1.5-7B-Chat",
        "microsoft/Phi-2",
        "google/gemma-7b-it"
    ]
    for model in models:
        print(f"  - {model}")
    print("\nFor Qwen models, you need to accept terms of use on HuggingFace and provide a token.")
    print("Visit: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct")

def start_vllm_server(model_name, host, port, gpu_memory_utilization, max_model_len, tensor_parallel_size, trust_remote_code):
    """Start the VLLM server with the specified model and configuration"""
    
    # Check if the port is already in use
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
    except socket.error:
        print(f"Port {port} is already in use. Please choose a different port or stop the existing service.")
        return False
    
    print(f"Starting VLLM server with {model_name} on {host}:{port}...")
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len)
    ]
    
    if tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    try:
        process = subprocess.Popen(cmd)
        print(f"VLLM server started. Process ID: {process.pid}")
        print(f"API endpoint: http://{host}:{port}/v1/chat/completions")
        print("Press Ctrl+C to stop the server")
        
        # Keep the script running until user interrupts
        process.wait()
        return True
    except KeyboardInterrupt:
        print("Stopping VLLM server...")
        process.terminate()
        process.wait()
        print("VLLM server stopped.")
        return True
    except Exception as e:
        print(f"Error starting VLLM server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Start a VLLM server with Qwen model')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', 
                        help='Model name or path (default: Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host to bind server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Port to bind server to (default: 8000)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8, 
                        help='GPU memory utilization (default: 0.8)')
    parser.add_argument('--max-model-len', type=int, default=8192, 
                        help='Maximum model length for context (default: 8192)')
    parser.add_argument('--tensor-parallel-size', type=int, default=4, 
                        help='Tensor parallel size for multi-GPU inference (default: 4)')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace access token for authentication (required for Qwen models)')
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='Trust remote code when downloading models (required for some models)')
    parser.add_argument('--list-models', action='store_true',
                        help='List some available models and exit')
    args = parser.parse_args()
    
    # Check if just listing models
    if args.list_models:
        list_available_models()
        return
    
    # Check if VLLM is installed
    if not check_vllm_installed():
        return
    
    # Set up HuggingFace token if provided
    if not setup_huggingface_token(args.hf_token):
        return
    
    # Start VLLM server
    start_vllm_server(
        args.model,
        args.host,
        args.port,
        args.gpu_memory_utilization,
        args.max_model_len,
        args.tensor_parallel_size,
        args.trust_remote_code
    )

if __name__ == "__main__":
    main() 