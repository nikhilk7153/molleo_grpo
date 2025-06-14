#!/usr/bin/env python3
"""
Serve Fine-tuned DPO Model

This script serves the DPO fine-tuned model using vLLM with LoRA adapter support.
"""

import argparse
import os
import subprocess
import time

def check_vllm_installation():
    """Check if vLLM is properly installed"""
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
        return True
    except ImportError:
        print("❌ vLLM not found. Please install vLLM first.")
        return False

def check_port_available(port):
    """Check if port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def serve_finetuned_model(adapter_path, base_model="Qwen/Qwen2.5-7B-Instruct", port=8001):
    """Serve the fine-tuned model with vLLM"""
    
    print("🚀 Starting Fine-tuned DPO Model Server")
    print("="*50)
    print(f"Base Model: {base_model}")
    print(f"LoRA Adapter: {adapter_path}")
    print(f"Port: {port}")
    print("="*50)
    
    # Check if adapter path exists
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter path not found: {adapter_path}")
        return False
    
    # Check if port is available
    if not check_port_available(port):
        print(f"❌ Port {port} is already in use. Please choose a different port or stop the existing service.")
        return False
    
    # Construct vLLM command with LoRA support
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--enable-lora",
        "--lora-modules", f"dpo_finetuned={adapter_path}",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--tensor-parallel-size", "4",  # Use all 4 GPUs
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "4096",
        "--disable-log-requests",
        "--trust-remote-code",
    ]
    
    print("🔧 Command to execute:")
    print(" ".join(cmd))
    print("\n⏳ Starting server (this may take a few minutes)...")
    
    try:
        # Start the server
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(30)  # Give it time to load
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ Server started successfully on port {port}")
            print(f"🌐 API endpoint: http://localhost:{port}/v1/chat/completions")
            print(f"📋 To use the fine-tuned model, specify model name: 'dpo_finetuned'")
            print("\n🔄 Server is running. Press Ctrl+C to stop.")
            
            # Keep the process running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                process.wait()
                print("✅ Server stopped.")
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Serve Fine-tuned DPO Model')
    parser.add_argument('--adapter_path', type=str, 
                        default='./trl_checkpoints_optimized/final',
                        help='Path to the LoRA adapter')
    parser.add_argument('--base_model', type=str, 
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Base model name')
    parser.add_argument('--port', type=int, default=8001,
                        help='Port to serve the model on')
    
    args = parser.parse_args()
    
    # Check vLLM installation
    if not check_vllm_installation():
        return
    
    # Convert to absolute path
    adapter_path = os.path.abspath(args.adapter_path)
    
    # Serve the model
    serve_finetuned_model(adapter_path, args.base_model, args.port)

if __name__ == "__main__":
    main() 