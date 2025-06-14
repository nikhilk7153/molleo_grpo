#!/usr/bin/env python3
"""
Debug script to test vLLM server connection
"""

import requests
import json

def test_server():
    server_url = "http://localhost:8000/v1/chat/completions"
    
    print(f"Testing server at: {server_url}")
    
    # Test payload
    payload = {
        "model": "./dpo_optimized_model",
        "messages": [
            {"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."},
            {"role": "user", "content": "Generate a drug-like molecule with high JNK3 inhibition activity"}
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "stop": ["User:", "System:"]
    }
    
    print("Sending request...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            server_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Error text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    test_server() 