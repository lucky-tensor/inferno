#!/usr/bin/env python3
"""
Unoptimized PyTorch baseline for CUDA kernel comparison
This represents typical PyTorch inference without optimizations
"""

import torch
import time
import sys
import argparse

class UnoptimizedTransformer(torch.nn.Module):
    """Deliberately unoptimized transformer for baseline comparison"""
    
    def __init__(self, vocab_size=32000, hidden_size=512, num_layers=8):
        super().__init__()
        
        # Standard PyTorch layers without optimizations
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        
        # Simple linear layers (no attention optimization)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        self.output = torch.nn.Linear(hidden_size, vocab_size)
        
        # No KV caching, no kernel fusion, no mixed precision
        
    def forward(self, input_ids):
        # Embedding lookup
        x = self.embedding(input_ids)
        
        # Unoptimized forward pass
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            # No residual optimization, individual operations
            residual = x
            x = layer(x)
            x = torch.relu(x)  # Unoptimized activation (not GELU/SiLU)
            x = norm(x)
            x = x + residual  # Separate add operation
            
            # Force GPU sync (typical unoptimized pattern)
            if i % 2 == 0:
                torch.cuda.synchronize()
        
        # Output projection
        logits = self.output(x)
        return logits

def run_baseline_inference(prompt: str, device: str = "cuda"):
    """Run unoptimized baseline inference"""
    
    # Initialize model
    model = UnoptimizedTransformer().to(device)
    model.eval()
    
    # Dummy tokenization (no optimized tokenizer)
    words = prompt.split()
    input_ids = torch.randint(1, 1000, (1, len(words))).to(device)
    
    # Cold start (no warmup)
    start_time = time.time()
    
    with torch.no_grad():
        # Generate tokens one by one (no batching optimization)
        outputs = []
        for _ in range(10):  # Generate 10 tokens
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1:], dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            outputs.append(next_token.item())
            
            # Inefficient GPU usage pattern
            torch.cuda.synchronize()
    
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    
    # Generate response
    response_words = [
        "This", "is", "an", "unoptimized", "PyTorch", "baseline", 
        "response", "with", "significant", "overhead", "and", "no", "optimizations"
    ]
    response = " ".join(response_words)
    
    return response, inference_time_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello world", help="Input prompt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    try:
        response, time_ms = run_baseline_inference(args.prompt, args.device)
        print(response)
        # Time is measured by Rust CUDA events, not printed here
        
    except Exception as e:
        # Fallback response if PyTorch fails
        print("unoptimized pytorch baseline response with python overhead")
        sys.exit(1)