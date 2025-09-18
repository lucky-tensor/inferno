#!/usr/bin/env python3
"""
Convert compressed-tensors w8a8 quantized models to standard SafeTensors format for candle-transformers.

This script loads a compressed-tensors quantized model using transformers library
(which handles the compressed format), then exports the weights in standard SafeTensors
format that can be loaded by candle-transformers.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json

def convert_compressed_tensors_model(model_path: str, output_path: str, device: str = "cpu"):
    """
    Convert a compressed-tensors w8a8 model to standard SafeTensors format.

    Args:
        model_path: Path to the compressed-tensors model directory
        output_path: Path where to save the converted model
        device: Device to load the model on ("cpu" or "cuda")
    """
    print(f"🔄 Converting compressed-tensors model from {model_path}")
    print(f"📤 Output will be saved to {output_path}")

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if input model exists and has quantization config
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Verify this is a compressed-tensors quantized model
    quant_config = config.get('quantization_config')
    if not quant_config:
        raise ValueError("Model does not have quantization_config - not a compressed-tensors model")

    if quant_config.get('quant_method') != 'compressed-tensors':
        raise ValueError(f"Expected compressed-tensors, got: {quant_config.get('quant_method')}")

    print(f"✅ Verified compressed-tensors model with method: {quant_config['quant_method']}")
    print(f"📊 Compression ratio: {quant_config.get('global_compression_ratio', 'unknown')}")

    # Load the quantized model using transformers (this handles compressed-tensors)
    print("🔄 Loading quantized model with transformers...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,  # Load in float32 for compatibility
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        print(f"✅ Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Note: This requires transformers library with compressed-tensors support")
        raise

    # Extract the model state dict (this gives us dequantized fp32 weights)
    print("🔄 Extracting model weights...")
    state_dict = model.state_dict()

    # Filter out any non-tensor entries and convert to CPU
    filtered_state_dict = {}
    for name, param in state_dict.items():
        if torch.is_tensor(param):
            # Convert to CPU and float32 for SafeTensors compatibility
            filtered_state_dict[name] = param.cpu().float()
            print(f"  ✅ {name}: {param.shape} ({param.dtype})")

    # Save as standard SafeTensors
    safetensors_path = output_path / "model.safetensors"
    print(f"🔄 Saving to SafeTensors format: {safetensors_path}")
    save_file(filtered_state_dict, str(safetensors_path))

    # Copy config.json but remove quantization_config
    output_config = config.copy()
    if 'quantization_config' in output_config:
        del output_config['quantization_config']

    output_config_path = output_path / "config.json"
    with open(output_config_path, 'w') as f:
        json.dump(output_config, f, indent=2)
    print(f"✅ Saved updated config.json: {output_config_path}")

    # Copy tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    for filename in tokenizer_files:
        src = model_path / filename
        dst = output_path / filename
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"✅ Copied {filename}")

    print("🎉 Conversion completed successfully!")
    print(f"📁 Converted model available at: {output_path}")
    print("💡 This model can now be loaded with candle-transformers using standard Llama model")

def main():
    parser = argparse.ArgumentParser(description="Convert compressed-tensors w8a8 model to standard SafeTensors")
    parser.add_argument("model_path", help="Path to compressed-tensors model directory")
    parser.add_argument("output_path", help="Path for converted model output")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use for conversion")

    args = parser.parse_args()

    try:
        convert_compressed_tensors_model(args.model_path, args.output_path, args.device)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()