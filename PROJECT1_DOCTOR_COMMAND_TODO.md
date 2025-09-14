# Project 1: CLI Doctor Subcommand

## Overview
Create a CLI subcommand called "doctor" which will check the system for dependencies necessary for running inference engines.

## Requirements
- Check if NVIDIA or AMD GPUs can be found and their driver/accelerator versions
- Verify compatibility with our implementation  
- Check CPU inference capability if no GPU accelerator found
- Detect models downloaded in safetensors format
- Check if models have been optimized for target architecture
- Provide compatibility score and table showing model/accelerator compatibility

## Task Checklist

### ✅ Completed Tasks
- [x] **Explore existing codebase structure** - Analyzed CLI framework (Clap-based) and patterns
- [x] **Design doctor subcommand architecture** - Planned integration with existing health checking system
- [x] **Implement GPU detection (NVIDIA and AMD)** - Successfully detects NVIDIA GPUs via nvidia-smi, AMD GPUs via rocm-smi, and fallback lspci detection
- [x] **Implement driver and accelerator version checking** - Checks NVIDIA driver versions, CUDA versions, ROCm versions, and validates compatibility
- [x] **Implement CPU inference capability checking** - Detects CPU specs, core count, and instruction set support (AVX, AVX2, AVX512)
- [x] **Implement model detection in safetensors format** - Scans model directories and detects SafeTensors, PyTorch, ONNX, and GGUF formats
- [x] **Implement model optimization status checking** - Checks for quantized models and optimization markers
- [x] **Create compatibility scoring system** - Implemented comprehensive scoring based on hardware and software capabilities
- [x] **Implement results table display** - Created formatted output with compatibility matrix and system readiness status
- [x] **Add comprehensive error handling and user-friendly messages** - Proper error handling and informative output messages

### 🔄 In Progress Tasks
- [⏳] **Test the doctor command across different system configurations** - Currently testing on development system

### 📋 Pending Tasks

#### Final Testing and Validation
- [ ] **Test on systems with different GPU configurations**
  - Test on systems with multiple NVIDIA GPUs
  - Test on systems with AMD GPUs (ROCm installed vs not installed)
  - Test on CPU-only systems
  - Test with various CUDA versions and driver combinations

- [ ] **Test with different model configurations**
  - Test with no models present
  - Test with large model collections
  - Test with various model formats and sizes
  - Test edge cases (corrupted files, permissions issues)

## Implementation Notes

### Architecture
- **Location**: `/home/jeef/inferno/crates/cli/src/cli_options.rs`
- **Pattern**: Follow existing Clap-based subcommand structure
- **Integration**: Leverage existing health checking code from `/home/jeef/inferno/crates/inference/src/health.rs`

### Key Components to Build Upon
- **VLLMHealthChecker**: Existing health checking infrastructure
- **System detection**: Existing GPU device management code
- **Model management**: Existing model downloading and verification system
- **Shared options**: Use LoggingOptions, MetricsOptions from shared crate

### Expected Output Format
```
System Diagnostics Report
========================

Hardware Detection:
✅ NVIDIA RTX 4080 (Driver: 535.98, CUDA: 12.2)
⚠️  AMD GPU detected but ROCm not installed
✅ CPU: Intel i7-12700K (16 cores, AVX2 support)

Model Compatibility:
✅ TinyLlama-1.1B (safetensors) - Optimized for CPU/CUDA
⚠️  Llama-7B (safetensors) - Not optimized, may be slow
❌ Custom-model (GGUF) - Format not supported

Score: 7/10 checks passed

Compatibility Matrix:
Model                | CPU    | CUDA   | ROCm
---------------------|--------|--------|--------
TinyLlama-1.1B      | ✅ Fast| ✅ Fast| ❌ N/A
Llama-7B            | ⚠️ Slow| ✅ Fast| ❌ N/A
Custom-model        | ❌ N/A | ❌ N/A | ❌ N/A

Recommendations:
- Install ROCm drivers for AMD GPU acceleration
- Consider quantizing Llama-7B for better CPU performance
- Convert Custom-model to safetensors format
```

## Dependencies
- System tools: `nvidia-smi`, `rocm-smi`, `lscpu`
- Rust crates: `sysinfo`, `clap`, existing inferno crates
- Hardware detection libraries as needed