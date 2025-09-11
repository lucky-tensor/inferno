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

### 🔄 In Progress Tasks
- [ ] Currently no tasks in progress

### 📋 Pending Tasks

#### Core Implementation
- [ ] **Implement GPU detection (NVIDIA and AMD)**
  - Detect NVIDIA GPUs using nvidia-ml-py or system calls
  - Detect AMD GPUs using ROCm tools or system calls  
  - Get GPU device information and capabilities

- [ ] **Implement driver and accelerator version checking**
  - Check NVIDIA driver version (nvidia-smi)
  - Check CUDA version and compatibility
  - Check AMD driver version (rocm-smi)
  - Verify versions are compatible with Burn framework

- [ ] **Implement CPU inference capability checking**  
  - Check CPU architecture and instruction sets (AVX, AVX2, etc.)
  - Verify CPU core count and memory availability
  - Test Burn CPU backend functionality

- [ ] **Implement model detection in safetensors format**
  - Scan model directories for .safetensors files
  - Parse model metadata and configuration
  - Verify model compatibility with inference engines

- [ ] **Implement model optimization status checking**
  - Check for optimized model variants (quantized, etc.)
  - Verify architecture-specific optimizations
  - Compare with available hardware capabilities

#### UI and Reporting
- [ ] **Create compatibility scoring system**
  - Define scoring criteria for hardware/software compatibility
  - Implement scoring algorithm
  - Generate overall system readiness score

- [ ] **Implement results table display**
  - Create formatted table showing model/accelerator compatibility matrix
  - Display compatibility scores and recommendations
  - Show warnings for potential issues

- [ ] **Add comprehensive error handling and user-friendly messages**
  - Handle cases where hardware detection fails
  - Provide clear error messages and troubleshooting tips
  - Add verbose mode for detailed diagnostics

#### Testing and Validation
- [ ] **Test the doctor command across different system configurations**
  - Test on systems with NVIDIA GPUs
  - Test on systems with AMD GPUs  
  - Test on CPU-only systems
  - Test with various model configurations

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