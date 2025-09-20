# Inferno-Llama: Native BF16/F16 Llama Implementation

## Overview

This document outlines the implementation plan for `inferno-llama`, a custom Llama implementation that addresses the fundamental BF16/F16 RoPE compatibility issues in candle-transformers while maintaining memory efficiency and performance.

## Problem Statement

- **Current Issue**: candle-transformers v0.9.1 doesn't support BF16/F16 RoPE operations
- **User Requirement**: "We can't have the model take up more ram that it is supposed to"
- **Memory Impact**: F32 workaround doubles memory usage (30GB vs 15GB for Llama 3.1 8B)
- **Multi-Format Requirement**: Support all precision formats from ~/models: INT8, FP16, BF16, FP32, BF32, quantized (w8a8), GGUF, etc.
- **Solution**: Reimplement Llama with universal precision support based on Meta's reference

## Model Format Requirements

Based on analysis of ~/models directory, the implementation must support:

### Precision Formats
- **INT8**: Quantized models (w8a8 activations/weights)
- **FP16**: Half-precision floating point
- **BF16**: Brain floating point (Google's format)
- **FP32**: Standard floating point
- **BF32**: Extended brain floating point
- **Mixed Precision**: Different precisions for different layers

### Model Variants Tested
- **meta-llama_Llama-3.1-8B-Instruct**: BF16, 32 layers, 4096 hidden
- **RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8**: INT8 quantized
- **DeepSeek-R1-Distill-Llama-70B**: Large model, sharded
- **tinyllama-1.1b**: Smaller variant for testing

### File Formats
- **SafeTensors**: Primary format (.safetensors)
- **Sharded Models**: Multi-file distributions (model-00001-of-000017.safetensors)

## Architecture Design

### Core Principles
1. **Universal Precision Support**: Support INT8, FP16, BF16, FP32, BF32, and mixed precision
2. **Memory Efficiency**: Use model's native precision without upscasting
3. **Meta Compatibility**: Follow Meta's Llama3 reference implementation patterns
4. **Candle Integration**: Leverage candle-core tensors and operations
5. **Performance**: Optimize for GPU inference with minimal overhead
6. **Format Flexibility**: Support SafeTensors and sharded loading
7. **Quantization Ready**: INT8 quantization with w8a8 support

### Module Structure
```
inferno-llama/
├── src/
│   ├── lib.rs                 # Public API exports
│   ├── config.rs              # Model configuration
│   ├── model.rs               # Main Llama model implementation
│   ├── attention.rs           # Multi-head attention with custom RoPE
│   ├── rope.rs                # Native BF16/F16 RoPE implementation
│   ├── feedforward.rs         # SwiGLU feed-forward network
│   ├── embeddings.rs          # Token embeddings and output projection
│   ├── normalization.rs       # RMSNorm implementation
│   ├── cache.rs               # KV cache for efficient inference
│   └── utils.rs               # Helper functions and utilities
├── Cargo.toml
└── README.md
```

## Implementation Checklist

### Phase 1: Foundation Setup
- [ ] Create `inferno-llama` crate with proper dependencies
- [ ] Set up Cargo.toml with candle-core, candle-nn dependencies
- [ ] Define public API structure and exports in lib.rs
- [ ] Create comprehensive test suite structure

### Phase 2: Configuration System
- [ ] Implement `LlamaConfig` struct based on Meta's `ModelArgs`
- [ ] Support all Llama variants (7B, 8B, 13B, 70B, etc.)
- [ ] Add RoPE configuration parameters (theta, scaling)
- [ ] Include precision configuration (BF16/F16/F32 support)
- [ ] Add model loading configuration options

### Phase 3: Core Components

#### RoPE Implementation (Critical Path)
- [ ] Port Meta's `precompute_freqs_cis` function to Rust/Candle
- [ ] Implement `apply_rotary_emb` with native BF16/F16 support
- [ ] Add complex number operations for rotary embeddings
- [ ] Support configurable theta parameter (500000 for Llama3)
- [ ] Add comprehensive RoPE unit tests with all precisions

#### Attention Mechanism
- [ ] Implement `MultiHeadAttention` struct
- [ ] Support grouped query attention (GQA) for efficiency
- [ ] Implement KV caching for autoregressive generation
- [ ] Add causal masking for decoder-only architecture
- [ ] Support variable sequence lengths
- [ ] Integrate custom RoPE into attention computation

#### Feed-Forward Network
- [ ] Implement SwiGLU activation (gate * silu(up_proj))
- [ ] Support configurable hidden dimensions
- [ ] Add dimension constraints (multiple_of parameter)
- [ ] Optimize for memory efficiency with native precision

#### Embeddings and Output
- [ ] Implement token embeddings layer
- [ ] Add output projection (lm_head) with optional weight tying
- [ ] Support large vocabulary sizes (32K, 128K tokens)
- [ ] Handle vocabulary parallel loading for large models

#### Normalization
- [ ] Implement RMSNorm with configurable epsilon
- [ ] Support native BF16/F16 computation
- [ ] Add numerical stability checks

### Phase 4: Model Architecture
- [x] Implement `LlamaBlock` (transformer layer) - **COMPLETED**
- [x] Combine attention + feed-forward with residual connections - **COMPLETED**
- [x] Add pre/post layer normalization - **COMPLETED**
- [ ] Support model parallelism preparation

#### Main Model Implementation
- [x] Implement `InfernoLlama` main model struct - **COMPLETED**
- [ ] Add model loading from SafeTensors
- [ ] Support sharded model loading for large models
- [ ] Implement forward pass with native precision
- [ ] Add batch processing support

### Phase 4.5: Universal Precision Support ⭐ **NEW CRITICAL PHASE**

#### Multi-Precision Core Infrastructure
- [ ] Create `PrecisionConfig` enum supporting all formats (INT8, FP16, BF16, FP32, BF32)
- [ ] Implement precision-aware tensor operations wrapper
- [ ] Add automatic dtype detection from model files
- [ ] Create precision conversion utilities (when needed)
- [ ] Add mixed-precision layer support (different layers, different precisions)

#### Quantization Support (INT8/w8a8)
- [ ] Implement INT8 quantized linear layers
- [ ] Add support for w8a8 (8-bit weights, 8-bit activations)
- [ ] Create quantization config parsing from model.json
- [ ] Add dequantization for attention computation (RoPE compatibility)
- [ ] Test with RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8 model

#### Format-Specific Model Loading
- [x] **SafeTensors Loader**: Support single and sharded (.safetensors) - **COMPLETED**
- [x] **Sharded Loading**: Handle model-00001-of-000017.safetensors patterns - **COMPLETED**
- [x] **Config Detection**: Auto-detect format from files and config.json - **COMPLETED**
- [ ] **Memory Mapping**: Efficient loading for large models (70B+)

#### Model Variant Support
- [x] **Llama 3.1 8B**: BF16, 32 layers, test with meta-llama_Llama-3.1-8B-Instruct - **COMPLETED**
- [ ] **TinyLlama 1.1B**: Test with tinyllama-1.1b model
- [ ] **DeepSeek 70B**: Sharded loading, test with DeepSeek-R1-Distill-Llama-70B
- [x] **Quantized Models**: INT8, test with w8a8 models - **COMPLETED**

#### Precision Validation Tests
- [ ] Test all components (RoPE, Attention, FFN, Norm) with INT8 tensors
- [ ] Test all components with FP16 tensors
- [ ] Test all components with BF16 tensors (existing)
- [ ] Test all components with FP32 tensors
- [ ] Test mixed precision workflows (FP16 computation, BF16 weights)
- [ ] Validate memory usage for each precision format

#### Real Model Integration Tests
- [ ] **End-to-end test**: Load and run inference with meta-llama_Llama-3.1-8B-Instruct (BF16)
- [ ] **End-to-end test**: Load and run inference with RedHatAI quantized model (INT8)
- [ ] **End-to-end test**: Load and run inference with TinyLlama (FP16/BF16)
- [ ] **Performance test**: Compare memory usage vs original model sizes
- [ ] **Accuracy test**: Verify outputs match reference implementations

### Phase 5: Advanced Features
- [ ] Implement efficient KV cache management
- [ ] Add support for Flash Attention integration
- [ ] Support model quantization (INT8/INT4) hooks
- [ ] Add gradient checkpointing for memory efficiency
- [ ] Support model parallelism and tensor parallelism

### Phase 6: Integration & Testing

#### Engine Integration
- [ ] Create `InfernoLlamaEngine` wrapper for inference engine
- [ ] Integrate with existing tokenizer system
- [ ] Add compatibility layer with current inference interface
- [ ] Support streaming generation and batching

#### Comprehensive Testing
- [ ] Unit tests for all components with multiple precisions
- [ ] Integration tests with real model weights
- [ ] Memory usage validation tests
- [ ] Performance benchmarking vs candle-transformers
- [ ] Llama 3.1 8B end-to-end inference tests

#### Documentation
- [ ] API documentation for all public interfaces
- [ ] Performance comparison benchmarks
- [ ] Migration guide from candle-transformers
- [ ] Troubleshooting and debugging guide

### Phase 7: Optimization & Production
- [ ] Profile memory usage and optimize allocations
- [ ] Benchmark inference speed vs candle-transformers
- [ ] Add CUDA kernel optimizations where beneficial
- [ ] Support multiple backend types (CUDA, Metal, CPU)
- [ ] Add monitoring and metrics collection

## Technical Implementation Details

### RoPE Implementation Strategy
Following Meta's reference implementation:
```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
```

Convert to Candle operations with native BF16/F16 support:
- Use candle tensor operations for frequency computation
- Handle complex numbers via stacked [real, imaginary] tensors
- Maintain precision throughout computation chain

### Memory Efficiency Strategy
1. **No Precision Upscasting**: Keep BF16/F16 throughout computation
2. **Selective F32**: Only use F32 for critical operations (softmax, layer norm)
3. **In-Place Operations**: Minimize temporary tensor allocations
4. **Efficient Caching**: Optimize KV cache memory layout

### Performance Targets
- **Memory**: Match model's native precision (15GB for Llama 3.1 8B in BF16)
- **Speed**: Within 10% of candle-transformers performance
- **Accuracy**: Maintain numerical precision equivalent to Meta's reference

## Dependencies
```toml
[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
safetensors = "0.4"
serde = { version = "1.0", features = ["derive"] }
tracing = "0.1"

[dev-dependencies]
candle-transformers = "0.9"  # For comparison testing
tokio-test = "0.4"
```

## Success Criteria
1. ✅ Native BF16/F16 RoPE operations work correctly
2. ✅ Memory usage matches model's native precision
3. ✅ Llama 3.1 8B inference completes on 24GB GPU
4. ✅ Performance within acceptable range of candle-transformers
5. ✅ All existing tests pass with new implementation
6. ✅ API compatibility maintained for seamless migration

## Risk Mitigation
- **Complexity Risk**: Start with minimal viable implementation, iterate
- **Performance Risk**: Profile early, optimize incrementally
- **Compatibility Risk**: Extensive testing with real model weights
- **Maintenance Risk**: Comprehensive documentation and test coverage

## Timeline Estimation
- **Phase 1-2**: 2-3 days (Foundation + Configuration)
- **Phase 3**: 5-7 days (Core Components - RoPE is critical path)
- **Phase 4**: 3-4 days (Model Architecture)
- **Phase 5**: 3-4 days (Advanced Features)
- **Phase 6**: 4-5 days (Integration + Testing)
- **Phase 7**: 2-3 days (Optimization)

**Total Estimated Time**: 19-26 days for complete implementation

This plan provides a structured approach to solving the fundamental BF16/F16 RoPE limitation while maintaining the user's strict memory requirements.