# Llama Implementation Plan: Generic Engine with Selective Candle Integration

## Overview

This document outlines the implementation plan for transforming the current `crates/inferno-llama` into a fully generic Llama inference engine as specified in `GENERIC_LLAMA_SPECIFICATION.md`. **Key insight**: We leverage Candle's excellent tensor system and backends while extending/forking only the components that need customization for our generic requirements.

## Pragmatic Strategy: Keep vs Fork vs Extend

### ✅ Keep from Candle (Proven, Stable)
- **`candle_core`**: Tensor, Device, DType, backends (CPU/CUDA/Metal)
- **Core tensor operations**: matmul, softmax, silu, basic ops
- **Memory management**: Candle's efficient memory pools
- **Backend abstractions**: Well-tested GPU/CPU implementations
- **SafeTensors loading**: Existing efficient weight loading

### 🔄 Fork/Extend from Candle
- **Model implementations**: Fork `candle-transformers/llama.rs` for generic support
- **Quantization**: Extend candle's basic quantization for w8a8/compressed-tensors
- **Configuration parsing**: Extend for all model variants (TinyLlama, distilled)
- **RoPE implementation**: Fork for precise BF16/F16 handling

### 🆕 Create New (Our Unique Requirements)
- **Model diagnostics**: Auto-detection and analysis system
- **Generic configuration**: Unified config for all Llama variants
- **Advanced quantization**: Compressed-tensors, per-layer dtypes
- **Model loading orchestration**: Smart loading strategies for different formats

## Current State Analysis

### Existing Structure (Good Foundation)
- **Location**: `crates/inferno-llama/`
- **Current Status**: Moderate Candle dependency (47 usages - manageable)
- **Architecture**: Already modular, well-designed:
  - `attention.rs` - Multi-head attention ✅ Keep structure, extend
  - `feed_forward.rs` - MLP layers ✅ Keep structure, extend
  - `rope.rs` - Rotary position embedding 🔄 Fork for precision
  - `normalization.rs` - RMS normalization ✅ Keep, minor extensions
  - `model.rs` - Complete Llama model 🔄 Major extension needed
  - `precision.rs` - Precision management ✅ Excellent foundation
  - `loader.rs` - Model loading utilities 🔄 Extend significantly
  - `tokenizer.rs` - Tokenization support ✅ Keep, extend formats

### Candle Integration Strategy
**Smart approach**: Use `candle_core` types throughout, extend `candle-transformers` components
- Keep: `candle_core::{DType, Device, Tensor, Result}`
- Keep: `candle_nn::{Module, VarBuilder, Linear, Embedding}`
- Extend: Model implementations for generic support
- Fork: Quantization and specialized operations

## Implementation Plan

### Phase 1: Foundation - Candle Integration & Extensions
**Duration**: 1-2 weeks
**Goal**: Smart Candle integration with our extensions

#### 1.1 Fork and Extend Candle Components
**Files to create**:
```
crates/inferno-llama/src/candle_extensions/
├── mod.rs              # Re-exports and integration
├── llama_models.rs     # Forked from candle-transformers/llama.rs
├── quantized_ops.rs    # Extended quantization operations
├── rope_precision.rs   # High-precision RoPE for BF16/F16
└── var_builder_ext.rs  # Extended VarBuilder for our needs
```

**Key approach**:
- Fork `candle-transformers/models/llama.rs` → our `llama_models.rs`
- Extend `candle_core::DType` support (keep existing, add quantized types metadata)
- Use `candle_nn::VarBuilder` but extend for quantized loading
- Keep all Candle tensor operations, extend where needed

#### 1.2 Enhanced Data Type System (Candle-Compatible)
**Files to create**:
```
crates/inferno-llama/src/dtype_extensions/
├── mod.rs               # Data type extensions
├── quantized.rs         # W8A8, CompressedTensors support
├── precision_config.rs  # Per-layer dtype configuration
└── conversion.rs        # Safe dtype conversions
```

**Strategy**: Extend `candle_core::DType` with metadata, don't replace
```rust
// Use candle_core::DType + our extensions
pub struct ExtendedDType {
    pub base: candle_core::DType,
    pub quantization: Option<QuantizationSpec>,
    pub precision_config: Option<PrecisionConfig>,
}
```

#### 1.3 Update Existing Modules (Minimal Changes)
**Priority order** (much smaller scope now):
1. **precision.rs**: Extend (not replace) `candle_core::DType` support
2. **loader.rs**: Add quantized weight loading on top of existing SafeTensors
3. **rope.rs**: Fork Candle's RoPE for our precision requirements
4. **model.rs**: Import and extend forked Llama models
5. **config.rs**: Extend existing config parsing for all model variants
6. **error.rs**: Keep `candle_core::Error`, add our specific error types

### Phase 2: Model Discovery and Diagnostics (New Capability)
**Duration**: 1-2 weeks
**Goal**: Auto-discovery and analysis of any Llama-like model

#### 2.1 Model Diagnostic System
**Files to create**:
```
crates/inferno-llama/src/diagnostic/
├── mod.rs              # Main diagnostic exports
├── detector.rs         # Model variant detection (Meta/Tiny/Distilled)
├── config_parser.rs    # Multi-format config parsing
├── weight_analyzer.rs  # SafeTensors inspection + quantization detection
└── memory_estimator.rs # Smart memory requirement calculation
```

**Key functionality**:
- Auto-detect: Meta Llama 3.1/3.2, TinyLlama, DeepSeek distilled, w8a8 quantized
- Parse configs: Handle all variations (different rope_scaling, vocab_size, etc.)
- Weight analysis: Detect dtypes, quantization schemes, sharding patterns
- Memory estimation: Accurate requirements + optimization suggestions

#### 2.2 Generic Configuration System (Candle + Extensions)
**Strategy**: Extend `candle-transformers` config parsing
```rust
// Extend existing Candle LlamaConfig
pub struct GenericLlamaConfig {
    pub base: candle_transformers::models::llama::Config,
    pub variant: LlamaVariant,  // Meta/Tiny/Distilled/Custom
    pub quantization: Option<QuantizationConfig>,
    pub memory_layout: ModelMemoryLayout,
}
```

### Phase 3: Enhanced Model Loading (Build on Candle)
**Duration**: 1-2 weeks
**Goal**: Load all model formats leveraging Candle's SafeTensors support

#### 3.1 Smart Weight Loading (Candle + Extensions)
**Strategy**: Use `candle_nn::VarBuilder` + our extensions
```
crates/inferno-llama/src/loading/
├── mod.rs               # Loading orchestration
├── weight_loader.rs     # Extend Candle SafeTensors loading
├── quantized_loader.rs  # w8a8, compressed-tensors support
├── sharded_loader.rs    # Multi-file model loading
└── model_mapper.rs      # Weight name mapping between variants
```

**Key features**:
- Use Candle's proven SafeTensors loading as base
- Add quantized weight dequantization on load
- Handle sharded models (like meta-llama 8B with 4 files)
- Map weight names between different model formats (TinyLlama vs Meta Llama)

#### 3.2 Tokenizer Integration (Keep + Extend)
**Strategy**: Keep existing `tokenizers` crate integration, extend support
- Keep current tokenizer loading (already works well)
- Add support for different tokenizer formats in different models
- Ensure tokenizer compatibility validation

### Phase 4: Generic Inference Engine (Fork + Extend Candle Models)
**Duration**: 1-2 weeks
**Goal**: Universal forward pass supporting all model variants

#### 4.1 Forked Model Implementation
**Strategy**: Fork `candle-transformers/models/llama.rs` and extend
```rust
// Our enhanced version of Candle's Llama model
pub struct GenericLlama {
    config: GenericLlamaConfig,
    layers: Vec<LlamaLayer>,  // Keep Candle's layer structure
    model_type: LlamaVariant,
    quantization_handler: Option<QuantizationHandler>,
}
```

**Key extensions**:
- Support TinyLlama architecture (different layer counts, head configs)
- Handle quantized weights during forward pass
- Flexible RoPE scaling for different model variants
- Per-layer dtype handling

#### 4.2 Enhanced Operations (Minimal Changes)
**Strategy**: Use Candle ops, fork only where necessary
- **RoPE**: Fork Candle's implementation for precise BF16/F16 (high priority)
- **Attention**: Use Candle's attention, add quantization support
- **Feed-Forward**: Use Candle's SiLU, extend for quantized weights
- **Layer Norm**: Keep Candle's RMSNorm (already good)

### Phase 5: Advanced Features (Build on Solid Foundation)
**Duration**: 1-2 weeks
**Goal**: Production-ready features using proven Candle base

#### 5.1 Generation Integration
**Strategy**: Use existing generation patterns, extend for model variants
```
crates/inferno-llama/src/generation/
├── mod.rs               # Generation coordination
├── generic_pipeline.rs  # Unified pipeline for all model types
├── sampling_ext.rs      # Extended sampling for different models
└── cache_manager.rs     # Smart KV cache for different architectures
```

#### 5.2 Quantization Pipeline (Candle + Our Extensions)
**Strategy**: Build on Candle's quantization support
- Use Candle's existing int8 quantization as base
- Add compressed-tensors format support
- Implement w8a8 dequantization during inference
- Per-layer quantization configuration

### Phase 6: Production Readiness
**Duration**: 1 week
**Goal**: Polish, testing, documentation

#### 6.1 Comprehensive Testing
**Test all supported models** (leveraging existing model samples):
- `~/models/meta-llama_Llama-3.1-8B-Instruct/` (sharded, BF16)
- `~/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8/` (quantized)
- `~/models/tinyllama-1.1b/` (distilled, small)
- `~/models/DeepSeek-R1-Distill-Llama-70B/` (large distilled)

#### 6.2 Performance Validation
**Benchmarks against current implementation**:
- Inference speed (should match or exceed current)
- Memory usage (should respect theoretical limits)
- Output quality (should match reference implementations)
- Multi-model support (new capability)

## Revised Implementation Schedule (Much Faster!)

### Week 1: Foundation + Candle Extensions
- [ ] Fork `candle-transformers/llama.rs` → `src/candle_extensions/llama_models.rs`
- [ ] Create model diagnostic system (auto-detection)
- [ ] Extend configuration parsing for all model variants
- [ ] Setup quantization metadata types

### Week 2: Model Loading + Testing
- [ ] Implement weight loading for quantized models (w8a8, compressed-tensors)
- [ ] Add sharded model loading support
- [ ] Test loading all model types in `~/models/`
- [ ] Create unified model factory

### Week 3: Inference Engine
- [ ] Integrate forked Llama models with quantization support
- [ ] Fork and enhance RoPE implementation for BF16/F16 precision
- [ ] Implement generation pipeline for all model variants
- [ ] Add per-layer dtype configuration

### Week 4: Production Polish
- [ ] Comprehensive testing across all model types
- [ ] Performance benchmarking vs current implementation
- [ ] Documentation and examples
- [ ] Integration with existing inference crate

## Dependencies and Libraries (Much Simpler!)

### Keep All Existing Candle Dependencies
```toml
[dependencies]
# Keep - these are excellent and proven
candle-core = { version = "0.9.1" }
candle-nn = { version = "0.9.1" }

# Add for our extensions
candle-transformers = { version = "0.9.1" }  # To fork from

# Existing (keep all)
serde = { workspace = true }
serde_json = { workspace = true }
tokenizers = { workspace = true }
half = { workspace = true }
thiserror = { workspace = true }
safetensors = { workspace = true }
hf-hub = { workspace = true }

# Minimal additions for quantization
bytemuck = { version = "1.0" }  # For safe type casting
```

### No Dependencies to Remove!
All existing Candle dependencies stay - we build on top of them.

## Risk Mitigation (Much Lower Risk Now!)

### Lower-Risk Areas (Using Proven Candle Base)
1. **Tensor Operations**: Using Candle's proven, tested operations
   - **Low Risk**: Candle handles all the complex tensor math
   - **Our Focus**: Only fork where we need specific precision/quantization behavior

2. **Memory Management**: Leverage Candle's efficient memory pools
   - **Low Risk**: Candle's memory management is battle-tested
   - **Our Addition**: Just quantization-specific memory patterns

3. **Performance**: Building on optimized Candle foundation
   - **Low Risk**: Start with Candle's performance, extend from there
   - **Target**: Match Candle performance + add new model support

### Remaining Medium-Risk Areas
1. **Quantization Accuracy**: Ensuring w8a8/compressed-tensors correctness
   - **Mitigation**: Test against RedHat AI reference outputs
   - **Validation**: Token-by-token comparison with HuggingFace

2. **Model Variant Support**: Ensuring TinyLlama/distilled models work correctly
   - **Mitigation**: Extensive testing with actual model files in `~/models/`
   - **Validation**: Compare with original implementations

### Testing Strategy (Simplified)
- **Model Loading Tests**: Verify all models in `~/models/` load correctly
- **Inference Tests**: Forward pass validation for each model type
- **Quantization Tests**: Accuracy validation for quantized models
- **Performance Tests**: Benchmark against current Candle implementation

## Success Criteria (More Achievable)

### Core Functionality
- [ ] Load and run inference on Meta Llama 3.1/3.2 (standard + sharded)
- [ ] Load and run inference on TinyLlama distilled models
- [ ] Load and run inference on w8a8 quantized models
- [ ] Load and run inference on DeepSeek distilled models

### Technical Goals
- [ ] Maintain Candle performance (no regression)
- [ ] Support all major Llama variants through single API
- [ ] Proper memory usage (leverage Candle's efficient patterns)
- [ ] Numerical accuracy matching reference (for non-quantized)

### Code Quality
- [ ] Build on proven Candle foundation (low risk)
- [ ] Clean extension patterns (fork only where needed)
- [ ] Comprehensive testing with real model files
- [ ] Clear documentation for new generic capabilities

## Implementation Approach Summary

**Key Insight**: This revised plan is much more pragmatic and achievable:

1. **✅ Keep Candle's Strengths**: Tensor ops, memory management, backends
2. **🔄 Fork Selectively**: Only Llama model implementation for our generic needs
3. **🆕 Add Strategically**: Model diagnostics, quantization support, multi-variant loading
4. **📉 Reduce Risk**: Build on proven foundation instead of replacing it
5. **⚡ Faster Timeline**: 4 weeks instead of 12 weeks

This approach leverages the excellent work already done in Candle while adding the specific generic Llama capabilities we need. Much more pragmatic and likely to succeed!