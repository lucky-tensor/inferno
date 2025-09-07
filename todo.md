# SmolLM2-135M Burn CPU Inference Implementation Plan

## Current Status: Infrastructure Ready - Real Inference Implementation Required

**Goal**: Replace stub implementation with production-quality SmolLM2-135M inference using Burn framework and SafeTensors weights.

### Model Architecture Details (SmolLM2-135M)
- **Architecture**: LlamaForCausalLM (Llama-based transformer)
- **Layers**: 30 transformer blocks
- **Hidden Size**: 576
- **Attention Heads**: 9 (with 3 key-value heads for grouped-query attention)
- **Intermediate Size**: 1536 (MLP feedforward)
- **Vocabulary Size**: 49152
- **Max Context**: 8192 tokens
- **Weights**: 269MB SafeTensors (bfloat16 → f32 conversion needed)

---

## Phase 1: Model Architecture Implementation ⚠️ CRITICAL PATH

### 1.1 Core Model Configuration
- [ ] **Create `LlamaConfig` struct** matching SmolLM2-135M architecture
  - Map config.json parameters to Burn-compatible structure
  - Handle hidden_size=576, num_layers=30, num_heads=9, etc.
  - Add validation for architecture consistency

### 1.2 Attention Mechanism Implementation
- [ ] **Implement Multi-Head Attention with Grouped-Query Attention**
  - Support 9 query heads, 3 key-value heads (GQA optimization)
  - Efficient attention computation with proper scaling
  - Memory-optimized attention pattern for CPU inference
  
- [ ] **Add Rotary Positional Encoding (RoPE)**
  - Implement RoPE with theta=100000 from config
  - Support interleaved=false setting
  - Efficient sin/cos computation and caching

### 1.3 Transformer Block Components
- [ ] **Implement RMSNorm Layer**
  - RMS normalization with eps=1e-05
  - Efficient variance computation for CPU
  - Proper gradient flow for potential fine-tuning

- [ ] **Create SwiGLU MLP Layer**
  - SiLU (Swish) activation function
  - Gate mechanism: `SiLU(xW1) ⊙ xW2`
  - Hidden size 576 → intermediate 1536 → output 576

### 1.4 Complete Model Architecture
- [ ] **Build Transformer Block**
  - Combine attention + MLP with residual connections
  - Pre-norm architecture (RMSNorm before attention/MLP)
  - Efficient memory layout for 30 layer stack

- [ ] **Create Full Llama Model**
  - Token embedding layer (vocab_size=49152, hidden_size=576)
  - Stack of 30 transformer blocks
  - Final RMSNorm and language modeling head
  - Tie input/output embeddings (tie_word_embeddings=true)

---

## Phase 2: SafeTensors Integration ⚠️ CRITICAL PATH

### 2.1 Weight Loading Infrastructure
- [ ] **Implement SafeTensors Reader**
  - Load 269MB model.safetensors file
  - Handle bfloat16 → f32 conversion for CPU inference
  - Memory-efficient loading without full duplication

### 2.2 Weight Mapping and Initialization
- [ ] **Create HuggingFace → Burn weight mapper**
  - Map tensor names: `model.embed_tokens.weight` → embedding layer
  - Transform attention weights: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Handle MLP weights: `gate_proj`, `up_proj`, `down_proj`
  - Load normalization weights: `input_layernorm`, `post_attention_layernorm`

- [ ] **Add Model Initialization**
  - Initialize Burn model with correct tensor shapes
  - Load and assign SafeTensors weights to model parameters
  - Validate weight shapes match model architecture

---

## Phase 3: Text Generation Pipeline ⚠️ CRITICAL PATH

### 3.1 Autoregressive Generation
- [ ] **Implement Text Generation Loop**
  - Token-by-token generation with model forward pass
  - Proper logit computation and sampling
  - Stop token handling (eos_token_id=0)

### 3.2 Sampling Strategies
- [ ] **Add Temperature and Top-P Sampling**
  - Temperature scaling for logit distribution
  - Nucleus (top-p) sampling implementation
  - Deterministic sampling for testing (temperature=0.0)

### 3.3 Performance Optimizations
- [ ] **Implement KV-Cache**
  - Cache key-value tensors for efficient sequential generation
  - Memory management for long sequences
  - Sliding window for max context length

---

## Phase 4: Integration and Testing ⚠️ CRITICAL PATH

### 4.1 Replace Stub Implementation
- [ ] **Update `burn_hello_world.rs`**
  - Remove hardcoded response generation
  - Integrate real Llama model inference
  - Use loaded SafeTensors weights for predictions

### 4.2 Comprehensive Testing
- [ ] **Unit Tests for Model Components**
  - Test each layer: attention, MLP, RMSNorm individually
  - Verify tensor shapes and mathematical correctness
  - Compare outputs with known reference implementations

- [ ] **Integration Tests**
  - End-to-end inference: "Hello" → actual model continuation
  - Math reasoning: "What is 2+2?" → contextual response
  - Long sequence handling within 8192 token limit

### 4.3 Performance Benchmarks
- [ ] **Criterion Benchmarks**
  - Single token inference latency
  - Full sequence generation throughput
  - Memory usage profiling
  - CPU utilization optimization

---

## Code Organization Strategy

### Module Structure
```
crates/inference/src/
├── models/
│   ├── llama/
│   │   ├── mod.rs              # Public API
│   │   ├── config.rs           # LlamaConfig struct
│   │   ├── model.rs            # Full LlamaModel
│   │   ├── transformer.rs      # TransformerBlock
│   │   ├── attention.rs        # MultiHeadAttention + RoPE
│   │   ├── mlp.rs             # SwiGLU MLP
│   │   ├── norm.rs            # RMSNorm
│   │   └── generation.rs      # Text generation pipeline
│   └── safetensors_loader.rs   # SafeTensors integration
└── inference/
    └── burn_hello_world.rs     # Updated engine
```

### Key Burn APIs to Use
- **Model Definition**: `burn::nn::*` for layers
- **SafeTensors**: `burn::import::onnx::*` or custom loader
- **Tensor Operations**: `burn::tensor::Tensor`
- **Backend**: `burn::backend::ndarray::NdArray<f32>`
- **Device**: `burn::backend::ndarray::NdArrayDevice`

---

## Error Handling Strategy

### Comprehensive Error Types
- **ModelLoadError**: SafeTensors loading, weight shape mismatches
- **InferenceError**: Forward pass failures, numerical instabilities  
- **GenerationError**: Sampling failures, context length exceeded
- **ConfigError**: Invalid model configuration parameters

### Validation Points
- Pre-inference: Validate input token sequences
- Runtime: Check tensor shapes at each layer
- Post-inference: Verify output logits and sampling results
- Memory: Monitor allocation limits for CPU inference

---

## Performance Optimization Targets

### CPU Inference Optimizations
- **Memory Layout**: Contiguous tensors, minimal allocations
- **SIMD**: Leverage `burn-ndarray` SIMD optimizations
- **Batching**: Single sequence optimization (typical CPU use case)
- **Precision**: f32 precision balance (accuracy vs performance)

### Performance Baselines (Target)
- **Single Token**: <50ms latency on modern CPU
- **Short Sequence** (10 tokens): <200ms total time
- **Memory Usage**: <1GB peak for model + intermediate tensors
- **Cold Start**: <2s model loading time

---

## Testing Strategy

### Test-Driven Development Approach
1. **Write tests first** for each component before implementation
2. **Property-based testing** using `proptest` for tensor operations
3. **Integration testing** with real model weights and known outputs
4. **Performance regression tests** with `criterion`

### Validation Approach
- **Mathematical Correctness**: Compare with reference PyTorch implementation
- **Deterministic Testing**: Fixed seeds for reproducible outputs
- **Edge Cases**: Empty prompts, maximum context length, special tokens
- **Memory Safety**: No leaks, proper tensor lifetime management

---

## Success Criteria

### Functional Requirements ✅ MUST PASS
- [ ] **Real Text Generation**: "Hello" → contextual continuation (not hardcoded)
- [ ] **Math Reasoning**: "What is 2+2?" → model-generated response
- [ ] **Long Context**: Handle sequences up to 8192 tokens
- [ ] **Deterministic**: Same input + seed → identical output

### Performance Requirements ✅ MUST PASS
- [ ] **Latency**: Single token inference <50ms
- [ ] **Memory**: Peak usage <1GB for full pipeline
- [ ] **Accuracy**: Generated text quality comparable to reference implementation
- [ ] **Stability**: No crashes or numerical instabilities

### Code Quality Requirements ✅ MUST PASS
- [ ] **Linting**: `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] **Formatting**: `cargo fmt --check`
- [ ] **Dependencies**: `cargo machete` (no unused deps)
- [ ] **Testing**: `cargo test` (90%+ coverage)

---

## Current Implementation Status

### ✅ COMPLETED - Infrastructure Ready
- [x] SafeTensors model downloaded (269MB)
- [x] Tokenizer working with real HuggingFace tokenizer
- [x] Basic Burn tensor operations tested
- [x] Build system and dependencies configured
- [x] Test infrastructure in place

### ❌ NOT IMPLEMENTED - Critical Missing Components
- [ ] Llama model architecture (30 transformer layers)
- [ ] SafeTensors weight loading into Burn model  
- [ ] Autoregressive text generation loop
- [ ] Real inference replacing hardcoded responses

### Next Immediate Steps (Priority Order)
1. **Start with model configuration and basic layers** (RMSNorm, SwiGLU)
2. **Implement attention mechanism with RoPE**
3. **Build complete transformer block and full model**
4. **Add SafeTensors loading and weight mapping**
5. **Implement text generation pipeline**
6. **Replace stub implementation with real inference**

---

**CRITICAL**: This is a production-quality implementation requiring proper abstractions, comprehensive error handling, and performance optimization. Every component must have extensive tests and documentation before integration.