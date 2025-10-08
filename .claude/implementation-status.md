# Inferno GPU Inference Engine - Implementation Status

## Date: 2025-10-07

## Executive Summary

The inferno project has substantial infrastructure in place but needs GPU-only OpenAI model support. The existing codebase uses:
- **Candle framework**: 5076 lines of implementation for Llama models
- **CLI infrastructure**: Complete with download, play, doctor commands
- **Backend architecture**: Service discovery, metrics, health checks
- **Hardware**: RTX 4090 with 24GB VRAM available

## Current State Analysis

### ✅ What's Already Implemented

1. **Project Structure** (Complete)
   - Workspace with 7 crates: shared, proxy, backend, governator, cli, inference, benchmarking
   - Proper Cargo.toml with candle-cuda feature support
   - Build system working (compiles with `--features candle-cuda`)

2. **CLI Commands** (80% Complete)
   - `inferno download`: Implemented with hf-hub integration, Git LFS/Xet support
   - `inferno play`: Interactive mode with rustyline, statistics tracking
   - `inferno doctor`: System diagnostics
   - Model selection UI and validation

3. **Candle Inference Engine** (60% Complete - Llama only)
   - `CandleInferenceEngine`: Full Llama model support with quantization
   - Tokenizer integration: CandleTokenizer with streaming decode
   - Quantized model support: CompressedTensorsLoader, HybridQuantizedLlama
   - Backend selection: CPU/CUDA/Metal device management
   - KV cache: Implemented for Llama architecture

4. **Infrastructure** (90% Complete)
   - Memory management: CudaMemoryPool, GpuAllocator, MemoryTracker
   - Service discovery: SWIM gossip protocol
   - Health checks and metrics
   - Error handling and logging
   - Async architecture with Tokio

### ❌ What's Missing for OpenAI Model Support

1. **OpenAI Model Architecture** (0% Complete)
   - No OpenAI-specific transformer implementation
   - Current code only supports Llama architecture
   - Need: OpenAI-specific layer implementations (different from Llama)
   - Need: OpenAI model config parser

2. **Safetensors Loader for OpenAI** (30% Complete)
   - Existing loader is Llama-specific (expects Llama weight names)
   - Need: OpenAI weight name mapping
   - Need: Multi-file shard support for 20B model
   - Need: VarBuilder pattern for OpenAI architecture

3. **OpenAI-Specific Layers** (0% Complete)
   - Need: OpenAI Embedding layer
   - Need: OpenAI RMSNorm (may differ from Llama)
   - Need: OpenAI RoPE implementation
   - Need: OpenAI Attention (GQA configuration)
   - Need: OpenAI MLP (SwiGLU activation)
   - Need: OpenAI TransformerBlock assembly

4. **Inference Pipeline** (40% Complete)
   - Basic inference loop exists for Llama
   - Need: Adapt for OpenAI model specifics
   - Need: Sampling strategies (temperature, top-k, top-p) - partially done
   - Need: Streaming token generation - exists but needs validation

5. **Testing** (20% Complete)
   - No TDD tests for new OpenAI implementation
   - Existing tests are Llama-specific
   - Need: Unit tests for each layer
   - Need: Integration tests for full model
   - Need: Property-based tests with proptest

6. **Performance Benchmarks** (10% Complete)
   - Criterion benchmarks exist but not for OpenAI model
   - Need: Tokens/sec benchmarks
   - Need: Memory usage profiling
   - Need: Latency measurements (TTFT, decode latency)

## Gap Analysis

### Critical Path Items (Must Have)

1. **OpenAI Model Configuration**
   - Parse config.json from OpenAI model repos
   - Map to internal config struct
   - Validate against hardware constraints

2. **OpenAI Transformer Layers**
   - Implement all layers from scratch following OpenAI architecture
   - Ensure GPU-only operations (no CPU fallback)
   - Use Candle's CUDA kernels

3. **Weight Loading**
   - Map OpenAI safetensors weight names to layer structure
   - Handle multi-file sharding
   - Load directly to GPU memory

4. **Inference Engine Integration**
   - Plug OpenAI model into existing engine infrastructure
   - Reuse KV cache management
   - Adapt sampling and generation logic

5. **End-to-End Testing**
   - Download small OpenAI model (e.g., 3B for testing)
   - Test full pipeline: download → load → inference
   - Validate output quality

### Architecture Decisions Needed

1. **Should we create a new crate `inferno-models-openai`?**
   - Pro: Clean separation, focused implementation
   - Pro: Easier to maintain and test
   - Con: More crate overhead
   - **Recommendation**: Yes, create `crates/models-openai/`

2. **Reuse existing Candle engine or create new?**
   - Current `CandleInferenceEngine` is tightly coupled to Llama
   - **Recommendation**: Refactor to support multiple model architectures
   - Create trait `ModelArchitecture` with implementations for Llama and OpenAI

3. **Target model size for initial implementation?**
   - 20B model requires ~40GB VRAM in FP16 (won't fit on 24GB)
   - **Recommendation**: Start with smaller OpenAI model (3B-7B) for development
   - Add quantization support (INT8) to fit 20B on RTX 4090

## Implementation Strategy

### Phase 1: Architecture Setup (Days 1-2)
- [ ] Create `crates/models-openai/` crate
- [ ] Define OpenAI model config structs
- [ ] Implement config parsing from HuggingFace format
- [ ] Write tests for config parsing (TDD)

### Phase 2: Transformer Layers (Days 3-5)
- [ ] Implement Embedding layer + tests
- [ ] Implement RMSNorm + tests
- [ ] Implement RoPE + tests
- [ ] Implement Attention (GQA) + tests
- [ ] Implement MLP (SwiGLU) + tests
- [ ] Implement TransformerBlock + tests
- [ ] Assemble complete model + tests

### Phase 3: Weight Loading (Days 6-7)
- [ ] Implement safetensors loader with OpenAI weight mapping
- [ ] Add multi-file shard support
- [ ] Implement VarBuilder pattern
- [ ] Test loading with real OpenAI model weights

### Phase 4: Inference Integration (Days 8-9)
- [ ] Integrate OpenAI model into CandleInferenceEngine
- [ ] Adapt KV cache for OpenAI architecture
- [ ] Test forward pass on GPU
- [ ] Implement sampling strategies
- [ ] Test generation loop

### Phase 5: CLI Integration (Day 10)
- [ ] Update `inferno download` for OpenAI models
- [ ] Update `inferno play` to support OpenAI models
- [ ] Add model auto-detection
- [ ] Test end-to-end user workflow

### Phase 6: Optimization & Benchmarking (Days 11-12)
- [ ] Profile GPU utilization
- [ ] Optimize memory usage
- [ ] Benchmark tokens/sec (target: >20)
- [ ] Add FlashAttention if needed
- [ ] Document performance characteristics

## Risk Mitigation

### Risk 1: OpenAI architecture differs significantly from Llama
**Mitigation**:
- Study OpenAI's model card and config.json carefully
- Compare with Llama architecture to identify differences
- Implement small test cases for each layer

### Risk 2: 20B model won't fit on 24GB VRAM
**Mitigation**:
- Start with smaller model (3B-7B) for development
- Implement INT8 quantization to reduce memory
- Consider model sharding across layers

### Risk 3: Performance below 20 tokens/sec
**Mitigation**:
- Use FlashAttention for optimized attention
- Profile early and identify bottlenecks
- Use fused kernels where possible
- Optimize KV cache layout

### Risk 4: Complex debugging on GPU
**Mitigation**:
- Implement comprehensive logging
- Add tensor shape validation at each layer
- Create CPU fallback for debugging (dev only)
- Use small test inputs for layer validation

## Success Criteria

### Minimum Viable Product (MVP)
1. ✅ `inferno download openai/<model-id>` successfully downloads model
2. ✅ Model loads to GPU without errors (no CPU usage)
3. ✅ `inferno play` starts and allows interaction
4. ✅ Model generates coherent text (not gibberish)
5. ✅ Performance: >20 tokens/sec on RTX 4090
6. ✅ All validation checks pass: clippy, fmt, machete, tests

### Quality Gates
1. Every function has comprehensive documentation
2. Test coverage >80% for new code
3. No clippy warnings
4. Formatted with rustfmt
5. No unused dependencies (cargo machete)
6. Changelog updated with performance metrics

## Timeline Estimate

**Total: 12-14 days** (single developer, full-time)

- Days 1-2: Architecture setup
- Days 3-5: Layer implementation
- Days 6-7: Weight loading
- Days 8-9: Inference integration
- Day 10: CLI integration
- Days 11-12: Optimization
- Days 13-14: Testing & documentation

## Next Immediate Steps

1. Create `crates/models-openai/` crate structure
2. Download a small OpenAI model (3B) for testing
3. Analyze model's config.json and weight structure
4. Implement config parsing with tests (TDD)
5. Begin layer-by-layer implementation with tests

## Open Questions

1. **Which specific OpenAI model to target?**
   - Need model ID and HuggingFace repo URL
   - Recommend starting with smallest available for faster iteration

2. **What are the exact architectural differences vs Llama?**
   - Need to study OpenAI's config.json format
   - May need reference implementation to compare

3. **Should we support both quantized and full precision?**
   - Full precision: Better quality, higher memory
   - Quantized (INT8): Fits larger models on 24GB
   - Recommend: Implement full precision first, add quantization later

4. **Streaming vs batch generation?**
   - Existing code has streaming support
   - Recommend: Keep streaming for better UX in play mode
