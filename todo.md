# SmolLM3 + Burn Framework Implementation Plan

## Quick Win: Hello World Real Inference

### Phase 1: Setup and Model Download
- [ ] Add SmolLM3 model download functionality
- [ ] Update .gitignore to exclude ./models/ directory
- [ ] Choose smallest SmolLM3 variant (135M parameters)
- [ ] Implement model caching to ./models/smollm3-135m/

### Phase 2: Burn Framework Integration
- [ ] Add proper Burn framework dependencies (avoiding edition2024 conflicts)
- [ ] Implement real model loading with Burn tensors
- [ ] Replace placeholder tokenization with real tokenizer
- [ ] Implement actual forward pass inference

### Phase 3: Core Implementation
- [ ] Update BurnInferenceEngine::download_real_model() for SmolLM3
- [ ] Update BurnInferenceEngine::load_burn_model() with real loading
- [ ] Update BurnInferenceEngine::burn_framework_inference() with real inference
- [ ] Ensure deterministic outputs with fixed seed

### Phase 4: Testing
- [ ] Create basic "hello world" inference test
- [ ] Test simple prompt: "Hello" -> expected deterministic response
- [ ] Ensure test passes with real model download and inference
- [ ] Verify no mocking or simulation remains

### Success Criteria
- [ ] SmolLM3-135M model downloads to ./models/
- [ ] Real Burn framework tensor operations
- [ ] Deterministic text generation
- [ ] Basic unit test passes: prompt -> real model response
- [ ] Cargo lint passes
- [ ] No mocking/simulation code

### Model Choice: SmolLM3-135M
- **Size**: ~135M parameters (~270MB download)
- **Architecture**: Llama-like transformer
- **Tokenizer**: SentencePiece/Llama tokenizer
- **Deterministic**: Yes, with temperature=0.0 and fixed seed
- **CPU-friendly**: Optimized for CPU inference

This is our "hello world" for real Burn framework inference - minimal viable implementation with actual model loading and text generation.