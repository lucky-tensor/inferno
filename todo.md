# Inference Crate Cleanup and Refactoring Tasks

## Summary
The latest commit (f9205ce) successfully demonstrates SmolLM2 inference working with Burn framework. Now we need to consolidate and clean up the redundant demo/test files.

## Test Files Analysis

### Keep (Working Implementation)
- **`tests/real_burn_inference.rs`** - KEEP
  - Contains the complete SmolLM2 implementation with proper architecture
  - Has attention, MLP, layernorm, and full transformer layers
  - Successfully demonstrates inference with tokenizer integration
  - Tests: `test_weight_loading_only`, `test_real_smollm2_inference`, `test_complete_inference_architecture`, `test_model_components`

### Remove (Redundant/Demo Files)
1. **`tests/burn_clean_demo.rs`** - REMOVE
   - Simple demo showing Burn abstractions vs manual implementation
   - Redundant now that we have full SmolLM2 working

2. **`tests/burn_proper_inference.rs`** - REMOVE  
   - Intermediate demo with SimpleLanguageModel
   - Superseded by real_burn_inference.rs

3. **`tests/minimal_hello_world.rs`** - REMOVE
   - Manual SafeTensors loading demo
   - Not using proper Burn abstractions
   - Functionality covered in real_burn_inference.rs

4. **`tests/real_inference_tests.rs`** - REVIEW
   - Tests for VLLM/Llama integration
   - May be needed for different model support
   - Check if still relevant to project goals

5. **`tests/integration_basic.rs`** - KEEP
   - Basic integration tests for VLLM backend
   - Tests config creation and validation
   - Useful for CI/CD

## Source Files Analysis

### Inference Module (`src/inference/`)
1. **`burn_engine.rs`** - REVIEW
   - Complex Burn engine implementation
   - Check if needed alongside SmolLM2

2. **`burn_engine_simple.rs`** - REMOVE
   - Simplified version, likely redundant

3. **`burn_hello_world.rs`** - REMOVE  
   - Demo/tutorial code, not needed in production

4. **`mod.rs`** - KEEP
   - Module definitions and exports

### Empty Directories
- **`src/models/llama/`** - REMOVE or IMPLEMENT
  - Currently empty, either implement or remove

## Refactoring Tasks

### 1. Consolidate SmolLM2 Implementation
- [ ] Move SmolLM2 model from `tests/real_burn_inference.rs` to proper location in `src/`
- [ ] Create `src/models/smollm2/mod.rs` with:
  - `SmolLM2Config`
  - `SmolLM2Model` 
  - `SmolLM2Attention`
  - `SmolLM2MLP`
  - `SmolLM2Layer`

### 2. Clean Test Structure
- [ ] Remove redundant test files (burn_clean_demo.rs, burn_proper_inference.rs, minimal_hello_world.rs)
- [ ] Keep only working tests in real_burn_inference.rs
- [ ] Move integration tests to proper structure

### 3. Update Module Exports
- [ ] Update `src/lib.rs` to export SmolLM2 model
- [ ] Clean up unused exports from removed files
- [ ] Add proper documentation

### 4. SafeTensors Weight Loading
- [ ] Implement proper weight loading with burn-import
- [ ] Add support for loading from HuggingFace format
- [ ] Create weight conversion utilities if needed

### 5. Inference API
- [ ] Create clean inference API using SmolLM2
- [ ] Implement proper batching support
- [ ] Add streaming generation support

### 6. Documentation
- [ ] Update README with SmolLM2 usage examples
- [ ] Document model architecture
- [ ] Add inference benchmarks

## Priority Order
1. **High Priority**: Remove redundant test files
2. **High Priority**: Move SmolLM2 to src/models/
3. **Medium Priority**: Clean up src/inference/ module
4. **Medium Priority**: Implement proper weight loading
5. **Low Priority**: Documentation and benchmarks

## Commands to Execute
```bash
# Remove redundant test files
git rm crates/inference/tests/burn_clean_demo.rs
git rm crates/inference/tests/burn_proper_inference.rs
git rm crates/inference/tests/minimal_hello_world.rs

# Remove redundant source files
git rm crates/inference/src/inference/burn_engine_simple.rs
git rm crates/inference/src/inference/burn_hello_world.rs

# Create new model structure
mkdir -p crates/inference/src/models/smollm2
```

## Notes
- The `real_burn_inference.rs` test successfully demonstrates:
  - Tokenizer integration
  - Model architecture (embedding, attention, MLP, layernorm)
  - Forward pass through 30 transformer layers
  - Next token prediction
  - SafeTensors weight loading structure (needs full implementation)
- Current implementation uses random weights for demo, needs actual weight loading
- Consider keeping VLLM infrastructure if planning to support multiple model types