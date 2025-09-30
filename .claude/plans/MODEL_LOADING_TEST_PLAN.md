# Model Loading Test Coverage Plan
## Engineering Analysis & Implementation Roadmap

**Date:** September 20, 2025
**Focus:** Complete unit test and end-to-end test coverage for model loading only
**Scope:** Based on GENERIC_LLAMA_SPECIFICATION requirements

---

## ✅ Current Status Analysis

### Completed Infrastructure
- ✅ **Weight Name Mapping**: 11 comprehensive unit tests, all passing (1.49μs per tensor)
- ✅ **UnifiedModelFactory**: Model detection working for quantized and standard models
- ✅ **WeightAnalyzer**: Correctly identifies dtypes, quantization schemes, sharding
- ✅ **SafeTensors Integration**: Real weight loading (no placeholders)
- ✅ **Configuration Parsing**: Handles multiple model variants correctly

### Current Blocker
❌ **VarBuilder Name Mapping Issue**: SafeTensors loads with HuggingFace names, model expects InfernoLlama names

---

## 🎯 Critical Issue: VarBuilder Name Mapping

The fundamental blocker preventing model loading is the name mapping between SafeTensors and InfernoLlama:

- **SafeTensors contains**: `model.embed_tokens.weight`
- **InfernoLlama expects**: `embed_tokens.weight`
- **Current VarBuilder**: Loads raw SafeTensors names without mapping

**Solution Required**: Custom VarBuilder wrapper that applies name mapping during tensor retrieval.

---

## 📋 Model Loading Test Plan

### Priority 1: Fix VarBuilder Name Mapping (Day 1)
**Critical Path Item**

**Implementation Approach:**
1. Create `MappedVarBuilder` wrapper around standard VarBuilder
2. Override tensor retrieval methods to apply name mapping
3. Use existing `map_weight_name` function for transformations
4. Test with actual SafeTensors loading

**Success Criteria:**
- [ ] Meta Llama 3.2 quantized model loads without "cannot find tensor" errors
- [ ] Weight mapping applied correctly during VarBuilder tensor access
- [ ] No performance regression in tensor loading

### Priority 2: SafeTensors Loading Unit Tests (Day 2)
**File**: `test_safetensors_loading.rs` (enhance existing)

**Required Tests:**
- [ ] `test_single_safetensors_direct_loading` - Load model.safetensors directly
- [ ] `test_sharded_safetensors_loading` - Load from multiple files
- [ ] `test_quantized_safetensors_dtypes` - Validate I8/BF16 tensor loading
- [ ] `test_safetensors_name_mapping_integration` - Ensure mapping works
- [ ] `test_safetensors_memory_efficiency` - Validate memory usage
- [ ] `test_safetensors_error_handling` - Missing files, corrupted data

### Priority 3: End-to-End Model Loading Tests (Days 3-4)
**File**: `test_e2e_model_loading.rs` (enhance existing)

**Real Model Testing:**
- [ ] `test_load_meta_llama_31_8b_complete` - Full pipeline with ~/models/meta-llama_Llama-3.1-8B-Instruct
- [ ] `test_load_quantized_w8a8_complete` - Full pipeline with ~/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8
- [ ] `test_load_tinyllama_complete` - Full pipeline with ~/models/tinyllama-1.1b
- [ ] `test_parameter_count_validation` - Verify loaded model parameter counts match expectations
- [ ] `test_memory_usage_validation` - Compare predicted vs actual memory usage

### Priority 4: ModelLoader Integration Tests (Day 5)
**File**: `test_model_loader_integration.rs` (new)

**Integration Testing:**
- [ ] `test_model_loader_full_pipeline` - Config → Weights → Model creation
- [ ] `test_dtype_handling_consistency` - Ensure dtypes match throughout pipeline
- [ ] `test_error_propagation` - Validate error messages and handling
- [ ] `test_performance_benchmarking` - Compare against existing implementation

---

## 🧪 Test Data Strategy

### Real Model Files Available
- **meta-llama_Llama-3.1-8B-Instruct**: BF16, potentially sharded, ~16GB
- **RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8**: W8A8, single file, ~2GB, 259 tensors
- **tinyllama-1.1b**: F16, unknown sharding, ~2.2GB

### Test Approach
- **Unit Tests**: Use synthetic small SafeTensors files for fast execution
- **Integration Tests**: Use real models but with parameter validation, not full inference
- **Performance Tests**: Full model loading with memory and timing measurements

---

## 🔧 Implementation Plan

### Day 1: Critical VarBuilder Fix
1. **Morning**: Analyze VarBuilder API and create MappedVarBuilder wrapper
2. **Afternoon**: Implement name mapping integration
3. **Evening**: Test with one model type (quantized 1B model - smallest)

### Day 2: SafeTensors Unit Tests
1. **Morning**: Enhance existing test_safetensors_loading.rs
2. **Afternoon**: Add quantized model specific tests
3. **Evening**: Add error handling and edge cases

### Day 3-4: End-to-End Testing
1. **Day 3 Morning**: Set up E2E test infrastructure
2. **Day 3 Afternoon**: Meta Llama 3.1 8B model loading
3. **Day 4 Morning**: TinyLlama model loading
4. **Day 4 Afternoon**: Parameter count and memory validation

### Day 5: Integration & Performance
1. **Morning**: ModelLoader integration tests
2. **Afternoon**: Performance benchmarking vs existing implementation
3. **Evening**: Error handling and regression testing

---

## 📊 Success Criteria

### Must-Have (End of Week)
- [ ] **At least 1 model type loads completely** without errors
- [ ] **VarBuilder name mapping working** for all supported models
- [ ] **90%+ test coverage** for model loading components
- [ ] **Memory usage validation** matches WeightAnalyzer predictions

### Should-Have (Stretch Goals)
- [ ] **All 3 model types loading** (Meta Llama, TinyLlama, Quantized)
- [ ] **Performance parity** with existing implementation
- [ ] **Comprehensive error handling** with clear messages
- [ ] **Regression test suite** for future changes

---

## 🚨 Risk Analysis

### Critical Risks
1. **VarBuilder Wrapper Complexity**: May require deep Candle API understanding
   - **Mitigation**: Start with simplest possible implementation, iterate
2. **Memory Constraints**: Large models may exceed system memory during testing
   - **Mitigation**: Use parameter count validation instead of full loading for large models
3. **Hardware Dependencies**: Quantized model loading may require specific hardware
   - **Mitigation**: Graceful fallbacks and clear error messages

### Technical Dependencies
1. **Candle VarBuilder API**: Must support wrapping/delegation patterns
2. **SafeTensors Format**: Must handle both single-file and sharded models
3. **Model File Access**: Requires ~/models/ directory with test files

---

This focused plan addresses the critical VarBuilder name mapping blocker while building comprehensive test coverage for model loading. The 5-day timeline prioritizes getting at least one model type working end-to-end before expanding coverage.