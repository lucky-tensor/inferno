# CUDA Kernel Evaluation Plan

## Overview

This plan outlines how to systematically evaluate CUDA kernel performance across different Rust inference libraries to optimize GPU utilization for business inference workloads. Based on our product goals, we prioritize sustained throughput over cold-start performance.

## Libraries to Evaluate

### 1. Current Implementation (lm.rs)
- **CUDA Support**: None (CPU-only)
- **Kernel Type**: N/A
- **Purpose**: Baseline CPU performance reference

### 2. Candle Framework (Hugging Face)
- **CUDA Support**: Full CUDA support with cuDNN optimization
- **Kernel Type**: cuDNN optimized kernels + custom CUDA kernels
- **Backend**: `candle-cuda-backend`
- **Tensor Core**: Supported on compatible hardware

### 3. Burn Framework + CubeCL
- **CUDA Support**: CUDA backend through CubeCL with tensor core support
- **Kernel Type**: Pre-compiled CUDA kernels with runtime optimization (CubeCL abstractions)
- **Backend**: `burn::backend::CudaBackend`
- **Tensor Core**: Hardware-optimized tensor operations
- **Note**: Research needed to confirm actual JIT capabilities vs pre-compiled kernels

### 4. TensorRT-LLM (FFI Integration)
- **CUDA Support**: Industry-leading optimized CUDA kernels
- **Kernel Type**: Fused kernels, precision-optimized (FP16/INT8)
- **Backend**: Custom FFI to TensorRT-LLM C++ runtime
- **Tensor Core**: Full utilization with automatic optimization

### 5. Custom JIT-CUDA Strategy (Cloudflare-inspired)
- **CUDA Support**: Runtime-compiled kernels
- **Kernel Type**: JIT-compiled kernels optimized per workload
- **Backend**: Custom CUDA runtime with dynamic compilation
- **Tensor Core**: Adaptive kernel selection based on hardware

## CUDA Kernel Evaluation Criteria

### Primary Performance Metrics

#### 1. Computational Efficiency
- **FLOPS Utilization**: Percentage of theoretical peak FLOPS achieved
- **Tensor Core Utilization**: Specific tensor core usage metrics (A100/H100)
- **Memory Bandwidth**: Effective memory bandwidth vs theoretical peak
- **Kernel Occupancy**: GPU occupancy percentage per kernel launch

#### 2. Latency Characteristics
- **Kernel Launch Overhead**: Time from API call to kernel execution
- **Time to First Token (TTFT)**: Critical for interactive workloads
- **Per-token Latency**: Sustained inference latency per token
- **Batched Inference Latency**: Latency scaling with batch size

#### 3. Throughput Metrics
- **Tokens/Second**: Raw throughput for single and batched requests
- **Requests/Second**: Concurrent request handling capacity
- **Batch Efficiency**: Throughput scaling factor with batch size
- **Sustained Throughput**: Performance under continuous load

### Secondary Optimization Metrics

#### 4. Memory Efficiency
- **VRAM Usage**: Peak and sustained GPU memory consumption
- **Memory Pool Utilization**: Efficiency of GPU memory allocation
- **KV Cache Efficiency**: Attention cache memory patterns
- **Memory Bandwidth Utilization**: Actual vs theoretical bandwidth

#### 5. Kernel Optimization Features
- **Kernel Fusion**: Number of operations fused per kernel
- **Precision Optimization**: FP32/FP16/INT8 performance comparison
- **Dynamic Batching**: Adaptive batch size optimization
- **Multi-GPU Scaling**: Tensor/pipeline parallelism efficiency

## Benchmarking Methodology

### Phase 1: Kernel Profiling Setup

```rust
// CUDA profiling infrastructure
pub struct CudaKernelProfiler {
    nvprof: NvProfiler,
    nsight_compute: NsightCompute,
    custom_timers: HashMap<String, CudaTimer>,
}

impl CudaKernelProfiler {
    pub fn profile_inference(&self, engine: &dyn InferenceEngine, input: &str) -> KernelProfile {
        let start = CudaEvent::new();
        let end = CudaEvent::new();
        
        // Profile kernel execution
        start.record();
        let result = engine.infer(input);
        end.record();
        
        KernelProfile {
            execution_time: start.elapsed_time(&end),
            kernel_launches: self.nvprof.get_kernel_count(),
            memory_transfers: self.nvprof.get_memory_stats(),
            occupancy: self.nsight_compute.get_occupancy_metrics(),
            tensor_core_usage: self.nsight_compute.get_tensor_core_metrics(),
        }
    }
}
```

### Phase 2: Comparative Kernel Analysis

#### Test Matrix
| Library | Model | Batch Size | Precision | Hardware | Workload Type |
|---------|-------|------------|-----------|----------|---------------|
| Candle | Llama 3.2 1B | 1,4,16,32 | FP16/FP32 | RTX 4090 | Single/Batch |
| Burn | Llama 3.2 1B | 1,4,16,32 | FP16/FP32 | RTX 4090 | Single/Batch |
| TensorRT-LLM | Llama 3.2 1B | 1,4,16,32 | FP16/INT8 | RTX 4090 | Single/Batch |
| Custom JIT | Llama 3.2 1B | 1,4,16,32 | FP16/FP32 | RTX 4090 | Single/Batch |

#### Kernel-Specific Benchmarks

```rust
#[derive(Debug)]
pub struct KernelBenchmarkSuite {
    pub attention_kernels: AttentionKernelTests,
    pub matmul_kernels: MatMulKernelTests, 
    pub layer_norm_kernels: LayerNormKernelTests,
    pub activation_kernels: ActivationKernelTests,
    pub tokenization_kernels: TokenizationKernelTests,
}

impl KernelBenchmarkSuite {
    pub async fn benchmark_all_libraries(&self) -> LibraryComparisonReport {
        let libraries = vec![
            Box::new(CandleEngine::new()) as Box<dyn InferenceEngine>,
            Box::new(BurnEngine::new()) as Box<dyn InferenceEngine>,
            Box::new(TensorRTEngine::new()) as Box<dyn InferenceEngine>,
            Box::new(CustomJitEngine::new()) as Box<dyn InferenceEngine>,
        ];
        
        let mut results = HashMap::new();
        
        for library in libraries {
            results.insert(
                library.name(),
                self.benchmark_library(library).await
            );
        }
        
        LibraryComparisonReport::new(results)
    }
}
```

### Phase 3: Workload-Specific Evaluation

#### Business Inference Workloads
1. **Sustained High-Throughput**: Continuous batched inference (32+ requests)
2. **Interactive Chat**: Low-latency single request processing
3. **Document Processing**: Long-context inference workloads
4. **Code Generation**: Variable-length output generation

```rust
pub enum WorkloadType {
    SustainedHighThroughput { batch_size: usize, duration_minutes: u32 },
    InteractiveChat { max_latency_ms: u32, concurrent_users: usize },
    DocumentProcessing { context_length: usize, batch_size: usize },
    CodeGeneration { max_output_tokens: usize, creativity_temp: f32 },
}

pub struct WorkloadBenchmark {
    pub workload_type: WorkloadType,
    pub test_prompts: Vec<String>,
    pub success_criteria: PerformanceCriteria,
}
```

## Implementation Timeline

### Week 1: Infrastructure Setup
- [ ] CUDA profiling tools integration (nvprof, Nsight Compute)
- [ ] Benchmark harness implementation
- [ ] Test data generation (prompts, expected outputs)
- [ ] Hardware monitoring setup (GPU utilization, temperature, power)

### Week 2: Library Integration
- [ ] Candle CUDA backend integration and testing
- [ ] Burn CubeCL backend setup and validation
- [ ] TensorRT-LLM FFI bindings (proof of concept)
- [ ] Custom JIT-CUDA strategy prototype

### Week 3: Kernel Profiling
- [ ] Individual kernel performance profiling
- [ ] Memory access pattern analysis
- [ ] Tensor core utilization measurement
- [ ] Kernel fusion opportunity identification

### Week 4: Comparative Analysis
- [ ] Cross-library performance comparison
- [ ] Workload-specific optimization recommendations
- [ ] Production deployment strategy
- [ ] Cost-benefit analysis for each approach

## Expected Deliverables

### 1. Kernel Performance Report
```rust
pub struct KernelPerformanceReport {
    pub library_rankings: Vec<LibraryRanking>,
    pub kernel_specific_winners: HashMap<KernelType, LibraryName>,
    pub workload_recommendations: HashMap<WorkloadType, LibraryName>,
    pub optimization_opportunities: Vec<OptimizationRecommendation>,
}
```

### 2. Production Recommendations
- **Primary Library Choice**: Data-driven recommendation for production
- **Hybrid Strategy**: Using different libraries for different workloads
- **Migration Path**: Step-by-step transition from current CPU implementation
- **Performance Targets**: Specific metrics for success criteria

### 3. Optimization Roadmap
- **Short-term**: Immediate kernel optimizations (1-2 months)
- **Medium-term**: Advanced batching and memory optimization (3-6 months)  
- **Long-term**: Custom kernel development and multi-GPU scaling (6+ months)

## Success Criteria

### Performance Targets (vs current CPU baseline)
- **Throughput**: 5-10x improvement in tokens/second
- **Latency**: <100ms time-to-first-token for interactive workloads
- **Efficiency**: >70% GPU utilization under sustained load
- **Cost**: 50%+ reduction in $/token through better GPU utilization

### Technical Validation
- **Kernel Occupancy**: >75% occupancy on target hardware
- **Memory Bandwidth**: >80% of theoretical peak utilization
- **Tensor Core Usage**: >60% utilization on compatible operations
- **Multi-batch Scaling**: Linear throughput scaling up to hardware limits

## Risk Mitigation

### Technical Risks
- **Library Maturity**: Burn framework still in development
- **Integration Complexity**: TensorRT-LLM FFI may be complex
- **Hardware Dependencies**: CUDA-specific optimizations limit portability

### Mitigation Strategies
- **Parallel Development**: Implement multiple approaches simultaneously
- **Fallback Strategy**: Maintain CPU implementation as backup
- **Incremental Migration**: Gradual transition with A/B testing
- **Vendor Diversification**: Evaluate both NVIDIA and AMD solutions

This evaluation plan will provide data-driven insights for optimizing CUDA kernel performance across different Rust inference libraries, aligned with our product goals of maximizing GPU utilization for business inference workloads.