# CUDA Kernel Benchmarking Implementation Plan

## Overview

This plan outlines different methodologies for benchmarking CUDA kernel performance across our candidate libraries (Candle, Burn, TensorRT-LLM, Custom implementations). We'll implement multiple approaches to get comprehensive performance insights.

## Benchmarking Methodologies

### 1. NVIDIA Profiling Tools Integration

#### 1.1 Nsight Compute (Kernel-Level Profiling)
```rust
// crates/cuda-profiler/src/nsight_compute.rs
use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NsightComputeMetrics {
    pub kernel_name: String,
    pub duration_ms: f64,
    pub achieved_occupancy: f64,
    pub theoretical_occupancy: f64,
    pub memory_throughput_gbps: f64,
    pub compute_throughput_percent: f64,
    pub tensor_core_utilization: f64,
    pub l1_cache_hit_rate: f64,
    pub l2_cache_hit_rate: f64,
    pub registers_per_thread: u32,
    pub shared_memory_per_block: u32,
}

pub struct NsightComputeProfiler {
    executable_path: String,
    output_dir: String,
}

impl NsightComputeProfiler {
    pub fn new(executable: &str, output_dir: &str) -> Self {
        Self {
            executable_path: executable.to_string(),
            output_dir: output_dir.to_string(),
        }
    }
    
    pub async fn profile_inference(&self, library: &str, model_path: &str, input: &str) -> Result<NsightComputeMetrics, ProfilerError> {
        let output_file = format!("{}/nsight_profile_{}_{}.csv", self.output_dir, library, chrono::Utc::now().timestamp());
        
        let output = Command::new("ncu")
            .args(&[
                "--csv",
                "--log-file", &output_file,
                "--metrics", "sm__cycles_elapsed.avg,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,gpu__time_duration.sum",
                "--target-processes", "all",
                &self.executable_path,
                "--library", library,
                "--model", model_path,
                "--input", input
            ])
            .output()
            .await
            .map_err(|e| ProfilerError::ExecutionFailed(e.to_string()))?;
            
        if !output.status.success() {
            return Err(ProfilerError::ProfilingFailed(String::from_utf8_lossy(&output.stderr).to_string()));
        }
        
        self.parse_nsight_output(&output_file).await
    }
    
    async fn parse_nsight_output(&self, file_path: &str) -> Result<NsightComputeMetrics, ProfilerError> {
        // Parse CSV output and extract metrics
        // Implementation details...
        todo!("Parse Nsight Compute CSV output")
    }
}
```

#### 1.2 Nsight Systems (Timeline Profiling)
```rust
// crates/cuda-profiler/src/nsight_systems.rs
#[derive(Debug, Serialize, Deserialize)]
pub struct NsightSystemsTrace {
    pub total_gpu_time_ms: f64,
    pub total_cpu_time_ms: f64,
    pub kernel_launches: Vec<KernelLaunch>,
    pub memory_transfers: Vec<MemoryTransfer>,
    pub cpu_gpu_overlap_percent: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KernelLaunch {
    pub name: String,
    pub start_time_us: u64,
    pub duration_us: u64,
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
}

pub struct NsightSystemsProfiler {
    executable_path: String,
}

impl NsightSystemsProfiler {
    pub async fn profile_full_pipeline(&self, library: &str, workload: &InferenceWorkload) -> Result<NsightSystemsTrace, ProfilerError> {
        let trace_file = format!("trace_{}_{}.nsys-rep", library, chrono::Utc::now().timestamp());
        
        let output = Command::new("nsys")
            .args(&[
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--output", &trace_file,
                "--force-overwrite=true",
                &self.executable_path,
                "--library", library,
                "--workload", &serde_json::to_string(workload).unwrap()
            ])
            .output()
            .await
            .map_err(|e| ProfilerError::ExecutionFailed(e.to_string()))?;
            
        self.parse_nsys_trace(&trace_file).await
    }
}
```

### 2. Custom CUDA Event-Based Timing

#### 2.1 High-Precision Kernel Timing
```rust
// crates/cuda-profiler/src/cuda_events.rs
use cudarc::driver::{CudaDevice, CudaEvent};
use std::collections::HashMap;

pub struct CudaEventProfiler {
    device: CudaDevice,
    event_pool: Vec<CudaEvent>,
    timing_results: HashMap<String, Vec<f32>>,
}

impl CudaEventProfiler {
    pub fn new(device: CudaDevice) -> Result<Self, CudaError> {
        let event_pool = (0..1000)
            .map(|_| device.create_event())
            .collect::<Result<Vec<_>, _>>()?;
            
        Ok(Self {
            device,
            event_pool,
            timing_results: HashMap::new(),
        })
    }
    
    pub fn start_timing(&mut self, kernel_name: &str) -> Result<TimingHandle, CudaError> {
        let start_event = self.event_pool.pop().ok_or(CudaError::OutOfEvents)?;
        let end_event = self.event_pool.pop().ok_or(CudaError::OutOfEvents)?;
        
        start_event.record(&self.device.default_stream())?;
        
        Ok(TimingHandle {
            kernel_name: kernel_name.to_string(),
            start_event,
            end_event,
            profiler: self,
        })
    }
}

pub struct TimingHandle<'a> {
    kernel_name: String,
    start_event: CudaEvent,
    end_event: CudaEvent,
    profiler: &'a mut CudaEventProfiler,
}

impl<'a> Drop for TimingHandle<'a> {
    fn drop(&mut self) {
        // Record end event and calculate timing
        self.end_event.record(&self.profiler.device.default_stream()).unwrap();
        self.profiler.device.synchronize().unwrap();
        
        let elapsed_ms = self.start_event.elapsed_time(&self.end_event).unwrap();
        
        self.profiler.timing_results
            .entry(self.kernel_name.clone())
            .or_insert_with(Vec::new)
            .push(elapsed_ms);
            
        // Return events to pool
        self.profiler.event_pool.push(self.start_event);
        self.profiler.event_pool.push(self.end_event);
    }
}
```

### 3. Memory Bandwidth Benchmarking

#### 3.1 Memory Throughput Analysis
```rust
// crates/cuda-profiler/src/memory_profiler.rs
#[derive(Debug)]
pub struct MemoryBandwidthBenchmark {
    device: CudaDevice,
    test_sizes: Vec<usize>, // Different tensor sizes to test
}

impl MemoryBandwidthBenchmark {
    pub async fn benchmark_library_memory_patterns(&self, library: &dyn InferenceEngine) -> MemoryProfileResult {
        let mut results = MemoryProfileResult::new();
        
        for &size in &self.test_sizes {
            let test_data = self.generate_test_tensor(size);
            
            // Measure memory allocation patterns
            let alloc_timing = self.measure_memory_allocation(library, size).await?;
            results.allocation_timings.insert(size, alloc_timing);
            
            // Measure transfer bandwidth
            let transfer_bw = self.measure_transfer_bandwidth(library, &test_data).await?;
            results.transfer_bandwidths.insert(size, transfer_bw);
            
            // Measure kernel memory access patterns
            let access_pattern = self.analyze_memory_access_pattern(library, &test_data).await?;
            results.access_patterns.insert(size, access_pattern);
        }
        
        results
    }
    
    async fn measure_memory_allocation(&self, library: &dyn InferenceEngine, size: usize) -> Result<AllocationTiming, ProfilerError> {
        let start = std::time::Instant::now();
        let _tensor = library.allocate_tensor(size).await?;
        let allocation_time = start.elapsed();
        
        let start = std::time::Instant::now();
        drop(_tensor);
        let deallocation_time = start.elapsed();
        
        Ok(AllocationTiming {
            allocation_time,
            deallocation_time,
            size_bytes: size * 4, // Assuming f32
        })
    }
}

#[derive(Debug)]
pub struct MemoryProfileResult {
    pub allocation_timings: HashMap<usize, AllocationTiming>,
    pub transfer_bandwidths: HashMap<usize, f64>, // GB/s
    pub access_patterns: HashMap<usize, MemoryAccessPattern>,
}
```

### 4. Comparative Kernel Performance Analysis

#### 4.1 Operation-Specific Benchmarks
```rust
// crates/cuda-profiler/src/operation_benchmarks.rs
#[derive(Debug, Clone)]
pub enum LLMOperation {
    MatrixMultiplication { m: usize, n: usize, k: usize },
    Attention { seq_len: usize, num_heads: usize, head_dim: usize },
    LayerNorm { hidden_size: usize },
    Activation { activation_type: ActivationType, size: usize },
    Embedding { vocab_size: usize, hidden_size: usize },
    Softmax { size: usize },
}

pub struct OperationBenchmarkSuite {
    operations: Vec<LLMOperation>,
    libraries: Vec<Box<dyn InferenceEngine>>,
    profiler: CudaEventProfiler,
}

impl OperationBenchmarkSuite {
    pub async fn benchmark_all_operations(&mut self) -> OperationComparisonReport {
        let mut results = HashMap::new();
        
        for operation in &self.operations {
            let mut op_results = HashMap::new();
            
            for library in &self.libraries {
                let performance = self.benchmark_operation(library.as_ref(), operation).await
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to benchmark {} for {:?}: {}", library.name(), operation, e);
                        OperationPerformance::default()
                    });
                    
                op_results.insert(library.name().to_string(), performance);
            }
            
            results.insert(operation.clone(), op_results);
        }
        
        OperationComparisonReport { results }
    }
    
    async fn benchmark_operation(&mut self, library: &dyn InferenceEngine, operation: &LLMOperation) -> Result<OperationPerformance, ProfilerError> {
        let test_data = self.generate_test_data_for_operation(operation)?;
        
        // Warmup runs
        for _ in 0..5 {
            library.execute_operation(operation, &test_data).await?;
        }
        
        // Actual benchmark runs
        let mut timings = Vec::new();
        let mut memory_usage = Vec::new();
        
        for _ in 0..100 {
            let memory_before = self.get_gpu_memory_usage()?;
            
            let timing_handle = self.profiler.start_timing(&format!("{}_{:?}", library.name(), operation))?;
            library.execute_operation(operation, &test_data).await?;
            drop(timing_handle);
            
            let memory_after = self.get_gpu_memory_usage()?;
            memory_usage.push(memory_after - memory_before);
        }
        
        // Calculate statistics
        let kernel_timings = self.profiler.timing_results.get(&format!("{}_{:?}", library.name(), operation))
            .ok_or(ProfilerError::MissingTimingData)?;
            
        Ok(OperationPerformance {
            mean_time_ms: statistical_mean(kernel_timings),
            std_dev_ms: statistical_std_dev(kernel_timings),
            min_time_ms: kernel_timings.iter().copied().fold(f32::INFINITY, f32::min),
            max_time_ms: kernel_timings.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            mean_memory_mb: statistical_mean(&memory_usage) / (1024.0 * 1024.0),
            throughput_gflops: self.calculate_theoretical_gflops(operation) / (statistical_mean(kernel_timings) / 1000.0),
        })
    }
}
```

### 5. Automated Benchmarking Pipeline

#### 5.1 Continuous Integration Benchmarking
```rust
// crates/cuda-profiler/src/benchmark_runner.rs
pub struct AutomatedBenchmarkRunner {
    config: BenchmarkConfig,
    profilers: Vec<Box<dyn KernelProfiler>>,
    libraries: Vec<LibraryConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub model_path: String,
    pub test_prompts: Vec<String>,
    pub batch_sizes: Vec<usize>,
    pub sequence_lengths: Vec<usize>,
    pub precision_modes: Vec<PrecisionMode>,
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub output_format: OutputFormat,
}

impl AutomatedBenchmarkRunner {
    pub async fn run_full_benchmark_suite(&self) -> Result<BenchmarkReport, BenchmarkError> {
        let mut report = BenchmarkReport::new();
        
        for library_config in &self.libraries {
            let library = self.load_library(library_config).await?;
            
            for &batch_size in &self.config.batch_sizes {
                for &seq_len in &self.config.sequence_lengths {
                    for precision in &self.config.precision_modes {
                        let test_config = TestConfiguration {
                            batch_size,
                            sequence_length: seq_len,
                            precision_mode: precision.clone(),
                        };
                        
                        let results = self.run_single_benchmark(&*library, &test_config).await?;
                        report.add_result(library_config.name.clone(), test_config, results);
                    }
                }
            }
        }
        
        Ok(report)
    }
    
    async fn run_single_benchmark(&self, library: &dyn InferenceEngine, config: &TestConfiguration) -> Result<BenchmarkResults, BenchmarkError> {
        let mut combined_results = BenchmarkResults::new();
        
        // Run all profilers in parallel
        let profiling_tasks: Vec<_> = self.profilers.iter()
            .map(|profiler| {
                let library_clone = library.clone_boxed();
                let config_clone = config.clone();
                tokio::spawn(async move {
                    profiler.profile_configuration(library_clone.as_ref(), &config_clone).await
                })
            })
            .collect();
            
        let profiling_results = futures::future::join_all(profiling_tasks).await;
        
        for result in profiling_results {
            match result? {
                Ok(profile_data) => combined_results.merge(profile_data),
                Err(e) => eprintln!("Profiling failed: {}", e),
            }
        }
        
        Ok(combined_results)
    }
}
```

### 6. Performance Regression Detection

#### 6.1 Historical Performance Tracking
```rust
// crates/cuda-profiler/src/regression_detection.rs
pub struct PerformanceRegressionDetector {
    historical_data: HistoricalPerformanceDatabase,
    alert_thresholds: RegressionThresholds,
}

#[derive(Debug)]
pub struct RegressionThresholds {
    pub performance_degradation_percent: f64, // Alert if >5% slower
    pub memory_increase_percent: f64,         // Alert if >10% more memory
    pub statistical_significance: f64,       // p-value threshold
}

impl PerformanceRegressionDetector {
    pub fn analyze_benchmark_results(&self, current_results: &BenchmarkReport, library: &str) -> RegressionAnalysis {
        let historical_baseline = self.historical_data.get_baseline_performance(library);
        
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        
        for (test_config, current_perf) in &current_results.results {
            if let Some(baseline_perf) = historical_baseline.get(test_config) {
                let performance_change = self.calculate_performance_delta(baseline_perf, current_perf);
                
                match performance_change {
                    PerformanceChange::Regression(severity) => {
                        regressions.push(RegressionAlert {
                            test_configuration: test_config.clone(),
                            severity,
                            current_performance: current_perf.clone(),
                            baseline_performance: baseline_perf.clone(),
                            change_description: self.describe_change(baseline_perf, current_perf),
                        });
                    }
                    PerformanceChange::Improvement(magnitude) => {
                        improvements.push(ImprovementReport {
                            test_configuration: test_config.clone(),
                            magnitude,
                            details: self.describe_change(baseline_perf, current_perf),
                        });
                    }
                    PerformanceChange::NoSignificantChange => {}
                }
            }
        }
        
        RegressionAnalysis {
            regressions,
            improvements,
            overall_assessment: self.assess_overall_performance_trend(&current_results, &historical_baseline),
        }
    }
}
```

## Implementation Timeline

### Week 1: Core Infrastructure
- [ ] CUDA event timing system
- [ ] Basic Nsight integration
- [ ] Memory profiling foundation
- [ ] Test data generation

### Week 2: Library Integration
- [ ] Candle profiling integration
- [ ] Burn/CubeCL profiling setup
- [ ] TensorRT-LLM profiling (if available)
- [ ] Custom implementation benchmarking

### Week 3: Advanced Profiling
- [ ] Operation-specific benchmarks
- [ ] Memory bandwidth analysis
- [ ] Kernel occupancy optimization
- [ ] Multi-batch performance scaling

### Week 4: Analysis & Reporting
- [ ] Automated report generation
- [ ] Performance regression detection
- [ ] Comparative analysis dashboard
- [ ] Production recommendations

This comprehensive benchmarking approach will give us detailed insights into CUDA kernel performance across different libraries and help identify the optimal approach for our inference engine.