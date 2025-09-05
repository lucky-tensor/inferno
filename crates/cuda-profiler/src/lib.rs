//! CUDA Kernel Performance Profiling Library
//! 
//! This crate provides comprehensive benchmarking and profiling capabilities
//! for CUDA kernels across different Rust inference libraries.

pub mod cuda_events;
pub mod memory_profiler;
pub mod nsight_compute;
pub mod nsight_systems;
pub mod operation_benchmarks;
pub mod benchmark_runner;
pub mod regression_detection;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProfilerError {
    #[error("CUDA operation failed: {0}")]
    CudaError(String),
    
    #[error("Profiling execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Profiling failed: {0}")]
    ProfilingFailed(String),
    
    #[error("Missing timing data")]
    MissingTimingData,
    
    #[error("Out of CUDA events")]
    OutOfEvents,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Common trait for all kernel profilers
#[async_trait::async_trait]
pub trait KernelProfiler: Send + Sync {
    async fn profile_configuration(
        &self, 
        library: &dyn InferenceEngine, 
        config: &TestConfiguration
    ) -> Result<ProfileData, ProfilerError>;
    
    fn profiler_name(&self) -> &str;
}

/// Common trait for inference engines being benchmarked
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn infer(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    
    async fn infer_batch(&self, prompts: Vec<&str>) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;
    
    async fn execute_operation(
        &self, 
        operation: &LLMOperation, 
        data: &TestData
    ) -> Result<TestOutput, Box<dyn std::error::Error + Send + Sync>>;
    
    async fn allocate_tensor(&self, size: usize) -> Result<Box<dyn TensorHandle>, Box<dyn std::error::Error + Send + Sync>>;
    
    fn name(&self) -> &str;
    
    fn clone_boxed(&self) -> Box<dyn InferenceEngine>;
}

pub trait TensorHandle: Send + Sync {
    fn size_bytes(&self) -> usize;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfiguration {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub precision_mode: PrecisionMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionMode {
    FP32,
    FP16,
    INT8,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMOperation {
    MatrixMultiplication { m: usize, n: usize, k: usize },
    Attention { seq_len: usize, num_heads: usize, head_dim: usize },
    LayerNorm { hidden_size: usize },
    Activation { activation_type: ActivationType, size: usize },
    Embedding { vocab_size: usize, hidden_size: usize },
    Softmax { size: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU,
    Tanh,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileData {
    pub kernel_timings: HashMap<String, Vec<f32>>,
    pub memory_usage: MemoryUsageStats,
    pub gpu_utilization: f64,
    pub occupancy_metrics: OccupancyMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_bandwidth_gbps: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OccupancyMetrics {
    pub achieved_occupancy: f64,
    pub theoretical_occupancy: f64,
    pub limiting_factor: String,
}

#[derive(Debug)]
pub struct TestData {
    pub input_tensors: Vec<Vec<f32>>,
    pub expected_shapes: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub struct TestOutput {
    pub output_tensors: Vec<Vec<f32>>,
    pub execution_time_ms: f32,
}

/// Statistical helper functions
pub fn statistical_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

pub fn statistical_std_dev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        0.0
    } else {
        let mean = statistical_mean(values);
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_statistical_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(statistical_mean(&data), 3.0);
        assert!((statistical_std_dev(&data) - 1.5811388).abs() < 0.0001);
    }
}