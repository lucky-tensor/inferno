//! High-precision CUDA event-based timing for kernel performance measurement

use crate::{ProfilerError, KernelProfiler, InferenceEngine, TestConfiguration, ProfileData};
use cudarc::driver::{CudaDevice, CudaEvent, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct CudaEventProfiler {
    device: Arc<CudaDevice>,
    event_pool: Arc<Mutex<Vec<CudaEvent>>>,
    timing_results: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    stream: CudaStream,
}

impl CudaEventProfiler {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, ProfilerError> {
        let event_pool: Result<Vec<CudaEvent>, _> = (0..1000)
            .map(|_| device.create_event())
            .collect();
        
        let event_pool = event_pool
            .map_err(|e| ProfilerError::CudaError(format!("Failed to create events: {:?}", e)))?;
        
        let stream = device.create_stream()
            .map_err(|e| ProfilerError::CudaError(format!("Failed to create stream: {:?}", e)))?;
        
        Ok(Self {
            device,
            event_pool: Arc::new(Mutex::new(event_pool)),
            timing_results: Arc::new(Mutex::new(HashMap::new())),
            stream,
        })
    }
    
    pub async fn time_kernel_execution<F, Fut>(&self, kernel_name: &str, kernel_fn: F) -> Result<f32, ProfilerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    {
        let mut event_pool = self.event_pool.lock().await;
        let start_event = event_pool.pop().ok_or(ProfilerError::OutOfEvents)?;
        let end_event = event_pool.pop().ok_or(ProfilerError::OutOfEvents)?;
        drop(event_pool);
        
        // Record start event
        start_event.record(&self.stream)
            .map_err(|e| ProfilerError::CudaError(format!("Failed to record start event: {:?}", e)))?;
        
        // Execute kernel
        kernel_fn().await
            .map_err(|e| ProfilerError::CudaError(format!("Kernel execution failed: {}", e)))?;
        
        // Record end event
        end_event.record(&self.stream)
            .map_err(|e| ProfilerError::CudaError(format!("Failed to record end event: {:?}", e)))?;
        
        // Synchronize and calculate elapsed time
        self.device.synchronize()
            .map_err(|e| ProfilerError::CudaError(format!("Failed to synchronize device: {:?}", e)))?;
        
        let elapsed_ms = start_event.elapsed_time(&end_event)
            .map_err(|e| ProfilerError::CudaError(format!("Failed to calculate elapsed time: {:?}", e)))?;
        
        // Store timing result
        let mut timing_results = self.timing_results.lock().await;
        timing_results.entry(kernel_name.to_string())
            .or_insert_with(Vec::new)
            .push(elapsed_ms);
        
        // Return events to pool
        let mut event_pool = self.event_pool.lock().await;
        event_pool.push(start_event);
        event_pool.push(end_event);
        
        Ok(elapsed_ms)
    }
    
    pub async fn get_timing_statistics(&self, kernel_name: &str) -> Option<KernelTimingStats> {
        let timing_results = self.timing_results.lock().await;
        let timings = timing_results.get(kernel_name)?;
        
        if timings.is_empty() {
            return None;
        }
        
        let mean = crate::statistical_mean(timings);
        let std_dev = crate::statistical_std_dev(timings);
        let min = timings.iter().copied().fold(f32::INFINITY, f32::min);
        let max = timings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        // Calculate percentiles
        let mut sorted_timings = timings.clone();
        sorted_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50 = percentile(&sorted_timings, 0.5);
        let p90 = percentile(&sorted_timings, 0.9);
        let p99 = percentile(&sorted_timings, 0.99);
        
        Some(KernelTimingStats {
            mean_ms: mean,
            std_dev_ms: std_dev,
            min_ms: min,
            max_ms: max,
            p50_ms: p50,
            p90_ms: p90,
            p99_ms: p99,
            sample_count: timings.len(),
        })
    }
    
    pub async fn clear_timing_results(&self) {
        let mut timing_results = self.timing_results.lock().await;
        timing_results.clear();
    }
    
    pub async fn benchmark_kernel_series<F, Fut>(&self, kernel_name: &str, iterations: usize, kernel_fn: F) -> Result<KernelBenchmarkResult, ProfilerError>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    {
        // Warmup iterations
        for _ in 0..5 {
            kernel_fn().await
                .map_err(|e| ProfilerError::CudaError(format!("Warmup iteration failed: {}", e)))?;
        }
        
        // Clear any previous timing data
        self.clear_timing_results().await;
        
        // Benchmark iterations
        for _ in 0..iterations {
            self.time_kernel_execution(kernel_name, || kernel_fn()).await?;
        }
        
        let stats = self.get_timing_statistics(kernel_name).await
            .ok_or(ProfilerError::MissingTimingData)?;
        
        Ok(KernelBenchmarkResult {
            kernel_name: kernel_name.to_string(),
            timing_stats: stats,
            iterations_completed: iterations,
        })
    }
}

#[async_trait::async_trait]
impl KernelProfiler for CudaEventProfiler {
    async fn profile_configuration(
        &self,
        library: &dyn InferenceEngine,
        config: &TestConfiguration,
    ) -> Result<ProfileData, ProfilerError> {
        let kernel_name = format!("{}_{}", library.name(), serde_json::to_string(config)?);
        
        // Generate test prompt based on configuration
        let test_prompt = generate_test_prompt(config.sequence_length);
        
        // Create benchmark closure
        let library_name = library.name().to_string();
        let prompt_clone = test_prompt.clone();
        
        let benchmark_result = self.benchmark_kernel_series(
            &kernel_name,
            100, // 100 iterations
            || {
                let prompt = prompt_clone.clone();
                let lib_name = library_name.clone();
                async move {
                    // This would need actual library inference call
                    // For now, simulate with a delay
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    Ok(())
                }
            }
        ).await?;
        
        // Convert to ProfileData
        let mut kernel_timings = HashMap::new();
        let timing_results = self.timing_results.lock().await;
        for (name, timings) in timing_results.iter() {
            kernel_timings.insert(name.clone(), timings.clone());
        }
        
        Ok(ProfileData {
            kernel_timings,
            memory_usage: crate::MemoryUsageStats {
                peak_memory_mb: 0.0, // Would be measured by actual profiler
                average_memory_mb: 0.0,
                memory_bandwidth_gbps: 0.0,
            },
            gpu_utilization: 0.0, // Would be measured by actual profiler
            occupancy_metrics: crate::OccupancyMetrics {
                achieved_occupancy: 0.0,
                theoretical_occupancy: 0.0,
                limiting_factor: "Unknown".to_string(),
            },
        })
    }
    
    fn profiler_name(&self) -> &str {
        "CudaEventProfiler"
    }
}

#[derive(Debug)]
pub struct KernelTimingStats {
    pub mean_ms: f32,
    pub std_dev_ms: f32,
    pub min_ms: f32,
    pub max_ms: f32,
    pub p50_ms: f32,
    pub p90_ms: f32,
    pub p99_ms: f32,
    pub sample_count: usize,
}

#[derive(Debug)]
pub struct KernelBenchmarkResult {
    pub kernel_name: String,
    pub timing_stats: KernelTimingStats,
    pub iterations_completed: usize,
}

fn percentile(sorted_values: &[f32], percentile: f64) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    
    let index = (percentile * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

fn generate_test_prompt(sequence_length: usize) -> String {
    // Generate a test prompt of approximately the desired sequence length
    let words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs"];
    let mut prompt = String::new();
    
    while prompt.len() < sequence_length {
        for word in &words {
            prompt.push_str(word);
            prompt.push(' ');
            if prompt.len() >= sequence_length {
                break;
            }
        }
    }
    
    prompt.truncate(sequence_length);
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.5), 3.0);
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
    }
    
    #[test]
    fn test_generate_test_prompt() {
        let prompt = generate_test_prompt(50);
        assert!(prompt.len() <= 50);
        assert!(!prompt.is_empty());
    }
}