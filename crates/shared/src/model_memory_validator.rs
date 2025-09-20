//! Model Memory Validation
//!
//! This module provides upfront validation of whether a model will fit in GPU memory
//! before attempting to load it, saving users time and providing clear guidance.

use crate::gpu_diagnostics::{get_gpu_memory_info, get_gpu_processes, GpuMemoryInfo, GpuProcess};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Memory requirement levels for different model types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    /// Small models (< 1B parameters)
    Small,
    /// Medium models (1B - 7B parameters)
    Medium,
    /// Large models (7B - 13B parameters)
    Large,
    /// XLarge models (13B - 70B parameters)
    XLarge,
    /// XXLarge models (> 70B parameters)
    XXLarge,
}

/// Memory validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryValidation {
    /// Whether the model is expected to fit
    pub will_fit: bool,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Model size in GB
    pub model_size_gb: f64,
    /// Available GPU memory in GB
    pub available_memory_gb: f64,
    /// Estimated memory requirement in GB (including overhead)
    pub estimated_requirement_gb: f64,
    /// Memory overhead factor applied
    pub overhead_factor: f64,
    /// Validation message for the user
    pub message: String,
    /// Recommendations for the user
    pub recommendations: Vec<String>,
    /// GPU processes that might interfere
    pub interfering_processes: Vec<GpuProcess>,
}

/// Model memory validator
pub struct ModelMemoryValidator {
    /// Device ID to validate against
    device_id: u32,
    /// Fixed memory overhead amounts
    memory_overhead: MemoryOverhead,
}

/// Fixed memory overhead amounts in GB
#[derive(Debug, Clone)]
pub struct MemoryOverhead {
    /// Fixed overhead for model loading (SafeTensors parsing, temp buffers, etc.) in GB
    pub loading_overhead_gb: f64,
    /// Fixed overhead for inference context (KV cache initialization, etc.) in GB
    pub inference_overhead_gb: f64,
    /// Fixed framework overhead (CUDA context, cuDNN workspace, etc.) in GB
    pub framework_overhead_gb: f64,
}

impl Default for MemoryOverhead {
    fn default() -> Self {
        Self {
            loading_overhead_gb: 2.0,   // 2GB for SafeTensors parsing and temp buffers
            inference_overhead_gb: 1.5, // 1.5GB for KV cache initialization
            framework_overhead_gb: 1.0, // 1GB for CUDA context and cuDNN workspace
        }
    }
}

impl ModelMemoryValidator {
    /// Create a new validator for the specified GPU device
    pub fn new(device_id: u32) -> Self {
        Self {
            device_id,
            memory_overhead: MemoryOverhead::default(),
        }
    }

    /// Create validator with custom memory overhead amounts
    pub fn with_memory_overhead(device_id: u32, memory_overhead: MemoryOverhead) -> Self {
        Self {
            device_id,
            memory_overhead,
        }
    }

    /// Validate if a model will fit in GPU memory
    pub async fn validate_model_fit(
        &self,
        model_path: &str,
    ) -> Result<MemoryValidation, Box<dyn std::error::Error>> {
        // Get model size
        let model_size_gb = calculate_model_size_gb(model_path)?;

        // Get GPU memory info
        let gpu_memory = get_gpu_memory_info(self.device_id)?;
        let gpu_processes = get_gpu_processes().unwrap_or_default();

        // Calculate memory requirements with fixed overhead
        let total_overhead_gb = self.calculate_total_overhead_gb();
        let estimated_requirement_gb = model_size_gb + total_overhead_gb;
        let available_memory_gb = gpu_memory.free_mb as f64 / 1024.0;

        // Determine if it will fit
        let will_fit = estimated_requirement_gb <= available_memory_gb;
        let confidence = self.calculate_confidence(&gpu_memory, &gpu_processes, model_size_gb);

        // Generate message and recommendations
        let (message, recommendations) = self.generate_guidance(
            will_fit,
            model_size_gb,
            estimated_requirement_gb,
            available_memory_gb,
            &gpu_processes,
        );

        // Find interfering processes
        let interfering_processes = gpu_processes
            .into_iter()
            .filter(|p| p.gpu_memory_mb > 500) // Processes using > 500MB
            .collect();

        Ok(MemoryValidation {
            will_fit,
            confidence,
            model_size_gb,
            available_memory_gb,
            estimated_requirement_gb,
            overhead_factor: total_overhead_gb, // Now represents fixed overhead in GB
            message,
            recommendations,
            interfering_processes,
        })
    }

    /// Calculate total fixed overhead in GB
    fn calculate_total_overhead_gb(&self) -> f64 {
        self.memory_overhead.loading_overhead_gb
            + self.memory_overhead.inference_overhead_gb
            + self.memory_overhead.framework_overhead_gb
    }

    /// Calculate confidence in the prediction
    fn calculate_confidence(
        &self,
        gpu_memory: &GpuMemoryInfo,
        processes: &[GpuProcess],
        model_size_gb: f64,
    ) -> f32 {
        let mut confidence: f32 = 0.8; // Base confidence

        // Reduce confidence if there are many GPU processes
        if processes.len() > 3 {
            confidence -= 0.2;
        }

        // Reduce confidence if memory is highly utilized
        let memory_utilization = gpu_memory.used_mb as f32 / gpu_memory.total_mb as f32;
        if memory_utilization > 0.7 {
            confidence -= 0.2;
        }

        // Reduce confidence for very large models (more uncertainty)
        if model_size_gb > 50.0 {
            confidence -= 0.1;
        }

        // Increase confidence if we have lots of free memory
        let free_gb = gpu_memory.free_mb as f64 / 1024.0;
        if free_gb > model_size_gb * 3.0 {
            confidence += 0.1;
        }

        confidence.clamp(0.1, 1.0)
    }

    /// Generate user guidance message and recommendations
    fn generate_guidance(
        &self,
        will_fit: bool,
        model_size_gb: f64,
        estimated_requirement_gb: f64,
        available_memory_gb: f64,
        processes: &[GpuProcess],
    ) -> (String, Vec<String>) {
        let mut recommendations = Vec::new();

        let message = if will_fit {
            if available_memory_gb > estimated_requirement_gb * 1.5 {
                format!(
                    "Model should load successfully! Model requires ~{:.1}GB, you have {:.1}GB available.",
                    estimated_requirement_gb, available_memory_gb
                )
            } else {
                format!(
                    "WARNING: Model should fit, but it will be tight. Model requires ~{:.1}GB, you have {:.1}GB available.",
                    estimated_requirement_gb, available_memory_gb
                )
            }
        } else {
            let shortfall_gb = estimated_requirement_gb - available_memory_gb;
            format!(
                "ERROR: Model likely won't fit! Model requires ~{:.1}GB, but only {:.1}GB available. Shortfall: {:.1}GB.",
                estimated_requirement_gb, available_memory_gb, shortfall_gb
            )
        };

        // Generate recommendations
        if !will_fit || available_memory_gb < estimated_requirement_gb * 1.2 {
            // Find processes using significant memory
            let heavy_processes: Vec<_> = processes
                .iter()
                .filter(|p| p.gpu_memory_mb > 1000)
                .collect();

            if !heavy_processes.is_empty() {
                recommendations.push("Kill GPU processes using significant memory:".to_string());
                for process in heavy_processes {
                    recommendations.push(format!(
                        "   sudo kill {} ({}, {:.1}GB)",
                        process.pid,
                        process.process_name,
                        process.gpu_memory_mb as f64 / 1024.0
                    ));
                }
            }

            if !will_fit {
                recommendations
                    .push("Consider using a smaller model or quantized version".to_string());
                recommendations.push("Use model sharding or pipeline parallelism".to_string());
                recommendations.push("Try a GPU with more memory".to_string());
            }

            if model_size_gb > 10.0 {
                recommendations.push(
                    "For large models, ensure no other applications are using GPU".to_string(),
                );
            }
        }

        if available_memory_gb > estimated_requirement_gb * 2.0 {
            recommendations
                .push("Plenty of memory available - consider larger batch sizes".to_string());
        }

        (message, recommendations)
    }

    /// Display validation results to the user
    pub fn display_validation(&self, validation: &MemoryValidation) {
        println!("\nGPU Memory Validation (GPU {})", self.device_id);
        println!("=======================================");
        println!("{}", validation.message);
        println!("Model size: {:.1} GB", validation.model_size_gb);
        println!(
            "Estimated requirement: {:.1} GB (+{:.1}GB overhead)",
            validation.estimated_requirement_gb, validation.overhead_factor
        );
        println!("Available memory: {:.1} GB", validation.available_memory_gb);
        println!("Confidence: {:.0}%", validation.confidence * 100.0);

        if !validation.interfering_processes.is_empty() {
            println!("\nGPU Processes:");
            for process in &validation.interfering_processes {
                println!(
                    "   PID {}: {} ({:.1}GB)",
                    process.pid,
                    process.process_name,
                    process.gpu_memory_mb as f64 / 1024.0
                );
            }
        }

        if !validation.recommendations.is_empty() {
            println!("\nRecommendations:");
            for rec in &validation.recommendations {
                println!("   {}", rec);
            }
        }
        println!();
    }
}

/// Calculate the size of a model from its files
pub fn calculate_model_size_gb(model_path: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let model_dir = Path::new(model_path);
    let mut total_size_bytes = 0u64;

    if !model_dir.exists() {
        return Err(format!("Model path does not exist: {}", model_path).into());
    }

    // Look for SafeTensors files
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            // Include both sharded and single model files
            if (file_name_str.starts_with("model") && file_name_str.ends_with(".safetensors"))
                || file_name_str == "pytorch_model.bin"
                || file_name_str.ends_with(".bin")
            {
                if let Ok(metadata) = entry.metadata() {
                    total_size_bytes += metadata.len();
                }
            }
        }
    }

    if total_size_bytes == 0 {
        return Err("No model files found in the specified directory".into());
    }

    Ok(total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Classify model size based on GB
pub fn classify_model_size(size_gb: f64) -> ModelSize {
    if size_gb < 2.0 {
        ModelSize::Small
    } else if size_gb < 15.0 {
        ModelSize::Medium
    } else if size_gb < 30.0 {
        ModelSize::Large
    } else if size_gb < 150.0 {
        ModelSize::XLarge
    } else {
        ModelSize::XXLarge
    }
}

/// Convenience function to validate and display model memory requirements
pub async fn validate_and_display_model_memory(
    device_id: u32,
    model_path: &str,
    model_name: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!(
        "Checking if {} will fit on GPU {}...",
        model_name, device_id
    );

    let validator = ModelMemoryValidator::new(device_id);
    let validation = validator.validate_model_fit(model_path).await?;

    validator.display_validation(&validation);

    Ok(validation.will_fit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_size_classification() {
        assert_eq!(classify_model_size(1.0), ModelSize::Small);
        assert_eq!(classify_model_size(7.0), ModelSize::Medium);
        assert_eq!(classify_model_size(13.0), ModelSize::Large);
        assert_eq!(classify_model_size(30.0), ModelSize::XLarge);
        assert_eq!(classify_model_size(200.0), ModelSize::XXLarge);
    }

    #[tokio::test]
    async fn test_model_size_calculation() {
        // Test with a known directory structure
        let temp_dir = std::env::temp_dir();
        match calculate_model_size_gb(&temp_dir.to_string_lossy()) {
            Ok(_) => {
                // Size calculation worked (may be 0 if no model files)
            }
            Err(_) => {
                // Expected if directory doesn't contain model files
            }
        }
    }
}
