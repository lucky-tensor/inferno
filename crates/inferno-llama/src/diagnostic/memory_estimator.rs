//! Memory usage estimation and optimization recommendations
//!
//! Calculates memory requirements for different model configurations
//! and provides optimization suggestions.

use super::*;
use candle_core::DType;

/// Memory usage estimator
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate memory layout for a model configuration
    pub fn estimate_memory(
        param_count: u64,
        primary_dtype: DType,
        quantization: &QuantizationConfig,
    ) -> ModelMemoryLayout {
        let bytes_per_param = Self::bytes_per_param(primary_dtype, quantization);
        let base_memory = param_count * bytes_per_param as u64;

        // Add overhead for activations, KV cache, etc. (rough estimate)
        let overhead_multiplier = 1.5; // 50% overhead
        let estimated_memory_bytes = (base_memory as f64 * overhead_multiplier) as u64;

        let mut layout = ModelMemoryLayout {
            total_params: param_count,
            primary_dtype,
            estimated_memory_bytes,
            ..Default::default()
        };

        // Add optimization hints based on memory usage
        Self::add_optimization_hints(&mut layout);

        layout
    }

    /// Calculate bytes per parameter for a given configuration
    fn bytes_per_param(dtype: DType, quantization: &QuantizationConfig) -> usize {
        let base_bytes = match dtype {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::U8 => 1,
            _ => 4, // Default to F32 size
        };

        // Apply quantization reduction
        let reduction_factor = quantization.scheme.memory_reduction_factor();
        ((base_bytes as f32) / reduction_factor).max(1.0) as usize
    }

    /// Add optimization hints based on memory layout
    fn add_optimization_hints(layout: &mut ModelMemoryLayout) {
        let memory_gb = layout.total_memory_gb();

        // Suggest quantization for large models
        if memory_gb > 16.0 && layout.primary_dtype != DType::U8 {
            layout
                .add_optimization_hint(OptimizationHint::UseQuantization(QuantizationScheme::W8A8));
        }

        // Suggest sharding for very large models
        if memory_gb > 32.0 && !layout.is_sharded {
            let suggested_shards = (memory_gb / 16.0).ceil() as usize;
            layout.add_optimization_hint(OptimizationHint::UseSharding(suggested_shards));
        }

        // Suggest flash attention for models with many attention heads
        layout.add_optimization_hint(OptimizationHint::UseFlashAttention);
    }
}
