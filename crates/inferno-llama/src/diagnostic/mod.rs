//! Model diagnostic and auto-detection system
//!
//! This module provides comprehensive model analysis capabilities including:
//! - Automatic detection of Llama model variants
//! - Quantization scheme analysis
//! - Memory layout optimization
//! - Configuration parsing across different formats

pub mod config_parser;
pub mod detector;
pub mod memory_estimator;
pub mod weight_analyzer;

// Re-export key types
pub use config_parser::ConfigParser;
pub use detector::ModelDetector;
pub use memory_estimator::MemoryEstimator;
pub use weight_analyzer::{WeightAnalysisResult, WeightAnalyzer};

use candle_core::DType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantization schemes supported by the generic engine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// No quantization - full precision
    None,
    /// 8-bit weights, 8-bit activations
    W8A8,
    /// 4-bit weights, 16-bit activations
    W4A16,
    /// Compressed tensors format (various schemes)
    CompressedTensors(String),
    /// Custom quantization scheme
    Custom(String),
}

impl std::fmt::Display for QuantizationScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationScheme::None => write!(f, "None"),
            QuantizationScheme::W8A8 => write!(f, "W8A8"),
            QuantizationScheme::W4A16 => write!(f, "W4A16"),
            QuantizationScheme::CompressedTensors(scheme) => write!(f, "CompressedTensors({})", scheme),
            QuantizationScheme::Custom(scheme) => write!(f, "Custom({})", scheme),
        }
    }
}

/// Quantization configuration metadata
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// The quantization scheme being used
    pub scheme: QuantizationScheme,
    /// Per-layer quantization settings if available
    pub per_layer_config: Option<HashMap<String, LayerQuantizationConfig>>,
    /// Global quantization parameters
    pub global_params: Option<QuantizationParams>,
    /// Whether symmetric or asymmetric quantization is used
    pub symmetric: bool,
    /// Zero-point for asymmetric quantization
    pub zero_point: Option<i32>,
    /// Scale factor for quantization
    pub scale: Option<f32>,
}

/// Per-layer quantization configuration
#[derive(Debug, Clone)]
pub struct LayerQuantizationConfig {
    /// Data type for this layer
    pub dtype: DType,
    /// Layer-specific quantization scheme
    pub scheme: QuantizationScheme,
    /// Layer-specific parameters
    pub params: Option<QuantizationParams>,
}

/// Quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Quantization bits
    pub bits: u8,
    /// Group size for grouped quantization
    pub group_size: Option<usize>,
    /// Block size for block-wise quantization
    pub block_size: Option<usize>,
    /// Custom parameters for specific schemes
    pub custom_params: Option<HashMap<String, serde_json::Value>>,
}

/// Model memory layout and optimization information
#[derive(Debug, Clone)]
pub struct ModelMemoryLayout {
    /// Total number of parameters
    pub total_params: u64,
    /// Primary data type used by the model
    pub primary_dtype: DType,
    /// Whether the model is sharded across multiple files
    pub is_sharded: bool,
    /// Number of shards if sharded
    pub num_shards: usize,
    /// Estimated memory requirement in bytes
    pub estimated_memory_bytes: u64,
    /// Per-layer memory breakdown
    pub layer_memory_map: HashMap<String, LayerMemoryInfo>,
    /// Optimization recommendations
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Memory information for a specific layer
#[derive(Debug, Clone)]
pub struct LayerMemoryInfo {
    /// Number of parameters in this layer
    pub param_count: u64,
    /// Data type used by this layer
    pub dtype: DType,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Whether this layer is quantized
    pub is_quantized: bool,
}

/// Memory optimization recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationHint {
    /// Suggest using quantization to reduce memory
    UseQuantization(QuantizationScheme),
    /// Suggest model sharding for large models
    UseSharding(usize),
    /// Suggest using a different data type
    UseDifferentDType(DType),
    /// Suggest enabling flash attention
    UseFlashAttention,
    /// Suggest CPU offloading for some layers
    OffloadToCpu(Vec<String>),
    /// Custom optimization hint
    Custom(String),
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::None,
            per_layer_config: None,
            global_params: None,
            symmetric: true,
            zero_point: None,
            scale: None,
        }
    }
}

impl Default for ModelMemoryLayout {
    fn default() -> Self {
        Self {
            total_params: 0,
            primary_dtype: DType::F32,
            is_sharded: false,
            num_shards: 1,
            estimated_memory_bytes: 0,
            layer_memory_map: HashMap::new(),
            optimization_hints: Vec::new(),
        }
    }
}

impl QuantizationScheme {
    /// Check if this quantization scheme is supported
    pub fn is_supported(&self) -> bool {
        match self {
            QuantizationScheme::None => true,
            QuantizationScheme::W8A8 => true,
            QuantizationScheme::W4A16 => false, // TODO: Implement W4A16 support
            QuantizationScheme::CompressedTensors(_) => false, // TODO: Implement compressed tensors
            QuantizationScheme::Custom(_) => false,
        }
    }

    /// Get the typical memory reduction factor for this quantization scheme
    pub fn memory_reduction_factor(&self) -> f32 {
        match self {
            QuantizationScheme::None => 1.0,
            QuantizationScheme::W8A8 => 2.0, // Roughly 2x reduction from FP16
            QuantizationScheme::W4A16 => 4.0, // Roughly 4x reduction from FP16
            QuantizationScheme::CompressedTensors(_) => 2.0, // Conservative estimate
            QuantizationScheme::Custom(_) => 1.0, // Unknown, assume no reduction
        }
    }
}

impl ModelMemoryLayout {
    /// Calculate total memory usage in GB for human-readable output
    pub fn total_memory_gb(&self) -> f64 {
        self.estimated_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if the model fits in a given memory budget
    pub fn fits_in_memory(&self, available_bytes: u64) -> bool {
        self.estimated_memory_bytes <= available_bytes
    }

    /// Add an optimization hint if not already present
    pub fn add_optimization_hint(&mut self, hint: OptimizationHint) {
        if !self.optimization_hints.contains(&hint) {
            self.optimization_hints.push(hint);
        }
    }
}
