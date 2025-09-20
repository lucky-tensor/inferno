//! Weight file analysis and quantization detection
//!
//! Analyzes SafeTensors and other weight files to detect quantization schemes,
//! data types, and sharding patterns.

use super::*;
use candle_core::DType;
use std::path::Path;

/// Weight file analyzer
pub struct WeightAnalyzer;

impl WeightAnalyzer {
    /// Analyze weights in a model directory
    pub async fn analyze_weights(
        model_path: &str,
    ) -> Result<WeightAnalysisResult, Box<dyn std::error::Error>> {
        use safetensors::SafeTensors;
        use std::collections::HashMap;
        use std::fs;

        let path = Path::new(model_path);

        // Check if model directory exists
        if !path.exists() {
            return Err(format!("Model directory does not exist: {}", model_path).into());
        }

        // Get weight mapping from index file or detect single file
        let weight_mapping = Self::get_weight_mapping_flexible(path)?;

        // Check for sharding
        let (is_sharded, num_shards) = Self::is_sharded(model_path).await?;

        // Detect quantization
        let quantization = Self::detect_quantization(model_path).await?;

        // Analyze tensor properties from SafeTensors files
        let mut total_params = 0u64;
        let mut dtype_counts: HashMap<DType, usize> = HashMap::new();
        let mut total_memory_bytes = 0u64;

        // Group weights by file to avoid loading same file multiple times
        let mut files_to_weights: HashMap<String, Vec<String>> = HashMap::new();
        for (weight_name, filename) in &weight_mapping {
            files_to_weights
                .entry(filename.clone())
                .or_default()
                .push(weight_name.clone());
        }

        // Analyze each SafeTensors file
        for (filename, _weight_names) in files_to_weights {
            let file_path = path.join(&filename);

            if !file_path.exists() {
                continue; // Skip missing files
            }

            // Load SafeTensors metadata without loading actual data
            let buffer = fs::read(&file_path)?;
            let safetensors = SafeTensors::deserialize(&buffer)?;

            // Analyze each tensor in the file
            for tensor_name in safetensors.names() {
                let tensor_info = safetensors.tensor(tensor_name)?;

                // Calculate parameter count
                let shape = tensor_info.shape();
                let param_count: u64 = shape.iter().map(|&dim| dim as u64).product();
                total_params += param_count;

                // Map SafeTensors dtype to Candle DType
                let dtype = Self::safetensors_dtype_to_candle(tensor_info.dtype())?;
                *dtype_counts.entry(dtype).or_insert(0) += 1;

                // Calculate memory usage
                let bytes_per_element = Self::dtype_size_bytes(dtype);
                total_memory_bytes += param_count * bytes_per_element as u64;
            }
        }

        // Determine primary dtype (most common)
        let primary_dtype = dtype_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&dtype, _)| dtype)
            .unwrap_or(DType::F32);

        Ok(WeightAnalysisResult {
            primary_dtype,
            total_params,
            quantization,
            is_sharded,
            num_shards,
            estimated_memory_bytes: total_memory_bytes,
        })
    }

    /// Detect quantization scheme from weight files
    pub async fn detect_quantization(
        model_path: &str,
    ) -> Result<QuantizationConfig, Box<dyn std::error::Error>> {
        // Check for quantization indicators in model name/path
        let path_str = model_path.to_lowercase();

        if path_str.contains("w8a8") || path_str.contains("int8") {
            return Ok(QuantizationConfig {
                scheme: QuantizationScheme::W8A8,
                symmetric: true,
                ..Default::default()
            });
        }

        if path_str.contains("w4a16") || path_str.contains("int4") || path_str.contains("4bit") {
            return Ok(QuantizationConfig {
                scheme: QuantizationScheme::W4A16,
                symmetric: true,
                ..Default::default()
            });
        }

        // Default to no quantization
        Ok(QuantizationConfig::default())
    }

    /// Check if model uses sharded weights
    pub async fn is_sharded(model_path: &str) -> Result<(bool, usize), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);
        let mut shard_count = 0;

        let mut entries = tokio::fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_name = entry.file_name().to_string_lossy().to_lowercase();
            if file_name.contains("model")
                && (file_name.contains(".safetensors") || file_name.contains(".bin"))
                && (file_name.contains("-") || file_name.contains("_"))
            {
                shard_count += 1;
            }
        }

        Ok((shard_count > 1, shard_count.max(1)))
    }

    /// Maps SafeTensors dtype to Candle DType.
    fn safetensors_dtype_to_candle(
        dtype: safetensors::Dtype,
    ) -> Result<DType, Box<dyn std::error::Error>> {
        use safetensors::Dtype as STDtype;

        let candle_dtype = match dtype {
            STDtype::F32 => DType::F32,
            STDtype::F16 => DType::F16,
            STDtype::BF16 => DType::BF16,
            STDtype::U8 => DType::U8,
            STDtype::I8 => DType::U8, // Map I8 to U8 for compatibility
            STDtype::U32 => DType::U32,
            STDtype::I64 => DType::I64,
            _ => {
                return Err(format!("Unsupported SafeTensors dtype: {:?}", dtype).into());
            }
        };

        Ok(candle_dtype)
    }

    /// Returns the size in bytes for a given DType.
    fn dtype_size_bytes(dtype: DType) -> usize {
        match dtype {
            DType::U8 => 1,
            DType::U32 => 4,
            DType::I64 => 8,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }

    /// Get weight mapping, handling both sharded and single-file models.
    fn get_weight_mapping_flexible(
        model_path: &Path,
    ) -> Result<std::collections::HashMap<String, String>, Box<dyn std::error::Error>> {
        use crate::InfernoLlama;
        use std::collections::HashMap;

        // First try to get sharded mapping
        match InfernoLlama::get_weight_mapping(model_path) {
            Ok(mapping) => Ok(mapping),
            Err(_) => {
                // If no index file, check for single model.safetensors file
                let single_file = model_path.join("model.safetensors");
                if single_file.exists() {
                    // Read the single safetensors file to get all weight names
                    let buffer = std::fs::read(&single_file)?;
                    let safetensors = safetensors::SafeTensors::deserialize(&buffer)?;

                    let mut mapping = HashMap::new();
                    for tensor_name in safetensors.names() {
                        mapping.insert(tensor_name.to_string(), "model.safetensors".to_string());
                    }
                    Ok(mapping)
                } else {
                    Err("No model.safetensors.index.json or model.safetensors file found".into())
                }
            }
        }
    }
}

/// Result of weight file analysis
#[derive(Debug, Clone)]
pub struct WeightAnalysisResult {
    /// Primary data type detected
    pub primary_dtype: DType,
    /// Total parameter count estimated
    pub total_params: u64,
    /// Quantization configuration
    pub quantization: QuantizationConfig,
    /// Sharding information
    pub is_sharded: bool,
    /// Number of shard files
    pub num_shards: usize,
    /// Estimated memory usage
    pub estimated_memory_bytes: u64,
}
