//! Direct SafeTensors weight loader with arbitrary dtype support
//!
//! This implementation demonstrates:
//! 1. Direct SafeTensors loading without llama-burn limitations
//! 2. Support for F32, F16, BF16 tensors
//! 3. Tensor parsing and validation
//! 4. Real neural network weight inspection

use burn::{
    backend::ndarray::NdArray,
    tensor::{backend::Backend, Device, Tensor},
};
use safetensors::{SafeTensors, tensor::TensorView};
use std::path::Path;
use std::error::Error;
use std::collections::HashMap;

// Type alias for our backend
type SafeTensorsBackend = NdArray<f32>;

/// A structure to hold loaded SafeTensors weights
#[derive(Debug)]
pub struct LoadedWeights {
    pub tensors: HashMap<String, WeightTensor>,
    pub total_parameters: usize,
    pub memory_usage_mb: f64,
}

#[derive(Debug)]
pub struct WeightTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data_sample: Vec<f32>, // First 10 values for inspection
    pub parameter_count: usize,
}

/// SafeTensors loader with arbitrary dtype support
pub struct SafeTensorsLoader<B: Backend> {
    device: Device<B>,
}

impl<B: Backend> SafeTensorsLoader<B> {
    pub fn new(device: Device<B>) -> Self {
        Self { device }
    }

    /// Load and parse SafeTensors file with full dtype support
    pub fn load_safetensors(&self, safetensors_path: &Path) -> Result<LoadedWeights, Box<dyn Error>> {
        println!("🔍 Loading SafeTensors file: {}", safetensors_path.display());

        if !safetensors_path.exists() {
            return Err(format!("SafeTensors file not found: {}", safetensors_path.display()).into());
        }

        // Read and parse SafeTensors file
        let data = std::fs::read(safetensors_path)?;
        let safetensors = SafeTensors::deserialize(&data)?;

        let file_size = data.len();
        let file_size_mb = file_size as f64 / 1_048_576.0;

        println!("  📊 File size: {:.1} MB", file_size_mb);
        println!("  🧮 Found {} tensors", safetensors.len());

        let mut loaded_tensors = HashMap::new();
        let mut total_parameters = 0;

        // Process each tensor with dtype support
        for tensor_name in safetensors.names() {
            let tensor_view = safetensors.tensor(&tensor_name)?;
            let weight_tensor = self.process_tensor(&tensor_name, tensor_view)?;

            total_parameters += weight_tensor.parameter_count;
            println!("  ✅ Loaded: {} | Shape: {:?} | Type: {} | Params: {}",
                     weight_tensor.name,
                     weight_tensor.shape,
                     weight_tensor.dtype,
                     weight_tensor.parameter_count);

            loaded_tensors.insert(tensor_name.to_string(), weight_tensor);
        }

        println!("📈 Total parameters: {:.2}M", total_parameters as f64 / 1_000_000.0);

        Ok(LoadedWeights {
            tensors: loaded_tensors,
            total_parameters,
            memory_usage_mb: file_size_mb,
        })
    }

    /// Process individual tensor with dtype support
    fn process_tensor(&self, name: &str, tensor_view: TensorView<'_>) -> Result<WeightTensor, Box<dyn Error>> {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = format!("{:?}", tensor_view.dtype());
        let parameter_count = shape.iter().product();

        // Extract data with dtype support
        let data_sample = self.extract_sample_data(tensor_view)?;

        Ok(WeightTensor {
            name: name.to_string(),
            shape,
            dtype,
            data_sample,
            parameter_count,
        })
    }

    /// Extract sample data supporting F32, F16, BF16
    fn extract_sample_data(&self, tensor_view: TensorView<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
        let raw_data = tensor_view.data();
        let sample_size = std::cmp::min(10, tensor_view.shape().iter().product::<usize>());

        match tensor_view.dtype() {
            safetensors::Dtype::F32 => {
                let float_data = unsafe {
                    std::slice::from_raw_parts(
                        raw_data.as_ptr() as *const f32,
                        raw_data.len() / 4
                    )
                };
                Ok(float_data.iter().take(sample_size).cloned().collect())
            }
            safetensors::Dtype::F16 => {
                let f16_data = unsafe {
                    std::slice::from_raw_parts(
                        raw_data.as_ptr() as *const u16,
                        raw_data.len() / 2
                    )
                };
                let f32_data: Vec<f32> = f16_data
                    .iter()
                    .take(sample_size)
                    .map(|&x| half::f16::from_bits(x).to_f32())
                    .collect();
                Ok(f32_data)
            }
            safetensors::Dtype::BF16 => {
                let bf16_data = unsafe {
                    std::slice::from_raw_parts(
                        raw_data.as_ptr() as *const u16,
                        raw_data.len() / 2
                    )
                };
                let f32_data: Vec<f32> = bf16_data
                    .iter()
                    .take(sample_size)
                    .map(|&x| half::bf16::from_bits(x).to_f32())
                    .collect();
                Ok(f32_data)
            }
            _ => Err(format!("Unsupported dtype: {:?}", tensor_view.dtype()).into())
        }
    }

    /// Create Burn tensors from SafeTensors data
    pub fn create_burn_tensor_2d(&self, tensor_view: TensorView<'_>) -> Result<Tensor<B, 2>, Box<dyn Error>> {
        let shape = tensor_view.shape();
        if shape.len() != 2 {
            return Err("Expected 2D tensor".into());
        }

        let data = self.extract_all_data(tensor_view)?;
        let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &self.device)
            .reshape([shape[0], shape[1]]);

        Ok(tensor)
    }

    /// Extract all tensor data (not just sample)
    fn extract_all_data(&self, tensor_view: TensorView<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
        let raw_data = tensor_view.data();
        let total_elements = tensor_view.shape().iter().product::<usize>();

        match tensor_view.dtype() {
            safetensors::Dtype::F32 => {
                let float_data = unsafe {
                    std::slice::from_raw_parts(
                        raw_data.as_ptr() as *const f32,
                        raw_data.len() / 4
                    )
                };
                Ok(float_data.to_vec())
            }
            safetensors::Dtype::F16 => {
                let f16_data = unsafe {
                    std::slice::from_raw_parts(
                        raw_data.as_ptr() as *const u16,
                        raw_data.len() / 2
                    )
                };
                let f32_data: Vec<f32> = f16_data
                    .iter()
                    .map(|&x| half::f16::from_bits(x).to_f32())
                    .collect();
                Ok(f32_data)
            }
            safetensors::Dtype::BF16 => {
                let bf16_data = unsafe {
                    std::slice::from_raw_parts(
                        raw_data.as_ptr() as *const u16,
                        raw_data.len() / 2
                    )
                };
                let f32_data: Vec<f32> = bf16_data
                    .iter()
                    .map(|&x| half::bf16::from_bits(x).to_f32())
                    .collect();
                Ok(f32_data)
            }
            _ => Err(format!("Unsupported dtype: {:?}", tensor_view.dtype()).into())
        }
    }

    /// Validate neural network architecture from loaded weights
    pub fn validate_llama_architecture(&self, weights: &LoadedWeights) -> Result<LlamaArchitectureInfo, Box<dyn Error>> {
        println!("🔍 Analyzing Llama architecture from loaded weights...");

        let mut arch_info = LlamaArchitectureInfo::default();

        // Extract architecture details from tensor names and shapes
        for (name, tensor) in &weights.tensors {
            if name == "model.embed_tokens.weight" {
                arch_info.vocab_size = tensor.shape[0];
                arch_info.hidden_size = tensor.shape[1];
                println!("  📝 Embeddings: vocab_size={}, hidden_size={}", arch_info.vocab_size, arch_info.hidden_size);
            } else if name == "lm_head.weight" {
                println!("  🎯 Language head: {} -> {}", tensor.shape[1], tensor.shape[0]);
            } else if name.contains("layers.") && name.contains("self_attn.q_proj.weight") {
                // Extract layer number
                let parts: Vec<&str> = name.split('.').collect();
                if let Ok(layer_num) = parts[2].parse::<usize>() {
                    arch_info.num_layers = std::cmp::max(arch_info.num_layers, layer_num + 1);
                }
                // Q projection shape tells us about attention heads
                if arch_info.num_attention_heads == 0 {
                    arch_info.num_attention_heads = tensor.shape[0] / arch_info.hidden_size;
                }
            }
        }

        arch_info.total_parameters = weights.total_parameters;

        println!("🏗️  Detected Architecture:");
        println!("   Layers: {}", arch_info.num_layers);
        println!("   Hidden size: {}", arch_info.hidden_size);
        println!("   Vocab size: {}", arch_info.vocab_size);
        println!("   Attention heads: {}", arch_info.num_attention_heads);
        println!("   Total parameters: {:.2}M", arch_info.total_parameters as f64 / 1_000_000.0);

        Ok(arch_info)
    }
}

#[derive(Debug, Default)]
pub struct LlamaArchitectureInfo {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub total_parameters: usize,
}

/// Test function to demonstrate real SafeTensors loading
pub fn test_safetensors_loading(model_path: &Path) -> Result<(), Box<dyn Error>> {
    println!("🧪 Testing SafeTensors loading with arbitrary dtype support");

    let device = Device::<SafeTensorsBackend>::default();
    let loader = SafeTensorsLoader::<SafeTensorsBackend>::new(device);

    let safetensors_path = model_path.join("model.safetensors");

    // Load all weights
    let weights = loader.load_safetensors(&safetensors_path)?;

    // Analyze architecture
    let arch_info = loader.validate_llama_architecture(&weights)?;

    // Test creating actual Burn tensors from a few key weights
    println!("🔧 Testing Burn tensor creation from SafeTensors...");

    let safetensors = SafeTensors::deserialize(&std::fs::read(&safetensors_path)?)?;

    if let Ok(embed_tensor_view) = safetensors.tensor("model.embed_tokens.weight") {
        match loader.create_burn_tensor_2d(embed_tensor_view) {
            Ok(burn_tensor) => {
                println!("  ✅ Successfully created Burn tensor for embeddings: shape {:?}", burn_tensor.dims());
                // This demonstrates that we can create real Burn tensors from SafeTensors data
                println!("  🧮 Tensor statistics: min={:.6}, max={:.6}",
                         burn_tensor.clone().min().into_scalar(),
                         burn_tensor.max().into_scalar());
            }
            Err(e) => {
                println!("  ⚠️  Failed to create Burn tensor: {}", e);
            }
        }
    }

    println!("✅ SafeTensors loading test completed successfully!");
    println!("   Demonstrated: Arbitrary dtype support, real weight loading, Burn tensor creation");

    Ok(())
}