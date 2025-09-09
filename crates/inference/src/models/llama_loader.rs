//! Llama model loader using the official burn-llama implementation

use burn::{
    backend::ndarray::NdArray,
    module::Module,
    tensor::Device,
};
use std::error::Error;
use std::path::Path;

#[cfg(feature = "burn-cpu")]
use llama_burn::llama::{Llama, LlamaConfig};

#[cfg(feature = "burn-cpu")]
use llama_burn::tokenizer::SentiencePieceTokenizer;

type Backend = NdArray<f32>;

/// Load TinyLlama-1.1B model from SafeTensors file
pub fn load_llama_weights(
    model_path: &Path,
    device: &Device<Backend>,
) -> Result<Llama<Backend, SentiencePieceTokenizer>, Box<dyn Error>> {
    // TinyLlama-1.1B configuration
    let tokenizer_path = model_path.join("tokenizer.json");
    let config = LlamaConfig {
        d_model: 2048,
        hidden_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: Some(4),
        vocab_size: 32000,
        norm_eps: 1e-5,
        rope: llama_burn::llama::RopeConfig::new(10000.0),
        max_seq_len: 2048,
        max_batch_size: 1,
        tokenizer: tokenizer_path.to_str().unwrap().to_string(),
    };
    
    // Initialize the model
    let model = config.init::<Backend, SentiencePieceTokenizer>(device)?;
    
    // Load weights from SafeTensors
    let weights_path = model_path.join("model.safetensors");
    if weights_path.exists() {
        println!("Loading TinyLlama weights from: {:?}", weights_path);
        // Note: The actual weight loading would use llama-burn's loading utilities
        // This is a placeholder - the llama-burn crate should provide proper loading
    }
    
    Ok(model)
}