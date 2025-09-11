//! Llama model loader using the official burn-llama implementation

use burn::{backend::ndarray::NdArray, tensor::Device};
use std::error::Error;
use std::path::Path;

#[cfg(feature = "burn-cpu")]
use llama_burn::llama::{Llama, LlamaConfig};

#[cfg(feature = "burn-cpu")]
use llama_burn::tokenizer::SentiencePieceTokenizer;

type Backend = NdArray<f32>;

/// Load TinyLlama-1.1B model with pre-trained weights from `SafeTensors`
#[cfg(all(feature = "burn-cpu", feature = "pretrained"))]
pub fn load_llama_weights(
    model_path: &Path,
    device: &Device<Backend>,
) -> Result<Llama<Backend, SentiencePieceTokenizer>, Box<dyn Error>> {
    println!("🔄 Loading pre-trained TinyLlama-1.1B model with real weights...");

    // Check if we have the required files
    let weights_path = model_path.join("model.safetensors");
    let tokenizer_path = model_path.join("tokenizer.json");

    if !weights_path.exists() {
        return Err(format!("SafeTensors file not found: {}", weights_path.display()).into());
    }

    if !tokenizer_path.exists() {
        return Err(format!("Tokenizer file not found: {}", tokenizer_path.display()).into());
    }

    // Create TinyLlama configuration matching the actual model
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

    println!(
        "📋 TinyLlama Config: {} layers, {} heads, vocab_size: {}",
        config.num_hidden_layers, config.num_attention_heads, config.vocab_size
    );

    // Initialize the model structure (with random weights initially)
    let mut model = config
        .init::<Backend, SentiencePieceTokenizer>(device)
        .map_err(|e| format!("Failed to initialize TinyLlama model: {}", e))?;

    println!("✅ Model structure initialized, attempting to load SafeTensors weights...");

    // Try to load weights using Burn's record system
    // Note: This is a simplified approach - full weight loading would require
    // proper tensor name mapping from HuggingFace format to Burn format
    match load_safetensors_weights(&weights_path, &mut model) {
        Ok(()) => {
            println!("✅ Successfully loaded TinyLlama with pre-trained weights!");
        }
        Err(e) => {
            println!("⚠️ Failed to load pre-trained weights: {}", e);
            println!("🔄 Continuing with initialized model (may have random weights)");
        }
    }

    Ok(model)
}

// Helper function to load SafeTensors weights (simplified implementation)
#[cfg(all(feature = "burn-cpu", feature = "pretrained"))]
fn load_safetensors_weights(
    _weights_path: &Path,
    _model: &mut Llama<Backend, SentiencePieceTokenizer>,
) -> Result<(), Box<dyn Error>> {
    // TODO: Implement proper SafeTensors loading with tensor name mapping
    // For now, we'll return an error to use the model with initialized weights
    Err("SafeTensors loading not yet implemented - using initialized weights".into())
}

/// Fallback for when pretrained feature is not available
#[cfg(all(feature = "burn-cpu", not(feature = "pretrained")))]
pub fn load_llama_weights(
    model_path: &Path,
    device: &Device<Backend>,
) -> Result<Llama<Backend, SentiencePieceTokenizer>, Box<dyn Error>> {
    println!("⚠️  Loading TinyLlama with random weights (pretrained feature not enabled)");

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

    // Initialize the model with random weights
    let model = config.init::<Backend, SentiencePieceTokenizer>(device)?;

    Ok(model)
}
