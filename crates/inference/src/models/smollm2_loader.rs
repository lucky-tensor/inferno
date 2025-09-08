//! SafeTensors weight loader for SmolLM2 model

use burn::{
    backend::ndarray::NdArray,
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Device, Tensor, TensorData},
};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::SmolLM2Model;

type Backend = NdArray<f32>;

/// Load SmolLM2 weights from SafeTensors file
pub fn load_smollm2_weights(
    model_path: &Path,
    device: &Device<Backend>,
) -> Result<SmolLM2Model<Backend>, Box<dyn Error>> {
    // Read SafeTensors file
    let weights_path = model_path.join("model.safetensors");
    let mut file = File::open(&weights_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    let tensors = SafeTensors::deserialize(&buffer)?;
    println!("Loaded {} tensors from SafeTensors", tensors.len());
    
    // Create model structure
    let config = crate::SmolLM2Config::default();
    
    // Load embedding weights
    let embed_weight = load_tensor(&tensors, "model.embed_tokens.weight")?;
    let embedding = create_embedding_from_weights(embed_weight, device)?;
    
    // Load transformer layers
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let layer = load_transformer_layer(&tensors, i, &config, device)?;
        layers.push(layer);
    }
    
    // Load final layer norm
    let norm_weight = load_tensor(&tensors, "model.norm.weight")?;
    let norm = create_layernorm_from_weights(norm_weight, device)?;
    
    // For lm_head, SmolLM2 uses tied embeddings, so we reuse the embedding weights
    // We need to transpose the embedding weights for the lm_head
    let lm_head = create_lm_head_from_embedding(&tensors, &config, device)?;
    
    Ok(SmolLM2Model {
        embed_tokens: embedding,
        layers,
        norm,
        lm_head,
    })
}

fn load_tensor(tensors: &SafeTensors, name: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let tensor = tensors.tensor(name)?;
    let data = tensor.data();
    
    // Convert bytes to f32 (assuming little-endian)
    let mut result = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into()?;
        result.push(f32::from_le_bytes(bytes));
    }
    
    Ok(result)
}

fn create_embedding_from_weights(
    weights: Vec<f32>,
    device: &Device<Backend>,
) -> Result<Embedding<Backend>, Box<dyn Error>> {
    // SmolLM2 embedding shape: [vocab_size, hidden_size]
    let vocab_size = 49152;
    let hidden_size = 576;
    
    let tensor_data = TensorData::new(weights, [vocab_size, hidden_size]);
    let weight_tensor = Tensor::from_data(tensor_data, device);
    
    // Create embedding module and set weights
    let mut embedding = EmbeddingConfig::new(vocab_size, hidden_size).init(device);
    // Note: In practice, we'd need to use Burn's record system to properly set weights
    // For now, this is a placeholder
    
    Ok(embedding)
}

fn create_layernorm_from_weights(
    weights: Vec<f32>,
    device: &Device<Backend>,
) -> Result<LayerNorm<Backend>, Box<dyn Error>> {
    let hidden_size = weights.len();
    let layernorm = LayerNormConfig::new(hidden_size).init(device);
    // TODO: Set weights properly using Burn's record system
    Ok(layernorm)
}

fn create_lm_head_from_embedding(
    tensors: &SafeTensors,
    config: &crate::SmolLM2Config,
    device: &Device<Backend>,
) -> Result<Linear<Backend>, Box<dyn Error>> {
    // Since embeddings are tied, we reuse the embedding weights for lm_head
    // but transposed: embedding is [vocab_size, hidden_size]
    // lm_head needs [hidden_size, vocab_size]
    
    let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
        .with_bias(false)
        .init(device);
    
    // TODO: Set weights from embedding (transposed)
    
    Ok(lm_head)
}

fn load_transformer_layer(
    tensors: &SafeTensors,
    layer_idx: usize,
    config: &crate::SmolLM2Config,
    device: &Device<Backend>,
) -> Result<crate::SmolLM2Layer<Backend>, Box<dyn Error>> {
    let prefix = format!("model.layers.{}", layer_idx);
    
    // Load attention weights
    let q_proj = load_linear(tensors, &format!("{}.self_attn.q_proj", prefix), device)?;
    let k_proj = load_linear(tensors, &format!("{}.self_attn.k_proj", prefix), device)?;
    let v_proj = load_linear(tensors, &format!("{}.self_attn.v_proj", prefix), device)?;
    let o_proj = load_linear(tensors, &format!("{}.self_attn.o_proj", prefix), device)?;
    
    let self_attn = crate::SmolLM2Attention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
    };
    
    // Load MLP weights
    let gate_proj = load_linear(tensors, &format!("{}.mlp.gate_proj", prefix), device)?;
    let up_proj = load_linear(tensors, &format!("{}.mlp.up_proj", prefix), device)?;
    let down_proj = load_linear(tensors, &format!("{}.mlp.down_proj", prefix), device)?;
    
    let mlp = crate::SmolLM2MLP {
        gate_proj,
        up_proj,
        down_proj,
    };
    
    // Load layer norms
    let input_layernorm_weight = load_tensor(tensors, &format!("{}.input_layernorm.weight", prefix))?;
    let post_attention_layernorm_weight = load_tensor(tensors, &format!("{}.post_attention_layernorm.weight", prefix))?;
    
    let input_layernorm = create_layernorm_from_weights(input_layernorm_weight, device)?;
    let post_attention_layernorm = create_layernorm_from_weights(post_attention_layernorm_weight, device)?;
    
    Ok(crate::SmolLM2Layer {
        self_attn,
        mlp,
        input_layernorm,
        post_attention_layernorm,
    })
}

fn load_linear(
    tensors: &SafeTensors,
    name: &str,
    device: &Device<Backend>,
) -> Result<Linear<Backend>, Box<dyn Error>> {
    let weight = load_tensor(tensors, &format!("{}.weight", name))?;
    
    // Get dimensions from the tensor shape
    let tensor = tensors.tensor(&format!("{}.weight", name))?;
    let shape = tensor.shape();
    
    if shape.len() != 2 {
        return Err(format!("Expected 2D tensor for {}, got {:?}", name, shape).into());
    }
    
    let out_features = shape[0];
    let in_features = shape[1];
    
    let linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(device);
    
    // TODO: Set weights properly using Burn's record system
    
    Ok(linear)
}