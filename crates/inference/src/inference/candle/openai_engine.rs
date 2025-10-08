//! OpenAI model inference engine with CUDA GPU support
//!
//! Simplified engine focused on GPU-only inference for OpenAI OSS models

use super::openai_model::{OpenAIConfig, OpenAIModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json;
use std::path::Path;
use tokenizers::Tokenizer;

/// OpenAI inference engine with GPU support
pub struct OpenAIEngine {
    model: OpenAIModel,
    tokenizer: Tokenizer,
    device: Device,
    config: OpenAIConfig,
    kv_caches: Vec<Option<(Tensor, Tensor)>>,
}

impl OpenAIEngine {
    /// Load model from safetensors files on GPU
    pub fn load_from_safetensors<P: AsRef<Path>>(
        model_path: P,
        device: Device,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.as_ref();

        // Load config
        let config_path = model_path.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)?;

        // Parse as generic JSON first to handle duplicate fields
        let config_json: serde_json::Value = serde_json::from_str(&config_data)?;

        // Extract values with fallbacks
        let vocab_size = config_json["vocab_size"].as_u64().ok_or_else(|| anyhow::anyhow!("Missing vocab_size"))? as usize;
        let hidden_size = config_json["n_embd"].as_u64().or_else(|| config_json["hidden_size"].as_u64()).ok_or_else(|| anyhow::anyhow!("Missing hidden_size"))? as usize;
        let num_hidden_layers = config_json["n_layer"].as_u64().or_else(|| config_json["num_hidden_layers"].as_u64()).ok_or_else(|| anyhow::anyhow!("Missing num_hidden_layers"))? as usize;
        let num_attention_heads = config_json["n_head"].as_u64().or_else(|| config_json["num_attention_heads"].as_u64()).ok_or_else(|| anyhow::anyhow!("Missing num_attention_heads"))? as usize;
        let max_position_embeddings = config_json["n_positions"].as_u64().or_else(|| config_json["n_ctx"].as_u64()).or_else(|| config_json["max_position_embeddings"].as_u64()).unwrap_or(1024) as usize;
        // GPT-2 standard: intermediate_size = 4 * hidden_size
        let intermediate_size = config_json["n_inner"].as_u64()
            .or_else(|| config_json["intermediate_size"].as_u64())
            .unwrap_or((hidden_size * 4) as u64) as usize;
        let layer_norm_eps = config_json["layer_norm_epsilon"].as_f64().or_else(|| config_json["rms_norm_eps"].as_f64()).unwrap_or(1e-5);
        let rope_theta = config_json["rope_theta"].as_f64().unwrap_or(10000.0) as f32;

        let config = OpenAIConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: None,
            max_position_embeddings,
            rms_norm_eps: layer_norm_eps,
            rope_theta,
            use_bias: true,  // GPT-2 uses bias in all layers
        };

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Find safetensors files
        let safetensors_files = Self::find_safetensors_files(model_path)?;

        // Load weights onto GPU
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_files, DType::F32, &device)?
        };

        // Create model on GPU
        let model = OpenAIModel::new(&config, vb)?;

        // Initialize KV caches
        let kv_caches = vec![None; config.num_hidden_layers];

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            kv_caches,
        })
    }

    fn find_safetensors_files(model_path: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();

        // Check for single file
        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            files.push(single_file);
            return Ok(files);
        }

        // Check for sharded files
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("model") && name.ends_with(".safetensors") {
                    files.push(path);
                }
            }
        }

        if files.is_empty() {
            anyhow::bail!("No safetensors files found in {}", model_path.display());
        }

        files.sort();
        Ok(files)
    }

    /// Generate text from a prompt
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> anyhow::Result<String> {
        // Tokenize prompt
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_tokens = encoding.get_ids();

        // Convert to tensor on GPU
        let input_tensor = Tensor::new(input_tokens, &self.device)?
            .unsqueeze(0)?;

        let mut generated_tokens = Vec::new();
        let mut current_input = input_tensor;
        let mut seqlen_offset = 0;

        for i in 0..max_tokens {
            // Forward pass on GPU
            let logits = self.model.forward(&current_input, seqlen_offset, &mut self.kv_caches)?;

            // Get last token logits [batch, seq, vocab] -> [vocab]
            let last_idx = logits.dim(1)? - 1;
            let logits = logits.narrow(1, last_idx, 1)?.squeeze(0)?.squeeze(0)?;

            // Sample next token
            let next_token = if temperature > 0.0 {
                Self::sample_with_temperature(&logits, temperature)?
            } else {
                Self::sample_greedy(&logits)?
            };

            generated_tokens.push(next_token);

            // Check for EOS
            if self.is_eos_token(next_token) {
                break;
            }

            // Prepare next input for next iteration
            current_input = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?;

            // Update offset: first iteration processes full prompt, then +1 per token
            if i == 0 {
                seqlen_offset = input_tokens.len();
            } else {
                seqlen_offset += 1;
            }
        }

        // Decode generated tokens
        let output = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(output)
    }

    fn sample_greedy(logits: &Tensor) -> anyhow::Result<u32> {
        let logits_vec = logits.to_vec1::<f32>()?;
        let max_idx = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow::anyhow!("Empty logits"))?;
        Ok(max_idx as u32)
    }

    fn sample_with_temperature(logits: &Tensor, temperature: f32) -> anyhow::Result<u32> {
        let logits = (logits / temperature as f64)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let probs_vec = probs.to_vec1::<f32>()?;

        // Sample from distribution
        let sum: f32 = probs_vec.iter().sum();
        let mut rng = rand::thread_rng();
        let mut random = rand::Rng::gen::<f32>(&mut rng) * sum;

        for (idx, &prob) in probs_vec.iter().enumerate() {
            random -= prob;
            if random <= 0.0 {
                return Ok(idx as u32);
            }
        }

        Ok((probs_vec.len() - 1) as u32)
    }

    fn is_eos_token(&self, token: u32) -> bool {
        // Common EOS tokens
        token == 2 || token == 50256 // </s> or <|endoftext|>
    }

    /// Reset KV caches for new generation
    pub fn reset_caches(&mut self) {
        self.kv_caches = vec![None; self.config.num_hidden_layers];
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
