use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};

use super::attention::CausalSelfAttention;
use super::config::GptOssConfig;
use super::moe::TextMoe;

/// RMSNorm layer
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32;

        let x = x.to_dtype(internal_dtype)?;
        let eps_tensor = Tensor::new(&[self.eps as f32], x.device())?;
        let norm = x
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .broadcast_add(&eps_tensor)?
            .sqrt()?;
        let x_normed = x.broadcast_div(&norm)?;

        let weight = self.weight.to_dtype(internal_dtype)?;
        x_normed
            .broadcast_mul(&weight)?
            .to_dtype(x_dtype)
    }
}

/// Transformer block with MoE
pub struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: TextMoe,
}

impl Block {
    pub fn new(
        vb: VarBuilder<'_>,
        cfg: &GptOssConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let rms_1 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;

        let attn = CausalSelfAttention::new(
            vb.pp("self_attn"),
            cfg,
            layer_idx,
        )?;

        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let mlp = TextMoe::new(vb.pp("mlp"), cfg)?;

        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut (Tensor, Tensor),
    ) -> Result<Tensor> {
        // Pre-norm attention with residual
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, attention_mask, seqlen_offset, kv_cache)? + residual)?;

        // Pre-norm MoE with residual
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;

        Ok(x)
    }
}

/// Complete GPT-OSS model
pub struct GptOssModel {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: candle_nn::Linear,
    kv_caches: Vec<(Tensor, Tensor)>,
    config: GptOssConfig,
    device: Device,
}

impl GptOssModel {
    pub fn new(cfg: &GptOssConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let vb_model = vb.pp("model");

        // Token embeddings
        let wte = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_model.pp("embed_tokens"),
        )?;

        // Transformer blocks
        let blocks: Vec<Block> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::new(
                    vb_model.pp(&format!("layers.{}", i)),
                    cfg,
                    i,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Final layer norm
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_model.pp("norm"),
        )?;

        // Output head
        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(wte.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?
        };

        // Initialize KV caches
        let device = vb.device().clone();
        let kv_caches = vec![
            (
                Tensor::zeros((1, cfg.num_key_value_heads, 0, cfg.head_dim), DType::F32, &device)?,
                Tensor::zeros((1, cfg.num_key_value_heads, 0, cfg.head_dim), DType::F32, &device)?
            );
            cfg.num_hidden_layers
        ];

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            kv_caches,
            config: cfg.clone(),
            device,
        })
    }

    /// Forward pass through the model
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut x = self.wte.forward(input_ids)?;

        // Create causal mask
        let attention_mask = self.create_causal_mask(seq_len, seqlen_offset)?;

        // Pass through transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, Some(&attention_mask), seqlen_offset, &mut self.kv_caches[i])?;
        }

        // Final layer norm
        x = self.ln_f.forward(&x)?;

        // Project to vocabulary
        let logits = self.lm_head.forward(&x)?;

        Ok(logits)
    }

    /// Create causal attention mask
    fn create_causal_mask(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let total_len = seq_len + seqlen_offset;

        // Create mask matrix
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j > i + seqlen_offset {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        Tensor::from_vec(mask, (seq_len, total_len), &self.device)?
            .unsqueeze(0)?
            .unsqueeze(0)
    }

    /// Reset KV caches
    pub fn reset_caches(&mut self) -> Result<()> {
        for (k_cache, v_cache) in &mut self.kv_caches {
            *k_cache = Tensor::zeros(
                (1, self.config.num_key_value_heads, 0, self.config.head_dim),
                DType::F32,
                &self.device,
            )?;
            *v_cache = Tensor::zeros(
                (1, self.config.num_key_value_heads, 0, self.config.head_dim),
                DType::F32,
                &self.device,
            )?;
        }
        Ok(())
    }

    /// Get the device this model is on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model configuration
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        // TODO: Add unit tests for model creation
    }

    #[test]
    fn test_forward_pass() {
        // TODO: Add unit tests for forward pass
    }
}
