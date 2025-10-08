//! OpenAI GPT model implementation for Candle
//!
//! Based on learnings from mistral.rs, implements transformer architecture
//! for OpenAI's OSS models with CUDA GPU support.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, Module, VarBuilder};
use serde::Deserialize;

/// Conv1D layer (GPT-2 style) - weights are transposed compared to standard Linear
/// GPT-2 stores weights as [in_features, out_features] while Candle Linear expects [out_features, in_features]
#[derive(Debug)]
pub struct Conv1D {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Conv1D {
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder<'_>) -> Result<Self> {
        // GPT-2 weights are stored as [in_features, out_features]
        // We keep them as-is since we'll transpose during forward
        let weight = vb.get((in_features, out_features), "weight")?;
        let bias = vb.get(out_features, "bias").ok();
        Ok(Self { weight, bias })
    }
}

impl Module for Conv1D {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // self.weight is [in_features, out_features]
        // x is [batch, seq, in_features]
        let dims_x = x.dims();
        let in_features = dims_x[dims_x.len() - 1];

        // Reshape x to [batch*seq, in_features]
        let batch_seq = dims_x.iter().take(dims_x.len() - 1).product::<usize>();
        let x_2d = x.reshape((batch_seq, in_features))?;

        // Matmul: [batch*seq, in_features] @ [in_features, out_features] = [batch*seq, out_features]
        let out = x_2d.matmul(&self.weight)?;

        // Reshape back to [batch, seq, out_features]
        let out_features = self.weight.dim(1)?;
        let mut out_shape = dims_x[..dims_x.len()-1].to_vec();
        out_shape.push(out_features);
        let out = out.reshape(out_shape)?;

        match &self.bias {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }
}

/// OpenAI model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIConfig {
    pub vocab_size: usize,
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    #[serde(alias = "n_inner", default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    #[serde(alias = "n_head_kv")]
    pub num_key_value_heads: Option<usize>, // For GQA
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(alias = "layer_norm_epsilon", default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_use_bias")]
    pub use_bias: bool,
}

fn default_intermediate_size() -> usize {
    0 // Will be computed in impl
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_use_bias() -> bool {
    false
}

impl OpenAIConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn intermediate_size(&self) -> usize {
        if self.intermediate_size == 0 {
            // GPT-2 default: 4x hidden_size
            self.hidden_size * 4
        } else {
            self.intermediate_size
        }
    }
}

/// Rotary Positional Embeddings (RoPE)
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let sin = freqs.sin()?;
        let cos = freqs.cos()?;

        Ok(Self { sin, cos, dim })
    }

    pub fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _n_head, seq_len, _n_embd) = q.dims4()?;

        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        let q_embed = Self::rotate_half(q, &cos, &sin)?;
        let k_embed = Self::rotate_half(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_b, _h, seq_len, n_embd) = x.dims4()?;
        let half = n_embd / 2;

        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[out1, out2], 3)
    }
}

/// Multi-Head Attention (GPT-2 style)
#[derive(Debug)]
pub struct Attention {
    c_attn: Conv1D,  // Combined QKV projection
    c_proj: Conv1D,  // Output projection
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(cfg: &OpenAIConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();

        // GPT-2 uses "c_attn" for combined QKV projection (3 * hidden_size)
        let c_attn = Conv1D::new(hidden_size, 3 * hidden_size, vb.pp("c_attn"))?;
        // GPT-2 uses "c_proj" for output projection
        let c_proj = Conv1D::new(hidden_size, hidden_size, vb.pp("c_proj"))?;

        // Note: GPT-2 does NOT use RoPE - it uses learned positional embeddings
        // that are added to token embeddings before the transformer blocks

        Ok(Self {
            c_attn,
            c_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = hidden_states.dims3()?;

        // GPT-2 combines QKV in one projection
        let qkv = self.c_attn.forward(hidden_states)?;

        // Split into Q, K, V
        let q = qkv.narrow(2, 0, hidden_size)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?;
        let v = qkv.narrow(2, 2 * hidden_size, hidden_size)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // GPT-2 does NOT apply RoPE - positional information comes from learned embeddings

        // Update KV cache
        let (k, v) = match kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[&*prev_k, &k], 2)?.contiguous()?;
                let v = Tensor::cat(&[&*prev_v, &v], 2)?.contiguous()?;
                *kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
            None => {
                *kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
        };

        // Repeat KV heads for GQA
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = (q.matmul(&k_t)? * scale)?;

        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let v_cont = v.contiguous()?;
        let attn_output = attn_weights.matmul(&v_cont)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        self.c_proj.forward(&attn_output)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            Ok(x.clone())
        } else {
            let (b_sz, num_kv_heads, seq_len, head_dim) = x.dims4()?;
            let expanded = x.unsqueeze(2)?
                .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
                .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))?;
            expanded.contiguous()
        }
    }
}

/// MLP with GELU activation (GPT-2 style)
#[derive(Debug)]
pub struct MLP {
    c_fc: Conv1D,
    c_proj: Conv1D,
}

impl MLP {
    pub fn new(cfg: &OpenAIConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size();

        // GPT-2 uses "c_fc" for the first projection and "c_proj" for the output
        let c_fc = Conv1D::new(hidden_size, intermediate_size, vb.pp("c_fc"))?;
        let c_proj = Conv1D::new(intermediate_size, hidden_size, vb.pp("c_proj"))?;

        Ok(Self {
            c_fc,
            c_proj,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.c_fc.forward(hidden_states)?;
        // GPT-2 uses GELU activation
        let hidden_states = hidden_states.gelu()?;
        self.c_proj.forward(&hidden_states)
    }
}

/// Transformer block
#[derive(Debug)]
pub struct TransformerBlock {
    attention: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl TransformerBlock {
    pub fn new(cfg: &OpenAIConfig, vb: VarBuilder<'_>) -> Result<Self> {
        // GPT-2 uses "attn" for attention layer
        let attention = Attention::new(cfg, vb.pp("attn"))?;
        // GPT-2 uses "mlp" for MLP layer
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        // GPT-2 uses "ln_1" for pre-attention layer norm (LayerNorm with bias)
        let input_layernorm = layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ln_1"))?;
        // GPT-2 uses "ln_2" for pre-MLP layer norm (LayerNorm with bias)
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("ln_2"),
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // Pre-norm architecture: normalize before sublayer
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.attention.forward(&hidden_states, attention_mask, seqlen_offset, kv_cache)?;
        let hidden_states = (hidden_states + residual)?;

        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        hidden_states + residual
    }
}

/// Complete OpenAI transformer model
#[derive(Debug)]
pub struct OpenAIModel {
    embed_tokens: Embedding,
    embed_positions: Embedding,  // GPT-2 positional embeddings
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    lm_head: Option<Conv1D>,
    wte_weight: Tensor, // For weight tying
    device: Device,
    config: OpenAIConfig,
}

impl OpenAIModel {
    pub fn new(cfg: &OpenAIConfig, vb: VarBuilder<'_>) -> Result<Self> {
        // GPT-2 uses "wte" for token embeddings
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("wte"))?;
        // GPT-2 uses "wpe" for positional embeddings
        let embed_positions = embedding(cfg.max_position_embeddings, cfg.hidden_size, vb.pp("wpe"))?;

        // Get wte weights for potential weight tying with lm_head
        let wte_weight = vb.get((cfg.vocab_size, cfg.hidden_size), "wte.weight")?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        // GPT-2 uses "h.{i}" for layers
        let vb_layers = vb.pp("h");
        for i in 0..cfg.num_hidden_layers {
            let layer = TransformerBlock::new(cfg, vb_layers.pp(i))?;
            layers.push(layer);
        }

        // GPT-2 uses "ln_f" for final layer norm (LayerNorm with bias)
        let norm = layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ln_f"))?;

        // GPT-2 may have separate lm_head or tie weights with wte
        let lm_head = Conv1D::new(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head")).ok();

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            norm,
            lm_head,
            wte_weight,
            device: vb.device().clone(),
            config: cfg.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_caches: &mut Vec<Option<(Tensor, Tensor)>>,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        // Token embeddings
        let token_embeds = self.embed_tokens.forward(input_ids)?;

        // Position embeddings
        let position_ids: Vec<u32> = (seqlen_offset..seqlen_offset + seq_len)
            .map(|i| i as u32)
            .collect();
        let position_ids_tensor = Tensor::new(&position_ids[..], &self.device)?
            .unsqueeze(0)?;  // Add batch dimension
        let position_embeds = self.embed_positions.forward(&position_ids_tensor)?;

        // Combine token and position embeddings
        let mut hidden_states = token_embeds.broadcast_add(&position_embeds)?;

        // Create causal mask
        let mask = if seq_len > 1 {
            Some(self.create_causal_mask(seq_len, seqlen_offset)?)
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                mask.as_ref(),
                seqlen_offset,
                &mut kv_caches[i],
            )?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;

        // Use lm_head if available, otherwise use tied weights from wte
        let logits = match &self.lm_head {
            Some(lm_head) => lm_head.forward(&hidden_states)?,
            None => {
                // Weight tying: use wte weights [vocab_size, hidden_size]
                // hidden_states: [batch, seq, hidden_size]
                // Need: [batch, seq, hidden_size] @ [hidden_size, vocab_size] = [batch, seq, vocab_size]
                let dims = hidden_states.dims();
                let batch_seq = dims[0] * dims[1];
                let hidden_size = dims[2];
                let hidden_2d = hidden_states.reshape((batch_seq, hidden_size))?;
                let logits_2d = hidden_2d.matmul(&self.wte_weight.t()?)?;
                logits_2d.reshape((dims[0], dims[1], self.config.vocab_size))?
            }
        };

        Ok(logits)
    }

    fn create_causal_mask(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if i + seqlen_offset < j + seqlen_offset {
                        f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
            })
            .collect();
        Tensor::from_vec(mask, (seq_len, seq_len), &self.device)?
            .unsqueeze(0)?
            .unsqueeze(0)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &OpenAIConfig {
        &self.config
    }
}
