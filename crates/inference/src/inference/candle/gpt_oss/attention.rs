use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use super::config::GptOssConfig;

/// Rotary Positional Embeddings (RoPE) for GPT-OSS with extended theta
pub struct GptOssRoPE {
    sin: Tensor,
    cos: Tensor,
    head_dim: usize,
}

impl GptOssRoPE {
    pub fn new(cfg: &GptOssConfig, device: &candle_core::Device) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let theta = cfg.rope_theta;
        let max_seq_len = cfg.max_position_embeddings;

        // Generate frequency basis
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / head_dim as f32))
            .collect();

        let inv_freq_tensor = Tensor::new(inv_freq, device)?;

        // Generate position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions_tensor = Tensor::new(positions, device)?.reshape((max_seq_len, 1))?;

        // Compute frequencies: outer product of positions and inv_freq
        let freqs = positions_tensor.broadcast_mul(&inv_freq_tensor.reshape((1, head_dim / 2))?)?;

        // Compute sin and cos
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;

        Ok(Self { sin, cos, head_dim })
    }

    /// Apply rotary embeddings to query and key tensors
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b, _n_heads, seq_len, _head_dim) = q.dims4()?;

        // Extract sin/cos for the current sequence positions
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;

        // Apply rotation
        let q_rot = self.apply_rotation(q, &sin, &cos)?;
        let k_rot = self.apply_rotation(k, &sin, &cos)?;

        Ok((q_rot, k_rot))
    }

    fn apply_rotation(&self, x: &Tensor, sin: &Tensor, cos: &Tensor) -> Result<Tensor> {
        let (_b, _n_heads, seq_len, head_dim) = x.dims4()?;

        // Reshape sin/cos to match x dimensions
        let sin = sin.reshape((1, 1, seq_len, head_dim / 2))?;
        let cos = cos.reshape((1, 1, seq_len, head_dim / 2))?;

        // Split x into two halves
        let x1 = x.narrow(D::Minus1, 0, head_dim / 2)?;
        let x2 = x.narrow(D::Minus1, head_dim / 2, head_dim / 2)?;

        // Rotate: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let rotated1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[rotated1, rotated2], D::Minus1)
    }
}

/// Causal self-attention with sliding window support
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope: GptOssRoPE,
    sliding_window: Option<usize>,
    sinks: Option<Tensor>,
}

impl CausalSelfAttention {
    pub fn new(
        vb: VarBuilder<'_>,
        cfg: &GptOssConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let q_size = cfg.head_dim * cfg.num_attention_heads;
        let kv_size = cfg.head_dim * cfg.num_key_value_heads;

        let q_proj = if cfg.attention_bias {
            candle_nn::linear(hidden_size, q_size, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(hidden_size, q_size, vb.pp("q_proj"))?
        };

        let k_proj = if cfg.attention_bias {
            candle_nn::linear(hidden_size, kv_size, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(hidden_size, kv_size, vb.pp("k_proj"))?
        };

        let v_proj = if cfg.attention_bias {
            candle_nn::linear(hidden_size, kv_size, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(hidden_size, kv_size, vb.pp("v_proj"))?
        };

        let o_proj = if cfg.attention_bias {
            candle_nn::linear(q_size, hidden_size, vb.pp("o_proj"))?
        } else {
            candle_nn::linear_no_bias(q_size, hidden_size, vb.pp("o_proj"))?
        };

        let sliding_window = if cfg.is_sliding_window(layer_idx) {
            Some(cfg.sliding_window)
        } else {
            None
        };

        let rope = GptOssRoPE::new(cfg, vb.device())?;

        // Load attention sinks if present
        let sinks = vb.get((cfg.num_attention_heads,), "sinks").ok();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope,
            sliding_window,
            sinks,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut (Tensor, Tensor),
    ) -> Result<Tensor> {
        let (b, seq_len, _hidden_size) = x.dims3()?;

        // Project to Q, K, V
        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let mut v = self.v_proj.forward(x)?;

        // Reshape to (batch, num_heads, seq_len, head_dim)
        q = q
            .reshape((b, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        (q, k) = self.rope.forward(&q, &k, seqlen_offset)?;

        // Update KV cache
        let (k_cache, v_cache) = kv_cache;
        let k = if seqlen_offset == 0 {
            *k_cache = k.clone();
            k
        } else {
            let k = Tensor::cat(&[k_cache.clone(), k], 2)?;
            *k_cache = k.clone();
            k
        };

        let v = if seqlen_offset == 0 {
            *v_cache = v.clone();
            v
        } else {
            let v = Tensor::cat(&[v_cache.clone(), v], 2)?;
            *v_cache = v.clone();
            v
        };

        // Repeat KV for Grouped Query Attention
        let n_kv_groups = self.num_attention_heads / self.num_key_value_heads;
        let k = self.repeat_kv(k, n_kv_groups)?;
        let v = self.repeat_kv(v, n_kv_groups)?;

        // Compute attention scores
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        attn_weights = (attn_weights * scale)?;

        // Apply causal mask
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        // Apply attention sinks if present
        if let Some(ref sinks) = self.sinks {
            let sinks_expanded = sinks
                .reshape((1, self.num_attention_heads, 1, 1))?
                .broadcast_as(attn_weights.shape())?;
            attn_weights = Tensor::cat(&[&attn_weights, &sinks_expanded], D::Minus1)?;
        }

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Drop sinks after softmax
        let attn_weights = if self.sinks.is_some() {
            attn_weights.narrow(D::Minus1, 0, attn_weights.dim(D::Minus1)? - 1)?
        } else {
            attn_weights
        };

        // Apply attention to values
        let mut y = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, hidden_size)
        y = y.transpose(1, 2)?.reshape((b, seq_len, ()))?;

        // Output projection
        self.o_proj.forward(&y)
    }

    /// Repeat KV tensors for Grouped Query Attention
    fn repeat_kv(&self, x: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x);
        }

        let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((b, n_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b, n_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_shapes() {
        // TODO: Add unit tests for RoPE
    }

    #[test]
    fn test_attention_shapes() {
        // TODO: Add unit tests for attention
    }
}
