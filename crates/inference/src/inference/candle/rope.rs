//! Custom `RoPE` (Rotary Position Embedding) implementation
//!
//! Based on Meta's Llama3 implementation: <https://github.com/meta-llama/llama3/blob/main/llama/model.py>
//! Handles BF16/F16 tensors properly by converting to F32 for computation and back to original dtype

use candle_core::{DType, Device, Result, Tensor};

/// Precompute rotary embedding frequencies
/// Equivalent to Meta's `precompute_freqs_cis` function
pub fn precompute_freqs_cis(
    dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &Device,
) -> Result<Tensor> {
    // Generate frequency values: freqs = 1.0 / (theta ** (arange(0, dim, 2) / dim))
    let half_dim = dim / 2;
    #[allow(clippy::cast_precision_loss)]
    let indices: Vec<f64> = (0..half_dim).map(|i| (i * 2) as f64 / dim as f64).collect();

    let freqs: Vec<f64> = indices.iter().map(|&idx| 1.0 / theta.powf(idx)).collect();

    let freqs_tensor = Tensor::from_vec(freqs, (half_dim,), device)?.to_dtype(DType::F32)?;

    // Create position indices: t = arange(max_seq_len)
    #[allow(clippy::cast_precision_loss)]
    let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let positions_tensor = Tensor::from_vec(positions, (max_seq_len,), device)?;

    // Compute outer product: freqs = outer(t, freqs)
    let freqs_expanded = positions_tensor
        .unsqueeze(1)? // Shape: (max_seq_len, 1)
        .broadcast_mul(&freqs_tensor.unsqueeze(0)?)?; // Shape: (max_seq_len, half_dim)

    // Convert to complex representation using cos and sin
    // In PyTorch: freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    let cos_freqs = freqs_expanded.cos()?;
    let sin_freqs = freqs_expanded.sin()?;

    // Stack cos and sin to create complex representation
    // Shape: (max_seq_len, half_dim, 2) where last dim is [cos, sin]
    let freqs_cis = Tensor::stack(&[&cos_freqs, &sin_freqs], 2)?;

    Ok(freqs_cis)
}

/// Apply rotary embeddings to query and key tensors
/// Equivalent to Meta's `apply_rotary_emb` function
pub fn apply_rotary_emb(
    xq: &Tensor,
    xk: &Tensor,
    freqs_cis: &Tensor,
    start_pos: usize,
) -> Result<(Tensor, Tensor)> {
    let original_dtype = xq.dtype();
    let seq_len = xq.dims()[1]; // Assuming shape: (batch, seq_len, heads, head_dim)

    // Convert inputs to F32 for computation (like Meta's .float())
    let xq_f32 = xq.to_dtype(DType::F32)?;
    let xk_f32 = xk.to_dtype(DType::F32)?;

    // Get the relevant frequency slice for current sequence
    let freqs_slice = freqs_cis.narrow(0, start_pos, seq_len)?;

    // Reshape tensors for complex operations
    let (batch_size, seq_len, num_heads, head_dim) = xq.dims4()?;
    let half_head_dim = head_dim / 2;

    // Reshape to prepare for complex multiplication
    // Split head_dim into pairs for complex representation
    let xq_reshaped = xq_f32.reshape((batch_size, seq_len, num_heads, half_head_dim, 2))?;
    let xk_reshaped = xk_f32.reshape((batch_size, seq_len, num_heads, half_head_dim, 2))?;

    // Extract real and imaginary parts
    let xq_real = xq_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
    let xq_imag = xq_reshaped.narrow(4, 1, 1)?.squeeze(4)?;
    let xk_real = xk_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
    let xk_imag = xk_reshaped.narrow(4, 1, 1)?.squeeze(4)?;

    // Get cos and sin from precomputed frequencies
    let cos_vals = freqs_slice.narrow(2, 0, 1)?.squeeze(2)?; // Shape: (seq_len, half_head_dim)
    let sin_vals = freqs_slice.narrow(2, 1, 1)?.squeeze(2)?; // Shape: (seq_len, half_head_dim)

    // Broadcast to match tensor dimensions
    let cos_broadcast = cos_vals
        .unsqueeze(0)? // batch dim
        .unsqueeze(2)? // heads dim
        .broadcast_as(xq_real.shape())?;
    let sin_broadcast = sin_vals
        .unsqueeze(0)? // batch dim
        .unsqueeze(2)? // heads dim
        .broadcast_as(xq_real.shape())?;

    // Apply rotation: (real + i*imag) * (cos + i*sin) = (real*cos - imag*sin) + i*(real*sin + imag*cos)
    let xq_real_cos = xq_real.mul(&cos_broadcast)?;
    let xq_imag_sin = xq_imag.mul(&sin_broadcast)?;
    let xq_real_sin = xq_real.mul(&sin_broadcast)?;
    let xq_imag_cos = xq_imag.mul(&cos_broadcast)?;

    let xq_real_rotated = xq_real_cos.sub(&xq_imag_sin)?;
    let xq_imag_rotated = xq_real_sin.add(&xq_imag_cos)?;

    let xk_real_cos = xk_real.mul(&cos_broadcast)?;
    let xk_imag_sin = xk_imag.mul(&sin_broadcast)?;
    let xk_real_sin = xk_real.mul(&sin_broadcast)?;
    let xk_imag_cos = xk_imag.mul(&cos_broadcast)?;

    let xk_real_rotated = xk_real_cos.sub(&xk_imag_sin)?;
    let xk_imag_rotated = xk_real_sin.add(&xk_imag_cos)?;

    // Reconstruct the tensors by interleaving real and imaginary parts
    let xq_out = Tensor::stack(&[&xq_real_rotated, &xq_imag_rotated], 4)?
        .reshape((batch_size, seq_len, num_heads, head_dim))?;
    let xk_out = Tensor::stack(&[&xk_real_rotated, &xk_imag_rotated], 4)?
        .reshape((batch_size, seq_len, num_heads, head_dim))?;

    // Convert back to original dtype (like Meta's .type_as(xq))
    let xq_final = if original_dtype == DType::F32 {
        xq_out
    } else {
        xq_out.to_dtype(original_dtype)?
    };

    let xk_final = if original_dtype == DType::F32 {
        xk_out
    } else {
        xk_out.to_dtype(original_dtype)?
    };

    Ok((xq_final, xk_final))
}

/// Utility function to reshape frequencies for broadcasting
/// Equivalent to Meta's `reshape_for_broadcast` function
pub fn reshape_for_broadcast(freqs_cis: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let mut broadcast_shape = vec![1; target_shape.len()];
    let freqs_dims = freqs_cis.dims();

    // Copy the relevant dimensions
    let copy_len = freqs_dims.len().min(broadcast_shape.len());
    broadcast_shape[..copy_len].copy_from_slice(&freqs_dims[..copy_len]);

    freqs_cis.reshape(broadcast_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_precompute_freqs_cis() {
        let device = Device::Cpu;
        let dim = 128;
        let max_seq_len = 512;
        let theta = 10000.0;

        let freqs = precompute_freqs_cis(dim, max_seq_len, theta, &device).unwrap();
        assert_eq!(freqs.dims(), &[max_seq_len, dim / 2, 2]);
    }

    #[test]
    fn test_apply_rotary_emb() {
        let device = Device::Cpu;
        let batch_size = 1;
        let seq_len = 10;
        let num_heads = 8;
        let head_dim = 64;

        // Create dummy tensors
        let xq = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let xk = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let freqs_cis = precompute_freqs_cis(head_dim, 512, 10000.0, &device).unwrap();

        let (xq_rot, xk_rot) = apply_rotary_emb(&xq, &xk, &freqs_cis, 0).unwrap();

        assert_eq!(xq_rot.dims(), xq.dims());
        assert_eq!(xk_rot.dims(), xk.dims());
        assert_eq!(xq_rot.dtype(), xq.dtype());
        assert_eq!(xk_rot.dtype(), xk.dtype());
    }
}
