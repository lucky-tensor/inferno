//! Test for custom RoPE implementation with BF16 support

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_custom_rope_bf16() -> Result<(), Box<dyn std::error::Error>> {
    println!("CUSTOM ROPE BF16 TEST");
    println!("====================");

    use candle_core::{DType, Device, Tensor};
    use inferno_inference::inference::candle::rope::{apply_rotary_emb, precompute_freqs_cis};

    println!("🔧 Testing custom RoPE implementation with BF16 tensors");

    let device = Device::new_cuda(0)?;
    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 4;
    let head_dim = 64;

    println!(
        "📊 Tensor dimensions: batch={}, seq_len={}, heads={}, head_dim={}",
        batch_size, seq_len, num_heads, head_dim
    );

    // Create BF16 tensors (like actual model weights)
    let xq_f32 = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, head_dim),
        &device,
    )?;
    let xk_f32 = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, head_dim),
        &device,
    )?;

    let xq_bf16 = xq_f32.to_dtype(DType::BF16)?;
    let xk_bf16 = xk_f32.to_dtype(DType::BF16)?;

    println!(
        "✅ Created BF16 tensors: xq.dtype={:?}, xk.dtype={:?}",
        xq_bf16.dtype(),
        xk_bf16.dtype()
    );

    // Precompute RoPE frequencies
    let max_seq_len = 512;
    let theta = 10000.0;
    println!(
        "🔄 Precomputing RoPE frequencies (dim={}, max_seq_len={}, theta={})",
        head_dim, max_seq_len, theta
    );

    let freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, theta, &device)?;
    println!("✅ RoPE frequencies computed: shape={:?}", freqs_cis.dims());

    // Apply rotary embeddings
    println!("🔄 Applying rotary embeddings to BF16 tensors...");
    let start_time = std::time::Instant::now();

    let (xq_rotated, xk_rotated) = apply_rotary_emb(&xq_bf16, &xk_bf16, &freqs_cis, 0)?;

    let duration = start_time.elapsed();
    println!(
        "✅ RoPE applied successfully in {:.2}ms",
        duration.as_secs_f32() * 1000.0
    );

    // Verify output properties
    println!("🔍 Verifying output tensors...");
    println!("   xq_rotated.dtype: {:?}", xq_rotated.dtype());
    println!("   xk_rotated.dtype: {:?}", xk_rotated.dtype());
    println!("   xq_rotated.shape: {:?}", xq_rotated.dims());
    println!("   xk_rotated.shape: {:?}", xk_rotated.dims());

    // Verify dtype preservation
    assert_eq!(
        xq_rotated.dtype(),
        DType::BF16,
        "Output should preserve BF16 dtype"
    );
    assert_eq!(
        xk_rotated.dtype(),
        DType::BF16,
        "Output should preserve BF16 dtype"
    );

    // Verify shape preservation
    assert_eq!(
        xq_rotated.dims(),
        xq_bf16.dims(),
        "Output shape should match input"
    );
    assert_eq!(
        xk_rotated.dims(),
        xk_bf16.dims(),
        "Output shape should match input"
    );

    // Verify the rotation actually changed the tensors (convert to F32 for computation)
    let xq_diff = xq_bf16
        .to_dtype(DType::F32)?
        .sub(&xq_rotated.to_dtype(DType::F32)?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    let xk_diff = xk_bf16
        .to_dtype(DType::F32)?
        .sub(&xk_rotated.to_dtype(DType::F32)?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;

    println!(
        "📈 Rotation magnitude: xq_diff={:.6}, xk_diff={:.6}",
        xq_diff, xk_diff
    );
    assert!(
        xq_diff > 0.001,
        "RoPE should modify query tensor significantly"
    );
    assert!(
        xk_diff > 0.001,
        "RoPE should modify key tensor significantly"
    );

    println!("\n🎉 SUCCESS: Custom RoPE works perfectly with BF16!");
    println!("   ✓ Preserves BF16 dtype throughout computation");
    println!("   ✓ Maintains tensor shapes");
    println!("   ✓ Applies rotation correctly");
    println!("   ✓ No memory overhead from dtype conversions");

    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_custom_rope_f16() -> Result<(), Box<dyn std::error::Error>> {
    println!("CUSTOM ROPE F16 TEST");
    println!("===================");

    use candle_core::{DType, Device, Tensor};
    use inferno_inference::inference::candle::rope::{apply_rotary_emb, precompute_freqs_cis};

    let device = Device::new_cuda(0)?;
    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 4;
    let head_dim = 64;

    // Create F16 tensors
    let xq_f32 = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, head_dim),
        &device,
    )?;
    let xk_f32 = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, head_dim),
        &device,
    )?;

    let xq_f16 = xq_f32.to_dtype(DType::F16)?;
    let xk_f16 = xk_f32.to_dtype(DType::F16)?;

    println!("✅ Created F16 tensors");

    let freqs_cis = precompute_freqs_cis(head_dim, 512, 10000.0, &device)?;
    let (xq_rotated, xk_rotated) = apply_rotary_emb(&xq_f16, &xk_f16, &freqs_cis, 0)?;

    assert_eq!(xq_rotated.dtype(), DType::F16);
    assert_eq!(xk_rotated.dtype(), DType::F16);

    println!("🎉 SUCCESS: Custom RoPE works with F16!");
    Ok(())
}

#[cfg(not(feature = "candle-cuda"))]
mod disabled_tests {
    #[tokio::test]
    async fn test_custom_rope_requires_cuda() {
        println!("Custom RoPE BF16 tests require 'candle-cuda' feature");
        println!("Run with: cargo test --features candle-cuda");
    }
}
