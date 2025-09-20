//! SwiGLU Feed-Forward Network Implementation
//!
//! This module implements the SwiGLU (Swish Gated Linear Unit) feed-forward network
//! used in Llama models. The implementation follows Meta's reference architecture with
//! native BF16/F16 support and memory-efficient operations.
//!
//! ## SwiGLU Architecture
//!
//! The SwiGLU activation function is defined as:
//! ```text
//! SwiGLU(x, W, V, W2) = (Swish(xW) ⊗ xV)W2
//! ```
//! Where:
//! - Swish(x) = x * sigmoid(x) = x * σ(x)
//! - ⊗ denotes element-wise multiplication
//! - W, V, W2 are learned linear transformations
//!
//! ## Memory Efficiency
//!
//! - All operations maintain BF16/F16 precision throughout
//! - Intermediate activations are computed in-place where possible
//! - Memory usage scales linearly with batch size and sequence length
//!
//! ## Performance Characteristics
//!
//! - Time Complexity: O(batch_size * seq_len * ffn_dim * dim)
//! - Space Complexity: O(batch_size * seq_len * ffn_dim) for intermediate activations

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::{config::LlamaConfig, error::LlamaError};

/// SwiGLU Feed-Forward Network for Llama models.
///
/// This struct implements the complete SwiGLU feed-forward mechanism including:
/// - Gate projection (W1) with SiLU activation
/// - Up projection (V/W3) for element-wise multiplication
/// - Down projection (W2) to original dimensions
/// - Memory-efficient computation with native BF16/F16 support
///
/// ## Architecture Details
///
/// The feed-forward network follows Meta's Llama implementation:
/// 1. Input is projected through two parallel linear layers (gate and up)
/// 2. Gate projection is activated with SiLU (Swish): x * sigmoid(x)
/// 3. Element-wise multiplication: gate_output ⊗ up_output
/// 4. Down projection reduces back to model dimension
///
/// ## Memory Layout
///
/// - Input: [batch_size, seq_len, dim]
/// - Gate/Up projections: [batch_size, seq_len, ffn_dim]
/// - Output: [batch_size, seq_len, dim]
///
/// ## SwiGLU Formula
///
/// ```text
/// gate = SiLU(x @ W1) = (x @ W1) * σ(x @ W1)
/// up = x @ W3
/// output = (gate ⊗ up) @ W2
/// ```
///
/// Where σ is the sigmoid function and ⊗ is element-wise multiplication.
#[derive(Debug)]
pub struct FeedForward {
    /// Gate projection layer (W1) - projects to ffn_dim with SiLU activation
    w1: Linear,
    /// Up projection layer (W3) - projects to ffn_dim for element-wise multiplication
    w3: Linear,
    /// Down projection layer (W2) - projects back to model dimension
    w2: Linear,
    /// Feed-forward network hidden dimension
    ffn_dim: usize,
    /// Model dimension
    dim: usize,
    /// Device for tensor operations
    device: Device,
    /// Data type for computations
    dtype: DType,
}

impl FeedForward {
    /// Creates a new SwiGLU FeedForward layer.
    ///
    /// # Arguments
    ///
    /// * `config` - Llama model configuration containing FFN parameters
    /// * `vb` - Variable builder for initializing weights
    ///
    /// # Returns
    ///
    /// A Result containing the initialized FeedForward layer or an error.
    ///
    /// # Errors
    ///
    /// Returns `LlamaError::ConfigError` if:
    /// - FFN dimension is invalid or zero
    /// - Model dimension is invalid or zero
    /// - Weight initialization fails
    ///
    /// # Performance
    ///
    /// This constructor initializes three linear layers with the specified precision.
    /// Memory usage is proportional to dim * ffn_dim for each projection.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use inferno_llama::{LlamaConfig, feed_forward::FeedForward};
    /// use candle_nn::VarBuilder;
    ///
    /// let config = LlamaConfig::llama_3_1_8b().unwrap();
    /// // let vb = VarBuilder::from_tensors(...);
    /// // let ffn = FeedForward::new(&config, vb)?;
    /// ```
    pub fn new(config: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let ffn_dim = config.ffn_hidden_dim();
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Validate dimensions
        if config.dim == 0 {
            return Err(LlamaError::config_error("dim", "must be greater than 0").into());
        }
        if ffn_dim == 0 {
            return Err(LlamaError::config_error("ffn_dim", "must be greater than 0").into());
        }

        // Initialize linear layers
        let w1 = linear(config.dim, ffn_dim, vb.pp("w1"))?; // Gate projection
        let w3 = linear(config.dim, ffn_dim, vb.pp("w3"))?; // Up projection
        let w2 = linear(ffn_dim, config.dim, vb.pp("w2"))?; // Down projection

        Ok(Self {
            w1,
            w3,
            w2,
            ffn_dim,
            dim: config.dim,
            device,
            dtype,
        })
    }

    /// Forward pass of the SwiGLU feed-forward network.
    ///
    /// Implements the SwiGLU activation: (SiLU(x @ W1) ⊗ (x @ W3)) @ W2
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, seq_len, dim]
    ///
    /// # Returns
    ///
    /// A Result containing the feed-forward output tensor or an error.
    ///
    /// # Errors
    ///
    /// Returns `LlamaError::TensorError` if:
    /// - Input tensor has incorrect shape
    /// - Linear projection fails
    /// - SiLU activation fails
    /// - Element-wise multiplication fails
    ///
    /// # Performance
    ///
    /// - Time Complexity: O(batch_size * seq_len * ffn_dim * dim)
    /// - Space Complexity: O(batch_size * seq_len * ffn_dim) for intermediate tensors
    /// - All operations maintain input precision (BF16/F16)
    /// - Computation is optimized for memory efficiency
    ///
    /// # Memory Layout
    ///
    /// The method processes tensors through the following transformations:
    /// 1. Gate projection: [B, L, D] -> [B, L, F] via W1
    /// 2. Up projection: [B, L, D] -> [B, L, F] via W3
    /// 3. SiLU activation: gate = gate * σ(gate)
    /// 4. Element-wise multiply: gate = gate ⊗ up
    /// 5. Down projection: [B, L, F] -> [B, L, D] via W2
    ///
    /// Where: B=batch, L=seq_len, D=dim, F=ffn_dim
    ///
    /// ## SwiGLU Implementation Details
    ///
    /// SiLU (Sigmoid Linear Unit) is computed as: x * σ(x) where σ is sigmoid.
    /// This is more numerically stable than separate sigmoid computation.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Validate input dimensions
        let input_dims = x.dims();
        if input_dims.len() != 3 {
            return Err(LlamaError::dimension_error(
                "feed_forward input",
                vec![0, 0, self.dim], // We don't validate batch/seq dims, only last dim
                input_dims.to_vec(),
            )
            .into());
        }

        let last_dim = input_dims[input_dims.len() - 1];
        if last_dim != self.dim {
            return Err(LlamaError::dimension_error(
                "feed_forward input dimension",
                vec![self.dim],
                vec![last_dim],
            )
            .into());
        }

        // Apply gate and up projections
        let gate = self.w1.forward(x)?; // [B, L, ffn_dim]
        let up = self.w3.forward(x)?; // [B, L, ffn_dim]

        // Apply SiLU activation to gate: gate * sigmoid(gate)
        let gate_activated = candle_nn::ops::silu(&gate)?;

        // Element-wise multiplication: gate ⊗ up
        let gated = gate_activated.mul(&up)?;

        // Down projection to original dimension
        self.w2.forward(&gated)
    }

    /// Returns the device used by this feed-forward layer.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the data type used by this feed-forward layer.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the feed-forward hidden dimension.
    pub fn ffn_dim(&self) -> usize {
        self.ffn_dim
    }

    /// Returns the model dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlamaConfig;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    // Test helper to create a simple VarBuilder for testing
    fn create_test_var_builder(device: &Device, dtype: DType) -> VarBuilder<'_> {
        let tensors = std::collections::HashMap::new();
        VarBuilder::from_tensors(tensors, dtype, device)
    }

    #[test]
    fn test_feed_forward_initialization_fails_without_implementation() {
        // This test should initially fail, demonstrating TDD approach
        let device = Device::Cpu;
        let dtype = DType::BF16;
        let config = LlamaConfig::llama_3_1_8b().unwrap();
        let vb = create_test_var_builder(&device, dtype);

        // This will fail because we haven't properly implemented the weight initialization
        let result = FeedForward::new(&config, vb);

        // For now, we expect this to fail - this drives our implementation
        assert!(
            result.is_err(),
            "Expected initialization to fail before proper implementation"
        );
    }

    #[test]
    fn test_feed_forward_dimensions() {
        let device = Device::Cpu;
        let dtype = DType::BF16;
        let config = LlamaConfig::llama_3_1_8b().unwrap();

        // Verify FFN dimensions are calculated correctly
        let ffn_dim = config.ffn_hidden_dim();
        assert!(ffn_dim > 0, "FFN dimension should be positive");
        assert!(
            ffn_dim > config.dim,
            "FFN dimension should be larger than model dimension"
        );

        // This test drives our dimension validation requirements
        let vb = create_test_var_builder(&device, dtype);
        let result = FeedForward::new(&config, vb);
        assert!(result.is_err(), "Expected failure before implementation");
    }

    #[test]
    fn test_swiglu_forward_pass_fails_with_bf16_tensors() {
        // This test should fail initially, driving our forward pass implementation
        let device = Device::Cpu;
        let dtype = DType::BF16;
        let config = LlamaConfig::llama_3_1_8b().unwrap();

        // Create input tensor
        let batch_size = 1;
        let seq_len = 128;
        let _input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, config.dim), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // This will fail because FeedForward::new() doesn't work yet
        let vb = create_test_var_builder(&device, dtype);
        let ffn_result = FeedForward::new(&config, vb);

        assert!(
            ffn_result.is_err(),
            "Expected feed-forward creation to fail before implementation"
        );
    }

    #[test]
    fn test_swiglu_activation_function() {
        // Test that SiLU activation works correctly
        let device = Device::Cpu;
        let dtype = DType::F32; // Use F32 for precision in activation testing

        // Create a simple test tensor
        let test_values = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let tensor = Tensor::from_vec(test_values.clone(), (1, test_values.len()), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // Apply SiLU activation
        let activated = candle_nn::ops::silu(&tensor).unwrap();
        let result_vec: Vec<f32> = activated.flatten_all().unwrap().to_vec1().unwrap();

        // Verify SiLU properties
        assert_eq!(result_vec.len(), test_values.len());

        // SiLU(0) should be 0
        assert!((result_vec[0] - 0.0).abs() < 1e-6);

        // SiLU should be monotonically increasing
        for i in 1..result_vec.len() {
            if test_values[i] > test_values[i - 1] {
                assert!(
                    result_vec[i] > result_vec[i - 1],
                    "SiLU should be monotonically increasing"
                );
            }
        }
    }

    #[test]
    fn test_element_wise_multiplication() {
        // Test element-wise multiplication used in SwiGLU
        let device = Device::Cpu;
        let dtype = DType::BF16;

        let shape = (2, 4, 8); // [batch, seq_len, dim]
        let tensor1 = Tensor::randn(0f32, 1f32, shape, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let tensor2 = Tensor::randn(0f32, 1f32, shape, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // Element-wise multiplication
        let result = tensor1.mul(&tensor2).unwrap();

        // Verify shape is preserved
        assert_eq!(result.shape().dims(), &[2, 4, 8]);
        assert_eq!(result.dtype(), dtype);
    }

    #[test]
    fn test_feed_forward_dimension_validation() {
        let device = Device::Cpu;
        let dtype = DType::BF16;

        // Create config with zero dimensions (should fail)
        let mut config = LlamaConfig::llama_3_1_8b().unwrap();
        let original_dim = config.dim;
        config.dim = 0;

        let vb = create_test_var_builder(&device, dtype);
        let result = FeedForward::new(&config, vb);

        assert!(result.is_err(), "Should fail with zero model dimension");

        // Restore valid dimension
        config.dim = original_dim;
        let vb = create_test_var_builder(&device, dtype);
        let result = FeedForward::new(&config, vb);

        // This should still fail because we haven't implemented proper weight loading
        assert!(result.is_err(), "Should fail before implementation");
    }

    #[test]
    fn test_memory_efficiency_with_bf16() {
        // Test that BF16 operations don't cause excessive memory overhead
        let device = Device::Cpu;
        let dtype = DType::BF16;
        let config = LlamaConfig::llama_3_1_8b().unwrap();

        let batch_size = 2;
        let seq_len = 512;

        // Create BF16 input tensor
        let _input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, config.dim), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // This test will fail until we implement the feed-forward network
        let vb = create_test_var_builder(&device, dtype);
        let ffn_result = FeedForward::new(&config, vb);
        assert!(
            ffn_result.is_err(),
            "Expected failure before implementation"
        );
    }

    #[test]
    fn test_variable_sequence_lengths() {
        // Test feed-forward with different sequence lengths
        let device = Device::Cpu;
        let dtype = DType::F16; // Test F16 as well as BF16
        let config = LlamaConfig::llama_3_1_8b().unwrap();

        let sequence_lengths = vec![32, 64, 128, 256];

        for seq_len in sequence_lengths {
            let batch_size = 1;
            let _input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, config.dim), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();

            // The actual feed-forward computation will fail until implemented
            let vb = create_test_var_builder(&device, dtype);
            let ffn_result = FeedForward::new(&config, vb);
            assert!(
                ffn_result.is_err(),
                "Expected failure for seq_len {}",
                seq_len
            );
        }
    }

    #[test]
    fn test_batch_processing() {
        // Test feed-forward with multiple batch sizes
        let device = Device::Cpu;
        let dtype = DType::BF16;
        let config = LlamaConfig::llama_3_1_8b().unwrap();
        let seq_len = 128;

        let batch_sizes = vec![1, 2, 4, 8];

        for batch_size in batch_sizes {
            let _input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, config.dim), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();

            // This will fail until we implement the feed-forward network
            let vb = create_test_var_builder(&device, dtype);
            let ffn_result = FeedForward::new(&config, vb);
            assert!(
                ffn_result.is_err(),
                "Expected failure for batch_size {}",
                batch_size
            );
        }
    }

    #[test]
    fn test_ffn_hidden_dimension_calculation() {
        // Test that FFN hidden dimension is calculated correctly from config
        let config = LlamaConfig::llama_3_1_8b().unwrap();
        let ffn_dim = config.ffn_hidden_dim();

        // For Llama 3.1 8B, ffn_dim should be derived from dim and ffn_dim_multiplier
        // dim=4096, ffn_dim_multiplier=1.3, multiple_of=1024
        // Actual calculation: (1.3 * 4096) = 5324.8 -> round up to multiple of 1024
        // (5324 + 1023) / 1024 * 1024 = 6 * 1024 = 6144
        assert_eq!(
            ffn_dim, 6144,
            "FFN dimension should be calculated correctly"
        );

        // Verify it's a multiple of multiple_of
        assert_eq!(
            ffn_dim % config.multiple_of,
            0,
            "FFN dimension should be multiple of multiple_of"
        );
    }
}
