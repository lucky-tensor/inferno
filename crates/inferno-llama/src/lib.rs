//! # Inferno Llama
//!
//! A native BF16/F16 Llama implementation designed for high-performance distributed systems.
//! This crate provides memory-efficient Llama model components that work correctly with
//! BF16 and F16 precision without the dtype issues present in candle-transformers.
//!
//! ## Key Features
//!
//! - Native BF16/F16 support for all operations, especially RoPE (Rotary Position Embedding)
//! - Memory-efficient implementation that respects model parameter memory requirements
//! - Zero-allocation patterns where possible
//! - Comprehensive error handling and validation
//! - Performance-focused design for distributed inference workloads
//!
//! ## Memory Requirements
//!
//! This implementation is designed to respect the theoretical memory requirements of models:
//! - Llama 3.1 8B should use ~15GB, not 30GB due to precision issues
//! - All operations maintain precision without unnecessary F32 conversions
//!
//! ## Architecture
//!
//! The crate is organized around the core Llama components:
//! - `rope`: Rotary Position Embedding with native BF16/F16 support
//! - `config`: Model configuration matching Meta's reference implementation
//! - `attention`: Multi-head attention mechanism
//! - `feed_forward`: Feed-forward network layers
//! - `model`: Complete Llama model implementation

pub mod attention;
pub mod config;
pub mod error;
pub mod feed_forward;
pub mod loader;
pub mod model;
pub mod normalization;
pub mod precision;
pub mod rope;
pub mod simple_loader;
pub mod tokenizer;
pub mod transformer_block;

// Re-export key types for convenience
pub use attention::MultiHeadAttention;
pub use config::LlamaConfig;
pub use error::{LlamaError, Result};
pub use feed_forward::FeedForward;
pub use loader::ModelLoader;
pub use model::InfernoLlama;
pub use normalization::RMSNorm;
pub use precision::{
    DetectedModelInfo, ModelDetector, ModelFormat, PrecisionConfig, PrecisionError,
    QuantizationConfig, TensorOps,
};
pub use rope::{apply_rotary_emb, precompute_freqs_cis};
pub use tokenizer::{load_tokenizer_from_path, TokenizedInfernoLlama};
pub use transformer_block::TransformerBlock;

/// Version information for the crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_version_available() {
        // Check that version is available and non-empty
        assert!(!VERSION.is_empty());
    }
}
