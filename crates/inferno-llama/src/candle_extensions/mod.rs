//! Candle extensions and forks for generic Llama support
//!
//! This module contains selective forks and extensions of candle-transformers
//! components that require customization for our generic Llama engine.
//!
//! We leverage Candle's proven tensor operations and memory management while
//! extending only the components that need modification for:
//! - Multiple model variant support (Meta Llama, TinyLlama, distilled models)
//! - Advanced quantization (w8a8, compressed-tensors)
//! - High-precision RoPE for BF16/F16 models
//! - Generic configuration parsing

pub mod llama_models;

// Re-export key types for easy access
pub use llama_models::{
    Config as GenericConfig, GenericLlamaConfig, Llama3RopeConfig, Llama3RopeType, LlamaEosToks,
    LlamaVariant,
};
