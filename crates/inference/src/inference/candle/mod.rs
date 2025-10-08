//! Candle inference engine implementation split into modular components

pub mod backend;
pub mod engine;
pub mod model_config;
pub mod openai_engine;
pub mod openai_model;
pub mod quantized_llama;
pub mod quantized_model;
pub mod simple_quantized_llama;
pub mod tokenizer;

#[cfg(test)]
pub mod debug_tensors;

#[cfg(test)]
pub mod test;

#[cfg(test)]
pub mod tokenizer_test;

pub use backend::CandleBackendType;
pub use engine::CandleInferenceEngine;
pub use model_config::CandleModelConfig;
pub use openai_engine::OpenAIEngine;
pub use openai_model::{OpenAIConfig, OpenAIModel};
pub use quantized_model::{CompressedTensorsLoader, QuantizedModelConfig};
pub use tokenizer::CandleTokenizer;
