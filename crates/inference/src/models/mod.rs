//! Model implementations

pub mod llama_loader;
pub mod tests;

#[cfg(feature = "burn-cpu")]
pub use llama_loader::load_llama_weights;
