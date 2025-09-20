//! Multi-backend inference engine implementation
//!
//! This module implements inference engines using both Burn and Candle frameworks
//! for flexible model deployment across different hardware backends.

// Common inference traits and types
pub mod traits;
pub use traits::*;

// Burn framework engine (temporarily disabled due to upstream compilation issues)
// pub mod burn_engine;

// Candle framework engine (new optimized implementation)
pub mod candle;

// Re-export engines - always available
// BurnInferenceEngine temporarily disabled due to upstream Burn compilation issues
// pub use burn_engine::BurnInferenceEngine;
pub use candle::{CandleBackendType, CandleInferenceEngine};

/// Create an inference engine based on the specified engine type
pub fn create_engine(engine_type: EngineType) -> Box<dyn InferenceEngine<Error = InferenceError>> {
    match engine_type {
        EngineType::BurnCpu => {
            panic!("Burn engine temporarily disabled due to upstream compilation issues")
        }
        EngineType::CandleCpu => {
            Box::new(CandleInferenceEngine::with_backend(CandleBackendType::Cpu))
        }
        EngineType::CandleCuda => {
            Box::new(CandleInferenceEngine::with_backend(CandleBackendType::Cuda))
        }
        EngineType::CandleMetal => Box::new(CandleInferenceEngine::with_backend(
            CandleBackendType::Metal,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        // Test that we can create engines (Burn temporarily disabled)
        let _engine = create_engine(EngineType::CandleCpu);
        // let _engine = create_engine(EngineType::BurnCpu); // Temporarily disabled
    }

    #[test]
    fn test_inference_types() {
        let request = create_test_request("Hello, world!");
        assert_eq!(request.prompt, "Hello, world!");
        assert_eq!(request.max_tokens, 50);

        let math_request = create_math_test_request();
        assert_eq!(math_request.prompt, "What is 2+2? Answer with number only:");
        assert!((math_request.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_engine_type_display() {
        assert_eq!(format!("{}", EngineType::BurnCpu), "Burn-CPU");
        assert_eq!(format!("{}", EngineType::CandleCpu), "Candle-CPU");
        assert_eq!(format!("{}", EngineType::CandleCuda), "Candle-CUDA");
    }
}
