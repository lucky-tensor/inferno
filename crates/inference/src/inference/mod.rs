//! Multi-backend inference engine implementation
//!
//! This module implements inference engines using both Burn and Candle frameworks
//! for flexible model deployment across different hardware backends.

// Common inference traits and types
pub mod traits;
pub use traits::*;

// Burn framework engine (original implementation)
#[cfg(feature = "burn-cpu")]
pub mod burn_engine;

// Candle framework engine (new optimized implementation)
#[cfg(any(
    feature = "candle-cpu",
    feature = "candle-cuda",
    feature = "candle-metal"
))]
pub mod candle;

// Re-export engines when features are enabled
#[cfg(feature = "burn-cpu")]
pub use burn_engine::BurnInferenceEngine;

#[cfg(any(
    feature = "candle-cpu",
    feature = "candle-cuda",
    feature = "candle-metal"
))]
pub use candle::{CandleBackendType, CandleInferenceEngine};

/// Create an inference engine based on the specified engine type
pub fn create_engine(engine_type: EngineType) -> Box<dyn InferenceEngine<Error = InferenceError>> {
    match engine_type {
        EngineType::BurnCpu => {
            #[cfg(feature = "burn-cpu")]
            {
                Box::new(BurnInferenceEngine::new())
            }
            #[cfg(not(feature = "burn-cpu"))]
            {
                panic!("Burn CPU engine requested but burn-cpu feature not enabled")
            }
        }
        EngineType::CandleCpu => {
            #[cfg(any(
                feature = "candle-cpu",
                feature = "candle-cuda",
                feature = "candle-metal"
            ))]
            {
                Box::new(CandleInferenceEngine::with_backend(CandleBackendType::Cpu))
            }
            #[cfg(not(any(
                feature = "candle-cpu",
                feature = "candle-cuda",
                feature = "candle-metal"
            )))]
            {
                panic!("Candle CPU engine requested but candle features not enabled")
            }
        }
        #[cfg(feature = "candle-cuda")]
        EngineType::CandleCuda => {
            Box::new(CandleInferenceEngine::with_backend(CandleBackendType::Cuda))
        }
        #[cfg(all(feature = "candle-metal", target_os = "macos"))]
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
        // Test that we can create engines when features are enabled
        #[cfg(any(
            feature = "candle-cpu",
            feature = "candle-cuda",
            feature = "candle-metal"
        ))]
        {
            let _engine = create_engine(EngineType::CandleCpu);
        }

        #[cfg(feature = "burn-cpu")]
        {
            let _engine = create_engine(EngineType::BurnCpu);
        }
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

        #[cfg(feature = "candle-cuda")]
        {
            assert_eq!(format!("{}", EngineType::CandleCuda), "Candle-CUDA");
        }
    }
}
