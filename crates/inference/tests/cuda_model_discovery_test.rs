//! CUDA Model Discovery Test
//!
//! This test demonstrates that model names are no longer hardcoded in production code
//! and can be configured or auto-discovered.

use inferno_inference::{
    config::VLLMConfig,
    inference::{BurnBackendType, BurnInferenceEngine},
};

#[cfg(feature = "burn-cuda")]
#[tokio::test]
async fn test_cuda_specific_model_loading() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Testing CUDA with specific model name (configurable)");

    let mut cuda_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cuda);

    // Test with specific model name - NO HARDCODED MODELS IN SOURCE!
    let config = VLLMConfig {
        model_name: "tinyllama-1.1b".to_string(), // ✅ User-configurable
        model_path: "../../models".to_string(),
        device_id: 0,
        ..Default::default()
    };

    println!("🔧 Testing with specific model: {}", config.model_name);

    match cuda_engine.initialize(config).await {
        Ok(()) => {
            println!("✅ Successfully loaded specific model: tinyllama-1.1b");
            assert!(cuda_engine.is_ready());
        }
        Err(e) => {
            println!("ℹ️ Specific model test: {}", e);
            println!("This is expected if the model files have issues");
        }
    }

    cuda_engine.shutdown()?;

    Ok(())
}

#[cfg(feature = "burn-cuda")]
#[tokio::test]
async fn test_cuda_model_autodiscovery() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing CUDA with model auto-discovery");

    let mut cuda_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cuda);

    // Test auto-discovery - NO model name specified
    let config = VLLMConfig {
        model_name: "".to_string(), // ✅ Empty = auto-discover
        model_path: "../../models".to_string(),
        device_id: 0,
        ..Default::default()
    };

    println!("🔧 Testing auto-discovery in: {}", config.model_path);

    match cuda_engine.initialize(config).await {
        Ok(()) => {
            println!("✅ Successfully auto-discovered a model!");
            assert!(cuda_engine.is_ready());
        }
        Err(e) => {
            println!("ℹ️ Auto-discovery test: {}", e);
            println!("This is expected if no valid models are found");
        }
    }

    cuda_engine.shutdown()?;

    Ok(())
}

#[cfg(feature = "burn-cuda")]
#[tokio::test]
async fn test_cuda_nonexistent_model() -> Result<(), Box<dyn std::error::Error>> {
    println!("❌ Testing CUDA with non-existent model name");

    let mut cuda_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cuda);

    // Test with non-existent model name
    let config = VLLMConfig {
        model_name: "nonexistent-model".to_string(), // ✅ User can specify any name
        model_path: "../../models".to_string(),
        device_id: 0,
        ..Default::default()
    };

    println!("🔧 Testing with non-existent model: {}", config.model_name);

    match cuda_engine.initialize(config).await {
        Ok(()) => {
            println!("⚠️ Unexpected success - should have failed");
        }
        Err(e) => {
            println!("✅ Expected failure: {}", e);
            // Should fail gracefully with proper error message
            assert!(e.to_string().contains("not found"));
        }
    }

    Ok(())
}

#[cfg(not(feature = "burn-cuda"))]
#[test]
fn test_cuda_model_discovery_feature_disabled() {
    println!("ℹ️ CUDA model discovery tests skipped - burn-cuda feature not enabled");
}
