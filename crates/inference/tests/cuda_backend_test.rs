//! Minimal CUDA Backend Test
//!
//! This test verifies CUDA backend creation and basic functionality
//! without requiring full model loading.

use inferno_inference::inference::{BurnBackendType, BurnInferenceEngine};

#[cfg(feature = "burn-cuda")]
#[test]
fn test_cuda_backend_creation() {
    println!("🧪 Testing CUDA backend creation");

    // Test CUDA backend creation
    let cuda_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cuda);
    println!("✅ CUDA engine created successfully");

    // Verify backend type
    assert_eq!(*cuda_engine.backend_type(), BurnBackendType::Cuda);
    println!("✅ CUDA backend type verified");

    // Test smart constructor
    let smart_engine = BurnInferenceEngine::with_cuda();
    println!("✅ Smart CUDA engine created (with fallback)");

    // Since we have CUDA available, it should prefer CUDA
    assert_eq!(*smart_engine.backend_type(), BurnBackendType::Cuda);
    println!("✅ Smart constructor correctly selected CUDA backend");

    // Test engine state
    assert!(
        !cuda_engine.is_ready(),
        "Engine should not be ready without initialization"
    );
    assert!(
        !smart_engine.is_ready(),
        "Smart engine should not be ready without initialization"
    );

    println!("🎉 CUDA backend creation test passed!");
}

#[cfg(feature = "burn-cuda")]
#[test]
fn test_cuda_backend_comparison() {
    println!("🔄 Testing backend type comparison");

    let cpu_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cpu);
    let cuda_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cuda);

    // Test PartialEq implementation
    assert_eq!(*cpu_engine.backend_type(), BurnBackendType::Cpu);
    assert_eq!(*cuda_engine.backend_type(), BurnBackendType::Cuda);
    assert_ne!(*cpu_engine.backend_type(), *cuda_engine.backend_type());

    println!("✅ Backend type comparison works correctly");
}

#[cfg(not(feature = "burn-cuda"))]
#[test]
fn test_cuda_feature_disabled() {
    println!("ℹ️ CUDA backend tests skipped - burn-cuda feature not enabled");
    println!("   Run with: cargo test --features burn-cuda");
}
