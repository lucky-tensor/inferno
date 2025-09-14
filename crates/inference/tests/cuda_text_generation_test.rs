//! Real CUDA Text Generation Test
//!
//! This test performs actual text generation with realistic prompts
//! to demonstrate the full inference pipeline working with CUDA acceleration.

use inferno_inference::{
    config::VLLMConfig,
    inference::{BurnBackendType, BurnInferenceEngine, InferenceRequest},
};

#[cfg(feature = "burn-cuda")]
#[tokio::test]
async fn test_cuda_real_text_generation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing CUDA Real Text Generation with SafeTensors");
    println!("==================================================");

    // Create CUDA engine
    let mut cuda_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cuda);

    // Verify CUDA backend is selected
    assert_eq!(*cuda_engine.backend_type(), BurnBackendType::Cuda);
    println!("✅ CUDA backend confirmed");

    // Create config with specific model name
    let config = VLLMConfig {
        model_name: "smollm2-135m".to_string(),
        model_path: "../../models".to_string(),
        device_id: 0, // Use GPU 0
        ..Default::default()
    };

    println!("🔧 Initializing CUDA engine with SafeTensors...");
    println!("📁 Model path: {}", config.model_path);
    println!("🤖 Model: {}", config.model_name);

    // Initialize the engine
    cuda_engine.initialize(config).await?;
    println!("🎉 CUDA engine initialized successfully!");

    // Verify engine is ready
    assert!(
        cuda_engine.is_ready(),
        "CUDA engine should be ready after initialization"
    );
    println!("✅ CUDA engine is ready for inference");

    // Test case 1: Simple completion
    println!("\n🎯 Test 1: Simple completion");
    println!("----------------------------");
    let request1 = InferenceRequest {
        request_id: 1,
        prompt: "The capital of France is".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        top_p: 0.9,
        seed: Some(42),
    };

    let response1 = cuda_engine.process(request1)?;
    println!("📝 Prompt: 'The capital of France is'");
    println!("🤖 Response: '{}'", response1.generated_text);
    println!("⚡ Inference time: {:.2}ms", response1.inference_time_ms);
    println!("🔢 Tokens: {}", response1.generated_tokens);

    assert!(
        !response1.generated_text.is_empty(),
        "Response should not be empty"
    );
    assert!(
        response1.inference_time_ms > 0.0,
        "Inference time should be positive"
    );
    assert!(
        response1.generated_tokens > 0,
        "Should generate at least one token"
    );

    // Test case 2: Story beginning
    println!("\n🎯 Test 2: Creative story beginning");
    println!("----------------------------------");
    let request2 = InferenceRequest {
        request_id: 2,
        prompt: "Once upon a time, in a magical forest,".to_string(),
        max_tokens: 20,
        temperature: 0.8,
        top_p: 0.95,
        seed: Some(123),
    };

    let response2 = cuda_engine.process(request2)?;
    println!("📝 Prompt: 'Once upon a time, in a magical forest,'");
    println!("🤖 Response: '{}'", response2.generated_text);
    println!("⚡ Inference time: {:.2}ms", response2.inference_time_ms);
    println!("🔢 Tokens: {}", response2.generated_tokens);

    assert!(
        !response2.generated_text.is_empty(),
        "Story response should not be empty"
    );
    assert!(
        response2.generated_tokens > 0,
        "Should generate story tokens"
    );

    // Test case 3: Technical explanation
    println!("\n🎯 Test 3: Technical explanation");
    println!("-------------------------------");
    let request3 = InferenceRequest {
        request_id: 3,
        prompt: "Machine learning is".to_string(),
        max_tokens: 25,
        temperature: 0.5,
        top_p: 0.9,
        seed: Some(456),
    };

    let response3 = cuda_engine.process(request3)?;
    println!("📝 Prompt: 'Machine learning is'");
    println!("🤖 Response: '{}'", response3.generated_text);
    println!("⚡ Inference time: {:.2}ms", response3.inference_time_ms);
    println!("🔢 Tokens: {}", response3.generated_tokens);

    assert!(
        !response3.generated_text.is_empty(),
        "Technical response should not be empty"
    );

    // Test case 4: Conversation
    println!("\n🎯 Test 4: Conversational response");
    println!("---------------------------------");
    let request4 = InferenceRequest {
        request_id: 4,
        prompt: "Hello! How are you doing today?".to_string(),
        max_tokens: 15,
        temperature: 0.6,
        top_p: 0.9,
        seed: Some(789),
    };

    let response4 = cuda_engine.process(request4)?;
    println!("📝 Prompt: 'Hello! How are you doing today?'");
    println!("🤖 Response: '{}'", response4.generated_text);
    println!("⚡ Inference time: {:.2}ms", response4.inference_time_ms);
    println!("🔢 Tokens: {}", response4.generated_tokens);

    assert!(
        !response4.generated_text.is_empty(),
        "Conversation response should not be empty"
    );

    // Performance summary
    println!("\n📊 Performance Summary");
    println!("=====================");
    let avg_time = (response1.inference_time_ms
        + response2.inference_time_ms
        + response3.inference_time_ms
        + response4.inference_time_ms)
        / 4.0;
    let total_tokens = response1.generated_tokens
        + response2.generated_tokens
        + response3.generated_tokens
        + response4.generated_tokens;

    println!("🏃 Average inference time: {:.2}ms", avg_time);
    println!("🔢 Total tokens generated: {}", total_tokens);
    println!(
        "⚡ Tokens per second: {:.1}",
        total_tokens as f64 / (avg_time / 1000.0)
    );

    // Cleanup
    cuda_engine.shutdown()?;
    println!("\n✅ CUDA engine shut down cleanly");
    println!("🎉 Real text generation test completed successfully!");

    Ok(())
}

#[cfg(feature = "burn-cuda")]
#[tokio::test]
async fn test_cuda_longer_generation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing CUDA Longer Text Generation");
    println!("=====================================");

    let mut cuda_engine = BurnInferenceEngine::with_backend(BurnBackendType::Cuda);

    let config = VLLMConfig {
        model_name: "smollm2-135m".to_string(),
        model_path: "../../models".to_string(),
        device_id: 0,
        ..Default::default()
    };

    cuda_engine.initialize(config).await?;

    // Test longer generation
    let request = InferenceRequest {
        request_id: 5,
        prompt: "In the year 2030, artificial intelligence will".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        seed: Some(999),
    };

    println!("📝 Prompt: '{}'", request.prompt);
    println!("🎯 Generating {} tokens...", request.max_tokens);

    let start = std::time::Instant::now();
    let response = cuda_engine.process(request)?;
    let end = start.elapsed();

    println!("\n🤖 Generated text:");
    println!("=================");
    println!("{}", response.generated_text);

    println!("\n📊 Statistics:");
    println!("⚡ Total time: {:.2}ms", end.as_millis());
    println!("🔢 Tokens generated: {}", response.generated_tokens);
    println!(
        "🏃 Tokens/sec: {:.1}",
        response.generated_tokens as f64 / end.as_secs_f64()
    );
    println!(
        "💾 Model inference time: {:.2}ms",
        response.inference_time_ms
    );

    assert!(
        response.generated_tokens > 10,
        "Should generate substantial text"
    );
    assert!(
        !response.generated_text.trim().is_empty(),
        "Should generate meaningful text"
    );

    cuda_engine.shutdown()?;
    println!("\n✅ Longer generation test completed!");

    Ok(())
}

#[cfg(not(feature = "burn-cuda"))]
#[test]
fn test_cuda_feature_required() {
    println!("❌ CUDA text generation test skipped - burn-cuda feature not enabled");
    println!("   Run with: cargo test --features burn-cuda");
    panic!("burn-cuda feature required for CUDA tests");
}
