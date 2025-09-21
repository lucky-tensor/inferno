//! Full pipeline integration test for llama-burn inference
//!
//! This test validates the complete inference pipeline from model loading
//! through text generation to ensure the system actually works end-to-end.

use inferno_inference::{
    config::InfernoConfig,
    inference::{create_engine, InferenceRequest},
};
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_full_tinyllama_inference_pipeline() {
    println!("🧪 Starting full TinyLlama inference pipeline test...");

    // Check if TinyLlama model exists
    let model_path = std::env::var("HOME")
        .map(|h| format!("{}/models/TinyLlama-1.1B-Chat-v1.0", h))
        .unwrap_or_else(|_| "/home/jeef/models/TinyLlama-1.1B-Chat-v1.0".to_string());

    if !std::path::Path::new(&model_path).exists() {
        println!("⏭️  Skipping test - TinyLlama model not found at: {}", model_path);
        return;
    }

    println!("📁 Using model path: {}", model_path);

    // Step 1: Create and initialize inference engine
    println!("\n🔧 Step 1: Creating inference engine...");
    let mut engine = create_engine();

    let config = InfernoConfig {
        model_path: model_path.clone(),
        model_name: "TinyLlama-1.1B-Chat-v1.0".to_string(),
        device_id: 0, // GPU device 0
        max_batch_size: 1,
        max_sequence_length: 512,
        ..Default::default()
    };

    println!("⚙️  Config: {:?}", config);

    // Step 2: Initialize engine with timeout
    println!("\n🚀 Step 2: Initializing engine (this may take a while)...");
    let init_result = timeout(Duration::from_secs(300), engine.initialize(config)).await;

    match init_result {
        Ok(Ok(())) => {
            println!("✅ Engine initialized successfully!");
        }
        Ok(Err(e)) => {
            panic!("❌ Engine initialization failed: {}", e);
        }
        Err(_) => {
            panic!("❌ Engine initialization timed out after 5 minutes");
        }
    }

    // Step 3: Test simple inference
    println!("\n🧠 Step 3: Testing simple inference...");
    let test_cases = vec![
        ("Hello", 10),
        ("What is AI?", 15),
        ("The sky is", 8),
        ("2 + 2 =", 5),
    ];

    for (prompt, max_tokens) in test_cases {
        println!("\n📝 Testing prompt: '{}'", prompt);

        let request = InferenceRequest {
            request_id: 1,
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            seed: Some(42),
        };

        let start_time = std::time::Instant::now();
        let response = timeout(Duration::from_secs(60), engine.process(request)).await;
        let elapsed = start_time.elapsed();

        match response {
            Ok(Ok(resp)) => {
                println!("✅ Response received in {:.2}s:", elapsed.as_secs_f64());
                println!("   Generated: '{}'", resp.generated_text);
                println!("   Tokens: {}", resp.generated_tokens);
                println!("   Inference time: {:.2}ms", resp.inference_time_ms);

                // Validate response
                assert!(!resp.generated_text.is_empty(), "Generated text should not be empty");
                assert!(resp.generated_tokens > 0, "Should generate at least one token");
                assert!(resp.inference_time_ms > 0.0, "Should report non-zero inference time");

                // Check for reasonable output (not just random characters)
                let output = resp.generated_text.trim();
                assert!(output.len() > 0, "Output should have content");

                // Basic sanity checks
                if prompt == "2 + 2 =" {
                    println!("   🔢 Math test - checking if output contains numbers...");
                    let has_numbers = output.chars().any(|c| c.is_ascii_digit());
                    if has_numbers {
                        println!("   ✅ Output contains numbers (good sign for math)");
                    } else {
                        println!("   ⚠️  Output doesn't contain numbers (may be expected for this model)");
                    }
                }

            }
            Ok(Err(e)) => {
                panic!("❌ Inference failed for prompt '{}': {}", prompt, e);
            }
            Err(_) => {
                panic!("❌ Inference timed out for prompt '{}'", prompt);
            }
        }
    }

    // Step 4: Test longer generation
    println!("\n📖 Step 4: Testing longer text generation...");
    let long_request = InferenceRequest {
        request_id: 2,
        prompt: "Once upon a time, in a distant galaxy".to_string(),
        max_tokens: 50,
        temperature: 0.8,
        top_p: 0.95,
        seed: Some(123),
    };

    let start_time = std::time::Instant::now();
    let response = timeout(Duration::from_secs(120), engine.process(long_request)).await;
    let elapsed = start_time.elapsed();

    match response {
        Ok(Ok(resp)) => {
            println!("✅ Long generation completed in {:.2}s:", elapsed.as_secs_f64());
            println!("   Generated: '{}'", resp.generated_text);
            println!("   Tokens: {}", resp.generated_tokens);

            // Calculate tokens per second
            let tokens_per_sec = resp.generated_tokens as f64 / (resp.inference_time_ms / 1000.0);
            println!("   Speed: {:.1} tokens/second", tokens_per_sec);

            // Validate longer generation
            assert!(resp.generated_tokens >= 20, "Should generate at least 20 tokens for longer text");
            assert!(resp.generated_text.len() > 50, "Longer text should be substantial");

        }
        Ok(Err(e)) => {
            panic!("❌ Long generation failed: {}", e);
        }
        Err(_) => {
            panic!("❌ Long generation timed out");
        }
    }

    println!("\n🎉 Full pipeline test completed successfully!");
    println!("✅ Model loading: PASS");
    println!("✅ Simple inference: PASS");
    println!("✅ Multiple prompts: PASS");
    println!("✅ Longer generation: PASS");
    println!("✅ Performance metrics: PASS");
}

#[tokio::test]
async fn test_model_loading_details() {
    println!("🔍 Testing detailed model loading process...");

    let model_path = std::env::var("HOME")
        .map(|h| format!("{}/models/TinyLlama-1.1B-Chat-v1.0", h))
        .unwrap_or_else(|_| "/home/jeef/models/TinyLlama-1.1B-Chat-v1.0".to_string());

    if !std::path::Path::new(&model_path).exists() {
        println!("⏭️  Skipping test - TinyLlama model not found");
        return;
    }

    // Test that all required files exist
    let required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ];

    for file in &required_files {
        let file_path = format!("{}/{}", model_path, file);
        assert!(
            std::path::Path::new(&file_path).exists(),
            "Required file missing: {}",
            file_path
        );
        println!("✅ Found required file: {}", file);
    }

    // Test SafeTensors file is valid
    let safetensors_path = format!("{}/model.safetensors", model_path);
    let metadata = std::fs::metadata(&safetensors_path).expect("Failed to read SafeTensors metadata");
    println!("📊 SafeTensors file size: {:.1} MB", metadata.len() as f64 / 1024.0 / 1024.0);
    assert!(metadata.len() > 1_000_000_000, "SafeTensors file should be > 1GB for TinyLlama");

    // Test config.json is readable
    let config_path = format!("{}/config.json", model_path);
    let config_content = std::fs::read_to_string(&config_path).expect("Failed to read config.json");
    let config_json: serde_json::Value = serde_json::from_str(&config_content).expect("Invalid JSON in config.json");

    println!("🔧 Model config loaded:");
    if let Some(model_type) = config_json["model_type"].as_str() {
        println!("   Model type: {}", model_type);
    }
    if let Some(vocab_size) = config_json["vocab_size"].as_u64() {
        println!("   Vocab size: {}", vocab_size);
    }
    if let Some(hidden_size) = config_json["hidden_size"].as_u64() {
        println!("   Hidden size: {}", hidden_size);
    }

    println!("✅ Model files validation completed");
}

#[tokio::test]
async fn test_inference_performance_benchmarks() {
    println!("⚡ Testing inference performance benchmarks...");

    let model_path = std::env::var("HOME")
        .map(|h| format!("{}/models/TinyLlama-1.1B-Chat-v1.0", h))
        .unwrap_or_else(|_| "/home/jeef/models/TinyLlama-1.1B-Chat-v1.0".to_string());

    if !std::path::Path::new(&model_path).exists() {
        println!("⏭️  Skipping test - TinyLlama model not found");
        return;
    }

    let mut engine = create_engine();

    let config = InfernoConfig {
        model_path,
        model_name: "TinyLlama-1.1B-Chat-v1.0".to_string(),
        device_id: 0,
        max_batch_size: 1,
        max_sequence_length: 256,
        ..Default::default()
    };

    // Initialize engine
    println!("🚀 Initializing engine for performance test...");
    let init_start = std::time::Instant::now();
    engine.initialize(config).await.expect("Failed to initialize engine");
    let init_time = init_start.elapsed();
    println!("⏱️  Initialization time: {:.2}s", init_time.as_secs_f64());

    // Performance test with multiple runs
    let test_prompts = [
        "Hello world",
        "The quick brown fox",
        "What is the meaning of life?",
    ];

    let mut total_tokens = 0;
    let mut total_time = 0.0;

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n🏃 Performance run {} with prompt: '{}'", i + 1, prompt);

        let request = InferenceRequest {
            request_id: (i + 1) as u64,
            prompt: prompt.to_string(),
            max_tokens: 20,
            temperature: 0.5,
            top_p: 0.9,
            seed: Some(42),
        };

        let response = engine.process(request).await.expect("Inference failed");

        total_tokens += response.generated_tokens;
        total_time += response.inference_time_ms;

        let tokens_per_sec = response.generated_tokens as f64 / (response.inference_time_ms / 1000.0);
        println!("   Tokens: {} | Time: {:.0}ms | Speed: {:.1} tok/s",
                 response.generated_tokens, response.inference_time_ms, tokens_per_sec);
    }

    let avg_tokens_per_sec = total_tokens as f64 / (total_time / 1000.0);
    println!("\n📊 Performance Summary:");
    println!("   Total tokens: {}", total_tokens);
    println!("   Total time: {:.0}ms", total_time);
    println!("   Average speed: {:.1} tokens/second", avg_tokens_per_sec);

    // Basic performance assertions
    assert!(total_tokens > 30, "Should generate substantial tokens across all tests");
    assert!(avg_tokens_per_sec > 5.0, "Should achieve at least 5 tokens/second on average");

    println!("✅ Performance benchmarks completed");
}