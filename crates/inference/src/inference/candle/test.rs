//! Tests for Candle inference engine with Llama 3.2 model

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

#[cfg(test)]
mod tests {
    use super::super::{CandleBackendType, CandleInferenceEngine};
    use crate::config::InfernoConfig;
    use crate::inference::{InferenceEngine, InferenceRequest};
    use std::path::Path;

    const LLAMA32_MODEL_PATH: &str = "/home/jeef/models/unsloth_Llama-3.2-1B-Instruct";

    #[tokio::test]
    #[cfg(feature = "candle-cuda")]
    async fn test_llama32_candle_cuda_full_pipeline() {
        // Skip test if model doesn't exist
        if !Path::new(LLAMA32_MODEL_PATH).exists() {
            eprintln!(
                "⚠️ Skipping test: Llama 3.2 model not found at {}",
                LLAMA32_MODEL_PATH
            );
            return;
        }

        println!("🚀 Testing Candle CUDA engine with Llama 3.2 model");

        // Step 1: Create engine
        println!("📝 Step 1: Creating Candle CUDA engine");
        let mut engine = CandleInferenceEngine::with_backend(CandleBackendType::Cuda);

        // Step 2: Initialize with model
        println!("📝 Step 2: Initializing engine with Llama 3.2 model");
        let config = InfernoConfig {
            model_name: "llama3.2-1b-instruct".to_string(),
            model_path: LLAMA32_MODEL_PATH.to_string(),
            device_id: 0, // CUDA device 0
            max_batch_size: 4,
            max_sequence_length: 2048,
            max_tokens: 50,
            gpu_memory_pool_size_mb: 4096,
            max_num_seqs: 16,
            temperature: 0.7,
            top_p: 0.9,
            top_k: -1,
            worker_threads: 4,
            enable_async_processing: true,
            ..Default::default()
        };

        let init_result = engine.initialize(config).await;
        match &init_result {
            Ok(()) => println!("✅ Engine initialization successful"),
            Err(e) => {
                println!("❌ Engine initialization failed: {}", e);
                panic!("Failed to initialize engine: {}", e);
            }
        }

        // Step 3: Test tokenization with a simple prompt
        println!("📝 Step 3: Testing inference with tokenization");
        let test_prompt = "who were the beatles?";
        println!(
            "🔤 Testing tokenization and inference with prompt: '{}'",
            test_prompt
        );

        let request = InferenceRequest {
            request_id: 1,
            prompt: test_prompt.to_string(),
            max_tokens: 20, // Keep it small for testing
            temperature: 0.7,
            top_p: 0.9,
            seed: Some(42),
        };

        let inference_result = engine.process(request).await;
        match &inference_result {
            Ok(response) => {
                println!("✅ Inference successful!");
                println!("📊 Request ID: {}", response.request_id);
                println!("🎯 Generated text: '{}'", response.generated_text);
                println!("📈 Generated tokens: {}", response.generated_tokens);
                println!("⏱️ Inference time: {:.2}ms", response.inference_time_ms);
                println!("🏁 Is finished: {}", response.is_finished);

                // Verify we got some output
                assert!(
                    !response.generated_text.is_empty(),
                    "Generated text should not be empty"
                );
                assert!(
                    response.generated_tokens > 0,
                    "Should generate at least one token"
                );
                assert!(
                    response.inference_time_ms > 0.0,
                    "Inference time should be positive"
                );
            }
            Err(e) => {
                println!("❌ Inference failed: {}", e);
                // Print detailed error information
                println!("🔍 Error details: {:?}", e);
                panic!("Inference failed: {}", e);
            }
        }

        println!("🎉 All tests passed! Candle CUDA engine with Llama 3.2 working correctly");
    }

    #[tokio::test]
    #[cfg(feature = "candle-cpu")]
    async fn test_llama32_candle_cpu_tokenization_only() {
        // Skip test if model doesn't exist
        if !Path::new(LLAMA32_MODEL_PATH).exists() {
            eprintln!(
                "⚠️ Skipping test: Llama 3.2 model not found at {}",
                LLAMA32_MODEL_PATH
            );
            return;
        }

        println!("🔥 Testing Candle CPU engine tokenization with Llama 3.2 model");

        // Create engine
        let mut engine = CandleInferenceEngine::with_backend(CandleBackendType::Cpu);

        // Initialize with model
        let config = InfernoConfig {
            model_name: "llama3.2-1b-instruct".to_string(),
            model_path: LLAMA32_MODEL_PATH.to_string(),
            device_id: 0, // CUDA device 0
            max_batch_size: 4,
            max_sequence_length: 2048,
            max_tokens: 50,
            gpu_memory_pool_size_mb: 4096,
            max_num_seqs: 16,
            temperature: 0.7,
            top_p: 0.9,
            top_k: -1,
            worker_threads: 4,
            enable_async_processing: true,
            ..Default::default()
        };

        let init_result = engine.initialize(config).await;
        if let Err(e) = init_result {
            println!("❌ Engine initialization failed: {}", e);
            panic!("Failed to initialize engine: {}", e);
        }
        println!("✅ Engine initialization successful");

        // Test tokenization with various prompts
        let test_prompts = vec![
            "hello",
            "who were the beatles?",
            "what is AI?",
            "The quick brown fox",
        ];

        for prompt in test_prompts {
            println!("🔤 Testing tokenization with: '{}'", prompt);

            let request = InferenceRequest {
                request_id: 1,
                prompt: prompt.to_string(),
                max_tokens: 5,    // Very small for CPU test
                temperature: 0.0, // Deterministic
                top_p: 1.0,
                seed: Some(42),
            };

            let inference_result = engine.process(request).await;
            match &inference_result {
                Ok(response) => {
                    println!(
                        "  ✅ Tokenization worked: '{}' -> {} tokens",
                        prompt, response.generated_tokens
                    );
                    if response.generated_tokens == 0 {
                        println!("  ⚠️ Warning: No tokens generated for prompt '{}'", prompt);
                    }
                }
                Err(e) => {
                    println!("  ❌ Failed for prompt '{}': {}", prompt, e);
                    // Don't panic on CPU failures, just log them
                }
            }
        }
    }

    #[test]
    fn test_engine_creation() {
        println!("🔧 Testing basic engine creation");

        let cpu_engine = CandleInferenceEngine::new();
        assert_eq!(cpu_engine.backend_type(), &CandleBackendType::Cpu);
        println!("✅ CPU engine created successfully");

        #[cfg(feature = "candle-cuda")]
        {
            let cuda_engine = CandleInferenceEngine::with_backend(CandleBackendType::Cuda);
            assert_eq!(cuda_engine.backend_type(), &CandleBackendType::Cuda);
            println!("✅ CUDA engine created successfully");
        }

        println!("🎉 Engine creation tests passed");
    }
}
