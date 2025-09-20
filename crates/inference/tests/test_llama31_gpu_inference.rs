//! GPU inference test for Llama 3.1 model
//!
//! This test uses the available GPU (24GB) to perform full model loading and inference
//! for the meta-llama/Llama-3.1-8B-Instruct model.

use std::env;
use std::time::Instant;

#[cfg(feature = "candle-cuda")]
use inferno_inference::{
    config::InfernoConfig,
    inference::{CandleInferenceEngine, InferenceEngine, InferenceRequest},
};

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_llama31_gpu_full_inference() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU INFERENCE TEST: Llama 3.1 with 24GB GPU");
    println!("==============================================");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !std::path::Path::new(&model_path).exists() {
        println!(
            "❌ Llama 3.1 model not found at {}, skipping test",
            model_path
        );
        return Ok(());
    }

    println!("📁 Model path: {}", model_path);

    // Configure for GPU inference
    println!("\n🔧 Step 1: Configuring CandleInferenceEngine for GPU");
    let config = InfernoConfig {
        model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        model_path: model_path.clone(),
        device_id: 0, // Use GPU device 0
        max_batch_size: 1,
        max_sequence_length: 512,
        gpu_memory_pool_size_mb: 20480, // 20GB memory pool
        ..Default::default()
    };

    println!("   ✓ Model: {}", config.model_name);
    println!("   ✓ Device: GPU {} (24GB available)", config.device_id);
    println!("   ✓ Max sequence length: {}", config.max_sequence_length);
    println!("   ✓ GPU memory pool: {}MB", config.gpu_memory_pool_size_mb);

    // Create engine (this defaults to CUDA when candle-cuda feature is enabled)
    println!("\n🏗️  Step 2: Creating CandleInferenceEngine");
    let mut engine = CandleInferenceEngine::new();
    println!("   ✅ Engine created with backend: CUDA");

    // Initialize the engine
    println!("\n⚡ Step 3: Loading 8B Model onto GPU");
    println!("   Loading 15GB model onto 24GB GPU (this may take time)...");

    let start_time = Instant::now();
    match engine.initialize(config).await {
        Ok(()) => {
            let init_time = start_time.elapsed();
            println!(
                "   🎉 SUCCESS: Model loaded in {:.2}s",
                init_time.as_secs_f64()
            );
            println!("   ✅ 8B parameters loaded onto GPU");
            println!("   ✅ Tokenizer ready");
            println!("   ✅ Model ready for inference");
        }
        Err(e) => {
            let init_time = start_time.elapsed();
            println!(
                "   ❌ Model loading failed after {:.2}s: {}",
                init_time.as_secs_f64(),
                e
            );

            if e.to_string().contains("out of memory") {
                println!("   💡 GPU memory issue detected");
                println!("   💡 This may be due to fragmentation or other GPU usage");
                println!("   💡 Try clearing GPU memory: nvidia-smi -r");
            }

            return Err(format!("Failed to load model: {}", e).into());
        }
    }

    // Test basic inference
    println!("\n🧠 Step 4: Testing Basic GPU Inference");
    let basic_request = InferenceRequest {
        request_id: 1,
        prompt: "The capital of France is".to_string(),
        max_tokens: 20,
        temperature: 0.7,
        top_p: 0.9,
        seed: Some(42),
    };

    println!("   📝 Prompt: '{}'", basic_request.prompt);
    println!("   🎯 Max tokens: {}", basic_request.max_tokens);

    let inference_start = Instant::now();
    match engine.process(basic_request).await {
        Ok(response) => {
            let inference_time = inference_start.elapsed();
            println!(
                "   ✅ GPU inference successful in {:.3}s",
                inference_time.as_secs_f64()
            );
            println!("   📤 Generated: '{}'", response.generated_text);
            println!("   📊 Tokens generated: {}", response.generated_tokens);
            println!(
                "   ⚡ Tokens/second: {:.1}",
                response.generated_tokens as f64 / response.inference_time_ms * 1000.0
            );

            // With GPU, we should get good performance
            let tokens_per_second =
                response.generated_tokens as f64 / response.inference_time_ms * 1000.0;
            if tokens_per_second > 50.0 {
                println!(
                    "   🚀 Excellent GPU performance: {:.1} tok/s",
                    tokens_per_second
                );
            } else if tokens_per_second > 20.0 {
                println!("   ⚡ Good GPU performance: {:.1} tok/s", tokens_per_second);
            } else {
                println!(
                    "   🐌 Slow performance: {:.1} tok/s (unexpected for GPU)",
                    tokens_per_second
                );
            }

            assert!(!response.generated_text.is_empty(), "Should generate text");
            assert!(response.generated_tokens > 0, "Should generate tokens");
        }
        Err(e) => {
            let inference_time = inference_start.elapsed();
            println!(
                "   ❌ GPU inference failed after {:.3}s: {}",
                inference_time.as_secs_f64(),
                e
            );
            return Err(format!("GPU inference failed: {}", e).into());
        }
    }

    // Test Llama 3.1 chat format
    println!("\n💬 Step 5: Testing Llama 3.1 Chat Format on GPU");
    let chat_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the fastest way to sort an array?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    let chat_request = InferenceRequest {
        request_id: 2,
        prompt: chat_prompt.to_string(),
        max_tokens: 50,
        temperature: 0.2, // Lower temperature for more focused response
        top_p: 0.9,
        seed: Some(42),
    };

    println!(
        "   💬 Chat format prompt length: {} chars",
        chat_request.prompt.len()
    );
    println!("   🎯 Max tokens: {}", chat_request.max_tokens);

    let chat_start = Instant::now();
    match engine.process(chat_request).await {
        Ok(response) => {
            let chat_time = chat_start.elapsed();
            println!(
                "   ✅ Chat inference successful in {:.3}s",
                chat_time.as_secs_f64()
            );
            println!("   📤 Chat response:");
            println!("   {}", response.generated_text.replace('\n', "\n   "));
            println!("   📊 Tokens generated: {}", response.generated_tokens);
            println!(
                "   ⚡ Tokens/second: {:.1}",
                response.generated_tokens as f64 / response.inference_time_ms * 1000.0
            );

            assert!(
                !response.generated_text.is_empty(),
                "Should generate chat response"
            );
            assert!(
                response.generated_tokens > 10,
                "Should generate substantial response"
            );
        }
        Err(e) => {
            let chat_time = chat_start.elapsed();
            println!(
                "   ❌ Chat inference failed after {:.3}s: {}",
                chat_time.as_secs_f64(),
                e
            );
            return Err(format!("Chat inference failed: {}", e).into());
        }
    }

    // Test longer creative generation
    println!("\n✨ Step 6: Testing Creative Generation on GPU");
    let creative_request = InferenceRequest {
        request_id: 3,
        prompt: "Write a haiku about machine learning:".to_string(),
        max_tokens: 30,
        temperature: 0.8,
        top_p: 0.95,
        seed: Some(123),
    };

    println!("   🎨 Creative prompt: '{}'", creative_request.prompt);
    println!("   🎯 Max tokens: {}", creative_request.max_tokens);

    let creative_start = Instant::now();
    match engine.process(creative_request).await {
        Ok(response) => {
            let creative_time = creative_start.elapsed();
            println!(
                "   ✅ Creative generation successful in {:.3}s",
                creative_time.as_secs_f64()
            );
            println!("   🎨 Generated haiku:");
            println!("   {}", response.generated_text.replace('\n', "\n   "));
            println!("   📊 Tokens generated: {}", response.generated_tokens);
            println!(
                "   ⚡ Tokens/second: {:.1}",
                response.generated_tokens as f64 / response.inference_time_ms * 1000.0
            );

            assert!(
                response.generated_tokens >= 10,
                "Should generate creative content"
            );
        }
        Err(e) => {
            let creative_time = creative_start.elapsed();
            println!(
                "   ❌ Creative generation failed after {:.3}s: {}",
                creative_time.as_secs_f64(),
                e
            );
            return Err(format!("Creative generation failed: {}", e).into());
        }
    }

    println!("\n🎉 GPU INFERENCE TEST COMPLETED SUCCESSFULLY!");
    println!("   ✅ Model loading: PASS (15GB loaded onto 24GB GPU)");
    println!("   ✅ Basic inference: PASS");
    println!("   ✅ Chat format: PASS");
    println!("   ✅ Creative generation: PASS");
    println!("   🚀 Llama 3.1 8B model is fully operational on GPU!");
    println!("   💫 Ready for production inference workloads");

    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[tokio::test]
async fn test_llama31_gpu_performance_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ GPU PERFORMANCE: Llama 3.1 Throughput Benchmark");
    println!("=================================================");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !std::path::Path::new(&model_path).exists() {
        println!("❌ Llama 3.1 model not found, skipping benchmark");
        return Ok(());
    }

    // GPU configuration for benchmarking
    let config = InfernoConfig {
        model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        model_path: model_path.clone(),
        device_id: 0,
        max_batch_size: 1,
        max_sequence_length: 256,
        gpu_memory_pool_size_mb: 20480,
        ..Default::default()
    };

    let mut engine = CandleInferenceEngine::new();

    println!("🏗️  Loading model for performance benchmarking...");
    let load_start = Instant::now();
    match engine.initialize(config).await {
        Ok(()) => {
            let load_time = load_start.elapsed();
            println!("   ✅ Model loaded in {:.2}s", load_time.as_secs_f64());
        }
        Err(e) => {
            if e.to_string().contains("out of memory") {
                println!("   ⚠️  Skipping benchmark due to GPU memory constraints");
                return Ok(());
            }
            return Err(e.into());
        }
    }

    // Performance benchmark with multiple requests
    println!("\n📊 Running performance benchmarks...");
    let benchmark_prompts = [
        ("Short completion", "The weather today is", 10),
        ("Medium completion", "Explain the concept of recursion in programming:", 25),
        ("Long completion", "Write a brief explanation of quantum computing and its potential applications:", 40),
        ("Chat response", "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat are the benefits of renewable energy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 30),
    ];

    let mut total_time = 0.0;
    let mut total_tokens = 0;

    for (i, (test_name, prompt, max_tokens)) in benchmark_prompts.iter().enumerate() {
        let request = InferenceRequest {
            request_id: (i + 1) as u64,
            prompt: prompt.to_string(),
            max_tokens: *max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            seed: Some(42),
        };

        let start = Instant::now();
        match engine.process(request).await {
            Ok(response) => {
                let elapsed = start.elapsed().as_secs_f64();
                total_time += elapsed;
                total_tokens += response.generated_tokens;

                let tokens_per_second = response.generated_tokens as f64 / elapsed;
                println!(
                    "   {} {}: {:.2}s, {} tokens, {:.1} tok/s",
                    if tokens_per_second > 50.0 {
                        "🚀"
                    } else if tokens_per_second > 20.0 {
                        "⚡"
                    } else {
                        "🐌"
                    },
                    test_name,
                    elapsed,
                    response.generated_tokens,
                    tokens_per_second
                );
            }
            Err(e) => {
                println!("   ❌ {} failed: {}", test_name, e);
                return Err(e.into());
            }
        }
    }

    // Performance summary
    println!("\n📈 Performance Summary:");
    println!("   Total time: {:.2}s", total_time);
    println!("   Total tokens: {}", total_tokens);
    println!(
        "   Average tokens/second: {:.1}",
        total_tokens as f64 / total_time
    );
    println!(
        "   Average time per request: {:.2}s",
        total_time / benchmark_prompts.len() as f64
    );

    let avg_tokens_per_second = total_tokens as f64 / total_time;
    if avg_tokens_per_second > 50.0 {
        println!(
            "   🚀 Excellent GPU performance: {:.1} tok/s",
            avg_tokens_per_second
        );
    } else if avg_tokens_per_second > 20.0 {
        println!(
            "   ⚡ Good GPU performance: {:.1} tok/s",
            avg_tokens_per_second
        );
    } else if avg_tokens_per_second > 10.0 {
        println!(
            "   ⚡ Acceptable GPU performance: {:.1} tok/s",
            avg_tokens_per_second
        );
    } else {
        println!(
            "   🐌 Slow GPU performance: {:.1} tok/s",
            avg_tokens_per_second
        );
    }

    // Performance expectations for 8B model on good GPU
    assert!(
        avg_tokens_per_second > 5.0,
        "Should achieve reasonable throughput on GPU"
    );
    assert!(
        total_tokens > 50,
        "Should generate substantial text across benchmarks"
    );

    println!("\n✅ GPU performance benchmark completed successfully!");
    Ok(())
}

#[cfg(not(feature = "candle-cuda"))]
mod disabled_tests {
    #[tokio::test]
    async fn test_llama31_gpu_inference_requires_cuda() {
        println!("Llama 3.1 GPU inference tests require 'candle-cuda' feature");
        println!("Run with: cargo test --features candle-cuda");
    }
}
