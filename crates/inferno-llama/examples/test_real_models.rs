//! Test precision detection with real models from ~/models directory
//!
//! This example demonstrates the model detection functionality by analyzing
//! real model files and detecting their precision configurations.

use inferno_llama::precision::ModelDetector;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = ModelDetector::new();

    println!("Testing model detection with real models from ~/models directory\n");

    // Test models we know exist
    let test_models = [
        ("meta-llama_Llama-3.1-8B-Instruct", "BF16 model from Meta"),
        (
            "RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8",
            "INT8 quantized model",
        ),
        ("unsloth_Llama-3.2-1B-Instruct", "Unsloth optimized model"),
        ("tinyllama-1.1b", "Small model for testing"),
    ];

    for (model_name, description) in &test_models {
        println!("=== {} ===", model_name);
        println!("Description: {}", description);

        let model_path = format!("/home/jeef/models/{}", model_name);
        let path = Path::new(&model_path);

        if !path.exists() {
            println!("❌ Model not found at: {}", model_path);
            continue;
        }

        match detector.detect_model_info(path) {
            Ok(info) => {
                println!("✅ Detection successful!");
                println!("   Precision: {}", info.precision);
                println!("   Format: {}", info.format);
                println!("   Quantized: {}", info.quantized);
                println!("   Shards: {}", info.num_shards);
                if let Some(quant_config) = &info.quantization_config {
                    println!("   Quantization method: {}", quant_config.method);
                    println!("   Bits: {}", quant_config.bits);
                    println!(
                        "   Activation quantized: {}",
                        quant_config.activation_quantized
                    );
                }
            }
            Err(e) => {
                println!("❌ Detection failed: {}", e);
            }
        }
        println!();
    }

    // Test config.json parsing specifically
    println!("=== Config.json Analysis ===");
    for (model_name, _) in &test_models {
        let config_path = format!("/home/jeef/models/{}/config.json", model_name);
        let path = Path::new(&config_path);

        if path.exists() {
            println!("Analyzing config.json for {}", model_name);
            match detector.detect_from_config(path) {
                Ok(info) => {
                    println!("   torch_dtype detected: {}", info.precision);
                    if info.quantized {
                        println!("   Quantization detected: Yes");
                        if let Some(q) = &info.quantization_config {
                            println!("   Method: {}, Bits: {}", q.method, q.bits);
                        }
                    }
                }
                Err(e) => {
                    println!("   Error: {}", e);
                }
            }
        } else {
            println!("No config.json found for {}", model_name);
        }
    }

    // Test SafeTensors file detection (if available)
    println!("\n=== SafeTensors Analysis ===");
    let safetensors_files = [
        "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8/model.safetensors",
        "/home/jeef/models/unsloth_Llama-3.2-1B-Instruct/model-00001-of-00002.safetensors",
    ];

    for safetensors_path in &safetensors_files {
        let path = Path::new(safetensors_path);
        if path.exists() {
            println!("Analyzing SafeTensors file: {}", safetensors_path);
            match detector.detect_from_safetensors(path) {
                Ok(info) => {
                    println!("   Detected precision: {}", info.precision);
                    println!("   Quantized: {}", info.quantized);
                }
                Err(e) => {
                    println!("   Error: {}", e);
                }
            }
        } else {
            println!("SafeTensors file not found: {}", safetensors_path);
        }
    }

    println!("\n=== Cache Performance Test ===");
    let test_path = Path::new("/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct");
    if test_path.exists() {
        // First access - should populate cache
        let start = std::time::Instant::now();
        let _ = detector.detect_model_info(test_path)?;
        let first_duration = start.elapsed();

        // Second access - should use cache
        let start = std::time::Instant::now();
        let _ = detector.detect_model_info(test_path)?;
        let cached_duration = start.elapsed();

        println!("First access: {:?}", first_duration);
        println!("Cached access: {:?}", cached_duration);
        println!(
            "Cache speedup: {:.2}x",
            first_duration.as_nanos() as f64 / cached_duration.as_nanos() as f64
        );
    }

    println!("\n🎉 Model detection testing complete!");
    Ok(())
}
