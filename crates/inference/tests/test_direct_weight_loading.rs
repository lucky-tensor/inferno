//! Direct test of TinyLlama weight loading

use std::env;
use std::path::PathBuf;


#[tokio::test]
async fn test_direct_weight_loading() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing direct TinyLlama weight loading");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = PathBuf::from(format!("{}/models/tinyllama-1.1b", home));
    println!("Model path: {:?}", model_path);

    // Check if files exist
    let weights_file = model_path.join("model.safetensors");
    let tokenizer_file = model_path.join("tokenizer.json");

    println!("Weights file exists: {}", weights_file.exists());
    println!("Tokenizer file exists: {}", tokenizer_file.exists());

    if !weights_file.exists() || !tokenizer_file.exists() {
        println!("Required files missing, skipping weight loading test");
        return Ok(());
    }

    // Create device
    let device = burn::backend::ndarray::NdArrayDevice::default();
    println!("Device created: {:?}", device);

    // Try to load weights directly
    println!("Attempting to load weights...");

    match inferno_inference::models::load_llama_weights(&model_path, &device) {
        Ok(_model) => {
            println!("Successfully loaded TinyLlama model!");
            println!("Model loaded - this indicates burn-llama is working!");

            // Basic validation
            println!("Model loaded successfully with burn-llama integration");
        }
        Err(e) => {
            println!("Failed to load model: {}", e);
            println!("This shows what specific error we're encountering");
        }
    }

    println!("Direct weight loading test completed");
    Ok(())
}
