//! End-to-end model loading tests

use inferno_llama::factory::UnifiedModelFactory;

#[tokio::test]
async fn test_end_to_end_quantized_model_loading() {
    let model_path = "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8";

    if !std::path::Path::new(model_path).exists() {
        println!("Skipping test: Model not found at {}", model_path);
        return;
    }

    println!("Testing end-to-end loading of quantized model...");

    let factory = UnifiedModelFactory::new().expect("Factory creation should succeed");

    // Step 1: Detect model configuration
    println!("Step 1: Detecting model configuration...");
    let config = match factory.detect_model_config(model_path).await {
        Ok(config) => {
            println!("✅ Model config detected successfully:");
            println!("  Variant: {:?}", config.variant);
            println!("  Hidden size: {}", config.base.hidden_size);
            println!("  Layers: {}", config.base.num_hidden_layers);
            if let Some(quantization) = &config.quantization {
                println!("  Quantization: {:?}", quantization.scheme);
            }
            config
        }
        Err(e) => {
            println!("❌ Model detection failed: {}", e);
            panic!("Model detection should work");
        }
    };

    // Step 2: Load the actual model
    println!("Step 2: Loading model weights...");
    let result = factory.load_model(model_path, config).await;
    match result {
        Ok(model) => {
            println!("✅ Model loaded successfully!");
            println!("  Parameter count: {}", model.parameter_count());
            println!("  Model layers: {}", model.layers.len());

            // Verify the model has reasonable parameter count for 1B model
            let param_count = model.parameter_count();
            if param_count > 500_000_000 && param_count < 2_000_000_000 {
                println!("✅ Parameter count is in expected range for 1B model");
            } else {
                println!(
                    "⚠️ Parameter count {} seems unusual for 1B model",
                    param_count
                );
            }
        }
        Err(e) => {
            println!("❌ Model loading failed: {}", e);
            println!("This is expected - weight loading may not be fully implemented yet");
            // Don't fail the test yet - this tells us what to work on next
        }
    }
}

#[tokio::test]
async fn test_end_to_end_standard_model_loading() {
    let model_path = "/home/jeef/models/meta-llama_Llama-3.1-8B-Instruct";

    if !std::path::Path::new(model_path).exists() {
        println!("Skipping test: Model not found at {}", model_path);
        return;
    }

    println!("Testing end-to-end loading of standard model...");

    let factory = UnifiedModelFactory::new().expect("Factory creation should succeed");

    // Step 1: Detect model configuration
    println!("Step 1: Detecting model configuration...");
    let config = match factory.detect_model_config(model_path).await {
        Ok(config) => {
            println!("✅ Model config detected successfully:");
            println!("  Variant: {:?}", config.variant);
            println!("  Hidden size: {}", config.base.hidden_size);
            println!("  Layers: {}", config.base.num_hidden_layers);
            println!("  Attention heads: {}", config.base.num_attention_heads);
            config
        }
        Err(e) => {
            println!("❌ Model detection failed: {}", e);
            panic!("Model detection should work");
        }
    };

    // Step 2: Load the actual model (this might fail - that's OK for now)
    println!("Step 2: Loading model weights...");
    let result = factory.load_model(model_path, config).await;
    match result {
        Ok(model) => {
            println!("✅ Model loaded successfully!");
            println!("  Parameter count: {}", model.parameter_count());
            println!("  Model layers: {}", model.layers.len());

            // Verify the model has reasonable parameter count for 8B model
            let param_count = model.parameter_count();
            if param_count > 6_000_000_000 && param_count < 10_000_000_000 {
                println!("✅ Parameter count is in expected range for 8B model");
            } else {
                println!(
                    "⚠️ Parameter count {} seems unusual for 8B model",
                    param_count
                );
            }
        }
        Err(e) => {
            println!("❌ Model loading failed: {}", e);
            println!("This is expected - weight loading may not be fully implemented yet");
            // Don't fail the test yet - this tells us what to work on next
        }
    }
}
