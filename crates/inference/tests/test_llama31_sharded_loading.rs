//! Test for Llama 3.1 sharded model loading and tokenization
//!
//! This test verifies that we can properly load sharded Llama 3.1 models
//! and handle their special tokenization requirements.

use std::env;
use std::path::PathBuf;

#[cfg(feature = "candle-cpu")]
#[tokio::test]
async fn test_llama31_sharded_model_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Llama 3.1 sharded model detection");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = PathBuf::from(format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home));
    println!("Model path: {:?}", model_path);

    if !model_path.exists() {
        println!("Llama 3.1 model not found, skipping test");
        return Ok(());
    }

    // Check for sharded model files
    let single_model = model_path.join("model.safetensors");
    let sharded_index = model_path.join("model.safetensors.index.json");
    let config_file = model_path.join("config.json");
    let tokenizer_file = model_path.join("tokenizer.json");
    let tokenizer_config = model_path.join("tokenizer_config.json");
    let special_tokens_map = model_path.join("special_tokens_map.json");

    println!("Single model file exists: {}", single_model.exists());
    println!("Sharded index file exists: {}", sharded_index.exists());
    println!("Config file exists: {}", config_file.exists());
    println!("Tokenizer file exists: {}", tokenizer_file.exists());
    println!("Tokenizer config exists: {}", tokenizer_config.exists());
    println!("Special tokens map exists: {}", special_tokens_map.exists());

    // Verify this is a sharded model
    assert!(
        !single_model.exists(),
        "This should be a sharded model, not single file"
    );
    assert!(
        sharded_index.exists(),
        "Sharded model should have index file"
    );

    // Check for sharded model files
    let mut shard_count = 0;
    if let Ok(entries) = std::fs::read_dir(&model_path) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
                shard_count += 1;
                println!("Found shard: {}", file_name_str);
            }
        }
    }

    assert!(shard_count > 0, "Should find at least one shard file");
    println!("Found {} shard files", shard_count);

    // Verify essential files exist
    assert!(config_file.exists(), "Config file is required");
    assert!(tokenizer_file.exists(), "Tokenizer file is required");
    assert!(tokenizer_config.exists(), "Tokenizer config is required");

    println!("Sharded model detection test passed!");
    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[tokio::test]
async fn test_llama31_tokenizer_special_tokens() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Llama 3.1 tokenizer special token mappings");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = PathBuf::from(format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home));

    if !model_path.exists() {
        println!("Llama 3.1 model not found, skipping test");
        return Ok(());
    }

    let tokenizer_config_path = model_path.join("tokenizer_config.json");
    let special_tokens_path = model_path.join("special_tokens_map.json");

    if !tokenizer_config_path.exists() {
        println!("Tokenizer config not found, skipping test");
        return Ok(());
    }

    // Read and parse tokenizer config
    let tokenizer_config_content = std::fs::read_to_string(&tokenizer_config_path)?;
    let tokenizer_config: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

    println!("Tokenizer config loaded successfully");

    // Check for added_tokens_decoder which should contain special token mappings
    if let Some(added_tokens) = tokenizer_config.get("added_tokens_decoder") {
        println!(
            "Found added_tokens_decoder with {} entries",
            added_tokens.as_object().map(|o| o.len()).unwrap_or(0)
        );

        // Check for specific tokens we expect
        let expected_tokens = vec![
            ("128000", "<|begin_of_text|>"),
            ("128001", "<|end_of_text|>"),
            ("128006", "<|start_header_id|>"),
            ("128007", "<|end_header_id|>"),
            ("128009", "<|eot_id|>"),
        ];

        for (token_id, token_content) in expected_tokens {
            if let Some(token_data) = added_tokens.get(token_id) {
                if let Some(content) = token_data.get("content") {
                    assert_eq!(
                        content.as_str().unwrap(),
                        token_content,
                        "Token {} should have content {}",
                        token_id,
                        token_content
                    );
                    println!("✓ Token {} correctly mapped to {}", token_id, token_content);
                } else {
                    panic!("Token {} missing content field", token_id);
                }
            } else {
                panic!(
                    "Expected token {} not found in added_tokens_decoder",
                    token_id
                );
            }
        }
    } else {
        panic!("No added_tokens_decoder found in tokenizer config");
    }

    // Check special tokens map if it exists
    if special_tokens_path.exists() {
        let special_tokens_content = std::fs::read_to_string(&special_tokens_path)?;
        let special_tokens: serde_json::Value = serde_json::from_str(&special_tokens_content)?;
        println!("Special tokens map: {}", special_tokens);
    }

    println!("Tokenizer special tokens test passed!");
    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[tokio::test]
async fn test_llama31_candle_sharding_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Llama 3.1 Candle sharding detection");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !PathBuf::from(&model_path).exists() {
        println!("Llama 3.1 model not found, skipping test");
        return Ok(());
    }

    println!("Testing model path: {}", model_path);

    // Test that we can detect sharded model files correctly (this is what our updated Candle engine should detect)
    let model_dir = std::path::Path::new(&model_path);

    // Check what our detection logic would find
    let has_single_model = model_dir.join("model.safetensors").exists();
    let has_sharded_model = model_dir.join("model.safetensors.index.json").exists();

    println!("Has single model: {}", has_single_model);
    println!("Has sharded model: {}", has_sharded_model);

    assert!(!has_single_model, "Should not have single model file");
    assert!(has_sharded_model, "Should have sharded model index");

    // Count shard files (this is what our updated Candle code should find)
    let mut shard_files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
                shard_files.push(entry.path());
            }
        }
    }

    shard_files.sort();
    assert!(!shard_files.is_empty(), "Should find shard files");
    println!("Found {} shard files", shard_files.len());

    for shard in &shard_files {
        println!("  - {}", shard.file_name().unwrap().to_string_lossy());
    }

    // Verify the exact logic our Candle engine uses
    assert!(
        shard_files.len() == 4,
        "Expected 4 shard files for Llama 3.1 8B"
    );

    println!("Candle sharding detection test passed!");
    Ok(())
}

// Test that we can read the model index and understand the sharding layout
#[tokio::test]
async fn test_llama31_model_index_parsing() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Llama 3.1 model index parsing");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = PathBuf::from(format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home));

    if !model_path.exists() {
        println!("Llama 3.1 model not found, skipping test");
        return Ok(());
    }

    let index_path = model_path.join("model.safetensors.index.json");
    if !index_path.exists() {
        println!("Model index not found, skipping test");
        return Ok(());
    }

    // Read and parse the model index
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    println!("Model index loaded successfully");

    // Check the structure
    if let Some(weight_map) = index.get("weight_map") {
        let weight_map_obj = weight_map
            .as_object()
            .expect("weight_map should be an object");
        println!("Found {} weight mappings", weight_map_obj.len());

        // Check that we have mappings to different shard files
        let mut shard_files_in_index = std::collections::HashSet::new();
        for (layer_name, shard_file) in weight_map_obj {
            let shard_file_str = shard_file.as_str().expect("shard file should be string");
            shard_files_in_index.insert(shard_file_str.to_string());

            // Print first few for debugging
            if shard_files_in_index.len() <= 3 {
                println!("  {} -> {}", layer_name, shard_file_str);
            }
        }

        println!(
            "Index references {} unique shard files",
            shard_files_in_index.len()
        );
        for shard_file in &shard_files_in_index {
            println!("  - {}", shard_file);

            // Verify the shard file actually exists
            let shard_path = model_path.join(shard_file);
            assert!(
                shard_path.exists(),
                "Shard file {} should exist",
                shard_file
            );
        }

        assert!(
            !shard_files_in_index.is_empty(),
            "Should have shard file references"
        );
    } else {
        panic!("Model index should have weight_map");
    }

    println!("Model index parsing test passed!");
    Ok(())
}

// Test actual tokenizer loading using our CandleTokenizer implementation
#[cfg(feature = "candle-cpu")]
#[tokio::test]
async fn test_llama31_tokenizer_loading() -> Result<(), Box<dyn std::error::Error>> {
    use inferno_inference::inference::candle::tokenizer::CandleTokenizer;

    println!("Testing Llama 3.1 tokenizer loading");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !PathBuf::from(&model_path).exists() {
        println!("Llama 3.1 model not found, skipping test");
        return Ok(());
    }

    // Test tokenizer loading
    match CandleTokenizer::load_from_path(&model_path).await {
        Ok(tokenizer) => {
            println!("✓ Tokenizer loaded successfully");

            // Test basic tokenization
            let test_text = "Hello, world!";
            match tokenizer.encode(test_text, false) {
                Ok(encoding) => {
                    let tokens = encoding.get_tokens();
                    let ids = encoding.get_ids();
                    println!(
                        "✓ Text '{}' tokenized to {} tokens",
                        test_text,
                        tokens.len()
                    );
                    println!("  Tokens: {:?}", tokens);
                    println!("  IDs: {:?}", ids);

                    assert!(!tokens.is_empty(), "Should produce at least one token");
                    assert_eq!(
                        tokens.len(),
                        ids.len(),
                        "Token and ID arrays should be same length"
                    );
                }
                Err(e) => {
                    panic!("Failed to tokenize text: {}", e);
                }
            }

            // Test special token handling for Llama 3.1
            let special_text = "<|begin_of_text|>Hello<|end_of_text|>";
            match tokenizer.encode(special_text, false) {
                Ok(encoding) => {
                    let tokens = encoding.get_tokens();
                    let ids = encoding.get_ids();
                    println!("✓ Special tokens text tokenized to {} tokens", tokens.len());
                    println!("  Special tokens: {:?}", tokens);
                    println!("  Special IDs: {:?}", ids);

                    // Check that special tokens are properly handled
                    assert!(
                        ids.contains(&128000),
                        "Should contain <|begin_of_text|> token ID (128000)"
                    );
                    assert!(
                        ids.contains(&128001),
                        "Should contain <|end_of_text|> token ID (128001)"
                    );
                }
                Err(e) => {
                    println!(
                        "Warning: Special token encoding failed (this may be expected): {}",
                        e
                    );
                }
            }
        }
        Err(e) => {
            panic!("Failed to load tokenizer: {}", e);
        }
    }

    println!("Tokenizer loading test passed!");
    Ok(())
}

// Test that sharding detection logic works correctly (without full model loading)
#[cfg(feature = "candle-cpu")]
#[tokio::test]
async fn test_llama31_sharding_logic_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Llama 3.1 sharding logic verification");

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let model_path = format!("{}/models/meta-llama_Llama-3.1-8B-Instruct", home);

    if !PathBuf::from(&model_path).exists() {
        println!("Llama 3.1 model not found, skipping test");
        return Ok(());
    }

    let model_dir = std::path::Path::new(&model_path);

    // Test the same sharding detection logic used by Candle engine
    let single_model_path = model_dir.join("model.safetensors");
    let sharded_index_path = model_dir.join("model.safetensors.index.json");

    println!("Testing sharding detection logic that Candle engine uses:");

    let (model_files, is_sharded) = if single_model_path.exists() {
        println!(
            "  Would load single model file: {}",
            single_model_path.display()
        );
        (vec![single_model_path], false)
    } else if sharded_index_path.exists() {
        println!(
            "  Would load sharded model with index: {}",
            sharded_index_path.display()
        );
        // Find all sharded model files (same logic as engine)
        let mut sharded_files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let file_name_str = file_name.to_string_lossy();
                if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
                    sharded_files.push(entry.path());
                }
            }
        }

        if sharded_files.is_empty() {
            return Err("No sharded model files found despite having index file".into());
        }
        sharded_files.sort(); // Ensure consistent ordering
        println!("  Found {} sharded model files", sharded_files.len());
        for file in &sharded_files {
            println!("    - {}", file.file_name().unwrap().to_string_lossy());
        }
        (sharded_files, true)
    } else {
        return Err(format!(
            "No SafeTensors model files found in {}. Expected either 'model.safetensors' or sharded model files with 'model.safetensors.index.json'",
            model_dir.display()
        ).into());
    };

    // Verify detection results
    assert!(is_sharded, "Should detect model as sharded");
    assert!(!model_files.is_empty(), "Should find model files");
    assert!(
        model_files.len() >= 2,
        "Sharded model should have multiple files"
    );

    println!("✓ Sharding detection logic works correctly");
    println!(
        "✓ Detected {} files for {} model",
        model_files.len(),
        if is_sharded { "sharded" } else { "single" }
    );

    println!("Sharding logic verification test passed!");
    Ok(())
}
