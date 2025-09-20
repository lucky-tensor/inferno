//! Generic tokenizer tests that work with any model
//! These tests validate tokenizer file parsing and basic tokenization functionality

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use std::env;
    use std::path::Path;

    fn get_small_model_path() -> String {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{}/models/Nikity_lille030m-instruct", home)
    }

    // ==== WORKING TESTS: File Access & JSON Parsing ====

    #[tokio::test]
    async fn test_model_files_exist() {
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);

        // Skip test if model doesn't exist
        if !model_path.exists() {
            eprintln!(
                "  Skipping test: Small model not found at {}",
                model_path_str
            );
            return;
        }

        println!("  WORKING: Testing file existence");

        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        let tokenizer_json_path = model_path.join("tokenizer.json");
        let config_json_path = model_path.join("config.json");

        assert!(
            tokenizer_config_path.exists(),
            "tokenizer_config.json should exist"
        );
        assert!(tokenizer_json_path.exists(), "tokenizer.json should exist");
        assert!(config_json_path.exists(), "config.json should exist");

        println!("    All required tokenizer files exist");
    }

    #[tokio::test]
    async fn test_json_parsing_working() {
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);

        // Skip test if model doesn't exist
        if !model_path.exists() {
            eprintln!(
                "  Skipping test: Small model not found at {}",
                model_path_str
            );
            return;
        }

        println!("  WORKING: Testing JSON file parsing");

        // Test tokenizer_config.json parsing
        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        let config_content = tokio::fs::read_to_string(&tokenizer_config_path)
            .await
            .expect("Failed to read tokenizer_config.json");

        let config: Value =
            serde_json::from_str(&config_content).expect("Failed to parse tokenizer_config.json");

        println!("    tokenizer_config.json parsed successfully");

        assert!(
            config.get("added_tokens_decoder").is_some(),
            "Should have added_tokens_decoder"
        );

        let tokenizer_class = config["tokenizer_class"].as_str().unwrap();
        println!("    - Tokenizer class: {}", tokenizer_class);
        // Accept both PreTrainedTokenizer and GPT2Tokenizer as valid tokenizer classes
        assert!(
            tokenizer_class == "PreTrainedTokenizer" || tokenizer_class == "GPT2Tokenizer",
            "Should be PreTrainedTokenizer or GPT2Tokenizer, got: {}",
            tokenizer_class
        );

        // Test model config.json parsing
        let model_config_path = model_path.join("config.json");
        let model_config_content = tokio::fs::read_to_string(&model_config_path)
            .await
            .expect("Failed to read config.json");

        let model_config: Value =
            serde_json::from_str(&model_config_content).expect("Failed to parse config.json");

        println!("    config.json parsed successfully");

        let vocab_size = model_config["vocab_size"].as_u64().unwrap();
        println!("    - Vocab size: {}", vocab_size);
        // Different models have different vocab sizes - just check it's reasonable
        assert!(
            vocab_size > 1000 && vocab_size < 200_000,
            "Should have a reasonable vocab size (1000-200000), got: {}",
            vocab_size
        );

        // Test tokenizer.json parsing
        let tokenizer_json_path = model_path.join("tokenizer.json");
        let tokenizer_content = tokio::fs::read_to_string(&tokenizer_json_path)
            .await
            .expect("Failed to read tokenizer.json");

        let tokenizer_data: Value =
            serde_json::from_str(&tokenizer_content).expect("Failed to parse tokenizer.json");

        println!("    tokenizer.json parsed successfully");
        assert!(
            tokenizer_data.get("model").is_some(),
            "Should have model section"
        );
        assert!(
            tokenizer_data.get("normalizer").is_some(),
            "Should have normalizer section"
        );
    }

    #[tokio::test]
    async fn test_direct_tokenizer_loading() {
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);

        // Skip test if model doesn't exist
        if !model_path.exists() {
            eprintln!(
                "  Skipping test: Small model not found at {}",
                model_path_str
            );
            return;
        }

        let tokenizer_json_path = model_path.join("tokenizer.json");

        // Try to load tokenizer directly using tokenizers crate
        {
            use tokenizers::Tokenizer;

            println!("    Loading tokenizer.json directly...");
            let direct_load_result = Tokenizer::from_file(&tokenizer_json_path);

            match direct_load_result {
                Ok(tokenizer) => {
                    println!("    Direct tokenizer.json loading successful!");

                    let vocab_size = tokenizer.get_vocab_size(false);
                    println!("    - Vocabulary size: {}", vocab_size);

                    // Test tokenization with direct loaded tokenizer
                    let test_text = "hello world";
                    println!("  🧪 Testing direct tokenization: '{}'", test_text);

                    let encoding_result = tokenizer.encode(test_text, false);
                    match encoding_result {
                        Ok(encoding) => {
                            let tokens = encoding.get_ids();
                            println!(
                                "      Direct tokenization successful: {} tokens",
                                tokens.len()
                            );
                            assert!(!tokens.is_empty(), "Should produce at least one token");
                        }
                        Err(e) => {
                            println!("      Direct tokenization failed: {}", e);
                            panic!("Tokenization should work with basic text");
                        }
                    }
                }
                Err(e) => {
                    println!("    Direct tokenizer.json loading failed: {}", e);
                    panic!("Direct tokenizer loading should work");
                }
            }
        }
    }
}
