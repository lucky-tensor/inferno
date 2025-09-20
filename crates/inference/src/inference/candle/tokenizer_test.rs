//! Dedicated tests for Candle tokenizer with small language model

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

#[cfg(test)]
mod tests {
    use super::super::tokenizer::CandleTokenizer;
    use serde_json::Value;
    use std::env;
    use std::path::Path;

    fn get_small_model_path() -> String {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{}/models/Nikity_lille-130m-instruct", home)
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

        let tokenizer_json = model_path.join("tokenizer.json");
        let tokenizer_config = model_path.join("tokenizer_config.json");
        let config_json = model_path.join("config.json");
        let model_safetensors = model_path.join("model.safetensors");

        assert!(tokenizer_json.exists(), "tokenizer.json should exist");
        assert!(
            tokenizer_config.exists(),
            "tokenizer_config.json should exist"
        );
        assert!(config_json.exists(), "config.json should exist");
        assert!(model_safetensors.exists(), "model.safetensors should exist");

        println!("    All required files exist");
    }

    #[tokio::test]
    async fn test_json_parsing_working() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  WORKING: Testing JSON file parsing");
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);

        // Test tokenizer_config.json parsing
        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        let config_content = tokio::fs::read_to_string(&tokenizer_config_path)
            .await
            .expect("Failed to read tokenizer_config.json");

        let config: Value =
            serde_json::from_str(&config_content).expect("Failed to parse tokenizer_config.json");

        println!("    tokenizer_config.json parsed successfully");

        // Verify expected fields
        assert!(
            config.get("tokenizer_class").is_some(),
            "Should have tokenizer_class"
        );
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

        // Check if it has vocabulary
        let has_vocab = tokenizer_data
            .get("model")
            .and_then(|m| m.get("vocab"))
            .is_some();
        println!("    - Has vocab in tokenizer.json: {}", has_vocab);
    }

    #[tokio::test]
    async fn test_special_tokens_parsing_working() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  WORKING: Testing special tokens parsing from JSON");
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);
        let tokenizer_config_path = model_path.join("tokenizer_config.json");

        let config_content = tokio::fs::read_to_string(&tokenizer_config_path)
            .await
            .expect("Failed to read tokenizer_config.json");

        let config: Value =
            serde_json::from_str(&config_content).expect("Failed to parse tokenizer_config.json");

        let added_tokens = config
            .get("added_tokens_decoder")
            .and_then(|v| v.as_object())
            .expect("Should have added_tokens_decoder object");

        println!("    Found {} special tokens", added_tokens.len());
        // Different models have different numbers of special tokens - just check it's reasonable
        assert!(
            !added_tokens.is_empty() && added_tokens.len() < 500,
            "Should have a reasonable number of special tokens (1-500), got: {}",
            added_tokens.len()
        );

        // Test parsing any special tokens that exist (model-agnostic)
        let mut token_count = 0;
        for (token_id, token_info) in added_tokens.iter().take(3) {
            if let Some(content) = token_info.get("content").and_then(|v| v.as_str()) {
                println!("      Token {}: '{}'", token_id, content);
                assert!(!content.is_empty(), "Token content should not be empty");
                token_count += 1;
            }
        }
        assert!(
            token_count > 0,
            "Should be able to parse at least one special token"
        );
    }

    #[tokio::test]
    async fn test_tokenizer_json_vocabulary_structure() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  WORKING: Testing tokenizer.json vocabulary structure");
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);
        let tokenizer_json_path = model_path.join("tokenizer.json");

        let tokenizer_content = tokio::fs::read_to_string(&tokenizer_json_path)
            .await
            .expect("Failed to read tokenizer.json");

        let tokenizer_data: Value =
            serde_json::from_str(&tokenizer_content).expect("Failed to parse tokenizer.json");

        println!("    tokenizer.json structure analysis:");

        // Check model type
        if let Some(model_type) = tokenizer_data.get("model").and_then(|m| m.get("type")) {
            println!("    - Model type: {}", model_type);
        }

        // Check if vocabulary exists and analyze it
        if let Some(vocab) = tokenizer_data.get("model").and_then(|m| m.get("vocab")) {
            if let Some(vocab_obj) = vocab.as_object() {
                println!("    - Vocabulary size: {}", vocab_obj.len());

                // Sample a few vocabulary entries
                let mut sample_count = 0;
                for (token, id) in vocab_obj {
                    if sample_count < 10 {
                        println!("      - '{}' -> {}", token, id);
                        sample_count += 1;
                    }
                }

                // Check for common English tokens
                let test_tokens = vec![
                    "hello", "world", "the", "a", "an", "and", "or", "who", "what",
                ];
                for token in test_tokens {
                    if let Some(id) = vocab_obj.get(token) {
                        println!("      Found common token '{}' -> {}", token, id);
                    } else {
                        println!("      Missing common token '{}'", token);
                    }
                }

                // Check vocab size matches expectation
                let expected_vocab_size = 128_256;
                if vocab_obj.len() == expected_vocab_size {
                    println!(
                        "      Vocabulary size matches expected: {}",
                        expected_vocab_size
                    );
                } else {
                    println!(
                        "      Vocabulary size mismatch: got {}, expected {}",
                        vocab_obj.len(),
                        expected_vocab_size
                    );
                }
            } else {
                println!("      Vocabulary is not an object");
            }
        } else {
            println!("      No vocabulary found in tokenizer.json");
        }

        // Check merges
        if let Some(merges) = tokenizer_data.get("model").and_then(|m| m.get("merges")) {
            if let Some(merges_array) = merges.as_array() {
                println!("    - Merges count: {}", merges_array.len());

                // Sample a few merges
                for (i, merge) in merges_array.iter().take(5).enumerate() {
                    println!("      - Merge {}: {}", i, merge);
                }
            }
        } else {
            println!("    - No merges found");
        }

        // Check added tokens
        if let Some(added_tokens) = tokenizer_data.get("added_tokens") {
            if let Some(added_array) = added_tokens.as_array() {
                println!("    - Added tokens count: {}", added_array.len());
            }
        }
    }

    #[tokio::test]
    async fn test_vocabulary_encoding_analysis() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  ANALYSIS: Testing vocabulary encoding in tokenizer.json");
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);
        let tokenizer_json_path = model_path.join("tokenizer.json");

        let tokenizer_content = tokio::fs::read_to_string(&tokenizer_json_path)
            .await
            .expect("Failed to read tokenizer.json");

        let tokenizer_data: Value =
            serde_json::from_str(&tokenizer_content).expect("Failed to parse tokenizer.json");

        if let Some(vocab) = tokenizer_data.get("model").and_then(|m| m.get("vocab")) {
            if let Some(vocab_obj) = vocab.as_object() {
                println!("    Analyzing vocabulary encoding:");
                println!("    - Total vocabulary size: {}", vocab_obj.len());

                // Look for tokens by ID range to understand the structure
                let mut found_tokens_by_range = std::collections::HashMap::new();

                for (token_str, id_val) in vocab_obj {
                    if let Some(id) = id_val.as_u64() {
                        let range = match id {
                            0..=999 => "0-999",
                            1000..=9999 => "1k-9k",
                            10000..=99999 => "10k-99k",
                            100_000..=127_999 => "100k-127k",
                            128_000..=128_255 => "special",
                            _ => "other",
                        };

                        found_tokens_by_range
                            .entry(range)
                            .or_insert_with(Vec::new)
                            .push((token_str.clone(), id));
                    }
                }

                for (range, tokens) in &found_tokens_by_range {
                    println!("    - Range {}: {} tokens", range, tokens.len());

                    // Show first 10 tokens in each range
                    for (token, id) in tokens.iter().take(10) {
                        // Try to display token with proper encoding
                        let display_token = if token
                            .chars()
                            .all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
                        {
                            format!("'{}'", token)
                        } else {
                            format!("'{}'", token.escape_debug())
                        };
                        println!("      {} -> {}", id, display_token);
                    }
                    if tokens.len() > 10 {
                        println!("      ... and {} more", tokens.len() - 10);
                    }
                }

                // Specifically look for common English words by searching the entire vocab
                println!("    Searching for common English words in vocabulary:");
                let common_words = vec![
                    "hello", "world", "the", "a", "an", "and", "or", "who", "what", "is", "was",
                    "are",
                ];

                for word in common_words {
                    let mut found = false;
                    for (token_str, id_val) in vocab_obj {
                        if token_str == word
                            || token_str == &format!(" {}", word)
                            || token_str == &format!("Ġ{}", word)
                        {
                            if let Some(id) = id_val.as_u64() {
                                println!(
                                    "      Found '{}' as token '{}' -> {}",
                                    word, token_str, id
                                );
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        println!("      Word '{}' not found in vocabulary", word);
                    }
                }

                // Look for byte-level tokens (common in GPT-style tokenizers)
                println!("    Looking for byte-level encodings:");
                let mut byte_tokens = 0;
                for (token_str, _) in vocab_obj {
                    if token_str.starts_with("Ġ") || token_str.len() == 1 {
                        byte_tokens += 1;
                        if byte_tokens <= 10 {
                            println!("    - Byte token example: '{}'", token_str);
                        }
                    }
                }
                println!("    - Total byte-level tokens: {}", byte_tokens);
            } else {
                println!("    Vocabulary is not an object");
            }
        } else {
            println!("    No vocabulary found in tokenizer.json");
        }
    }

    #[tokio::test]
    async fn test_direct_tokenizer_json_loading() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  ANALYSIS: Testing direct tokenizer.json loading with tokenizers crate");
        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);
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
                            println!("      Token IDs: {:?}", tokens);

                            if tokens.is_empty() {
                                println!("      Direct tokenization also produces 0 tokens");
                            } else {
                                println!("      DIRECT TOKENIZATION WORKS!");

                                // Try to decode back
                                let decode_result = tokenizer.decode(tokens, true);
                                match decode_result {
                                    Ok(decoded) => {
                                        println!("      Decoded text: '{}'", decoded);
                                        if decoded.trim() == test_text.trim() {
                                            println!("      Round-trip successful!");
                                        } else {
                                            println!(
                                                "      Round-trip mismatch: '{}' vs '{}'",
                                                test_text, decoded
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        println!("      Decoding failed: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            println!("      Direct tokenization failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("    Direct tokenizer.json loading failed: {}", e);
                }
            }
        }

        #[cfg(not(any(
        )))]
        {
            println!("    Tokenizers crate not available (no candle features)");
        }
    }

    // ==== BROKEN TESTS: Tokenizer Implementation ====

    #[tokio::test]
    async fn test_broken_tokenizer_loading() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  BROKEN: Testing Llama 3.2 tokenizer loading");

        let model_path_str = get_small_model_path();
        let tokenizer_result = CandleTokenizer::load_from_path(&model_path_str).await;

        match &tokenizer_result {
            Ok(tokenizer) => {
                println!("  Tokenizer loaded successfully");

                // Test basic tokenizer properties
                let vocab_size = tokenizer.get_vocab_size(false);
                println!("  Vocab size: {}", vocab_size);

                // Expected for Llama 3.2: 128,256 tokens (128,000 base + 256 special)
                if vocab_size != 128_256 {
                    println!("  Warning: Expected vocab size 128256, got {}", vocab_size);
                }
            }
            Err(e) => {
                println!("  Tokenizer loading failed: {}", e);
                panic!("Failed to load tokenizer: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_broken_tokenization_produces_zero_tokens() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  BROKEN: Testing Llama 3.2 basic tokenization - demonstrates 0 tokens");

        let model_path_str = get_small_model_path();
        let tokenizer = CandleTokenizer::load_from_path(&model_path_str)
            .await
            .expect("Failed to load tokenizer");

        // Test simple prompts
        let test_cases = vec![
            ("hello", "Simple greeting"),
            ("who were the beatles?", "Question about Beatles"),
            (
                "The quick brown fox jumps over the lazy dog",
                "Classic pangram",
            ),
            ("AI and machine learning", "Technical terms"),
            ("", "Empty string"),
            ("a", "Single character"),
        ];

        for (prompt, description) in test_cases {
            println!("🧪 Testing: '{}' ({})", prompt, description);

            let encoding_result = tokenizer.encode(prompt, false);

            match encoding_result {
                Ok(encoding) => {
                    let tokens = encoding.get_ids();
                    let token_count = tokens.len();

                    println!("    Tokenized successfully: {} tokens", token_count);
                    println!("    Token IDs: {:?}", tokens);

                    if token_count == 0 && !prompt.is_empty() {
                        println!("    WARNING: Non-empty prompt produced 0 tokens!");
                    }

                    // Try to decode back to text
                    let decode_result = tokenizer.decode(tokens, true);
                    match decode_result {
                        Ok(decoded_text) => {
                            println!("    Decoded text: '{}'", decoded_text);

                            if decoded_text != prompt && !prompt.is_empty() {
                                println!(
                                    "    WARNING: Round-trip failed! Original: '{}', Decoded: '{}'",
                                    prompt, decoded_text
                                );
                            }
                        }
                        Err(e) => {
                            println!("    Decoding failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("    Tokenization failed: {}", e);
                }
            }
            println!(); // Empty line for readability
        }
    }

    #[tokio::test]
    async fn test_broken_special_tokens_split_incorrectly() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  BROKEN: Testing Llama 3.2 special tokens - demonstrates incorrect splitting");

        let model_path_str = get_small_model_path();
        let tokenizer = CandleTokenizer::load_from_path(&model_path_str)
            .await
            .expect("Failed to load tokenizer");

        // Test special tokens that should be recognized
        let special_tokens = vec![
            ("<|begin_of_text|>", 128_000),
            ("<|end_of_text|>", 128_001),
            ("<|start_header_id|>", 128_006),
            ("<|end_header_id|>", 128_007),
            ("<|eot_id|>", 128_009),
        ];

        for (token_text, expected_id) in special_tokens {
            println!("  Testing special token: '{}'", token_text);

            let encoding_result = tokenizer.encode(token_text, true); // Allow special tokens

            match encoding_result {
                Ok(encoding) => {
                    let tokens = encoding.get_ids();
                    println!("    Token IDs: {:?}", tokens);

                    if tokens.len() == 1 {
                        let actual_id = tokens[0];
                        if actual_id == expected_id {
                            println!("    Correct special token ID: {}", actual_id);
                        } else {
                            println!(
                                "    Wrong special token ID: got {}, expected {}",
                                actual_id, expected_id
                            );
                        }
                    } else {
                        println!(
                            "    Special token split into {} tokens: {:?}",
                            tokens.len(),
                            tokens
                        );
                    }
                }
                Err(e) => {
                    println!("    Failed to tokenize special token: {}", e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_llama32_tokenizer_demonstrate_failure() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("🚨 Demonstrating Llama 3.2 tokenizer failure in default mode");

        let model_path_str = get_small_model_path();
        let tokenizer = CandleTokenizer::load_from_path(&model_path_str)
            .await
            .expect("Failed to load tokenizer");

        let test_prompt = "who were the beatles?";
        println!("  Testing prompt: '{}'", test_prompt);

        let encoding_result = tokenizer.encode(test_prompt, false);

        match encoding_result {
            Ok(encoding) => {
                let tokens = encoding.get_ids();
                let token_count = tokens.len();

                println!("  Tokenization result:");
                println!("  - Token count: {}", token_count);
                println!("  - Token IDs: {:?}", tokens);

                if token_count == 0 {
                    println!("🔴 FAILURE DEMONSTRATED: Tokenization produced 0 tokens for non-empty prompt!");
                    println!("  This explains why inference fails with 'cannot reshape tensor of 0 elements'");

                    // This should cause the test to fail, demonstrating the issue
                    assert!(
                        token_count > 0,
                        "Tokenization should produce at least one token for non-empty input"
                    );
                } else {
                    println!("  Tokenization produced tokens (issue may be resolved)");
                }
            }
            Err(e) => {
                println!(
                    "🔴 FAILURE DEMONSTRATED: Tokenization failed with error: {}",
                    e
                );
                panic!("Tokenization failed: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_tokenizer_files_inspection() {
        // Skip test if model doesn't exist
        let model_path_str = get_small_model_path();
        if !Path::new(&model_path_str).exists() {
            eprintln!(
                "  Skipping test: Llama 3.2 model not found at {}",
                &model_path_str
            );
            return;
        }

        println!("  Inspecting tokenizer files in Llama 3.2 model directory");

        let model_path_str = get_small_model_path();
        let model_path = Path::new(&model_path_str);

        // Check which tokenizer files exist
        let tokenizer_json = model_path.join("tokenizer.json");
        let tokenizer_config = model_path.join("tokenizer_config.json");
        let vocab_json = model_path.join("vocab.json");

        println!("  File existence check:");
        println!("  - tokenizer.json: {}", tokenizer_json.exists());
        println!("  - tokenizer_config.json: {}", tokenizer_config.exists());
        println!("  - vocab.json: {}", vocab_json.exists());

        if tokenizer_config.exists() {
            println!("  Reading tokenizer_config.json...");
            let config_content = tokio::fs::read_to_string(&tokenizer_config)
                .await
                .expect("Failed to read tokenizer_config.json");

            // Parse and display key information
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                if let Some(tokenizer_class) = config.get("tokenizer_class") {
                    println!("  - Tokenizer class: {}", tokenizer_class);
                }
                if let Some(vocab_size) = config.get("vocab_size") {
                    println!("  - Config vocab size: {}", vocab_size);
                }
                if let Some(model_max_length) = config.get("model_max_length") {
                    println!("  - Model max length: {}", model_max_length);
                }

                // Check for added_tokens_decoder (special tokens)
                if let Some(added_tokens) = config.get("added_tokens_decoder") {
                    if let Some(obj) = added_tokens.as_object() {
                        println!("  - Special tokens count: {}", obj.len());

                        // Show first few special tokens
                        let mut count = 0;
                        for (id, token_info) in obj {
                            if count < 5 {
                                if let Some(content) = token_info.get("content") {
                                    println!("    - ID {}: {}", id, content);
                                }
                                count += 1;
                            }
                        }
                        if obj.len() > 5 {
                            println!("    - ... and {} more", obj.len() - 5);
                        }
                    }
                }
            }
        }

        // This test just inspects files, so it should always pass
        assert!(
            tokenizer_config.exists(),
            "tokenizer_config.json should exist"
        );
    }
}
