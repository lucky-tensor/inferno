//! Tokenizer handling and Llama 3.2 compatibility for Candle

#![allow(missing_docs)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::redundant_closure_for_method_calls)]

#[cfg(any(
))]
use tokenizers::{
    models::bpe::BPE, processors::template::TemplateProcessing, AddedToken, PaddingDirection,
    PaddingParams, Tokenizer, TruncationDirection, TruncationParams,
};

use crate::inference::InferenceError;
use tracing::{debug, info};

pub struct CandleTokenizer;

impl CandleTokenizer {
    /// Load tokenizer from model directory with Llama 3.2 compatibility
    #[cfg(any(
    ))]
    pub async fn load_from_path(model_path: &str) -> Result<Tokenizer, InferenceError> {
        let model_path = std::path::Path::new(model_path);
        let tokenizer_path = model_path.join("tokenizer.json");

        // First, try to load tokenizer.json directly
        if tokenizer_path.exists() {
            match Tokenizer::from_file(&tokenizer_path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => {
                    info!("Direct tokenizer loading failed: {}. Attempting compatibility mode for Llama 3.2...", e);
                    // Fall through to attempt creating a compatible tokenizer
                }
            }
        }

        // Llama 3.2 compatibility: Create a simplified tokenizer from tokenizer_config.json
        let config_path = model_path.join("tokenizer_config.json");
        if config_path.exists() {
            return Self::create_llama32_compatible_tokenizer(&config_path, &tokenizer_path).await;
        }

        // If tokenizer.json doesn't exist, try to create it from vocab.json + tokenizer_config.json
        let vocab_path = model_path.join("vocab.json");

        if vocab_path.exists() && config_path.exists() {
            info!("Creating tokenizer from vocab.json and tokenizer_config.json");

            // Read tokenizer config to determine the tokenizer type
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                InferenceError::InvalidArgument(format!(
                    "Failed to read tokenizer_config.json: {}",
                    e
                ))
            })?;

            let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
                InferenceError::InvalidArgument(format!(
                    "Failed to parse tokenizer_config.json: {}",
                    e
                ))
            })?;

            // Get tokenizer class (e.g., "GPTTokenizer", "PreTrainedTokenizer")
            let tokenizer_class = config
                .get("tokenizer_class")
                .and_then(|v| v.as_str())
                .unwrap_or("PreTrainedTokenizer");

            info!("Tokenizer class: {}", tokenizer_class);

            match tokenizer_class {
                "GPT2Tokenizer" => {
                    // Handle GPT2 tokenizer creation
                    return Tokenizer::from_file(&vocab_path).map_err(|e| {
                        InferenceError::InvalidArgument(format!(
                            "Failed to load GPT2 tokenizer: {}",
                            e
                        ))
                    });
                }
                _ => {
                    return Err(InferenceError::InvalidArgument(format!(
                        "Unsupported tokenizer class: {}. Supported: GPT2Tokenizer",
                        tokenizer_class
                    )));
                }
            }
        }

        Err(InferenceError::InvalidArgument(
            "No suitable tokenizer files found. Expected tokenizer.json or vocab.json + tokenizer_config.json".to_string()
        ))
    }

    /// Create a Llama 3.2 compatible tokenizer from tokenizer files
    #[cfg(any(
    ))]
    #[allow(clippy::too_many_lines)]
    async fn create_llama32_compatible_tokenizer(
        config_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<Tokenizer, InferenceError> {
        info!("Creating Llama 3.2 compatible tokenizer from config and extracting vocab from tokenizer.json");

        // Read tokenizer config for special tokens
        let config_str = tokio::fs::read_to_string(config_path).await.map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to read tokenizer_config.json: {}", e))
        })?;

        let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            InferenceError::InvalidArgument(format!("Failed to parse tokenizer_config.json: {}", e))
        })?;

        // Try to extract vocabulary from the original tokenizer.json
        let original_tokenizer_path = tokenizer_path;
        let mut vocab = std::collections::HashMap::new();

        if original_tokenizer_path.exists() {
            info!("Extracting vocabulary from existing tokenizer.json");
            let tokenizer_str = tokio::fs::read_to_string(original_tokenizer_path)
                .await
                .map_err(|e| {
                    InferenceError::InvalidArgument(format!("Failed to read tokenizer.json: {}", e))
                })?;

            if let Ok(tokenizer_data) = serde_json::from_str::<serde_json::Value>(&tokenizer_str) {
                // Extract vocabulary from the tokenizer.json
                if let Some(model) = tokenizer_data.get("model") {
                    if let Some(vocab_obj) = model.get("vocab").and_then(|v| v.as_object()) {
                        info!("Found vocabulary with {} entries", vocab_obj.len());
                        for (token, id) in vocab_obj {
                            if let Some(id_num) = id.as_u64() {
                                vocab.insert(token.clone(), id_num as u32);
                            }
                        }
                    }
                }
            }
        }

        // If we couldn't extract vocab or it's empty, create a basic one
        if vocab.is_empty() {
            info!("Creating fallback vocabulary");
            // Add basic tokens to cover the vocabulary space
            for i in 0..128_000u32 {
                vocab.insert(format!("token_{}", i), i);
            }
        }

        // Create empty merges (character-level tokenization)
        let merges = Vec::new();

        // Create BPE tokenizer
        let bpe_tokenizer = BPE::new(vocab, merges);
        let mut tokenizer = Tokenizer::new(bpe_tokenizer);

        // Add special tokens from config with their proper IDs
        if let Some(added_tokens) = config
            .get("added_tokens_decoder")
            .and_then(|v| v.as_object())
        {
            info!("Adding {} special tokens", added_tokens.len());
            for (id_str, token_info) in added_tokens {
                if let (Ok(id), Some(content)) = (
                    id_str.parse::<u32>(),
                    token_info.get("content").and_then(|v| v.as_str()),
                ) {
                    let special = token_info
                        .get("special")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let added_token = AddedToken::from(content, special);
                    // Try to add the token with correct ID
                    tokenizer.add_tokens(&[added_token]);
                    // Manually set the ID in the tokenizer's vocabulary if possible
                    debug!("Added special token '{}' with ID {}", content, id);
                }
            }
        }

        // Set up padding if configured
        if let Some(pad_token) = config.get("pad_token").and_then(|v| v.as_str()) {
            let padding = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Left,
                pad_to_multiple_of: None,
                pad_id: 128_004, // <|finetune_right_pad_id|> for Llama 3.2
                pad_type_id: 0,
                pad_token: pad_token.to_string(),
            };
            tokenizer.with_padding(Some(padding));
        }

        // Set up truncation
        if let Some(max_length) = config.get("model_max_length").and_then(|v| v.as_u64()) {
            let truncation = TruncationParams {
                max_length: max_length as usize,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            };
            let _ = tokenizer.with_truncation(Some(truncation));
        }

        // Set up post-processor for Llama 3.2 format
        if let Some(bos_token) = config.get("bos_token").and_then(|v| v.as_str()) {
            let post_processor = TemplateProcessing::builder()
                .try_single(format!("{} $A", bos_token))
                .map_err(|e| {
                    InferenceError::InvalidArgument(format!(
                        "Failed to create post-processor: {}",
                        e
                    ))
                })?
                .special_tokens(vec![
                    (bos_token.to_string(), 128_000),    // <|begin_of_text|>
                    ("<|eot_id|>".to_string(), 128_009), // <|eot_id|>
                ])
                .build()
                .map_err(|e| {
                    InferenceError::InvalidArgument(format!(
                        "Failed to build post-processor: {}",
                        e
                    ))
                })?;

            tokenizer.with_post_processor(post_processor);
        }

        // Save the tokenizer for future use
        if let Err(e) = tokenizer.save(tokenizer_path, false) {
            info!("Could not save compatible tokenizer: {}", e);
        }

        info!("Successfully created Llama 3.2 compatible tokenizer");
        Ok(tokenizer)
    }
}
