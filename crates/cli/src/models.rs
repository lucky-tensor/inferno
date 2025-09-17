//! Shared model management functionality
//!
//! This module provides utilities for discovering and validating
//! machine learning models. It does not contain interactive functionality
//! to maintain separation between CLI-specific logic and shared utilities.

use inferno_shared::{InfernoError, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

/// Model file extension (we only support SafeTensors format)
const MODEL_EXTENSION: &str = ".safetensors";

/// Discovered model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Full path to the model file
    pub path: PathBuf,
    /// Display name (includes parent directory for context)
    pub name: String,
    /// File size in bytes
    pub size_bytes: u64,
}

/// Discover SafeTensors models in a directory recursively
pub fn discover_models(model_dir: &str) -> Result<Vec<ModelInfo>> {
    let path = Path::new(model_dir);

    if !path.exists() {
        return Ok(Vec::new());
    }

    if !path.is_dir() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();
    discover_models_recursive(path, &mut models)?;

    // Sort models by name for consistent ordering
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

/// Recursively discover models in directory and subdirectories
fn discover_models_recursive(dir: &Path, models: &mut Vec<ModelInfo>) -> Result<()> {
    let entries = fs::read_dir(dir).map_err(|e| {
        InfernoError::internal(
            format!("Failed to read directory '{}': {}", dir.display(), e),
            None,
        )
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            InfernoError::internal(format!("Failed to read directory entry: {}", e), None)
        })?;

        let file_path = entry.path();

        if file_path.is_dir() {
            // Recursively search subdirectories
            discover_models_recursive(&file_path, models)?;
        } else {
            // Check if file has the model extension (.safetensors)
            if let Some(extension) = file_path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    let ext_lower = format!(".{}", ext_str.to_lowercase());
                    if ext_lower == MODEL_EXTENSION {
                        // Get file metadata for size
                        let metadata = fs::metadata(&file_path).map_err(|e| {
                            InfernoError::internal(
                                format!("Failed to get file metadata: {}", e),
                                None,
                            )
                        })?;

                        // Create a more descriptive name that includes parent directory
                        let name = if let Some(parent) = file_path.parent() {
                            if let Some(parent_name) = parent.file_name() {
                                if let Some(file_name) = file_path.file_name() {
                                    format!(
                                        "{}/{}",
                                        parent_name.to_string_lossy(),
                                        file_name.to_string_lossy()
                                    )
                                } else {
                                    file_path.to_string_lossy().to_string()
                                }
                            } else {
                                file_path.to_string_lossy().to_string()
                            }
                        } else {
                            file_path.to_string_lossy().to_string()
                        };

                        models.push(ModelInfo {
                            path: file_path,
                            name,
                            size_bytes: metadata.len(),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

/// Format file size in human readable format
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{:.0} {}", size, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Result of model validation and discovery
#[derive(Debug)]
pub enum ModelValidationResult {
    /// A specific valid model file was found
    SingleModel(String),
    /// No models were found in the directory
    NoModels,
    /// Multiple models were found - caller needs to handle selection
    MultipleModels(Vec<ModelInfo>),
}

/// Validate and discover models based on the provided path
///
/// This function only handles validation and discovery, not user interaction.
/// The caller is responsible for handling the interactive selection logic.
pub fn validate_and_discover_models(model_path: &str) -> Result<ModelValidationResult> {
    let path = Path::new(model_path);

    // If model_path is already a specific file, validate and use it
    if path.is_file() {
        // Verify it's a SafeTensors file
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                let ext_lower = format!(".{}", ext_str.to_lowercase());
                if ext_lower == MODEL_EXTENSION {
                    info!("Using specified model file: {}", model_path);
                    return Ok(ModelValidationResult::SingleModel(model_path.to_string()));
                }
            }
        }

        return Err(InfernoError::configuration(
            format!(
                "Specified model file '{}' is not a SafeTensors (.safetensors) file",
                model_path
            ),
            None,
        ));
    }

    // Check if this directory contains a sharded model (treat as single model)
    if path.is_dir() && is_sharded_model_directory(path) {
        info!("Found sharded model directory: {}", model_path);
        return Ok(ModelValidationResult::SingleModel(model_path.to_string()));
    }

    // Discover models in the directory
    let models = discover_models(model_path)?;

    match models.len() {
        0 => Ok(ModelValidationResult::NoModels),
        1 => {
            // Only one model found, use it automatically
            let model = &models[0];
            info!(
                "Found single model, using: {} ({})",
                model.name,
                format_file_size(model.size_bytes)
            );
            Ok(ModelValidationResult::SingleModel(
                model.path.to_string_lossy().to_string(),
            ))
        }
        _ => Ok(ModelValidationResult::MultipleModels(models)),
    }
}

/// Check if a directory contains a sharded SafeTensors model
fn is_sharded_model_directory(dir: &Path) -> bool {
    // Must have index file and required config files
    let has_index = dir.join("model.safetensors.index.json").exists();
    let has_config = dir.join("config.json").exists();
    let has_tokenizer = dir.join("tokenizer.json").exists();

    if !has_index || !has_config || !has_tokenizer {
        return false;
    }

    // Must have at least one sharded model file
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    return true;
                }
            }
        }
    }

    false
}
