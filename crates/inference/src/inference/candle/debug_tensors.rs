//! Debug utility to inspect tensor names in quantized models
#![allow(dead_code)]

use safetensors::SafeTensors;

/// Inspects the structure and contents of `SafeTensors` files for debugging quantized models
#[allow(clippy::too_many_lines)]
pub fn inspect_safetensors(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let safetensors_path = std::path::Path::new(model_path).join("model.safetensors");
    let buffer = std::fs::read(&safetensors_path)?;
    let safetensors = SafeTensors::deserialize(&buffer)?;

    let names: Vec<_> = safetensors.names();
    println!("Total tensors: {}", names.len());

    println!("\nFirst 20 tensor names:");
    for (i, name) in names.iter().take(20).enumerate() {
        let tensor_info = safetensors.tensor(name)?;
        println!(
            "  {}: {} - {:?} dtype, shape {:?}",
            i + 1,
            name,
            tensor_info.dtype(),
            tensor_info.shape()
        );
    }

    // Look for embedding-related tensors
    let embed_names: Vec<_> = names.iter().filter(|name| name.contains("embed")).collect();
    println!("\nEmbedding tensors:");
    for name in &embed_names {
        let tensor_info = safetensors.tensor(name)?;
        println!(
            "  {} - {:?} dtype, shape {:?}",
            name,
            tensor_info.dtype(),
            tensor_info.shape()
        );
    }

    // Look for model prefix tensors
    let model_names: Vec<_> = names
        .iter()
        .filter(|name| name.starts_with("model."))
        .collect();
    println!("\nModel.* tensors (first 10):");
    for name in model_names.iter().take(10) {
        let tensor_info = safetensors.tensor(name)?;
        println!(
            "  {} - {:?} dtype, shape {:?}",
            name,
            tensor_info.dtype(),
            tensor_info.shape()
        );
    }

    // Look for quantized weight patterns
    let weight_tensors: Vec<_> = names
        .iter()
        .filter(|name| name.ends_with(".weight"))
        .collect();
    println!("\nWeight tensors (first 10):");
    for name in weight_tensors.iter().take(10) {
        let tensor_info = safetensors.tensor(name)?;
        println!(
            "  {} - {:?} dtype, shape {:?}",
            name,
            tensor_info.dtype(),
            tensor_info.shape()
        );
    }

    // Look for scale tensors
    let scale_tensors: Vec<_> = names
        .iter()
        .filter(|name| name.ends_with(".weight_scale"))
        .collect();
    println!("\nScale tensors (first 10):");
    for name in scale_tensors.iter().take(10) {
        let tensor_info = safetensors.tensor(name)?;
        println!(
            "  {} - {:?} dtype, shape {:?}",
            name,
            tensor_info.dtype(),
            tensor_info.shape()
        );
    }

    // Look for specific important tensors
    println!("\nImportant tensors:");
    let important_names = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ];
    for name in &important_names {
        if let Ok(tensor_info) = safetensors.tensor(name) {
            println!(
                "    {} - {:?} dtype, shape {:?}",
                name,
                tensor_info.dtype(),
                tensor_info.shape()
            );
        } else {
            println!("    {} - NOT FOUND", name);
        }
    }

    // Search for tensors containing "head" or "norm"
    let head_norm_tensors: Vec<_> = names
        .iter()
        .filter(|name| name.contains("head") || name.contains("norm"))
        .collect();
    println!("\nHead/Norm tensors:");
    for name in &head_norm_tensors {
        let tensor_info = safetensors.tensor(name)?;
        println!(
            "  {} - {:?} dtype, shape {:?}",
            name,
            tensor_info.dtype(),
            tensor_info.shape()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn inspect_quantized_model_tensors() {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let model_path = format!(
            "{}/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8",
            home
        );

        if std::path::Path::new(&model_path).exists() {
            println!("  Inspecting tensors in: {}", model_path);
            if let Err(e) = inspect_safetensors(&model_path) {
                println!("  Error inspecting tensors: {}", e);
            }
        } else {
            println!("  Model not found at: {}", model_path);
        }
    }
}
