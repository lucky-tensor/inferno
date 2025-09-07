//! Build script for SmolLM2-135M model download using SafeTensors format
//!
//! This script downloads the SmolLM2-135M model from Hugging Face
//! and prepares it for use with Burn framework.

#[cfg(feature = "burn-cpu")]
fn main() {
    use std::path::Path;

    println!("cargo:rerun-if-changed=build.rs");

    // Create models directory if it doesn't exist
    let models_dir = Path::new("../../models/smollm2-135m");
    std::fs::create_dir_all(models_dir).expect("Failed to create models directory");

    let model_files = [
        ("model.safetensors", "SafeTensors model"),
        ("tokenizer.json", "tokenizer"),
        ("config.json", "config"),
    ];

    for (filename, description) in &model_files {
        let file_path = models_dir.join(filename);
        if !file_path.exists() {
            println!(
                "cargo:warning=Downloading SmolLM2-135M {} ({})...",
                filename, description
            );
            if download_model_file(&file_path, filename) {
                println!("cargo:warning=Successfully downloaded {}", filename);
            } else {
                println!(
                    "cargo:warning=Failed to download {}, continuing...",
                    filename
                );
            }
        } else {
            println!("cargo:warning=Found cached {}", filename);
        }
    }

    // Check if we have the essential files
    let model_path = models_dir.join("model.safetensors");
    let tokenizer_path = models_dir.join("tokenizer.json");

    if model_path.exists() && tokenizer_path.exists() {
        println!("cargo:warning=SmolLM2-135M model ready for Burn framework");
    } else {
        println!("cargo:warning=Missing essential files for SmolLM2-135M model");
    }
}

#[cfg(feature = "burn-cpu")]
fn download_model_file(file_path: &std::path::Path, filename: &str) -> bool {
    let url = format!(
        "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/{}",
        filename
    );

    // Try curl first
    let output = std::process::Command::new("curl")
        .arg("-L") // Follow redirects
        .arg("-o")
        .arg(file_path.to_str().unwrap())
        .arg(&url)
        .output();

    match output {
        Ok(result) if result.status.success() => true,
        _ => {
            // Try wget as fallback
            let wget_result = std::process::Command::new("wget")
                .arg("-O")
                .arg(file_path.to_str().unwrap())
                .arg(&url)
                .output();

            match wget_result {
                Ok(result) if result.status.success() => true,
                _ => {
                    // Rust HTTP fallback
                    download_via_rust(&url, file_path)
                }
            }
        }
    }
}

#[cfg(feature = "burn-cpu")]
fn download_via_rust(url: &str, path: &std::path::Path) -> bool {
    use std::io::Write;

    match reqwest::blocking::get(url) {
        Ok(response) if response.status().is_success() => match std::fs::File::create(path) {
            Ok(mut file) => match response.bytes() {
                Ok(content) => file.write_all(&content).is_ok(),
                Err(_) => false,
            },
            Err(_) => false,
        },
        _ => false,
    }
}

#[cfg(not(feature = "burn-cpu"))]
fn main() {
    println!("cargo:warning=burn-cpu feature not enabled, skipping model download");
}
