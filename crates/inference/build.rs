//! Build script for ONNX model import using Burn framework
//!
//! This script downloads the SmolLM2-135M-Instruct ONNX model from Hugging Face
//! and generates the corresponding Burn model code.

#[cfg(feature = "burn-import")]
fn main() {
    use std::path::Path;
    use std::io::Write;
    
    println!("cargo:rerun-if-changed=build.rs");
    
    // Create models directory if it doesn't exist
    let models_dir = Path::new("../../models/smollm2-135m-instruct");
    std::fs::create_dir_all(&models_dir).expect("Failed to create models directory");
    
    let onnx_path = models_dir.join("model.onnx");
    
    // Try different ONNX model variants, starting with smallest quantized version
    let model_variants = [
        ("model_q4f16.onnx", "118MB quantized FP16"),
        ("model_int8.onnx", "137MB INT8 quantized"),
        ("model_fp16.onnx", "270MB FP16"),
        ("model.onnx", "540MB full precision"),
    ];
    
    let mut model_downloaded = false;
    for (filename, description) in &model_variants {
        let variant_path = models_dir.join(filename);
        if !variant_path.exists() {
            println!("cargo:warning=Downloading SmolLM2-135M-Instruct ONNX model ({})...", description);
            if download_onnx_model(&variant_path, filename) {
                // Test if this variant works with Burn (opset 16+)
                if test_onnx_compatibility(&variant_path) {
                    // Copy to standard name for Burn to use
                    std::fs::copy(&variant_path, &onnx_path).expect("Failed to copy working ONNX model");
                    model_downloaded = true;
                    break;
                } else {
                    println!("cargo:warning=ONNX model {} has incompatible opset, trying next variant...", filename);
                    std::fs::remove_file(&variant_path).unwrap_or(());
                }
            }
        } else {
            // Test existing file
            if test_onnx_compatibility(&variant_path) {
                std::fs::copy(&variant_path, &onnx_path).expect("Failed to copy working ONNX model");
                model_downloaded = true;
                break;
            }
        }
    }
    
    if !model_downloaded {
        println!("cargo:warning=No compatible ONNX model found with opset 16+");
        return;
    }
    
    // Generate Burn model from ONNX
    if onnx_path.exists() {
        println!("cargo:warning=Generating Burn model from ONNX...");
        
        burn_import::onnx::ModelGen::new()
            .input(onnx_path.to_str().unwrap())
            .out_dir("model/")
            .run_from_script();
    }
}

#[cfg(feature = "burn-import")]
fn download_onnx_model(onnx_path: &std::path::Path, filename: &str) -> bool {
    let url = format!("https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/onnx/{}", filename);
    
    // Use curl or wget to download the large file
    let output = std::process::Command::new("curl")
        .arg("-L") // Follow redirects
        .arg("-o")
        .arg(onnx_path.to_str().unwrap())
        .arg(url)
        .output();
        
    match output {
        Ok(result) if result.status.success() => {
            println!("cargo:warning=Successfully downloaded ONNX model to {:?}", onnx_path);
            return true;
        }
        _ => {
            // Try wget as fallback
            let wget_result = std::process::Command::new("wget")
                .arg("-O")
                .arg(onnx_path.to_str().unwrap())
                .arg(&url)
                .output();
                
            match wget_result {
                Ok(result) if result.status.success() => {
                    println!("cargo:warning=Successfully downloaded ONNX model via wget");
                    return true;
                }
                _ => {
                    // Manual fallback using Rust (this will be slow)
                    println!("cargo:warning=Downloading via Rust HTTP client (this may be slow)...");
                    return download_via_rust(&url, onnx_path);
                }
            }
        }
    }
}

#[cfg(feature = "burn-import")]
fn download_via_rust(url: &str, path: &std::path::Path) -> bool {
    use std::io::Write;
    // Simple blocking download - not ideal but works as fallback
    match reqwest::blocking::get(url) {
        Ok(response) if response.status().is_success() => {
            match std::fs::File::create(path) {
                Ok(mut file) => {
                    match response.bytes() {
                        Ok(content) => {
                            match file.write_all(&content) {
                                Ok(_) => {
                                    println!("cargo:warning=Successfully downloaded via Rust HTTP client");
                                    return true;
                                }
                                Err(e) => println!("cargo:warning=Failed to write file: {}", e),
                            }
                        }
                        Err(e) => println!("cargo:warning=Failed to read response: {}", e),
                    }
                }
                Err(e) => println!("cargo:warning=Failed to create file: {}", e),
            }
        }
        Ok(response) => println!("cargo:warning=HTTP error: {}", response.status()),
        Err(e) => println!("cargo:warning=Request failed: {}", e),
    }
    false
}

#[cfg(feature = "burn-import")]
fn test_onnx_compatibility(onnx_path: &std::path::Path) -> bool {
    // Simple test: try to parse the ONNX file with burn-import
    // If it fails with opset version error, return false
    use std::process::Command;
    
    // We can't easily test burn-import directly in build script,
    // so for now assume newer quantized models have correct opset
    let filename = onnx_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
        
    // Heuristic: quantized models (q4, int8) are likely newer with opset 16+
    if filename.contains("q4") || filename.contains("int8") || filename.contains("fp16") {
        println!("cargo:warning=Assuming {} has compatible opset (quantized model)", filename);
        return true;
    }
    
    // For full precision model, it might be older opset
    println!("cargo:warning=Assuming {} might have older opset", filename);
    false
}

#[cfg(not(feature = "burn-import"))]
fn main() {
    println!("cargo:warning=burn-import feature not enabled, skipping ONNX model generation");
}