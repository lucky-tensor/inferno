use safetensors::SafeTensors;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let safetensors_path = format!("{}/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8/model.safetensors", model_path);

    println!("🔍 Inspecting SafeTensors file: {}", safetensors_path);

    if !std::path::Path::new(&safetensors_path).exists() {
        println!("❌ SafeTensors file not found");
        return Ok(());
    }

    // Read the SafeTensors file
    let buffer = fs::read(&safetensors_path)?;
    let safetensors = SafeTensors::deserialize(&buffer)?;

    println!("✅ Successfully loaded SafeTensors file");
    println!("📊 Total tensors: {}", safetensors.len());

    let mut tensor_count = 0;
    let mut scale_keys = Vec::new();
    let mut zero_point_keys = Vec::new();
    let mut weight_keys = Vec::new();

    println!("\n🔍 Analyzing tensor keys and types:");

    for (name, tensor_info) in safetensors.tensors() {
        if tensor_count < 20 {
            println!("  {}: shape={:?}, dtype={:?}", name, tensor_info.shape(), tensor_info.dtype());
        }

        let name_lower = name.to_lowercase();
        if name_lower.contains("scale") {
            scale_keys.push(name.to_string());
        } else if name_lower.contains("zero") || name_lower.contains("zp") {
            zero_point_keys.push(name.to_string());
        } else if name_lower.contains("weight") {
            weight_keys.push(name.to_string());
        }

        tensor_count += 1;
    }

    println!("\n📈 Scale tensors found: {}", scale_keys.len());
    for (i, key) in scale_keys.iter().enumerate() {
        if i < 10 {
            if let Some(tensor_info) = safetensors.tensor(key) {
                println!("  {}: shape={:?}, dtype={:?}", key, tensor_info.shape(), tensor_info.dtype());
            }
        }
    }

    println!("\n🎯 Zero point tensors found: {}", zero_point_keys.len());
    for (i, key) in zero_point_keys.iter().enumerate() {
        if i < 10 {
            if let Some(tensor_info) = safetensors.tensor(key) {
                println!("  {}: shape={:?}, dtype={:?}", key, tensor_info.shape(), tensor_info.dtype());
            }
        }
    }

    println!("\n⚖️ Weight tensors found: {}", weight_keys.len());
    for (i, key) in weight_keys.iter().enumerate() {
        if i < 10 {
            if let Some(tensor_info) = safetensors.tensor(key) {
                println!("  {}: shape={:?}, dtype={:?}", key, tensor_info.shape(), tensor_info.dtype());
            }
        }
    }

    // Look for patterns in tensor naming
    println!("\n🔍 Analyzing tensor naming patterns:");
    let all_keys: Vec<String> = safetensors.tensors().map(|(name, _)| name.to_string()).collect();

    // Group by common prefixes
    let mut prefix_counts = std::collections::HashMap::new();
    for key in &all_keys {
        if let Some(first_dot) = key.find('.') {
            let prefix = &key[..first_dot];
            *prefix_counts.entry(prefix.to_string()).or_insert(0) += 1;
        }
    }

    println!("Common prefixes:");
    for (prefix, count) in prefix_counts.iter() {
        if *count > 1 {
            println!("  {}: {} tensors", prefix, count);
        }
    }

    Ok(())
}