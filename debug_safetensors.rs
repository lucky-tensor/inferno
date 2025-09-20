use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8/model.safetensors";

    let buffer = fs::read(model_path)?;
    let safetensors = safetensors::SafeTensors::deserialize(&buffer)?;

    let names: Vec<&str> = safetensors.names().collect();
    println!("Total tensors: {}", names.len());

    println!("\nFirst 20 tensor names:");
    for (i, name) in names.iter().take(20).enumerate() {
        let tensor = safetensors.tensor(name)?;
        println!("{}. {} - shape: {:?}, dtype: {:?}",
                 i + 1, name, tensor.shape(), tensor.dtype());
    }

    println!("\nEmbedding-related weights:");
    for name in &names {
        if name.to_lowercase().contains("embed") {
            let tensor = safetensors.tensor(name)?;
            println!("   {} - shape: {:?}, dtype: {:?}",
                     name, tensor.shape(), tensor.dtype());
        }
    }

    println!("\nFirst layer weights (first 10):");
    let mut layer_count = 0;
    for name in &names {
        if name.starts_with("model.layers.0.") {
            let tensor = safetensors.tensor(name)?;
            println!("   {} - shape: {:?}, dtype: {:?}",
                     name, tensor.shape(), tensor.dtype());
            layer_count += 1;
            if layer_count >= 10 { break; }
        }
    }

    println!("\nNormalization weights:");
    for name in &names {
        if name.contains("norm") {
            let tensor = safetensors.tensor(name)?;
            println!("   {} - shape: {:?}, dtype: {:?}",
                     name, tensor.shape(), tensor.dtype());
        }
    }

    Ok(())
}
