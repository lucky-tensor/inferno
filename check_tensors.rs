use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8/model.safetensors";

    // Load SafeTensors file
    let buffer = std::fs::read(model_path)?;
    let safetensors = safetensors::SafeTensors::deserialize(&buffer)?;

    println!("Total tensors in model: {}", safetensors.names().len());
    println!("\nFirst 20 tensor names:");

    let mut names: Vec<_> = safetensors.names().collect();
    names.sort();

    for (i, name) in names.iter().take(20).enumerate() {
        let tensor_info = safetensors.tensor(name)?;
        println!("  {}: {} - {:?} dtype, shape {:?}",
                 i + 1, name, tensor_info.dtype(), tensor_info.shape());
    }

    // Look for embedding tensors
    let embed_tensors: Vec<_> = names.iter().filter(|name| name.contains("embed")).collect();
    println!("\nEmbedding tensors:");
    for name in &embed_tensors {
        let tensor_info = safetensors.tensor(name)?;
        println!("  {} - {:?} dtype, shape {:?}", name, tensor_info.dtype(), tensor_info.shape());
    }

    // Look for quantized weight tensors
    let weight_tensors: Vec<_> = names.iter().filter(|name| name.ends_with(".weight")).collect();
    println!("\nFirst 10 weight tensors:");
    for name in weight_tensors.iter().take(10) {
        let tensor_info = safetensors.tensor(name)?;
        println!("  {} - {:?} dtype, shape {:?}", name, tensor_info.dtype(), tensor_info.shape());
    }

    // Look for scale tensors
    let scale_tensors: Vec<_> = names.iter().filter(|name| name.ends_with(".weight_scale")).collect();
    println!("\nFirst 10 scale tensors:");
    for name in scale_tensors.iter().take(10) {
        let tensor_info = safetensors.tensor(name)?;
        println!("  {} - {:?} dtype, shape {:?}", name, tensor_info.dtype(), tensor_info.shape());
    }

    Ok(())
}