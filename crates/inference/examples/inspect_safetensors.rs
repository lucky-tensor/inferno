//! Inspect SafeTensors file to understand weight names

use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../../models/smollm2-135m/model.safetensors";
    
    // Read the file
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // Parse SafeTensors
    let tensors = SafeTensors::deserialize(&buffer)?;
    
    println!("SafeTensors file contains {} tensors", tensors.len());
    println!("\nFirst 20 tensor names:");
    
    let names = tensors.names();
    let mut names: Vec<_> = names.into_iter().map(|s| s.to_string()).collect();
    names.sort();
    
    for (i, name) in names.iter().enumerate() {
        if i >= 20 { break; }
        let tensor = tensors.tensor(name)?;
        let shape = tensor.shape();
        println!("  {}: {:?}", name, shape);
    }
    
    println!("\nEmbedding-related tensors:");
    for name in &names {
        if name.contains("embed") {
            let tensor = tensors.tensor(name)?;
            let shape = tensor.shape();
            println!("  {}: {:?}", name, shape);
        }
    }
    
    println!("\nLayer 0 tensors:");
    for name in &names {
        if name.contains("layers.0.") {
            let tensor = tensors.tensor(name)?;
            let shape = tensor.shape();
            println!("  {}: {:?}", name, shape);
        }
    }
    
    println!("\nLM head and norm tensors:");
    for name in &names {
        if name.contains("lm_head") || (name.contains("norm") && !name.contains("layers")) {
            let tensor = tensors.tensor(name)?;
            let shape = tensor.shape();
            println!("  {}: {:?}", name, shape);
        }
    }
    
    println!("\nLast 10 tensor names:");
    let start = names.len().saturating_sub(10);
    for name in &names[start..] {
        let tensor = tensors.tensor(name)?;
        let shape = tensor.shape();
        println!("  {}: {:?}", name, shape);
    }
    
    Ok(())
}