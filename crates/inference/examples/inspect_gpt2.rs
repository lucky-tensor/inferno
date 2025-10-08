//! Inspect GPT-2 SafeTensors file to understand weight names

use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 {
        &args[1]
    } else {
        "~/.inferno/models/gpt2/model.safetensors"
    };

    // Expand ~ if needed
    let path = path.replace("~", &std::env::var("HOME").unwrap_or_else(|_| ".".to_string()));

    println!("Inspecting: {}\n", path);

    // Read the file
    let mut file = File::open(&path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Parse SafeTensors
    let tensors = SafeTensors::deserialize(&buffer)?;

    println!("SafeTensors file contains {} tensors\n", tensors.len());

    let names = tensors.names();
    let mut names: Vec<_> = names.into_iter().map(|s| s.to_string()).collect();
    names.sort();

    println!("ALL TENSOR NAMES AND SHAPES:");
    println!("============================");
    for name in &names {
        let tensor = tensors.tensor(name)?;
        let shape = tensor.shape();
        println!("  {}: {:?}", name, shape);
    }

    println!("\n\nANALYSIS:");
    println!("=========");

    // Find embeddings
    println!("\n1. Token/Position Embeddings:");
    for name in &names {
        if name.contains("wte") || name.contains("wpe") || name.contains("embed") {
            let tensor = tensors.tensor(name)?;
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    // Find layer 0 patterns
    println!("\n2. Layer 0 patterns:");
    for name in &names {
        if name.contains(".0.") || name.contains(".h.0") {
            let tensor = tensors.tensor(name)?;
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    // Find attention patterns
    println!("\n3. Attention patterns:");
    for name in &names {
        if name.contains("attn") {
            let tensor = tensors.tensor(name)?;
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    // Find MLP patterns
    println!("\n4. MLP patterns:");
    for name in &names {
        if name.contains("mlp") || (name.contains("c_fc") || name.contains("c_proj")) {
            let tensor = tensors.tensor(name)?;
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    // Final norm
    println!("\n5. Final norm:");
    for name in &names {
        if (name.contains("ln_f") || name.contains("norm")) && !name.contains(".h.") {
            let tensor = tensors.tensor(name)?;
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    // LM head
    println!("\n6. LM head:");
    for name in &names {
        if name.contains("lm_head") {
            let tensor = tensors.tensor(name)?;
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    Ok(())
}
