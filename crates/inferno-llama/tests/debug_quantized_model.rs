//! Debug test to examine actual dtypes in quantized model

use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;

#[test]
fn debug_quantized_model_dtypes() {
    let model_path =
        "/home/jeef/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8/model.safetensors";

    if !std::path::Path::new(model_path).exists() {
        println!("Skipping test: Model not found at {}", model_path);
        return;
    }

    println!("Loading SafeTensors file: {}", model_path);

    let buffer = fs::read(model_path).expect("Failed to read safetensors file");
    let safetensors = SafeTensors::deserialize(&buffer).expect("Failed to deserialize safetensors");

    let mut dtype_counts: HashMap<String, usize> = HashMap::new();

    println!("Analyzing {} tensors...", safetensors.names().len());

    for (i, tensor_name) in safetensors.names().iter().enumerate() {
        let tensor_info = safetensors
            .tensor(tensor_name)
            .expect("Failed to get tensor info");
        let dtype_str = format!("{:?}", tensor_info.dtype());
        *dtype_counts.entry(dtype_str).or_insert(0) += 1;

        if i < 10 {
            println!(
                "Tensor {}: {} - dtype: {:?}, shape: {:?}",
                i,
                tensor_name,
                tensor_info.dtype(),
                tensor_info.shape()
            );
        }
    }

    println!("\nDtype distribution:");
    for (dtype, count) in &dtype_counts {
        println!("  {}: {} tensors", dtype, count);
    }

    // This test is for debugging - always pass
    assert!(true);
}
