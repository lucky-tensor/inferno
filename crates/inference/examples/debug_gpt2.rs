//! Debug GPT-2 inference step by step

use candle_core::{Device, Tensor};
use inferno_inference::inference::candle::{tokenizer::Tokenizer, OpenAIEngine};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("HOME")? + "/.inferno/models/gpt2";

    println!("🔍 GPT-2 Debug Mode");
    println!("==================\n");

    // Load tokenizer
    println!("📝 Loading tokenizer...");
    let tokenizer = Tokenizer::from_file_or_config(&model_path)?;

    // Test encoding
    let test_prompt = "Hello, my name is";
    println!("Test prompt: \"{}\"", test_prompt);

    let encoding = tokenizer.encode(test_prompt, true)?;
    let input_ids = encoding.get_ids();
    println!("Token IDs: {:?}", input_ids);

    // Decode each token
    println!("\nToken breakdown:");
    for (i, &token_id) in input_ids.iter().enumerate() {
        let decoded = tokenizer.decode(&[token_id], false)?;
        println!("  [{}] {} -> \"{}\"", i, token_id, decoded);
    }

    // Load model
    println!("\n📦 Loading model...");
    let device = Device::new_cuda(0)?;
    let mut engine = OpenAIEngine::load_from_safetensors(&model_path, device.clone())?;
    println!("✅ Model loaded\n");

    // Generate one token at a time with inspection
    println!("🎯 Generating tokens:");
    let mut generated_ids = input_ids.to_vec();

    for step in 0..10 {
        // Get current input
        let current_input = if step == 0 {
            Tensor::new(input_ids, &device)?.unsqueeze(0)?
        } else {
            let last_token = generated_ids.last().unwrap();
            Tensor::new(&[*last_token], &device)?.unsqueeze(0)?
        };

        println!("\nStep {}:", step);
        println!("  Input shape: {:?}", current_input.shape());

        // Forward pass
        let logits = engine.forward_internal(&current_input, step > 0)?;
        println!("  Logits shape: {:?}", logits.shape());

        // Get last token logits
        let last_logits = logits.i((0, logits.dim(1)? - 1))?;

        // Find top 10 tokens
        let logits_vec = last_logits.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Top 10 predictions:");
        for (rank, &(token_id, score)) in indexed.iter().take(10).enumerate() {
            let decoded = tokenizer.decode(&[token_id as u32], false)?;
            println!(
                "    {}. [{}] score={:.2} -> \"{}\"",
                rank + 1,
                token_id,
                score,
                decoded.replace("\n", "\\n")
            );
        }

        let next_token = indexed[0].0 as u32;
        generated_ids.push(next_token);

        let decoded_token = tokenizer.decode(&[next_token], false)?;
        println!("  ✓ Selected: [{}] \"{}\"", next_token, decoded_token);
    }

    println!("\n📄 Final output:");
    let full_output = tokenizer.decode(&generated_ids, true)?;
    println!("{}", full_output);

    Ok(())
}
