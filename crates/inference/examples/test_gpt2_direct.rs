use candle_core::Device;
use inferno_inference::inference::candle::OpenAIEngine;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("HOME")? + "/.inferno/models/gpt2";
    let device = Device::new_cuda(0)?;
    let mut engine = OpenAIEngine::load_from_safetensors(&model_path, device)?;

    let prompts = vec![
        "The Eiffel Tower is located in",
        "Once upon a time",
        "Python is a programming language",
    ];

    for prompt in prompts {
        println!("\nPrompt: {}", prompt);
        let output = engine.generate(prompt, 30, 0.7)?;
        println!("Output: {}", output);
        engine.reset_caches();
    }

    Ok(())
}
