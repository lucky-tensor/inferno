use candle_core::Device;
use inferno_inference::inference::candle::OpenAIEngine;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("HOME")? + "/.inferno/models/gpt2";
    let device = Device::new_cuda(0)?;
    let mut engine = OpenAIEngine::load_from_safetensors(&model_path, device)?;

    let prompt = "The meaning of life is";

    for i in 1..=3 {
        println!("\n=== Generation {} ===", i);
        let output = engine.generate(prompt, 50, 0.7)?;
        println!("{}", output);
        engine.reset_caches();
    }

    Ok(())
}
