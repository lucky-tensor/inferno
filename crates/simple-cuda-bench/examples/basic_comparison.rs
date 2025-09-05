//! Basic comparison example for CUDA kernel performance

use simple_cuda_bench::{SimpleCudaBench, BenchResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Simple CUDA Kernel Benchmark");
    println!("================================");
    
    let bench = SimpleCudaBench::new()?;
    
    // Test different scenarios
    let test_prompts = [
        "Hello world",
        "Write a short story about AI",
        "Explain quantum computing in simple terms for beginners to understand",
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n📝 Test {}: \"{}\"", i + 1, 
            if prompt.len() > 30 { 
                format!("{}...", &prompt[..30]) 
            } else { 
                prompt.to_string() 
            });
        
        let results = bench.compare_libraries(prompt).await?;
        
        // Sort by performance (fastest first)
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| a.time_ms.partial_cmp(&b.time_ms).unwrap());
        
        for (rank, result) in sorted_results.iter().enumerate() {
            let emoji = match rank {
                0 => "🥇",
                1 => "🥈", 
                2 => "🥉",
                _ => "  ",
            };
            
            let speedup_vs_baseline = if let Some(baseline) = sorted_results.iter().find(|r| r.name == "pytorch_baseline") {
                format!(" ({:.1}x vs PyTorch)", baseline.time_ms / result.time_ms)
            } else {
                String::new()
            };
            
            println!("  {} {:<16}: {:>6.1}ms | {:>6.1} tok/s{}", 
                emoji, result.name, result.time_ms, result.throughput_tokens_per_sec, speedup_vs_baseline);
        }
        
        // Calculate speedup vs slowest
        if sorted_results.len() > 1 {
            let fastest = &sorted_results[0];
            let slowest = &sorted_results[sorted_results.len() - 1];
            let speedup = slowest.time_ms / fastest.time_ms;
            println!("  ⚡ Speedup: {:.1}x faster ({} vs {})", 
                speedup, fastest.name, slowest.name);
        }
    }
    
    println!("\n✅ Benchmark complete!");
    Ok(())
}

// Expected performance characteristics relative to PyTorch baseline
// PyTorch baseline: ~300ms (unoptimized Python + GPU overhead)
// lm.rs: ~100ms (optimized CPU, no Python overhead) 
// Candle: ~50ms (optimized GPU kernels)
// Burn: ~30ms (advanced GPU optimizations)
// TensorRT: ~20ms (industry-leading optimization)