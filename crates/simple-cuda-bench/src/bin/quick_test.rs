//! Quick 30-second kernel performance test

use simple_cuda_bench::SimpleCudaBench;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Quick CUDA Kernel Test (30 seconds)");
    
    let bench = SimpleCudaBench::new()?;
    
    // Single test case
    let prompt = "Hello, how are you doing today?";
    println!("\nTesting with: \"{}\"", prompt);
    
    let results = bench.compare_libraries(prompt).await?;
    
    // Simple output
    println!("\nResults:");
    for result in results {
        println!("• {}: {:.1}ms ({:.0} tok/s)", 
            result.name, result.time_ms, result.throughput_tokens_per_sec);
    }
    
    Ok(())
}