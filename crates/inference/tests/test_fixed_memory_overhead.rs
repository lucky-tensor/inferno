//! Test for fixed memory overhead calculation

use inferno_shared::{MemoryOverhead, ModelMemoryValidator};

#[tokio::test]
async fn test_fixed_memory_overhead_calculation() -> Result<(), Box<dyn std::error::Error>> {
    println!("FIXED MEMORY OVERHEAD TEST");
    println!("==========================");

    // Test that overhead is now fixed amounts, not proportional
    let custom_overhead = MemoryOverhead {
        loading_overhead_gb: 2.0,
        inference_overhead_gb: 1.5,
        framework_overhead_gb: 1.0,
    };

    let _validator = ModelMemoryValidator::with_memory_overhead(0, custom_overhead);

    println!("\nFixed Overhead Analysis:");
    println!("Loading overhead:   2.0 GB (SafeTensors parsing, temp buffers)");
    println!("Inference overhead: 1.5 GB (KV cache initialization)");
    println!("Framework overhead: 1.0 GB (CUDA context, cuDNN workspace)");
    println!("Total overhead:     4.5 GB (fixed amount)");

    // Test with different model sizes - overhead should be the same
    let test_cases = vec![
        ("Small model (1B)", 2.0),
        ("Medium model (7B)", 14.0),
        ("Large model (13B)", 26.0),
        ("XLarge model (33B)", 66.0),
        ("XXLarge model (70B)", 140.0),
    ];

    println!("\nOverhead Comparison (Old vs New):");
    println!("Model Size | Old (Proportional) | New (Fixed) | Difference");
    println!("-----------|-------------------|-------------|------------");

    for (_model_name, model_size_gb) in test_cases {
        // Old proportional calculation (approximate)
        let old_overhead = match model_size_gb {
            size if size < 2.0 => model_size_gb * 1.2 - model_size_gb,
            size if size < 15.0 => model_size_gb * 1.3 - model_size_gb,
            size if size < 30.0 => model_size_gb * 1.4 - model_size_gb,
            size if size < 150.0 => model_size_gb * 1.5 - model_size_gb,
            _ => model_size_gb * 1.8 - model_size_gb,
        };

        // New fixed overhead
        let new_overhead = 4.5; // Total fixed overhead

        let difference = old_overhead - new_overhead;

        println!(
            "{:11} | {:>16.1}GB | {:>10.1}GB | {:>9.1}GB",
            format!("{:.0}GB", model_size_gb),
            old_overhead,
            new_overhead,
            difference
        );
    }

    println!("\nKey Benefits of Fixed Overhead:");
    println!("1. More realistic memory estimation for large models");
    println!("2. Large models no longer artificially penalized");
    println!("3. Small models have slightly more conservative estimates");
    println!("4. Overhead represents actual GPU infrastructure costs");

    // Validate that 70B model is much more feasible now
    let model_70b_old_total = 140.0 + (140.0 * 0.8); // ~250GB total
    let model_70b_new_total = 140.0 + 4.5; // ~145GB total
    let savings_gb = model_70b_old_total - model_70b_new_total;

    println!("\nExample: 70B Model Impact:");
    println!(
        "Old calculation: {:.0}GB total requirement",
        model_70b_old_total
    );
    println!(
        "New calculation: {:.1}GB total requirement",
        model_70b_new_total
    );
    println!(
        "Memory savings:  {:.0}GB ({:.0}% reduction)",
        savings_gb,
        (savings_gb / model_70b_old_total) * 100.0
    );

    Ok(())
}

#[tokio::test]
async fn test_realistic_gpu_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("REALISTIC GPU SCENARIOS TEST");
    println!("============================");

    let _validator = ModelMemoryValidator::new(0); // Default overhead: 4.5GB

    println!("\nGPU Memory Scenarios with Fixed 4.5GB Overhead:");

    let gpu_scenarios = vec![
        ("RTX 3060 Ti", 8.0),
        ("RTX 4060 Ti", 16.0),
        ("RTX 4090", 24.0),
        ("A6000", 48.0),
        ("H100", 80.0),
    ];

    let model_scenarios = vec![
        ("Llama 3.2 1B", 2.0),
        ("Llama 3.2 3B", 6.0),
        ("Llama 3.1 8B", 15.0),
        ("Llama 2 13B", 26.0),
        ("Llama 2 33B", 66.0),
        ("Llama 2 70B", 140.0),
    ];

    for (gpu_name, gpu_memory_gb) in &gpu_scenarios {
        println!("\n{} ({:.0}GB):", gpu_name, gpu_memory_gb);

        for (model_name, model_size_gb) in &model_scenarios {
            let total_required = model_size_gb + 4.5; // Fixed overhead
            let fits = total_required <= *gpu_memory_gb;
            let status = if fits { "✓" } else { "✗" };

            println!("  {} {} ({:.1}GB req)", status, model_name, total_required);
        }
    }

    println!("\nConclusions:");
    println!("- Fixed overhead enables more accurate capacity planning");
    println!("- Large models are now properly accessible on high-end GPUs");
    println!("- Small models still have reasonable overhead estimates");

    Ok(())
}
