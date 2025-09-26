| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `./target/release/inferno-baseline play --prompt "explain machine learning briefly" --model-path "/home/jeef/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors"` | 1.989 ± 0.031 | 1.901 | 2.008 | 1.01 ± 0.03 |
| `./target/release/inferno-pgo play --prompt "explain machine learning briefly" --model-path "/home/jeef/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors"` | 1.972 ± 0.058 | 1.826 | 2.010 | 1.00 |
