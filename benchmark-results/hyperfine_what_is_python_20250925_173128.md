| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `./target/release/inferno-baseline play --prompt "what is python?" --model-path "/home/jeef/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors"` | 2.000 ± 0.005 | 1.990 | 2.008 | 1.01 ± 0.02 |
| `./target/release/inferno-pgo play --prompt "what is python?" --model-path "/home/jeef/.inferno/models/TinyLlama_TinyLlama-1.1B-Chat-v1.0/model.safetensors"` | 1.978 ± 0.045 | 1.851 | 1.999 | 1.00 |
