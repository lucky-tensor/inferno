# Claude Development Guidelines

## Code Style Requirements

### Logging and Output
- **No Emojis**: Do not use emojis in logging messages, console output, or any production code
- Use clear, descriptive text instead of visual symbols
- Example: Use "Loading model..." instead of "🔄 Loading model..."

### Memory and Performance Recommendations
- **No CPU Inference Recommendations**: Do not recommend CPU inference for large language models (>1B parameters)
- CPU inference is extremely slow and impractical for production LLM usage
- Instead recommend: quantized models, smaller models, model sharding, or better hardware

### Other Requirements
- Follow Rust best practices and clippy suggestions
- Maintain consistent formatting with rustfmt
- Write clear, self-documenting code