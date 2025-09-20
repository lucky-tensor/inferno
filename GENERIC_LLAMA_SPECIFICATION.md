# Generic Llama Inference Engine Specification

## Overview

The Generic Llama Inference Engine provides a unified interface for loading and running inference on Llama-like models with varying architectures, data types, and quantization schemes. This engine supports original Meta Llama models, distilled variants (TinyLlama), and quantized models (w8a8 compressed-tensors) with flexible data type handling across all components.

## Core Architecture

### 1. Model Discovery and Diagnostics

#### ModelDiagnostic
```rust
pub struct ModelDiagnostic {
    pub model_path: PathBuf,
    pub model_type: LlamaVariant,
    pub config: LlamaConfig,
    pub weight_info: WeightInfo,
    pub tokenizer_info: TokenizerInfo,
    pub quantization_config: Option<QuantizationConfig>,
    pub supported_dtypes: Vec<DataType>,
    pub memory_requirements: MemoryRequirements,
}

pub enum LlamaVariant {
    MetaLlama {
        version: String,  // "3.1", "3.2", etc.
        size: String,     // "1B", "8B", "70B", etc.
    },
    TinyLlama {
        version: String,  // "1.1B", etc.
    },
    Distilled {
        base_model: String,
        vendor: String,
    },
    Custom {
        name: String,
    },
}
```

#### Model Discovery Process
1. **Directory Scanning**: Detect model files (safetensors, bin, config.json, tokenizer files)
2. **Config Analysis**: Parse model configuration to determine architecture
3. **Weight Inspection**: Analyze tensor dtypes, shapes, and quantization schemes
4. **Capability Assessment**: Determine supported operations and memory requirements
5. **Compatibility Check**: Validate against available backends and hardware

### 2. Flexible Data Type System

#### DataType Support
```rust
pub enum DataType {
    // Floating Point
    F32,
    F16,
    BF16,

    // Integer Types
    I8,
    I4,
    U8,
    U4,

    // Quantized Types
    W8A8,          // 8-bit weights, 8-bit activations
    W4A16,         // 4-bit weights, 16-bit activations
    W8A16,         // 8-bit weights, 16-bit activations
    CompressedTensors, // RedHat AI format

    // Custom formats
    Custom(String),
}

pub struct LayerDataTypeConfig {
    pub embedding: DataType,
    pub attention_weights: HashMap<AttentionComponent, DataType>,
    pub mlp_weights: HashMap<MLPComponent, DataType>,
    pub layer_norm: DataType,
    pub activations: DataType,
}

pub enum AttentionComponent {
    QueryProjection,
    KeyProjection,
    ValueProjection,
    OutputProjection,
    RotaryEmbedding,
}

pub enum MLPComponent {
    GateProjection,
    UpProjection,
    DownProjection,
}
```

### 3. Generic Llama Configuration

#### Unified Configuration System
```rust
pub struct LlamaConfig {
    // Core Architecture
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: Option<u32>,
    pub max_position_embeddings: u32,

    // Activation and Normalization
    pub hidden_act: ActivationType,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub tie_word_embeddings: bool,

    // RoPE Configuration
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,

    // Token Configuration
    pub bos_token_id: u32,
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: Option<u32>,

    // Data Types
    pub default_dtype: DataType,
    pub layer_dtypes: Option<LayerDataTypeConfig>,

    // Quantization
    pub quantization_config: Option<QuantizationConfig>,
}

pub enum ActivationType {
    Silu,
    Gelu,
    Relu,
    Swish,
}

pub struct RopeScaling {
    pub rope_type: String,  // "llama3", "linear", "dynamic"
    pub factor: f64,
    pub low_freq_factor: Option<f64>,
    pub high_freq_factor: Option<f64>,
    pub original_max_position_embeddings: Option<u32>,
}
```

### 4. Quantization Support

#### Quantization Configuration
```rust
pub struct QuantizationConfig {
    pub method: QuantizationMethod,
    pub format: String,
    pub config_groups: HashMap<String, QuantizationGroup>,
    pub global_compression_ratio: Option<f64>,
    pub ignored_layers: Vec<String>,
}

pub enum QuantizationMethod {
    CompressedTensors,
    GPTQ,
    AWQ,
    GGUF,
    Custom(String),
}

pub struct QuantizationGroup {
    pub targets: Vec<String>,
    pub weights: QuantizationSpec,
    pub input_activations: Option<QuantizationSpec>,
    pub output_activations: Option<QuantizationSpec>,
}

pub struct QuantizationSpec {
    pub num_bits: u8,
    pub symmetric: bool,
    pub dynamic: bool,
    pub group_size: Option<u32>,
    pub strategy: String,  // "token", "channel", "tensor"
    pub observer: String,  // "minmax", "memoryless"
}
```

### 5. Generic Llama Inference Engine

#### Main Engine Interface
```rust
pub struct GenericLlamaEngine {
    config: LlamaConfig,
    model: Box<dyn LlamaModel>,
    tokenizer: Box<dyn LlamaTokenizer>,
    backend: Box<dyn InferenceBackend>,
    memory_manager: MemoryManager,
}

impl GenericLlamaEngine {
    pub async fn from_diagnostic(
        diagnostic: ModelDiagnostic,
        backend_config: BackendConfig,
    ) -> Result<Self, LlamaEngineError> {
        // Implementation
    }

    pub async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResponse, LlamaEngineError> {
        // Implementation
    }
}
```

#### Model Loading Strategy
```rust
pub trait LlamaModel: Send + Sync {
    fn load_weights(&mut self, weights: WeightMap) -> Result<(), ModelError>;
    fn forward(&self, input_ids: &[u32], cache: &mut KVCache) -> Result<Tensor, ModelError>;
    fn get_config(&self) -> &LlamaConfig;
}

pub struct WeightMap {
    pub embeddings: TensorMap,
    pub layers: Vec<LayerWeights>,
    pub final_norm: TensorMap,
    pub lm_head: Option<TensorMap>,
}

pub struct LayerWeights {
    pub self_attention: AttentionWeights,
    pub mlp: MLPWeights,
    pub input_layernorm: TensorMap,
    pub post_attention_layernorm: TensorMap,
}
```

### 6. Backend Abstraction

#### Backend Interface
```rust
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &str;
    fn supported_dtypes(&self) -> &[DataType];
    fn create_tensor(&self, data: &[u8], shape: &[usize], dtype: DataType) -> Result<Tensor, BackendError>;
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError>;
    fn apply_rope(&self, tensor: &Tensor, positions: &[u32]) -> Result<Tensor, BackendError>;
    fn layer_norm(&self, input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor, BackendError>;
}

pub enum BackendType {
    CandleCpu,
    CandleCuda,
    CandleMetal,
    Custom(String),
}
```

### 7. Memory Management

#### Memory Requirements and Optimization
```rust
pub struct MemoryRequirements {
    pub model_weights: usize,
    pub kv_cache: usize,
    pub activation_memory: usize,
    pub total_estimated: usize,
    pub recommendations: Vec<MemoryOptimization>,
}

pub enum MemoryOptimization {
    QuantizeToDataType(DataType),
    ReduceKVCacheSize(u32),
    EnableGradientCheckpointing,
    UseModelSharding(u32),
    ReduceBatchSize(u32),
}

pub struct MemoryManager {
    available_memory: usize,
    allocated_memory: usize,
    memory_pools: HashMap<DataType, MemoryPool>,
}
```

### 8. Tokenizer Abstraction

#### Generic Tokenizer Interface
```rust
pub trait LlamaTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>;
    fn vocab_size(&self) -> u32;
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;
}

pub enum TokenizerType {
    SentencePiece,
    BPE,
    Tiktoken,
    Custom(String),
}
```

## Usage Examples

### 1. Model Discovery and Initialization

```rust
// Discover and diagnose model
let diagnostic = ModelDiagnostic::analyze_directory("~/models/meta-llama_Llama-3.1-8B-Instruct").await?;

// Configure backend
let backend_config = BackendConfig::new(BackendType::CandleCuda)
    .with_memory_limit(16_000_000_000)  // 16GB
    .with_precision(DataType::BF16);

// Create engine
let engine = GenericLlamaEngine::from_diagnostic(diagnostic, backend_config).await?;
```

### 2. Quantized Model Loading

```rust
// Load w8a8 quantized model
let diagnostic = ModelDiagnostic::analyze_directory("~/models/RedHatAI_Llama-3.2-1B-Instruct-quantized.w8a8").await?;

let backend_config = BackendConfig::new(BackendType::CandleCuda)
    .with_quantization_support(true);

let engine = GenericLlamaEngine::from_diagnostic(diagnostic, backend_config).await?;
```

### 3. Mixed Precision Configuration

```rust
// Custom dtype configuration per layer
let layer_config = LayerDataTypeConfig {
    embedding: DataType::F16,
    attention_weights: HashMap::from([
        (AttentionComponent::QueryProjection, DataType::W8A16),
        (AttentionComponent::KeyProjection, DataType::W8A16),
        (AttentionComponent::ValueProjection, DataType::W8A16),
        (AttentionComponent::OutputProjection, DataType::BF16),
    ]),
    mlp_weights: HashMap::from([
        (MLPComponent::GateProjection, DataType::W8A16),
        (MLPComponent::UpProjection, DataType::W8A16),
        (MLPComponent::DownProjection, DataType::BF16),
    ]),
    layer_norm: DataType::F32,
    activations: DataType::BF16,
};

let config = LlamaConfig {
    layer_dtypes: Some(layer_config),
    ..diagnostic.config
};
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Model diagnostic system
- [ ] Unified configuration parsing
- [ ] Backend abstraction layer
- [ ] Basic tensor operations

### Phase 2: Model Loading
- [ ] SafeTensors weight loading
- [ ] Quantization format support
- [ ] Memory-efficient loading strategies
- [ ] Weight verification and validation

### Phase 3: Inference Engine
- [ ] Forward pass implementation
- [ ] KV cache management
- [ ] Generation strategies (greedy, sampling)
- [ ] Batch processing support

### Phase 4: Optimization
- [ ] Memory optimization
- [ ] Custom operation kernels
- [ ] Multi-GPU support
- [ ] Model sharding

### Phase 5: Advanced Features
- [ ] Custom quantization schemes
- [ ] Dynamic precision switching
- [ ] Model composition (MoE support)
- [ ] Streaming inference

## Model Compatibility Matrix

| Model Type | Quantization | Supported Backends | Memory Req | Status |
|------------|--------------|-------------------|------------|---------|
| Meta Llama 3.1/3.2 | None, BF16 | All | High | ✅ Planned |
| TinyLlama | None, BF16 | All | Low | ✅ Planned |
| Distilled Models | Various | CPU, CUDA | Medium | ✅ Planned |
| W8A8 Quantized | Compressed-Tensors | CUDA | Low | ✅ Planned |
| GPTQ | 4-bit | CUDA | Low | 🔄 Future |
| AWQ | 4-bit | CUDA | Low | 🔄 Future |
| GGUF | Various | CPU, Metal | Variable | 🔄 Future |

## Testing Strategy

### Model Validation Tests
1. **Correctness**: Compare outputs with reference implementations
2. **Performance**: Benchmark against optimized engines
3. **Memory**: Validate memory usage predictions
4. **Compatibility**: Test across different model variants

### Integration Tests
1. **End-to-end**: Full pipeline from model loading to generation
2. **Backend Switching**: Seamless backend transitions
3. **Quantization**: Accuracy preservation with quantized models
4. **Error Handling**: Graceful failure modes

This specification provides a comprehensive framework for building a generic Llama inference engine that can handle the diverse ecosystem of Llama-like models with flexible data type support and backend compatibility.