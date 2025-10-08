# Comprehensive Analysis of mistral.rs LLM Inference Engine

## Executive Summary

mistral.rs is a production-grade Rust inference engine for open-source LLMs built on top of the Candle ML framework. It implements a sophisticated architecture with support for multiple quantization formats (GGUF, safetensors), advanced attention mechanisms (FlashAttention, PagedAttention), and efficient scheduling. The codebase is well-architected with clear separation of concerns between model loading, inference execution, and text generation.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Complete Inference Flow](#complete-inference-flow)
3. [Model Loading Formats](#model-loading-formats)
4. [Transformer Implementation](#transformer-implementation)
5. [Attention Mechanisms](#attention-mechanisms)
6. [Text Generation Pipeline](#text-generation-pipeline)
7. [Key Abstractions and Patterns](#key-abstractions-and-patterns)
8. [Important Files Reference](#important-files-reference)

---

## High-Level Architecture

The inference engine is structured in several distinct layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Engine Layer                         │
│  (Scheduling, Sequence Management, Request Handling)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Pipeline Layer                         │
│    (Model-specific logic, Chat templating, I/O)        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                Transformer Models                       │
│      (LLaMA, Mistral, Phi, etc. implementations)       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Quantization Layer                         │
│     (GGUF, safetensors, GPTQ, HQQ, etc.)               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Candle Framework                           │
│        (Tensor operations, Device abstraction)          │
└─────────────────────────────────────────────────────────┘
```

### Core Components

1. **Engine** (`mistralrs-core/src/engine/mod.rs`)
   - Main event loop that processes requests
   - Manages the scheduler and sequence lifecycle
   - Coordinates between prompt and completion phases

2. **Pipeline** (`mistralrs-core/src/pipeline/mod.rs`)
   - Abstraction layer for different model types
   - Handles tokenization, chat template application
   - Processes inputs and generates responses

3. **Models** (`mistralrs-core/src/models/`)
   - Concrete transformer implementations
   - Each model (LLaMA, Mistral, etc.) has its own file
   - Implements forward pass and layer operations

4. **Quantization** (`mistralrs-quant/`)
   - Loading and dequantizing quantized weights
   - Multiple format support (GGUF, GPTQ, HQQ, etc.)

---

## Complete Inference Flow

### From User Prompt to Generated Tokens

Here's the complete flow of a request through the system:

#### 1. Request Creation and Submission

**File**: `mistralrs-core/src/request.rs`

```
User Input (String)
    ↓
NormalRequest {
    messages: RequestMessage,
    sampling_params: SamplingParams,
    response: Sender<Response>,
    ...
}
    ↓
Sent to Engine via mpsc channel
```

The request structure contains:
- **Messages**: Either chat messages, completion text, or tokens
- **Sampling params**: Temperature, top-k, top-p, penalties, etc.
- **Response channel**: For streaming or final responses
- **Constraints**: Optional grammar/regex constraints via llguidance

#### 2. Engine Processing Loop

**File**: `mistralrs-core/src/engine/mod.rs` (line 200-525)

The engine runs an infinite loop:

```rust
loop {
    // 1. Poll for new requests
    while let Ok(request) = self.rx.try_recv() {
        self.handle_request(request).await;
    }

    // 2. Schedule sequences (decide what to run)
    let scheduled = scheduler.schedule();

    // 3. Run scheduled sequences
    match scheduled {
        // Completion phase (generating tokens one at a time)
        SchedulerOutput::DefaultScheduler { prompt, completion } => {
            if !completion.is_empty() {
                pipeline.step(&mut completion, false, ...).await;
            }

            // Prompt phase (processing input tokens in batch)
            if !prompt.is_empty() {
                pipeline.step(&mut prompt, true, ...).await;
            }
        }
        // PagedAttention uses different scheduling
        SchedulerOutput::PagedAttention { ... } => { ... }
    }

    // 4. Free completed sequences
    scheduler.free_finished_sequence_groups();
}
```

**Key insight**: The engine alternates between two phases:
- **Prompt phase** (`is_prompt=true`): Process all input tokens at once (batch processing)
- **Completion phase** (`is_prompt=false`): Generate one token at a time (autoregressive)

#### 3. Sequence Creation

**File**: `mistralrs-core/src/sequence.rs` (line 377-600)

When a request is added, it creates a `Sequence`:

```rust
pub struct Sequence {
    // Immutable metadata
    id: usize,
    prompt: String,
    tokens: Vec<u32>,           // All tokens (prompt + generated)
    prompt_len: usize,
    max_len: Option<usize>,
    sampler: Arc<Sampler>,

    // KV cache for this sequence
    cache: Vec<Option<(Tensor, Tensor)>>,  // One per layer

    // State management
    state: RwLock<SequenceState>,  // Waiting, Running, Done

    // Generation tracking
    logprobs: Vec<Logprobs>,
    completion_bytes: Vec<u8>,

    // Multimodal data (images, audio)
    multimodal: MultimodalData,

    // For PagedAttention
    custom_metadata: SequenceCustomMetadata,
}
```

**Sequence States**:
- `Waiting`: Just created, not yet scheduled
- `RunningPrompt`: Processing input tokens
- `RunningCompletion`: Generating output tokens
- `Done(StopReason)`: Finished (EOS, length limit, stop string, etc.)

#### 4. Input Processing

**File**: `mistralrs-core/src/pipeline/inputs_processor.rs`

The inputs processor prepares data for the model:

```rust
// Tokenize text if needed
let tokens = tokenizer.encode(text)?;

// Create attention mask (causal for autoregressive)
let attention_mask = create_causal_mask(seq_len);

// Create position IDs
let position_ids = (0..seq_len).collect();

// For prompt: process all tokens at once
let input_ids = Tensor::from_slice(&tokens, (batch_size, seq_len), device)?;

// For completion: just the last generated token
let input_ids = Tensor::from_slice(&[last_token], (batch_size, 1), device)?;
```

**Context length handling**:
- `context_lens` tracks which part of logits to extract
- For prompt: extract logits for all positions
- For completion: extract logits only for the last position

#### 5. Model Forward Pass

**File**: `mistralrs-core/src/models/llama.rs` (example)

The transformer forward pass goes through these steps:

```rust
pub fn forward(
    &self,
    input_ids: &Tensor,
    seqlen_offsets: &[usize],
    context_lens: Vec<(usize, usize)>,
    metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    flash_params: &FlashParams,
) -> Result<Tensor> {
    // 1. Embedding lookup
    let mut x = self.wte.forward(input_ids)?;  // (batch, seq_len, hidden_size)

    // 2. Create causal mask
    let mask = CausalMasker.make_causal_mask_matrix(...)?;

    // 3. Pass through transformer blocks
    for (block_idx, block) in self.blocks.iter().enumerate() {
        x = block.forward(
            &x,
            &mask,
            seqlen_offsets,
            &mut cache[block_idx],
            metadata,
            flash_params,
        )?;
    }

    // 4. Final layer norm
    let x = self.ln_f.forward(&x)?;

    // 5. Project to vocabulary
    let logits = self.lm_head.forward(&x)?;  // (batch, seq_len, vocab_size)

    // 6. Extract relevant logits
    extract_logits(&logits, context_lens)
}
```

#### 6. Transformer Block (Layer)

**File**: `mistralrs-core/src/models/llama.rs` (line 248-279)

Each transformer block contains:

```rust
struct Block {
    rms_1: RmsNorm,           // Pre-attention normalization
    attn: CausalSelfAttention,
    rms_2: RmsNorm,           // Pre-MLP normalization
    mlp: Box<dyn MlpLayer>,
}

fn forward(&self, x: &Tensor, ...) -> Result<Tensor> {
    // 1. Self-attention with residual
    let residual = x;
    let x = self.rms_1.forward(x)?;
    let x = self.attn.forward(&x, ...)?;
    let x = (x + residual)?;

    // 2. MLP with residual
    let residual = &x;
    let x = self.rms_2.forward(&x)?;
    let x = self.mlp.forward(&x)?;
    let x = (x + residual)?;

    Ok(x)
}
```

**Key pattern**: Pre-normalization with residual connections (used in most modern LLMs)

#### 7. Attention Mechanism

**File**: `mistralrs-core/src/models/llama.rs` (line 66-172)

The attention layer performs:

```rust
fn forward(
    &self,
    x: &Tensor,
    attention_mask: &Option<Tensor>,
    seqlen_offsets: &[usize],
    kv_cache: &mut KvCache,
    metadata: Option<...>,
    flash_params: &FlashParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _) = x.dims3()?;

    // 1. Project to Q, K, V
    let q = self.q_proj.forward(x)?;  // (batch, seq_len, hidden_size)
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;

    // 2. Reshape for multi-head attention
    let q = q.reshape((b_sz, seq_len, num_heads, head_dim))?
             .transpose(1, 2)?;  // (batch, num_heads, seq_len, head_dim)
    let k = k.reshape((b_sz, seq_len, num_kv_heads, head_dim))?
             .transpose(1, 2)?;
    let v = v.reshape((b_sz, seq_len, num_kv_heads, head_dim))?
             .transpose(1, 2)?;

    // 3. Apply rotary embeddings (RoPE)
    let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

    // 4. Update KV cache and get all past K,V
    let (k, v) = kv_cache.append(&k, &v)?;

    // 5. Compute attention scores
    let y = Sdpa.run_attention(&q, &k, &v, attention_mask, ...)?;

    // 6. Output projection
    let y = self.o_proj.forward(&y)?;

    Ok(y)
}
```

#### 8. KV Cache Management

**File**: `mistralrs-core/src/kv_cache/single_cache.rs`

The KV cache stores past key-value pairs to avoid recomputation:

```rust
pub struct SingleCache {
    all_data: Option<Tensor>,      // (batch, num_heads, seq_len, head_dim)
    dim: usize,                     // Which dimension is sequence length
    current_seq_len: usize,         // How many tokens cached
    capacity_seq_len: usize,        // Current capacity
    max_seq_len: usize,             // Maximum allowed
}

pub fn append(&mut self, src: &Tensor) -> Result<()> {
    let seq_len = src.dim(self.dim)?;

    // Initialize cache on first use
    if self.all_data.is_none() {
        let mut shape = src.dims().to_vec();
        shape[self.dim] = self.capacity_seq_len;
        self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
    }

    // Grow cache if needed
    if self.current_seq_len + seq_len > self.capacity_seq_len {
        self.expand_cache()?;
    }

    // Append new keys/values
    let ad = self.all_data.as_mut().unwrap();
    ad.slice_set(src, self.dim, self.current_seq_len)?;
    self.current_seq_len += seq_len;

    Ok(())
}
```

**Cache growth strategy**:
- Starts with small capacity
- Grows in blocks of `CACHE_GROW_SIZE` (typically 256 tokens)
- Bounded by `max_seq_len`

#### 9. Sampling (Token Selection)

**File**: `mistralrs-core/src/sampler.rs` (line 838-906)

After getting logits, sample the next token:

```rust
pub fn sample(
    &self,
    logits: Tensor,           // (vocab_size,)
    context: &[u32],          // All previous tokens
    return_logprobs: bool,
    rng: Arc<Mutex<Isaac64Rng>>,
    sample_speculative: bool,
    multiple_sequences: bool,
) -> Result<Logprobs> {
    // 1. Apply penalties (frequency, presence, repetition, DRY)
    let mut logits = self.apply_penalties(logits.to_vec1()?, context)?;

    // 2. Apply custom logits processors
    for processor in &self.logits_processors {
        logits = processor.apply(&logits, context)?;
    }

    // 3. Sample based on temperature
    let next_token = match self.temperature {
        None => {
            // Greedy: argmax
            self.sample_argmax(logits, return_logprobs)?
        }
        Some(temperature) => {
            // Stochastic sampling
            let logits = (&logits / temperature)?;
            let probs = softmax_last_dim(&logits)?;
            let mut probs: Vec<f32> = probs.to_vec1()?;

            // Apply top-k, top-p, min-p
            self.sample_top_kp_min_p(
                &mut probs,
                &logits,
                self.top_k,
                self.top_p,
                self.min_p,
                return_logprobs,
                rng,
            )?
        }
    };

    Ok(next_token)
}
```

**Sampling strategies**:
- **Greedy** (temp=0): Always pick highest probability token
- **Temperature**: Scale logits before softmax (higher = more random)
- **Top-k**: Only sample from k highest probability tokens
- **Top-p (nucleus)**: Sample from smallest set of tokens with cumulative prob >= p
- **Min-p**: Filter out tokens with prob < (max_prob * min_p)

#### 10. Stop Condition Checking

**File**: `mistralrs-core/src/sequence.rs` (line 881-921)

After sampling, check if generation should stop:

```rust
pub fn is_done(
    &self,
    tok: u32,
    eos_tok: Option<&[u32]>,
    max_model_len: usize,
) -> Option<StopReason> {
    // EOS token
    if eos_tok.contains(&tok) {
        return Some(StopReason::Eos);
    }

    // User-specified stop tokens
    if self.stop_tokens.contains(&tok) {
        return Some(StopReason::StopTok(tok));
    }

    // Max length reached
    if self.tokens.len() - self.prompt_len + 1 >= self.max_len {
        return Some(StopReason::Length(self.max_len));
    }

    // Model's maximum context length
    if self.tokens.len() - self.prompt_len >= max_model_len {
        return Some(StopReason::ModelLength(max_model_len));
    }

    // Stop string found in decoded text
    for (idx, s) in self.stop_strings.iter().enumerate() {
        if let Some(pos) = find_substring(&self.completion_bytes, s.as_bytes()) {
            return Some(StopReason::StopString {
                stop_string_idx: idx,
                completion_bytes_pos: pos,
            });
        }
    }

    None
}
```

#### 11. Response Generation

**File**: `mistralrs-core/src/pipeline/response.rs`

Send responses back to the user:

```rust
// For streaming
if is_streaming {
    // Decode new token
    let delta = sequence.get_delta()?;

    // Send chunk
    sequence.responder().send(Response::Chunk(ChatCompletionChunkResponse {
        id: sequence.id.to_string(),
        choices: vec![ChunkChoice {
            delta: ResponseMessage { content: delta },
            finish_reason: None,
            index: 0,
        }],
        ...
    })).await?;
}

// For final response
if sequence.is_done() {
    // Get full completion text
    let text = String::from_utf8_lossy(&sequence.completion_bytes);

    // Send final response
    sequence.responder().send(Response::Done(ChatCompletionResponse {
        id: sequence.id.to_string(),
        choices: vec![Choice {
            message: ResponseMessage {
                content: text.to_string(),
                role: "assistant".to_string(),
            },
            finish_reason: stop_reason.to_string(),
            index: 0,
        }],
        usage: Usage {
            prompt_tokens: sequence.prompt_len,
            completion_tokens: sequence.tokens.len() - sequence.prompt_len,
            total_tokens: sequence.tokens.len(),
        },
        ...
    })).await?;
}
```

---

## Model Loading Formats

mistral.rs supports multiple model formats with different quantization schemes.

### 1. GGUF Format

**File**: `mistralrs-quant/src/gguf/mod.rs`

GGUF (GPT-Generated Unified Format) is a binary format for quantized models.

#### Structure

```rust
pub struct GgufMatMul {
    w: QMatMul,              // Quantized weight matrix
    b: Option<Tensor>,       // Optional bias (usually unquantized)
}

pub enum QMatMul {
    QTensor(Arc<QTensor>),   // Quantized tensor
    Tensor(Tensor),           // Full precision tensor
    TensorF16(Tensor),        // FP16 tensor
}
```

#### Quantization Types

GGUF supports multiple quantization formats:

```rust
pub enum GgmlDType {
    F32,    // 32-bit float (no quantization)
    F16,    // 16-bit float
    Q4_0,   // 4-bit quantization (block-wise, no zero point)
    Q4_1,   // 4-bit quantization (block-wise, with zero point)
    Q5_0,   // 5-bit quantization
    Q5_1,   // 5-bit quantization
    Q8_0,   // 8-bit quantization
    Q2K,    // 2-bit K-quant (improved block-wise)
    Q3K,    // 3-bit K-quant
    Q4K,    // 4-bit K-quant
    Q5K,    // 5-bit K-quant
    Q6K,    // 6-bit K-quant
    Q8K,    // 8-bit K-quant
    BF16,   // bfloat16
}
```

**K-quant formats**: More sophisticated quantization with:
- Separate scales for different parts of the weight block
- Better preservation of outlier weights
- Mixed precision within blocks

#### Loading Process

```rust
impl QuantMethod for GgufMatMul {
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        // Quantized matmul (specialized kernels)
        let x = self.w.forward(a)?;

        // Add bias if present
        if let Some(ref b) = self.b {
            x.broadcast_add(b)
        } else {
            Ok(x)
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        // Convert quantized weights back to FP32 (for ISQ or debugging)
        self.w.dequantize_f16()?.to_dtype(DType::F32)
    }
}
```

#### Serialization Format

GGUF files store tensors with this layout:

```
┌─────────────────────────────────────┐
│ UQFF Version (u32, little endian)   │
├─────────────────────────────────────┤
│ ISQ Type (u8) - 0 for GGUF          │
├─────────────────────────────────────┤
│ Tensor data length (u32)            │
├─────────────────────────────────────┤
│ Has bias (u8 boolean)               │
├─────────────────────────────────────┤
│ Quantized dtype (u32)               │
├─────────────────────────────────────┤
│ Num shape dims (u32)                │
├─────────────────────────────────────┤
│ Shape dims array (u32 each)         │
├─────────────────────────────────────┤
│ Quantized weight data (u8 array)    │
├─────────────────────────────────────┤
│ [OPTIONAL] Bias tensor              │
└─────────────────────────────────────┘
```

### 2. Safetensors Format

**File**: `mistralrs-quant/src/safetensors.rs`

Safetensors is a simple, safe format for storing tensors.

#### Structure

```rust
pub struct MmapedSafetensors {
    safetensors: Vec<Yoke<SafeTensors_<'static>, Mmap>>,
    routing: Option<HashMap<String, usize>>,  // Which file has which tensor
}
```

#### Memory Mapping

Safetensors uses memory-mapped files for efficient loading:

```rust
pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
    // Open file
    let file = std::fs::File::open(p)?;

    // Memory map it (doesn't load into RAM immediately)
    let mmap = memmap2::MmapOptions::new().map(&file)?;

    // Parse header (metadata about tensors)
    let safetensors = SafeTensors::deserialize(&mmap)?;

    Ok(Self {
        safetensors: vec![safetensors],
        routing: None,
    })
}
```

**Key advantage**: Tensors are loaded on-demand via memory mapping, reducing startup time and memory usage.

#### Loading Tensors

```rust
pub fn load(&self, name: &str, dev: &Device, dtype: Option<DType>) -> Result<Tensor> {
    // Get tensor view (doesn't copy data yet)
    let view = self.get(name)?;

    // Convert to Tensor (copies data to device if needed)
    convert(&view, dev, dtype)
}

fn convert(view: &TensorView, device: &Device, cast_dtype: Option<DType>) -> Result<Tensor> {
    match (view.dtype(), cast_dtype) {
        (Dtype::F16, Some(DType::F32)) => {
            // Convert FP16 to FP32 on load
            let conv = |x: half::f16| Ok(x.to_f32());
            convert_with_cast(view, device, conv)
        }
        (Dtype::BF16, None) => {
            // Load BF16 directly
            convert::<half::bf16>(view, device)
        }
        // ... other type combinations
    }
}
```

#### Multi-file Support

For large models split across multiple files:

```rust
pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
    let mut routing = HashMap::new();
    let mut safetensors = vec![];

    for (index, path) in paths.iter().enumerate() {
        let file = std::fs::File::open(path)?;
        let mmap = memmap2::MmapOptions::new().map(&file)?;
        let st = SafeTensors::deserialize(&mmap)?;

        // Record which file has which tensors
        for tensor_name in st.names() {
            routing.insert(tensor_name.to_string(), index);
        }

        safetensors.push(st);
    }

    Ok(Self {
        safetensors,
        routing: Some(routing),
    })
}
```

### 3. VarBuilder Pattern

**File**: `mistralrs-core/src/utils/varbuilder_utils.rs`

The VarBuilder abstracts away the underlying storage format:

```rust
pub trait TensorLoaderBackend {
    fn get_names(&self) -> Vec<String>;
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor>;
}

// Unified interface for loading tensors
struct SafetensorBackend(MmapedSafetensors);
struct PickleBackend(PthTensors);  // For PyTorch .pth files

impl TensorLoaderBackend for SafetensorBackend {
    fn load_name(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Tensor> {
        self.0.load(name, device, dtype)
    }
}
```

#### Hierarchical Naming

VarBuilder uses a hierarchical namespace:

```rust
// vb.pp("model") -> prefix with "model."
// vb.pp("layers").pp(0) -> prefix with "layers.0."

let vb_model = vb.pp("model");
let vb_layer0 = vb_model.pp("layers").pp(0);

// Load model.layers.0.mlp.gate_proj.weight
let gate_proj = vb_layer0.pp("mlp").get("gate_proj.weight")?;
```

This allows loading weights with automatic name mapping.

#### Device Mapping

For multi-GPU inference:

```rust
pub enum DeviceForLoadTensor {
    Base,           // Load to base device
    Idx(usize),     // Load to specific layer device
}

// Load tensors to different devices based on layer
let get_device = Arc::new(|name: String| {
    if name.starts_with("model.layers.0") {
        DeviceForLoadTensor::Idx(0)  // Layer 0 on device 0
    } else if name.starts_with("model.layers.1") {
        DeviceForLoadTensor::Idx(1)  // Layer 1 on device 1
    } else {
        DeviceForLoadTensor::Base
    }
});
```

---

## Transformer Implementation

### Core Architecture

**File**: `mistralrs-core/src/models/llama.rs`

The transformer follows the standard decoder-only architecture with some improvements.

### Configuration

```rust
pub struct Config {
    pub hidden_size: usize,              // Dimension of hidden states (e.g., 4096)
    pub intermediate_size: usize,         // MLP hidden dimension (e.g., 11008)
    pub vocab_size: usize,                // Size of vocabulary (e.g., 32000)
    pub num_hidden_layers: usize,         // Number of transformer blocks (e.g., 32)
    pub num_attention_heads: usize,       // Number of attention heads (e.g., 32)
    pub num_key_value_heads: usize,       // For GQA (e.g., 8)
    pub rms_norm_eps: f64,                // Epsilon for RMSNorm (e.g., 1e-6)
    pub rope_theta: f32,                  // Base for rotary embeddings (e.g., 10000.0)
    pub max_position_embeddings: usize,   // Max sequence length (e.g., 4096)
    pub rope_scaling: Option<RopeConfig>, // For extended context
    pub quantization_config: Option<QuantizedConfig>,
    pub tie_word_embeddings: bool,        // Share input/output embeddings
}
```

### Model Structure

```rust
pub struct Llama {
    wte: Embedding,                  // Token embeddings
    blocks: Vec<Block>,              // Transformer layers
    ln_f: RmsNorm,                   // Final layer norm
    lm_head: Arc<dyn QuantMethod>,   // Output projection to vocab
    kv_cache: EitherCache,           // KV cache for all layers
    device: Device,
    mapper: Box<dyn DeviceMapper>,   // For multi-GPU
    cfg: ModelConfigMetadata,
}
```

### 1. Embedding Layer

```rust
pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
    // input_ids: (batch_size, seq_len)
    // output: (batch_size, seq_len, hidden_size)
    self.wte.forward(input_ids)
}
```

**Implementation**: Simple lookup table
- Input: Token IDs (integers)
- Output: Dense vectors (embeddings)
- Weights: `(vocab_size, hidden_size)` matrix

### 2. RMSNorm (Root Mean Square Normalization)

**File**: `mistralrs-core/src/layers.rs` (line 203-258)

```rust
pub struct RmsNorm {
    eps: f64,
    weight: Tensor,  // Learnable scale parameter
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Compute RMS: sqrt(mean(x^2) + eps)
        let rms = x.powf(2.)?.mean_keepdim(D::Minus1)?;
        let rms = (rms + self.eps)?.sqrt()?;

        // 2. Normalize: x / rms
        let x_normed = x.broadcast_div(&rms)?;

        // 3. Scale: x_normed * weight
        x_normed.broadcast_mul(&self.weight)
    }
}
```

**Why RMSNorm instead of LayerNorm?**
- Simpler: No learned bias, no mean subtraction
- Faster: Fewer operations
- Equivalent performance for LLMs

### 3. Rotary Positional Embeddings (RoPE)

RoPE encodes position information directly into Q and K:

```rust
pub struct Llama3RotaryEmbedding {
    sin: Tensor,  // Precomputed sin values
    cos: Tensor,  // Precomputed cos values
    dim: usize,
}

impl Llama3RotaryEmbedding {
    pub fn forward(
        &self,
        q: &Tensor,  // (batch, num_heads, seq_len, head_dim)
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // Apply rotary embedding to half of head_dim
        let half_dim = self.dim / 2;

        // For each position, rotate Q and K by position-dependent angle
        let q_rot = self.apply_rotary_emb(q, seqlen_offsets)?;
        let k_rot = self.apply_rotary_emb(k, seqlen_offsets)?;

        Ok((q_rot, k_rot))
    }

    fn apply_rotary_emb(&self, x: &Tensor, offsets: &[usize]) -> Result<Tensor> {
        // Split into two halves
        let (x1, x2) = x.chunk(2, D::Minus1)?;

        // Get sin/cos for current positions
        let (sin, cos) = self.get_sin_cos(offsets)?;

        // Rotation:
        // [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        let x1_rot = (x1 * &cos - x2 * &sin)?;
        let x2_rot = (x1 * &sin + x2 * &cos)?;

        Tensor::cat(&[x1_rot, x2_rot], D::Minus1)
    }
}
```

**Key properties of RoPE**:
- Encodes relative positions (not absolute)
- Naturally extends to longer sequences
- Applied to Q and K (not V)
- Uses complex number rotations

### 4. MLP (Feed-Forward Network)

```rust
pub struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,  // (hidden_size, intermediate_size)
    up_proj: Arc<dyn QuantMethod>,    // (hidden_size, intermediate_size)
    down_proj: Arc<dyn QuantMethod>,  // (intermediate_size, hidden_size)
    act: Activation,                   // Usually SiLU (Swish)
}

impl MlpLayer for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU architecture:
        // output = down_proj(silu(gate_proj(x)) * up_proj(x))

        let gate = self.gate_proj.forward(x)?;
        let gate = self.act.forward(&gate)?;  // SiLU activation

        let up = self.up_proj.forward(x)?;

        let mlp_out = (gate * up)?;

        self.down_proj.forward(&mlp_out)
    }
}
```

**SwiGLU** (Swish-Gated Linear Unit):
- Gating mechanism: Controls information flow
- Two parallel projections (gate and up)
- Element-wise multiplication
- Better than standard FFN

### 5. Transformer Block

```rust
impl Block {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<...>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        // Pre-norm architecture (norm before sublayer)

        // 1. Self-attention
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = self.attn.forward(&x, ...)?;
        let x = (x + residual)?;  // Residual connection

        // 2. MLP
        let residual = &x;
        let x = self.rms_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;  // Residual connection

        Ok(x)
    }
}
```

**Pre-norm vs Post-norm**:
- **Pre-norm** (used here): Norm before sublayer, more stable training
- **Post-norm**: Norm after sublayer, used in original Transformer

---

## Attention Mechanisms

mistral.rs supports multiple attention implementations with different trade-offs.

### 1. Scaled Dot-Product Attention (Base Implementation)

**File**: `mistralrs-core/src/attention/mod.rs`

```rust
pub struct Sdpa;

impl Sdpa {
    pub fn run_attention(
        &self,
        q: &Tensor,    // (batch, num_heads, seq_len_q, head_dim)
        k: &Tensor,    // (batch, num_kv_heads, seq_len_k, head_dim)
        v: &Tensor,    // (batch, num_kv_heads, seq_len_k, head_dim)
        attention_mask: Option<&Tensor>,
        flash_params: Option<&FlashParams>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let scale = sdpa_params.softmax_scale;  // 1/sqrt(head_dim)
        let n_kv_groups = sdpa_params.n_kv_groups;

        // 1. Repeat KV heads for GQA (Grouped Query Attention)
        let k = repeat_kv(k, n_kv_groups)?;  // Match Q heads
        let v = repeat_kv(v, n_kv_groups)?;

        // 2. Compute attention scores: Q @ K^T / sqrt(d_k)
        let att = q.matmul(&k.t()?)?;
        let att = (att * scale)?;

        // 3. Apply causal mask (prevent attending to future)
        let att = if let Some(mask) = attention_mask {
            att.broadcast_add(mask)?  // Add -inf for future positions
        } else {
            att
        };

        // 4. Softmax
        let att = candle_nn::ops::softmax_last_dim(&att)?;

        // 5. Weighted sum of values: softmax(scores) @ V
        let y = att.matmul(&v)?;

        Ok(y)
    }
}
```

**Grouped Query Attention (GQA)**:
- Q heads: 32 (for example)
- KV heads: 8 (fewer than Q heads)
- Reduces KV cache size by 4x
- Each KV head is shared across multiple Q heads

### 2. FlashAttention

FlashAttention is a faster, more memory-efficient attention algorithm.

**Key ideas**:
1. **Tiling**: Process attention in blocks to fit in SRAM
2. **Recomputation**: Recompute softmax during backward pass instead of storing
3. **Fused kernels**: Combine operations to reduce memory traffic

**Implementation** (conceptual, actual kernels are in CUDA/HIP):

```rust
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: &FlashParams,
) -> Result<Tensor> {
    // Use cumulative sequence lengths for batched sequences
    let cumulative_seqlens_q = &flash_params.cumulative_seqlens_q;
    let cumulative_seqlens_k = &flash_params.cumulative_seqlens_k;

    // Call optimized kernel (implementation depends on backend)
    flash_attn_varlen(
        q,
        k,
        v,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
        flash_params.max_q,
        flash_params.max_k,
        softmax_scale,
        causal,
    )
}
```

**Benefits**:
- 2-4x faster than standard attention
- Uses O(N) memory instead of O(N^2)
- Exact (not an approximation)

### 3. PagedAttention

**File**: `mistralrs-core/src/paged_attention/`

PagedAttention manages KV cache like virtual memory, enabling:
- Efficient memory usage across multiple requests
- Dynamic allocation and deallocation
- Sharing KV cache between sequences (for parallel sampling)

#### Block Engine

```rust
pub struct BlockEngine {
    block_size: usize,              // Tokens per block (e.g., 16)
    num_blocks: usize,              // Total blocks available
    block_tables: HashMap<usize, Vec<Arc<Mutex<PhysicalTokenBlock>>>>,
    free_blocks: Vec<Arc<Mutex<PhysicalTokenBlock>>>,
    allocator: BlockAllocator,
}
```

#### Blocks

```rust
pub struct LogicalTokenBlock {
    block_id: usize,
    tokens: Vec<usize>,
    block_size: usize,
}

pub struct PhysicalTokenBlock {
    block_id: usize,
    ref_count: usize,  // For sharing blocks
    device: Device,
}
```

#### Allocation

```rust
impl BlockEngine {
    pub fn allocate(&mut self, sequence: &mut Sequence) -> Result<()> {
        let num_tokens = sequence.get_toks().len();
        let num_blocks_needed = num_tokens.div_ceil(self.block_size);

        // Allocate physical blocks
        let mut physical_blocks = Vec::new();
        for _ in 0..num_blocks_needed {
            let block = self.allocator.allocate()?;
            physical_blocks.push(block);
        }

        // Create logical blocks
        let mut logical_blocks = Vec::new();
        for _ in 0..num_blocks_needed {
            logical_blocks.push(LogicalTokenBlock::new(self.block_size));
        }

        // Populate with tokens
        for (i, &token) in sequence.get_toks().iter().enumerate() {
            let block_idx = i / self.block_size;
            logical_blocks[block_idx].append_token_id(token);
        }

        // Store mapping
        self.block_tables.insert(*sequence.id(), physical_blocks);

        Ok(())
    }

    pub fn free(&mut self, sequence_id: usize) -> Result<()> {
        if let Some(blocks) = self.block_tables.remove(&sequence_id) {
            for block in blocks {
                let mut block = block.lock().unwrap();
                block.ref_count -= 1;
                if block.ref_count == 0 {
                    self.free_blocks.push(Arc::new(Mutex::new(block)));
                }
            }
        }
        Ok(())
    }
}
```

#### Attention with Paged KV Cache

```rust
pub fn paged_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_cache: &Tensor,      // (num_blocks, block_size, num_heads, head_dim)
    value_cache: &Tensor,    // (num_blocks, block_size, num_heads, head_dim)
    block_tables: &Tensor,   // (batch_size, max_num_blocks_per_seq)
    context_lens: &Tensor,   // (batch_size,)
    input_metadata: &PagedAttentionInputMetadata,
) -> Result<Tensor> {
    // For each query position:
    // 1. Look up which blocks contain the relevant KV pairs
    // 2. Gather K,V from those blocks
    // 3. Compute attention with gathered K,V

    // This is done efficiently with specialized CUDA kernels
    paged_attention_kernel(
        q,
        k,
        v,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        ...
    )
}
```

**Benefits of PagedAttention**:
- **Memory efficiency**: Only allocate what's needed
- **Dynamic batching**: Add/remove sequences without reallocation
- **Sharing**: Multiple sequences can share KV cache (for beam search, parallel sampling)
- **Defragmentation**: Blocks can be swapped to disk if needed

---

## Text Generation Pipeline

### 1. Tokenization

**File**: `mistralrs-core/src/utils/tokenizer.rs`

Tokenization converts text to token IDs:

```rust
// Using HuggingFace tokenizers crate
let tokenizer = Tokenizer::from_file("tokenizer.json")?;

// Encode text
let encoding = tokenizer.encode(text, add_special_tokens)?;
let token_ids: Vec<u32> = encoding.get_ids().to_vec();

// Decode tokens
let text = tokenizer.decode(&token_ids, skip_special_tokens)?;
```

**Special tokens**:
- `<BOS>`: Beginning of sequence
- `<EOS>`: End of sequence
- `<PAD>`: Padding token
- `<UNK>`: Unknown token

### 2. Chat Template Application

**File**: `mistralrs-core/src/pipeline/chat_template.rs`

Chat templates format conversations for the model:

```rust
// Example: ChatML format
let template = r#"
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
<|im_start|>assistant
"#;

// Apply to messages
let messages = vec![
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"},
];

// Result:
// <|im_start|>user
// Hello!<|im_end|>
// <|im_start|>assistant
// Hi! How can I help?<|im_end|>
// <|im_start|>user
// What's the weather?<|im_end|>
// <|im_start|>assistant
```

Different models use different formats:
- **ChatML**: `<|im_start|>role\ncontent<|im_end|>`
- **Llama 2**: `[INST] user message [/INST]assistant message`
- **Mistral**: `[INST] user message [/INST]`

### 3. Sampling Parameters

**File**: `mistralrs-core/src/sampler.rs`

```rust
pub struct SamplingParams {
    pub temperature: Option<f64>,           // Randomness (0.0 = greedy)
    pub top_k: Option<usize>,               // Consider top K tokens
    pub top_p: Option<f64>,                 // Nucleus sampling
    pub min_p: Option<f64>,                 // Minimum probability
    pub frequency_penalty: Option<f32>,     // Penalize frequent tokens
    pub presence_penalty: Option<f32>,      // Penalize any repeated token
    pub repetition_penalty: Option<f32>,    // Scale repeated token logits
    pub stop_toks: Option<StopTokens>,      // Stop tokens/strings
    pub max_len: Option<usize>,             // Max completion length
    pub logits_bias: Option<HashMap<u32, f32>>,  // Bias specific tokens
    pub n_choices: usize,                   // Number of completions
    pub dry_params: Option<DrySamplingParams>,   // DRY (Don't Repeat Yourself) sampling
}
```

#### Temperature Scaling

```rust
fn apply_temperature(logits: &Tensor, temperature: f64) -> Result<Tensor> {
    // Lower temp -> sharper distribution (more deterministic)
    // Higher temp -> flatter distribution (more random)
    (logits / temperature)?
}
```

#### Top-K Sampling

```rust
fn apply_top_k(probs: &mut Vec<f32>, k: usize) {
    // Keep only the k highest probability tokens
    let mut indexed: Vec<(usize, f32)> = probs.iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Zero out all but top k
    for i in k..probs.len() {
        probs[indexed[i].0] = 0.0;
    }
}
```

#### Top-P (Nucleus) Sampling

```rust
fn apply_top_p(probs: &mut Vec<f32>, p: f64) {
    // Keep smallest set of tokens with cumulative probability >= p
    let mut indexed: Vec<(usize, f32)> = probs.iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, (_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero out tokens outside nucleus
    for i in cutoff_idx..indexed.len() {
        probs[indexed[i].0] = 0.0;
    }
}
```

#### Repetition Penalties

```rust
fn apply_repetition_penalty(
    logits: &mut [f32],
    context: &[u32],
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) {
    // Count token frequencies
    let mut counts = vec![0.0f32; logits.len()];
    for &token in context {
        counts[token as usize] += 1.0;
    }

    for (token_id, logit) in logits.iter_mut().enumerate() {
        let count = counts[token_id];

        // Frequency penalty: proportional to count
        *logit -= count * frequency_penalty;

        // Presence penalty: binary (has appeared or not)
        *logit -= (count > 0.0) as i32 as f32 * presence_penalty;

        // Repetition penalty: scale down if appeared
        if repetition_penalty != 1.0 && count > 0.0 {
            if *logit > 0.0 {
                *logit /= repetition_penalty;
            } else {
                *logit *= repetition_penalty;
            }
        }
    }
}
```

### 4. Token Decoding

**Streaming decoding**:

```rust
pub fn get_delta(&mut self) -> Result<Option<String>> {
    let new_decoded = String::from_utf8_lossy(
        &self.completion_bytes[self.stream_idx..]
    );

    // Check if sequence ends with invalid UTF-8
    if new_decoded.ends_with('�') {
        // Wait for more tokens (might be multi-byte sequence)
        return Ok(None);
    }

    // First token often starts with space - trim it
    let is_first = self.stream_idx == 0;
    let delta = if is_first {
        new_decoded.trim_start().to_string()
    } else {
        new_decoded.to_string()
    };

    self.stream_idx = self.completion_bytes.len();
    Ok(Some(delta))
}
```

**Key challenge**: Handling multi-byte UTF-8 sequences
- Some tokens produce partial UTF-8 sequences
- Must wait for complete sequence before decoding
- The `�` character indicates incomplete UTF-8

### 5. Stopping Conditions

Stop generation when any of these conditions are met:

1. **EOS token**: Model outputs end-of-sequence token
2. **Stop tokens**: User-specified tokens (e.g., `\n\n` for paragraphs)
3. **Stop strings**: Decoded text contains stop string
4. **Max length**: Reached maximum number of tokens
5. **Model length**: Exceeded model's context window
6. **Cancellation**: User cancelled the request

```rust
pub enum StopReason {
    Eos,                    // Model generated EOS token
    StopTok(u32),          // Hit user stop token
    Length(usize),         // Max length reached
    ModelLength(usize),    // Model context limit
    StopString {           // Stop string found
        stop_string_idx: usize,
        completion_bytes_pos: usize,
    },
    Canceled,              // User cancelled
    GeneratedImage,        // For image generation models
    GeneratedSpeech,       // For speech generation models
    ToolCalls,             // For function calling
}
```

---

## Key Abstractions and Patterns

### 1. Pipeline Trait

**File**: `mistralrs-core/src/pipeline/mod.rs` (line 368-786)

The `Pipeline` trait is the main abstraction for different model types:

```rust
#[async_trait]
pub trait Pipeline:
    Send
    + Sync
    + PreProcessingMixin       // Tokenization, chat templates
    + IsqPipelineMixin         // In-Situ Quantization
    + CacheManagerMixin        // KV cache management
    + MetadataMixin            // Device, tokenizer, metadata
    + AnyMoePipelineMixin      // Mixture-of-Experts support
{
    /// Forward pass through the model
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error>;

    /// Execute one step (prompt or completion)
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
    ) -> Result<Duration, candle_core::Error>;

    /// Sample tokens from logits
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error>;

    /// Get model category (Text, Vision, Diffusion, Audio, Speech)
    fn category(&self) -> ModelCategory;
}
```

**Benefits of this abstraction**:
- Same engine works with all model types
- Easy to add new model architectures
- Consistent interface for features (streaming, stop strings, etc.)

### 2. QuantMethod Trait

**File**: `mistralrs-quant/src/lib.rs`

Abstraction for quantized and unquantized weights:

```rust
pub trait QuantMethod: Send + Sync + Debug {
    /// Create from configuration
    fn new(method: QuantMethodConfig) -> Result<Self> where Self: Sized;

    /// Dequantize weights (for debugging or ISQ)
    fn dequantize_w(&self) -> Result<Tensor>;

    /// Forward pass (may use quantized kernels)
    fn forward(&self, a: &Tensor) -> Result<Tensor>;

    /// Add LoRA delta to weights
    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>>;

    /// Get dtype for activations (Some for quantized, None for unquantized)
    fn quantized_act_type(&self) -> Option<DType>;

    /// Apply in-situ quantization
    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>;
}
```

**Implementations**:
- `GgufMatMul`: GGUF quantized weights
- `GptqMatMul`: GPTQ quantization
- `HqqMatMul`: Half-Quadratic Quantization
- `UnquantLinear`: Full precision weights (baseline)

### 3. Sequence State Machine

**File**: `mistralrs-core/src/sequence.rs`

```rust
pub enum SequenceState {
    Waiting,               // Created, not scheduled yet
    RunningPrompt,         // Processing input tokens
    RunningCompletion,     // Generating output tokens
    RunningPrefillPrompt,  // Using prefix cache
    Done(StopReason),      // Finished generation
    Error,                 // Error occurred
    Swapped,               // PagedAttention: swapped to disk
    FinishedAborted,       // PagedAttention: aborted
    FinishedIgnored,       // PagedAttention: ignored
}
```

**State transitions**:
```
Waiting → RunningPrompt → RunningCompletion → Done(reason)
                    ↓
         RunningPrefillPrompt
```

### 4. Cache Abstraction

**File**: `mistralrs-core/src/kv_cache/mod.rs`

```rust
pub enum EitherCache {
    Normal(NormalCache),       // Standard KV cache
    PagedAttention,            // PagedAttention doesn't use this
}

pub struct NormalCache(pub Vec<Option<KvCache>>, Arc<Mutex<usize>>);

pub type KvCache = (Tensor, Tensor);  // (key, value)
```

**Two levels of caching**:
1. **Sequence-level**: Each sequence has its own cache
2. **Model-level**: Temporary cache during forward pass

**Cache operations**:
- `clone_in_cache`: Copy from sequence to model (before forward)
- `clone_out_cache`: Copy from model to sequence (after forward)
- `set_none_cache`: Reset model cache (before prompt)

### 5. Device Mapping

**File**: `mistralrs-core/src/device_map.rs`

For multi-GPU inference:

```rust
pub trait DeviceMapper: Send + Sync + Debug {
    /// Get device for a specific layer
    fn device_for(&self, layer: usize, is_non_granular: bool) -> Option<&Device>;

    /// Move tensor between devices
    fn map(&self, x: Tensor, layer: usize) -> Result<Tensor>;

    /// Set device for a VarBuilder
    fn set_device(&self, layer: usize, vb: ShardedVarBuilder, loading_isq: bool)
        -> ShardedVarBuilder;

    /// Get communication group for distributed inference
    fn get_comm_for(&self, layer: usize) -> Result<Arc<mistralrs_quant::Comm>>;
}
```

**Implementations**:
- `NormalDeviceMapper`: Single device (GPU or CPU)
- `LayerDeviceMapper`: Different layers on different devices
- `SequentialMapper`: Layers distributed across devices sequentially

---

## Important Files Reference

### Core Engine
- **`mistralrs-core/src/engine/mod.rs`**: Main inference loop, request handling, scheduling
- **`mistralrs-core/src/sequence.rs`**: Sequence state management, token storage, completion tracking
- **`mistralrs-core/src/request.rs`**: Request types (normal, tokenization, etc.)
- **`mistralrs-core/src/response.rs`**: Response types (completion, chunk, image, etc.)

### Pipeline Layer
- **`mistralrs-core/src/pipeline/mod.rs`**: Pipeline trait, forward pass orchestration
- **`mistralrs-core/src/pipeline/normal.rs`**: Standard text model pipeline
- **`mistralrs-core/src/pipeline/vision.rs`**: Vision model pipeline (LLaVA, etc.)
- **`mistralrs-core/src/pipeline/inputs_processor.rs`**: Input preparation, attention metadata
- **`mistralrs-core/src/pipeline/chat_template.rs`**: Chat template application

### Model Implementations
- **`mistralrs-core/src/models/llama.rs`**: LLaMA/LLaMA 2/LLaMA 3 implementation
- **`mistralrs-core/src/models/mistral.rs`**: Mistral model
- **`mistralrs-core/src/models/mixtral.rs`**: Mixtral MoE model
- **`mistralrs-core/src/models/phi3.rs`**: Phi-3 model
- **`mistralrs-core/src/models/qwen2.rs`**: Qwen 2 model

### Layers and Components
- **`mistralrs-core/src/layers.rs`**: Common layers (RMSNorm, Embedding, Linear, RoPE, MLP)
- **`mistralrs-core/src/attention/mod.rs`**: Attention implementations
- **`mistralrs-core/src/layers_masker.rs`**: Causal mask generation

### Quantization
- **`mistralrs-quant/src/gguf/mod.rs`**: GGUF quantization implementation
- **`mistralrs-quant/src/safetensors.rs`**: Safetensors loading with memory mapping
- **`mistralrs-quant/src/gptq/mod.rs`**: GPTQ quantization
- **`mistralrs-quant/src/hqq/mod.rs`**: HQQ quantization

### Sampling and Generation
- **`mistralrs-core/src/sampler.rs`**: Token sampling (temperature, top-k, top-p, penalties)
- **`mistralrs-core/src/pipeline/sampling.rs`**: Sampling parameter handling

### Memory Management
- **`mistralrs-core/src/kv_cache/single_cache.rs`**: Single KV cache implementation
- **`mistralrs-core/src/kv_cache/full_cache.rs`**: Full KV cache (all layers)
- **`mistralrs-core/src/paged_attention/block_engine.rs`**: PagedAttention block management
- **`mistralrs-core/src/paged_attention/cache_engine.rs`**: PagedAttention cache engine
- **`mistralrs-core/src/prefix_cacher.rs`**: Prefix caching for repeated prompts

### Scheduling
- **`mistralrs-core/src/scheduler/mod.rs`**: Sequence scheduling (which to run)
- **`mistralrs-core/src/paged_attention/scheduler.rs`**: PagedAttention-specific scheduling

### Utilities
- **`mistralrs-core/src/utils/varbuilder_utils.rs`**: VarBuilder creation, tensor loading
- **`mistralrs-core/src/utils/tokenizer.rs`**: Tokenizer utilities
- **`mistralrs-core/src/device_map.rs`**: Multi-GPU device mapping
- **`mistralrs-core/src/model_loader.rs`**: Model loading orchestration

### Distributed Inference
- **`mistralrs-core/src/distributed.rs`**: Distributed inference coordination
- **`mistralrs-quant/src/distributed/mod.rs`**: Distributed tensor operations

---

## Summary

mistral.rs demonstrates several key patterns for building a production LLM inference engine:

1. **Clear Separation of Concerns**: Engine, pipeline, model, and quantization layers are distinct

2. **Flexible Architecture**: Pipeline trait allows supporting different model types with same engine

3. **Efficient Memory Management**:
   - KV cache with dynamic growth
   - Memory-mapped model loading
   - PagedAttention for multi-request scenarios

4. **Multiple Quantization Formats**: Unified interface (QuantMethod) for different quantization schemes

5. **Advanced Attention**: Support for FlashAttention and PagedAttention

6. **Sophisticated Sampling**: Wide range of sampling strategies and penalties

7. **Production Features**:
   - Streaming generation
   - Stop strings and tokens
   - Chat templates
   - Multi-GPU support
   - Prefix caching

8. **Type Safety**: Rust's type system prevents many classes of errors at compile time

This architecture serves as an excellent reference for building your own inference engine, with clear patterns for each component.
