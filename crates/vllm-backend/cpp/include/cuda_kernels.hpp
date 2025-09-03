#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Memory operations
cudaError_t cuda_memcpy_async(void* dst, const void* src, size_t count, 
                              cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cuda_memset_async(void* ptr, int value, size_t count, cudaStream_t stream);
cudaError_t cuda_memory_prefetch(void* ptr, size_t count, int device_id, cudaStream_t stream);

// Tensor operations
cudaError_t cuda_tensor_copy(float* dst, const float* src, size_t num_elements, cudaStream_t stream);
cudaError_t cuda_tensor_add(float* result, const float* a, const float* b, size_t num_elements, cudaStream_t stream);
cudaError_t cuda_tensor_multiply(float* result, const float* a, const float* b, size_t num_elements, cudaStream_t stream);
cudaError_t cuda_tensor_scale(float* result, const float* input, float scale, size_t num_elements, cudaStream_t stream);

// Attention mechanisms
cudaError_t cuda_scaled_dot_product_attention(
    float* output,           // [batch_size, seq_len, head_dim]
    const float* query,      // [batch_size, seq_len, head_dim] 
    const float* key,        // [batch_size, seq_len, head_dim]
    const float* value,      // [batch_size, seq_len, head_dim]
    const float* mask,       // [batch_size, seq_len, seq_len] (optional, can be null)
    int batch_size,
    int seq_len,
    int head_dim,
    float scale,
    cudaStream_t stream
);

cudaError_t cuda_multi_head_attention(
    float* output,           // [batch_size, seq_len, hidden_dim]
    const float* input,      // [batch_size, seq_len, hidden_dim]
    const float* weight_q,   // [hidden_dim, hidden_dim]
    const float* weight_k,   // [hidden_dim, hidden_dim]
    const float* weight_v,   // [hidden_dim, hidden_dim]
    const float* weight_o,   // [hidden_dim, hidden_dim]
    const float* bias_q,     // [hidden_dim] (optional)
    const float* bias_k,     // [hidden_dim] (optional)
    const float* bias_v,     // [hidden_dim] (optional)
    const float* bias_o,     // [hidden_dim] (optional)
    const float* mask,       // [batch_size, seq_len, seq_len] (optional)
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    cudaStream_t stream
);

// Sampling operations
cudaError_t cuda_top_k_sampling(
    int* output_tokens,      // [batch_size]
    float* output_probs,     // [batch_size] (optional)
    const float* logits,     // [batch_size, vocab_size]
    int batch_size,
    int vocab_size,
    int k,
    float temperature,
    unsigned long long seed,
    cudaStream_t stream
);

cudaError_t cuda_top_p_sampling(
    int* output_tokens,      // [batch_size]
    float* output_probs,     // [batch_size] (optional)
    const float* logits,     // [batch_size, vocab_size]
    int batch_size,
    int vocab_size,
    float p,
    float temperature,
    unsigned long long seed,
    cudaStream_t stream
);

cudaError_t cuda_combined_sampling(
    int* output_tokens,      // [batch_size]
    float* output_probs,     // [batch_size] (optional)
    const float* logits,     // [batch_size, vocab_size]
    int batch_size,
    int vocab_size,
    int top_k,
    float top_p,
    float temperature,
    unsigned long long seed,
    cudaStream_t stream
);

// Layer operations
cudaError_t cuda_layer_norm(
    float* output,           // [batch_size, hidden_dim]
    const float* input,      // [batch_size, hidden_dim]
    const float* weight,     // [hidden_dim]
    const float* bias,       // [hidden_dim] (optional)
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
);

cudaError_t cuda_rms_norm(
    float* output,           // [batch_size, hidden_dim]
    const float* input,      // [batch_size, hidden_dim]
    const float* weight,     // [hidden_dim]
    int batch_size,
    int hidden_dim,
    float eps,
    cudaStream_t stream
);

cudaError_t cuda_gelu_activation(
    float* output,           // [batch_size, hidden_dim]
    const float* input,      // [batch_size, hidden_dim]
    int batch_size,
    int hidden_dim,
    cudaStream_t stream
);

cudaError_t cuda_swiglu_activation(
    float* output,           // [batch_size, hidden_dim]
    const float* gate,       // [batch_size, intermediate_dim]
    const float* up,         // [batch_size, intermediate_dim]
    int batch_size,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream
);

// Positional encoding
cudaError_t cuda_rotary_embedding(
    float* output,           // [batch_size, seq_len, num_heads, head_dim]
    const float* input,      // [batch_size, seq_len, num_heads, head_dim]
    const float* cos_cache,  // [max_seq_len, head_dim / 2]
    const float* sin_cache,  // [max_seq_len, head_dim / 2]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int position_offset,
    cudaStream_t stream
);

// Batching utilities
cudaError_t cuda_batch_sequences(
    float* output,           // [total_tokens, hidden_dim]
    const float** inputs,    // Array of [seq_len_i, hidden_dim] pointers
    const int* sequence_lengths, // [batch_size]
    const int* sequence_offsets, // [batch_size + 1] (cumulative)
    int batch_size,
    int hidden_dim,
    cudaStream_t stream
);

cudaError_t cuda_unbatch_sequences(
    float** outputs,         // Array of [seq_len_i, hidden_dim] pointers
    const float* input,      // [total_tokens, hidden_dim]
    const int* sequence_lengths, // [batch_size]
    const int* sequence_offsets, // [batch_size + 1] (cumulative)
    int batch_size,
    int hidden_dim,
    cudaStream_t stream
);

// Performance utilities
cudaError_t cuda_warmup_kernels(int device_id);
cudaError_t cuda_synchronize_stream(cudaStream_t stream);
cudaError_t cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes, int device_id);

// Half precision support (FP16)
#ifdef CUDA_HALF_AVAILABLE
cudaError_t cuda_convert_fp32_to_fp16(
    __half* output,          // [num_elements]
    const float* input,      // [num_elements]
    size_t num_elements,
    cudaStream_t stream
);

cudaError_t cuda_convert_fp16_to_fp32(
    float* output,           // [num_elements]
    const __half* input,     // [num_elements]
    size_t num_elements,
    cudaStream_t stream
);
#endif

#ifdef __cplusplus
}
#endif