#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    VLLM_SUCCESS = 0,
    VLLM_ERROR_INVALID_ARGUMENT = 1,
    VLLM_ERROR_OUT_OF_MEMORY = 2,
    VLLM_ERROR_CUDA_ERROR = 3,
    VLLM_ERROR_MODEL_NOT_LOADED = 4,
    VLLM_ERROR_INFERENCE_FAILED = 5,
    VLLM_ERROR_INITIALIZATION_FAILED = 6,
    VLLM_ERROR_SHUTDOWN_FAILED = 7,
} VLLMErrorCode;

// Handle types (opaque pointers)
typedef struct VLLMEngine* VLLMEngineHandle;
typedef struct VLLMRequest* VLLMRequestHandle;
typedef struct VLLMResponse* VLLMResponseHandle;

// Configuration structure
typedef struct {
    // Model configuration
    const char* model_path;
    const char* model_name;
    int device_id;
    
    // Inference parameters
    size_t max_batch_size;
    size_t max_sequence_length;
    size_t max_tokens;
    
    // Memory configuration
    size_t gpu_memory_pool_size_mb;
    size_t max_num_seqs;
    
    // Performance settings
    float temperature;
    float top_p;
    int top_k;
    
    // Threading and async
    size_t worker_threads;
    bool enable_async_processing;
    
} VLLMConfig;

// Request structure
typedef struct {
    const char* prompt;
    size_t max_tokens;
    float temperature;
    float top_p;
    int top_k;
    const char* stop_sequences;
    bool stream;
    uint64_t request_id;
} VLLMInferenceRequest;

// Response structure
typedef struct {
    uint64_t request_id;
    const char* generated_text;
    size_t generated_tokens;
    bool is_finished;
    VLLMErrorCode error_code;
    const char* error_message;
    
    // Performance metrics
    double inference_time_ms;
    double queue_time_ms;
    size_t total_tokens;
    size_t prompt_tokens;
} VLLMInferenceResponse;

// Memory statistics
typedef struct {
    size_t total_memory_bytes;
    size_t used_memory_bytes;
    size_t free_memory_bytes;
    size_t cached_memory_bytes;
    size_t fragmentation_bytes;
    double utilization_percentage;
    int device_id;
} VLLMMemoryStats;

// Engine lifecycle functions
VLLMErrorCode vllm_create_engine(const VLLMConfig* config, VLLMEngineHandle* engine);
VLLMErrorCode vllm_destroy_engine(VLLMEngineHandle engine);
VLLMErrorCode vllm_load_model(VLLMEngineHandle engine, const char* model_path);
VLLMErrorCode vllm_unload_model(VLLMEngineHandle engine);

// Inference functions
VLLMErrorCode vllm_submit_request(VLLMEngineHandle engine, 
                                  const VLLMInferenceRequest* request,
                                  VLLMRequestHandle* request_handle);
VLLMErrorCode vllm_get_response(VLLMRequestHandle request_handle, 
                                VLLMInferenceResponse* response);
VLLMErrorCode vllm_cancel_request(VLLMRequestHandle request_handle);
VLLMErrorCode vllm_free_response(VLLMInferenceResponse* response);

// Batch processing functions
VLLMErrorCode vllm_submit_batch(VLLMEngineHandle engine,
                                const VLLMInferenceRequest* requests,
                                size_t num_requests,
                                VLLMRequestHandle* request_handles);
VLLMErrorCode vllm_get_batch_responses(const VLLMRequestHandle* request_handles,
                                       size_t num_requests,
                                       VLLMInferenceResponse* responses);

// Memory management functions
VLLMErrorCode vllm_get_memory_stats(VLLMEngineHandle engine, VLLMMemoryStats* stats);
VLLMErrorCode vllm_clear_cache(VLLMEngineHandle engine);
VLLMErrorCode vllm_set_memory_pool_size(VLLMEngineHandle engine, size_t size_mb);

// Health and monitoring functions
VLLMErrorCode vllm_health_check(VLLMEngineHandle engine);
VLLMErrorCode vllm_get_engine_info(VLLMEngineHandle engine, char* info_buffer, size_t buffer_size);

// Utility functions
const char* vllm_get_error_string(VLLMErrorCode error_code);
VLLMErrorCode vllm_set_log_level(int level);
VLLMErrorCode vllm_get_cuda_device_count(int* device_count);
VLLMErrorCode vllm_get_cuda_device_info(int device_id, char* info_buffer, size_t buffer_size);

// Stream processing functions (for streaming responses)
typedef void (*VLLMStreamCallback)(uint64_t request_id, const char* token, bool is_finished, void* user_data);
VLLMErrorCode vllm_set_stream_callback(VLLMEngineHandle engine, VLLMStreamCallback callback, void* user_data);
VLLMErrorCode vllm_enable_streaming(VLLMEngineHandle engine, bool enable);

#ifdef __cplusplus
}
#endif