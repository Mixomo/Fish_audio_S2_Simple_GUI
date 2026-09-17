#pragma once
// s2_model.h — Slow-AR + Fast-AR model (Dual-AR Qwen3 transformer)
//
// Loads the Dual-AR portion from a unified GGUF file and provides
// prefill / step / fast decode operations with KV cache.
// Direct port from ggml/examples/fish-speech-slow-ar/main.cpp

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace s2 {

struct ModelLayer {
    ggml_tensor * attention_norm = nullptr;
    ggml_tensor * ffn_norm       = nullptr;
    ggml_tensor * q_norm         = nullptr;
    ggml_tensor * k_norm         = nullptr;
    ggml_tensor * wqkv           = nullptr;
    ggml_tensor * wo             = nullptr;
    ggml_tensor * w1             = nullptr;
    ggml_tensor * w2             = nullptr;
    ggml_tensor * w3             = nullptr;
};

struct ModelHParams {
    int32_t context_length     = 0;
    int32_t vocab_size         = 0;
    int32_t embedding_length   = 0;
    int32_t feed_forward_length = 0;
    int32_t block_count        = 0;
    int32_t head_count         = 0;
    int32_t head_count_kv      = 0;
    int32_t codebook_size      = 0;
    int32_t num_codebooks      = 0;
    int32_t semantic_begin_id  = 0;
    int32_t semantic_end_id    = 0;
    float   rope_freq_base     = 10000.0f;
    float   rms_norm_eps       = 1e-5f;
    bool    tie_word_embeddings = true;
    bool    attention_qk_norm  = false;
    bool    scale_codebook_embeddings = false;

    // Fast decoder
    int32_t fast_context_length     = 0;
    int32_t fast_embedding_length   = 0;
    int32_t fast_feed_forward_length = 0;
    int32_t fast_block_count        = 0;
    int32_t fast_head_count         = 0;
    int32_t fast_head_count_kv      = 0;
    int32_t fast_head_dim           = 0;
    float   fast_rope_freq_base     = 10000.0f;
    float   fast_rms_norm_eps       = 1e-5f;
    bool    fast_attention_qk_norm  = false;
    bool    fast_has_project_in     = false;
    bool    has_fast_decoder        = false;
};

struct ModelWeights {
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;

    ggml_tensor * embeddings           = nullptr;
    ggml_tensor * codebook_embeddings  = nullptr;
    ggml_tensor * norm                 = nullptr;
    ggml_tensor * fast_project_in      = nullptr;
    ggml_tensor * fast_embeddings      = nullptr;
    ggml_tensor * fast_norm            = nullptr;
    ggml_tensor * fast_output          = nullptr;

    std::vector<ModelLayer> layers;
    std::vector<ModelLayer> fast_layers;
};

struct StepResult {
    std::vector<float> hidden;   // (embedding_length,) last-token hidden state
    std::vector<float> logits;   // (vocab_size,) logits
};

class SlowARModel {
public:
    SlowARModel();
    ~SlowARModel();

    // Load model from GGUF. gpu_device=-1 means CPU only.
    bool load(const std::string & gguf_path, int32_t gpu_device = -1, int32_t backend_type = -1);

    // Load a native Fast-AR LoRA adapter exported by Easy GUI.
    // The adapter is intentionally kept separate from the base GGUF so it also
    // works with quantized model files.
    bool load_lora(const std::string & adapter_path);
    void clear_lora();

    // Initialize KV cache for generation
    bool init_kv_cache(int32_t max_seq_len);

    // Reset KV cache (for new generation)
    void reset();

    void clear_kv_cache();

    // Prefill: process prompt tokens. flat_tokens: (num_codebooks+1)*n_tokens
    bool prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                 int32_t n_threads, StepResult & result);

    // Step: process a single timestep. flat_tokens: (num_codebooks+1)
    bool step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
              StepResult & result);

    // Fast decoder: generate next codebook logits given hidden state and prefix codes
    bool fast_decode(const std::vector<float> & hidden,
                     const std::vector<int32_t> & prefix_codes,
                     int32_t n_threads,
                     std::vector<float> & logits_out);

    const ModelHParams & hparams() const { return hparams_; }

private:
    ModelHParams   hparams_;
    ModelWeights   weights_;
    ggml_backend_t backend_      = nullptr;
    ggml_gallocr_t allocr_       = nullptr;
    ggml_gallocr_t fast_allocr_  = nullptr;
    ggml_context * ctx_kv_      = nullptr;
    ggml_backend_buffer_t kv_buf_ = nullptr;
    ggml_tensor *  memory_k_   = nullptr;
    ggml_tensor *  memory_v_   = nullptr;
    int32_t        max_seq_len_ = 0;
    int32_t        n_past_     = 0;

    struct LoRAPair {
        ggml_tensor * a = nullptr;
        ggml_tensor * b = nullptr;
        float scale = 1.0f;
    };

    ggml_context * lora_ctx_ = nullptr;
    ggml_backend_buffer_t lora_buf_ = nullptr;
    std::unordered_map<std::string, LoRAPair> lora_pairs_;
    uint32_t lora_rank_ = 0;
    float lora_alpha_ = 0.0f;

    ggml_tensor * apply_lora_linear(ggml_context * ctx, const std::string & name,
                                    ggml_tensor * input, ggml_tensor * base_output);
    ggml_tensor * apply_lora_embedding(ggml_context * ctx, const std::string & name,
                                       ggml_tensor * ids, ggml_tensor * base_output);

    bool eval_cached(const std::vector<int32_t> & flat_tokens,
                     int32_t n_tokens, int32_t n_threads,
                     StepResult & result);
};

} // namespace s2
