#pragma once
// s2_tokenizer.h — BPE tokenizer for Qwen3/Fish Speech
//
// Reads HuggingFace tokenizer.json format.
// Supports special tokens: <|im_start|>, <|im_end|>, <|voice|>,
// <|semantic:N|>, <|speaker:N|>, etc.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace s2 {

struct TokenizerConfig {
    int32_t im_start_id       = 0;
    int32_t im_end_id         = 0;
    int32_t voice_id          = 0;
    int32_t pad_id            = 0;
    int32_t eos_id            = 0;
    int32_t semantic_begin_id = 0;
    int32_t semantic_end_id   = 0;
    int32_t audio_pad_id      = 0;
    int32_t num_codebooks     = 10;
    int32_t codebook_size     = 4096;
    int32_t vocab_size        = 155776;
};

class Tokenizer {
public:
    Tokenizer() = default;

    // Load from HuggingFace tokenizer.json
    bool load(const std::string & path);

    // Encode text to token IDs (handles special tokens inline)
    std::vector<int32_t> encode(const std::string & text) const;

    // Get a special token ID by its string representation
    int32_t token_to_id(const std::string & token) const;

    // Check if loaded
    bool is_loaded() const { return loaded_; }

    // Access config
    const TokenizerConfig & config() const { return config_; }
    TokenizerConfig & config() { return config_; }

private:
    bool loaded_ = false;
    TokenizerConfig config_;

    // BPE vocabulary: token string → token ID
    std::unordered_map<std::string, int32_t> vocab_;
    // Reverse map: token ID → token string
    std::unordered_map<int32_t, std::string> id_to_token_;

    // BPE merge rules: (pair) → priority (lower = higher priority)
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int32_t> merge_rank_;

    // Special tokens: text → ID (matched before BPE)
    std::vector<std::pair<std::string, int32_t>> special_tokens_;

    // Internal BPE encoding of a single word
    std::vector<int32_t> bpe_encode_word(const std::string & word) const;
};

} // namespace s2
