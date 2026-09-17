#pragma once
// s2_prompt.h — Prompt tensor builder for Fish Speech S2
//
// Builds the (num_codebooks+1, total_len) int32 prompt tensor
// that is fed to the Slow-AR model.
// Port of build_prompt_tensor() from ggml_pure.py.

#include "s2_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

// Build prompt tensor for voice cloning.
// Returns flat int32 array of shape (num_codebooks+1, total_len) in row-major order.
// prompt_codes: (num_codebooks, T_prompt) in row-major order.
struct PromptTensor {
    std::vector<int32_t> data;  // flat (num_codebooks+1) × total_len
    int32_t rows    = 0;        // num_codebooks + 1
    int32_t cols    = 0;        // total_len (timesteps)
};

PromptTensor build_prompt(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,   // (num_codebooks, T_prompt) row-major
    int32_t num_codebooks,
    int32_t T_prompt
);

} // namespace s2
