#pragma once

#include "s2_model.h"
#include "s2_sampler.h"
#include "s2_tokenizer.h"
#include "s2_prompt.h"

#include <cstdint>
#include <vector>
#include <functional>

namespace s2 {

struct GenerateParams {
    int32_t max_new_tokens          = 1024;
    float   temperature             = 0.8f;
    float   top_p                   = 0.8f;
    int32_t top_k                   = 30;
    int32_t min_tokens_before_end   = 0;
    int32_t n_threads               = 4;
    bool    verbose                 = true;
};

struct GenerateResult {
    std::vector<int32_t> codes;
    int32_t num_codebooks = 0;
    int32_t n_frames      = 0;
};

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
);

}
