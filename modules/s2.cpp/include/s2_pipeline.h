#pragma once

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <mutex>

namespace s2 {

struct PipelineParams {
    std::string model_path;
    std::string tokenizer_path;
    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;
    GenerateParams gen;
    int32_t gpu_device = -1;   // -1 = CPU only
    int32_t backend_type = -1; //0 = Vulkan; 1 = Cuda;
    // Codec backend policy. By default the codec follows the selected GPU backend.
    bool codec_follow_backend = true;
    bool codec_cpu = false;
    bool codec_cuda = false; // Backward-compatible alias: force CUDA codec.
    // GPU codec decode window. 0 = choose automatically from current free VRAM.
    // Supported explicit values are 96, 128, 160 and 192 frames.
    int32_t gpu_window_frames = 0;
    std::string lora_path;
    bool trim_silence = false;
    bool normalize_output = false;
    bool normalize_dynamic = false;
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineParams & params);
    bool synthesize(const PipelineParams & params);

    bool synthesize_to_memory(const PipelineParams & params, void** ref_audio_buffer, size_t* ref_audio_size, void** wav_buffer, size_t* wav_size);
    bool synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float> & audio_out);

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    mutable std::mutex synthesize_mutex_;
    bool initialized_ = false;
};

}
