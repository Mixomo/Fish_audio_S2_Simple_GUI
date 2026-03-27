#pragma once
// s2_codec.h — Audio codec (encoder/decoder) from unified GGUF
//
// Loads codec tensors (c.* prefix) from a unified GGUF and provides
// encode (audio → codes) and decode (codes → audio) operations.
// Direct port from ggml/examples/fish-speech-codec/main.cpp

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

class AudioCodec {
public:
    AudioCodec();
    ~AudioCodec();

    // Load codec from GGUF (unified or standalone). gpu_device=-1 = CPU only.
    bool load(const std::string & gguf_path, int32_t gpu_device = -1, int32_t gpu_device_type = -1);

    // Encode mono float32 audio to VQ codes. Returns (num_codebooks, T) flattened row-major.
    bool encode(const float * audio, int32_t n_samples, int32_t n_threads,
                std::vector<int32_t> & codes_out, int32_t & n_frames_out);

    // Decode VQ codes to mono float32 audio.
    // codes: (num_codebooks, n_frames) flattened row-major.
    bool decode(const int32_t * codes, int32_t n_frames, int32_t n_threads,
                std::vector<float> & audio_out);

    int32_t sample_rate()     const { return sample_rate_; }
    int32_t hop_length()      const { return hop_length_; }
    int32_t num_codebooks()   const { return num_codebooks_; }

    // Model state (opaque, holds all codec tensors)
    struct Impl;
    Impl * impl_ = nullptr;

private:

    int32_t sample_rate_    = 44100;
    int32_t hop_length_     = 512;
    int32_t num_codebooks_  = 10;
};

} // namespace s2
