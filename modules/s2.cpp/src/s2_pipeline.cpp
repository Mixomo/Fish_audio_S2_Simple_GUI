#include "../include/s2_pipeline.h"
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#if defined(GGML_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

namespace s2 {

static void safe_print_ln(const std::string& msg) {
    fputs(msg.c_str(), stdout);
    fputc('\n', stdout);
}

static void safe_print_error_ln(const std::string& msg) {
    fputs(msg.c_str(), stderr);
    fputc('\n', stderr);
}

struct GpuDecodeWindowPlan {
    int32_t window_frames = 96;
    int32_t history_frames = 32;
    int32_t stride_frames = 32;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    bool memory_known = false;
    bool explicit_override = false;
};

static bool is_supported_gpu_window(int32_t frames) {
    return frames == 96 || frames == 128 || frames == 160 || frames == 192;
}

static bool query_gpu_memory(int32_t backend_type, int32_t device,
                             size_t & free_bytes, size_t & total_bytes) {
    if (device < 0) return false;

#if defined(GGML_USE_CUDA)
    if (backend_type == 1) {
        ggml_backend_cuda_get_device_memory(device, &free_bytes, &total_bytes);
        return free_bytes > 0 && total_bytes > 0;
    }
#endif
#if defined(GGML_USE_VULKAN)
    if (backend_type == 0) {
        ggml_backend_vk_get_device_memory(device, &free_bytes, &total_bytes);
        return free_bytes > 0 && total_bytes > 0;
    }
#endif
    return false;
}

static int32_t choose_gpu_window_from_free_memory(size_t free_bytes) {
    // The codec's F16 weights are already resident when this is called.  Keep
    // a quarter of the remaining memory (and at least 1 GiB) for transient
    // CUDA/Vulkan allocations, the AR cache and other applications.
    constexpr size_t GiB = 1024ull * 1024ull * 1024ull;
    const size_t reserve = std::max(GiB, free_bytes / 4);
    const size_t usable = free_bytes > reserve ? free_bytes - reserve : 0;

    // Attention buffers grow faster than the waveform buffers, so the policy
    // is deliberately conservative. Allocation/compute failure still has a
    // runtime fallback to the next smaller candidate.
    if (usable >= 5 * GiB) return 192;
    if (usable >= 3 * GiB) return 160;
    if (usable >= 1500ull * 1024ull * 1024ull) return 128;
    return 96;
}

static GpuDecodeWindowPlan make_gpu_window_plan(const PipelineParams & params) {
    GpuDecodeWindowPlan plan;

    if (is_supported_gpu_window(params.gpu_window_frames)) {
        plan.window_frames = params.gpu_window_frames;
        plan.explicit_override = true;
    } else {
        const int32_t memory_backend = params.codec_cuda ? 1 : params.backend_type;
        const int32_t memory_device = params.codec_cuda ? 0 : params.gpu_device;
        plan.memory_known = query_gpu_memory(memory_backend, memory_device,
                                             plan.free_bytes, plan.total_bytes);
        if (plan.memory_known) {
            plan.window_frames = choose_gpu_window_from_free_memory(plan.free_bytes);
        }
    }

    plan.history_frames = 32;
    // For a target window W, W - 2H new frames are committed per graph while
    // H frames remain as causal context on both sides of the committed tail.
    plan.stride_frames = std::max(32, plan.window_frames - 2 * plan.history_frames);
    return plan;
}

static std::string format_mib(size_t bytes) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1)
       << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
    return ss.str();
}

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {}

bool Pipeline::init(const PipelineParams & params) {
    safe_print_ln("--- Pipeline Init ---");

    if (!tokenizer_.load(params.tokenizer_path)) {
        safe_print_error_ln("Pipeline error: could not load tokenizer from " + params.tokenizer_path);
        return false;
    }

    if (!model_.load(params.model_path, params.gpu_device, params.backend_type)) {
        safe_print_error_ln("Pipeline error: could not load model from " + params.model_path);
        return false;
    }

    // Keep the audio codec on the same backend as the transformer whenever possible.
    // Older Easy-GUI builds unconditionally loaded the codec on CPU and created a
    // second CUDA-only encoder. That forced reference codes and waveform decoding
    // through host memory. Current s2.cpp follows the selected GPU backend instead.
    int32_t codec_gpu = -1;
    int32_t codec_backend = -1;
    if (!params.codec_cpu) {
        if (params.codec_cuda) {
            codec_gpu = 0;
            codec_backend = 1;
        } else if (params.codec_follow_backend && params.gpu_device >= 0) {
            codec_gpu = params.gpu_device;
            codec_backend = params.backend_type;
        }
    }

    if (!codec_.load(params.model_path, codec_gpu, codec_backend)) {
        if (codec_gpu >= 0) {
            safe_print_error_ln("[Codec] GPU initialization failed; falling back to CPU codec.");
            if (!codec_.load(params.model_path, -1, -1)) {
                safe_print_error_ln("Pipeline error: could not load codec from " + params.model_path);
                return false;
            }
        } else {
            safe_print_error_ln("Pipeline error: could not load codec from " + params.model_path);
            return false;
        }
    }

    if (codec_gpu >= 0) {
        safe_print_ln(std::string("[Codec] GPU backend enabled: ") +
                      (codec_backend == 1 ? "CUDA" : "Vulkan") +
                      " (reference encoder + waveform decoder)");
    } else {
        safe_print_ln("[Codec] CPU backend enabled");
    }

    if (!params.lora_path.empty()) {
        if (!model_.load_lora(params.lora_path)) {
            safe_print_error_ln("Pipeline error: could not load LoRA adapter from " + params.lora_path);
            return false;
        }
        safe_print_ln("[LoRA] Native C++ adapter loaded: " + params.lora_path);
    }

    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig & tc    = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    initialized_ = true;
    return true;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (!params.prompt_audio_path.empty() && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: prompt audio was provided without prompt text.");
        return false;
    }

    if (!params.prompt_audio_path.empty()) {
        safe_print_ln("Loading reference audio: " + params.prompt_audio_path);
        if (!load_audio(params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
            safe_print_error_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    if (!this->synthesize_raw(params, ref_audio, audio_out)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec_.sample_rate());
    }

    if (!save_audio(params.output_path, audio_out, codec_.sample_rate(),
                    false, params.normalize_output)) {
        safe_print_error_ln("Pipeline error: save_audio failed to " + params.output_path);
        return false;
    }

    safe_print_ln("Saved audio to: " + params.output_path);
    return true;
}

bool Pipeline::synthesize_to_memory(const PipelineParams & params, void** ref_audio_buffer, size_t* ref_audio_size, void** wav_buffer, size_t* wav_size) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (ref_audio_buffer && ref_audio_size && *ref_audio_buffer && *ref_audio_size > 0 && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: reference audio was provided without reference text.");
        return false;
    }

    if (ref_audio_buffer && ref_audio_size && *ref_audio_buffer && *ref_audio_size > 0) {
        safe_print_ln("Loading reference audio...");
        if (!load_audio_from_memory(*ref_audio_buffer, *ref_audio_size, ref_audio, codec_.sample_rate())) {
            safe_print_error_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    if (!this->synthesize_raw(params, ref_audio, audio_out)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec_.sample_rate());
    } else if (params.normalize_output && !audio_out.empty()) {
        float peak = 0.0f;
        for (float s : audio_out) { float a = std::fabs(s); if (a > peak) peak = a; }
        if (peak > 1e-6f) { float scale = 0.95f / peak; for (float & s : audio_out) s *= scale; }
    }

    if (!audio_write_memory_wav(wav_buffer, wav_size, audio_out.data(), audio_out.size(), codec_.sample_rate())) {
        safe_print_error_ln("Pipeline error: audio_write_memory_wav failed");
        return false;
    }

    safe_print_ln("Audio synthesized");
    return true;
}

bool Pipeline::synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float>& audio_out) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    using clock = std::chrono::steady_clock;
    const auto elapsed_seconds = [](clock::time_point start) {
        return std::chrono::duration<double>(clock::now() - start).count();
    };

    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    model_.clear_kv_cache();

    safe_print_ln("--- Pipeline Synthesize ---");
    safe_print_ln("Text: " + params.text);

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;

    if (!ref_audio.samples.empty()) {
        const auto encode_start = clock::now();
        if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                            params.gen.n_threads, ref_codes, T_prompt)) {
            safe_print_error_ln("Pipeline warning: encode failed, running without reference audio.");
            ref_codes.clear();
            T_prompt = 0;
        }
        safe_print_ln("[Timing] Reference encode: " + std::to_string(elapsed_seconds(encode_start)) + " s");
    }

    PromptTensor prompt = build_prompt(
        tokenizer_, params.text, params.prompt_text,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_codebooks, T_prompt);

    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    if (!model_.init_kv_cache(max_seq_len)) {
        safe_print_error_ln("Pipeline error: init_kv_cache failed.");
        return false;
    }

    const auto generate_start = clock::now();
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);
    safe_print_ln("[Timing] AR generation: " + std::to_string(elapsed_seconds(generate_start)) + " s");

    if (res.n_frames == 0) {
        safe_print_error_ln("Pipeline error: generation produced no frames.");
        return false;
    }

    const auto decode_start = clock::now();
    safe_print_ln("[Codec] Decode phase starting (" + std::to_string(res.n_frames) + " frames)...");

    // PATCH-07: bounded windowed decode for GPU codecs.
    // Current upstream s2.cpp avoids decoding long utterances as one monolithic
    // codec graph. On CUDA, very large graphs (e.g. 500+ codec frames) can spend
    // effectively unbounded time inside ggml_backend_graph_compute on Windows.
    // Keep ALL codec math on the selected GPU, but execute bounded causal windows
    // and commit only the stable tail of each window.
    const bool gpu_codec = codec_.is_gpu_backend();
    if (gpu_codec && res.n_frames > 96) {
        const auto window_policy_start = std::chrono::steady_clock::now();
        const GpuDecodeWindowPlan plan = make_gpu_window_plan(params);
        const double window_policy_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - window_policy_start).count();
        std::ostringstream plan_line;
        plan_line << "[Codec] GPU window policy: " << plan.window_frames
                  << " frames, history=" << plan.history_frames
                  << ", stride=" << plan.stride_frames
                  << ", selection=" << std::fixed << std::setprecision(3)
                  << window_policy_ms << " ms";
        if (plan.explicit_override) {
            plan_line << " (explicit)";
        } else if (plan.memory_known) {
            plan_line << ", free VRAM=" << format_mib(plan.free_bytes)
                      << "/" << format_mib(plan.total_bytes);
        } else {
            plan_line << " (VRAM query unavailable; conservative default)";
        }
        safe_print_ln(plan_line.str());

        // A failed graph allocation/compute is recoverable: retry the complete
        // decode with the next smaller plan.  This matters on Windows because
        // driver allocations can be fragmented even when cudaMemGetInfo() has
        // reported enough nominal free memory.
        auto decode_gpu_windows = [&](int32_t target_window_frames) -> bool {
            const int32_t history_frames = 32;
            const int32_t stride_frames = std::max(32, target_window_frames - 2 * history_frames);
            const int32_t num_codebooks_local = model_.hparams().num_codebooks;
            const size_t samples_per_frame = static_cast<size_t>(std::max(1, codec_.hop_length()));
            std::vector<int32_t> window_codes;
            std::vector<float> window_audio;
            audio_out.clear();
            audio_out.reserve(static_cast<size_t>(res.n_frames) * samples_per_frame);

            // One initial window, zero or more full intermediate windows, and
            // one final window.  The count is only for a truthful progress bar.
            const int32_t total_batches = res.n_frames <= target_window_frames
                ? 1
                : 2 + std::max(0, (res.n_frames - 1 - target_window_frames) / stride_frames);
            int32_t committed_frames = 0;
            int32_t batch_index = 0;

            auto decode_window = [&](int32_t frames_available, bool finalize) -> bool {
                if (frames_available <= committed_frames) return true;
                const int32_t stable_frames = finalize
                    ? frames_available
                    : std::max(0, frames_available - history_frames);
                if (stable_frames <= committed_frames && !finalize) return true;

                const int32_t window_start = std::max(0, committed_frames - history_frames);
                const int32_t window_frames = frames_available - window_start;
                if (window_frames <= 0) return true;

                window_codes.resize(static_cast<size_t>(num_codebooks_local) * window_frames);
                for (int32_t cb = 0; cb < num_codebooks_local; ++cb) {
                    const int32_t * src = res.codes.data() + static_cast<size_t>(cb) * res.n_frames + window_start;
                    std::copy(src, src + window_frames,
                              window_codes.begin() + static_cast<size_t>(cb) * window_frames);
                }

                ++batch_index;
                if (params.gen.verbose) {
                    const int width = 30;
                    const double ratio = std::max(0.0, std::min(1.0,
                        static_cast<double>(batch_index) / std::max(1, total_batches)));
                    const int filled = static_cast<int>(ratio * width);
                    const int percent = static_cast<int>(ratio * 100.0);
                    std::ostringstream line;
                    line << "\r[Codec]    " << std::setw(3) << percent << "%|";
                    for (int i = 0; i < width; ++i) line << (i < filled ? '#' : '-');
                    line << "| " << batch_index << "/" << total_batches
                         << " windows · frames " << window_start << "-" << frames_available;
                    std::cout << line.str() << std::flush;
                }

                if (!codec_.decode(window_codes.data(), window_frames,
                                   params.gen.n_threads, window_audio)) {
                    return false;
                }

                // The GGUF hop metadata is not the final sample stride of this
                // decoder graph: the waveform decoder can expand each code
                // frame further. Derive the stitch stride from the actual graph
                // output so every window size produces the same final waveform.
                if (window_audio.size() % static_cast<size_t>(window_frames) != 0) {
                    safe_print_error_ln("[Codec] GPU window output is not frame-aligned; refusing to stitch it.");
                    return false;
                }
                const size_t output_samples_per_frame =
                    window_audio.size() / static_cast<size_t>(window_frames);

                const size_t emit_begin = static_cast<size_t>(std::max(0,
                    committed_frames - window_start)) * output_samples_per_frame;
                const size_t emit_end = finalize
                    ? window_audio.size()
                    : std::min(window_audio.size(), static_cast<size_t>(std::max(0,
                        stable_frames - window_start)) * output_samples_per_frame);
                if (emit_end > emit_begin) {
                    audio_out.insert(audio_out.end(), window_audio.begin() + emit_begin,
                                     window_audio.begin() + emit_end);
                }
                committed_frames = finalize ? frames_available : stable_frames;
                return true;
            };

            if (res.n_frames <= target_window_frames) {
                return codec_.decode(res.codes.data(), res.n_frames,
                                     params.gen.n_threads, audio_out);
            }

            if (!decode_window(target_window_frames, false)) return false;
            for (int32_t available = target_window_frames + stride_frames;
                 available < res.n_frames;
                 available += stride_frames) {
                if (!decode_window(available, false)) return false;
            }
            return decode_window(res.n_frames, true);
        };

        std::vector<int32_t> candidates;
        for (const int32_t candidate : {192, 160, 128, 96}) {
            if (candidate <= plan.window_frames) candidates.push_back(candidate);
        }

        bool decoded = false;
        for (size_t attempt = 0; attempt < candidates.size(); ++attempt) {
            const int32_t candidate = candidates[attempt];
            if (candidate != plan.window_frames) {
                safe_print_ln("[Codec] Retrying GPU decode with " +
                              std::to_string(candidate) + " frame windows.");
            }
            if (decode_gpu_windows(candidate)) {
                decoded = true;
                break;
            }
        }
        if (!decoded) {
            safe_print_error_ln("Pipeline error: all adaptive CUDA window plans failed.");
            return false;
        }
    } else {
        if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, audio_out)) {
            safe_print_error_ln("Pipeline error: decode failed.");
            return false;
        }
    }

    if (params.gen.verbose && gpu_codec && res.n_frames > 96) {
        std::cout << std::endl;
    }
    safe_print_ln("[Timing] Waveform decode: " + std::to_string(elapsed_seconds(decode_start)) + " s");
    safe_print_ln("[Codec] Decode phase complete (" + std::to_string(audio_out.size()) + " samples).");

    model_.clear_kv_cache();
    return true;
}

}
