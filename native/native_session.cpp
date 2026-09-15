// Persistent JSON-lines bridge for notebook evidence tests; no HTTP server needed.
#include "omni.h"
#include "common.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <chrono>
#include <stdexcept>
using json = nlohmann::json;
namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "usage: native-session MODEL_BUNDLE OUTPUT_DIR SEED CONTEXT_SIZE\n";
        return 2;
    }
    fs::path bundle = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
    fs::create_directories(output / "audio");
    std::ofstream events(output / "native_events.jsonl");
    std::mutex event_mutex;
    auto emit = [&](json value) {
        std::lock_guard<std::mutex> lock(event_mutex);
        value["t"] = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        events << value.dump() << '\n'; events.flush();
    };
    omni_context *ctx = nullptr;
    common_params params;
    int audio_id = 0, audio_turn = 0;
    try {
        params.model.path = (bundle / "MiniCPM-o-4_5-Q4_K_M.gguf").string();
        params.apm_model = (bundle / "audio/MiniCPM-o-4_5-audio-F16.gguf").string();
        params.tts_model = (bundle / "tts/MiniCPM-o-4_5-tts-F16.gguf").string();
        for (const auto &p : {params.model.path, params.apm_model, params.tts_model})
            if (!fs::exists(p)) throw std::runtime_error("Missing model: " + p);
        params.n_ctx = std::stoi(argv[4]); params.n_gpu_layers = 99;
        params.sampling.seed = std::stoul(argv[3]);
        common_init();
        ctx = omni_init(&params, 1, true, (bundle / "tts").string(), -1, "gpu:0", true,
                        nullptr, nullptr, output.string());
        if (!ctx) throw std::runtime_error("omni_init failed");
        ctx->async = true;
        ctx->audio_output_cb = [&](const float *pcm, int n, int rate, bool final) {
            const int id = audio_id++;
            auto path = output / "audio" / ("chunk_" + std::to_string(id) + ".f32");
            std::ofstream audio(path, std::ios::binary);
            audio.write(reinterpret_cast<const char *>(pcm), n * sizeof(float)); audio.close();
            emit({{"kind","audio"},{"audio_id",id},{"turn",audio_turn},{"path",path.string()},
                  {"samples",n},{"sample_rate",rate},{"final",final}});
            if (final) audio_turn++;
        };
        ctx->context_eval_cb = [&](int frame, const std::string &text, bool ok, int before, int after) {
            emit({{"kind","context_applied"},{"frame",frame},{"text",text},{"ok",ok},
                  {"kv_before",before},{"kv_after",after}});
        };
        if (!omni_duplex_session_begin(ctx, "", output.string()))
            throw std::runtime_error("session_begin failed");
        emit({{"kind","ready"},{"vision_loaded",ctx->ctx_vision != nullptr},
              {"tts",ctx->use_tts},{"seed",params.sampling.seed},{"context_size",params.n_ctx}});
        std::string line;
        int sequence = 0;
        while (std::getline(std::cin, line)) {
            auto command = json::parse(line);
            if (command.value("op", "") == "stop") break;
            if (command.value("op", "") != "frame") throw std::runtime_error("Unknown command");
            OmniDuplexFrame frame;
            frame.aud_fname = command.at("audio").get<std::string>();
            frame.user_text = command.value("text", ""); frame.user_seq = ++sequence;
            if (!fs::exists(frame.aud_fname)) throw std::runtime_error("Input audio missing");
            auto id = omni_duplex_push_frame(ctx, frame);
            if (id < 0) throw std::runtime_error("push failed");
            emit({{"kind","frame_pushed"},{"frame",id},{"phase",command.value("phase", "")}});
            OmniDuplexFrameResult result;
            if (!omni_duplex_wait_next_frame(ctx, &result, 120000))
                throw std::runtime_error("Frame timeout");
            emit({{"kind","frame_done"},{"frame",result.frame_id},{"phase",command.value("phase", "")},
                  {"ok",result.ok},{"speak",result.is_speak},{"text",result.text},
                  {"latency_ms",result.ms_total},{"kv_after",result.n_past_after}});
            if (!result.ok) throw std::runtime_error("Frame inference failed");
        }
        omni_duplex_session_end(ctx);
        omni_free(ctx); ctx = nullptr;
        emit({{"kind","closed"},{"frames",sequence}});
        return 0;
    } catch (const std::exception &ex) {
        emit({{"kind","error"},{"message",ex.what()}});
        if (ctx) omni_free(ctx);
        return 1;
    }
}

