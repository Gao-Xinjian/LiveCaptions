// Internal to lc_native: the capture state shared by lc_native.cpp (miniaudio) and lc_tap_macos.mm (Core Audio tap).
// Every file that includes this sees the same miniaudio configuration, so the miniaudio structs have one layout.
#pragma once

#include "lc_native.h"

#define MA_ENABLE_ONLY_SPECIFIC_BACKENDS
#ifdef _WIN32
#define MA_ENABLE_WASAPI   // the only Windows backend that can capture what the speakers play
#elif defined(__APPLE__)
#define MA_ENABLE_COREAUDIO
#else
#define MA_ENABLE_PULSEAUDIO
#define MA_ENABLE_ALSA
#endif
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#include "miniaudio.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

constexpr int kLcRate = 16000;  // Hz, what Whisper expects

struct lc_audio {
    ma_context ctx{};
    ma_device  dev{};
    bool       ctx_ok = false;
    bool       dev_ok = false;
    int        channel_mode = 0;   // 0 mix, 1 left, 2 right
    void *     tap = nullptr;      // macOS process tap state (LcTap) when capturing system audio there

    std::mutex              mu;
    std::condition_variable cv;
    std::vector<float>      ring;       // capacity = ring.size()
    size_t                  head = 0;   // read position
    size_t                  size = 0;   // samples waiting

    std::atomic<int64_t> last_ns{0};
    std::atomic<int64_t> dropped{0};
    std::atomic<int>     flags{0};
    std::atomic<bool>    stopping{false};
};

inline int64_t LcNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Appends frames of 16 kHz float audio with `step` interleaved channels, taking channel `pick` of each frame.
inline void LcPushSamples(lc_audio * a, const float * in, uint64_t frames, int step, int pick) {
    if (!in || frames == 0) return;
    const size_t cap = a->ring.size();
    {
        std::lock_guard<std::mutex> lock(a->mu);
        for (uint64_t i = 0; i < frames; i++) {
            if (a->size == cap) {   // full: drop the oldest sample so latency stays bounded
                a->head = (a->head + 1) % cap;
                a->size--;
                a->dropped++;
            }
            a->ring[(a->head + a->size) % cap] = in[i * (uint64_t) step + (uint64_t) pick];
            a->size++;
        }
    }
    a->last_ns = LcNowNs();
    a->cv.notify_one();
}

#ifdef __APPLE__
// lc_tap_macos.mm: system audio through a Core Audio process tap (macOS 14.2+).
bool LcTapAvailable();
bool LcTapOpen(lc_audio * a, const lc_audio_params * p, std::string & err);
bool LcTapStart(lc_audio * a);
void LcTapStop(lc_audio * a);
void LcTapClose(lc_audio * a);
#endif
