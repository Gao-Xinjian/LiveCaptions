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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

constexpr int kLcRate = 16000;           // Hz, what Whisper expects
constexpr int kLcLevelSamples = 160;     // one waveform column per 10 ms
constexpr size_t kLcLevelCap = 6000;     // columns kept: 60 s

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
    uint64_t                pushed = 0; // samples ever pushed; minus size = capture position of the next read
    // Peak per 10 ms as the audio arrives, for a live waveform that does not wait for the reader.
    std::vector<float>      levels;     // capacity = levels.size(), indexed by column % capacity
    uint64_t                level_total = 0;
    float                   level_peak  = 0.0f;
    int                     level_fill  = 0;

    std::atomic<int64_t> last_ns{0};
    std::atomic<int64_t> dropped{0};
    std::atomic<int>     flags{0};
    std::atomic<bool>    stopping{false};
};

inline int64_t LcNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double GetMsSince(int64_t t0_ns) { return (double) (LcNowNs() - t0_ns) / 1e6; }

constexpr int kLcLogInfo = 2;  // GGML_LOG_LEVEL_INFO: printed after lc_set_log_level(2) or lower

// Prints a printf-style message to stderr when `level` reaches the level set by lc_set_log_level.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void LcLog(int level, const char * fmt, ...);

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
            const float x = in[i * (uint64_t) step + (uint64_t) pick];
            a->ring[(a->head + a->size) % cap] = x;
            a->size++;
            a->level_peak = std::max(a->level_peak, std::fabs(x));
            if (++a->level_fill == kLcLevelSamples && !a->levels.empty()) {
                a->levels[a->level_total % a->levels.size()] = a->level_peak;
                a->level_total++;
                a->level_peak = 0.0f;
                a->level_fill = 0;
            }
        }
        a->pushed += frames;
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
