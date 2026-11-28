// lc_native: the native part of LiveCaptions, loaded from Python with ctypes.
// One shared library with miniaudio capture, whisper.cpp (statically linked) and whisper.cpp's Silero VAD.
// Every struct here is mirrored in livecaptions/native.py; lc_struct_sizes() lets Python check the layout.
#pragma once

#include <stdint.h>

#ifdef _WIN32
#define LC_API __declspec(dllexport)
#else
#define LC_API __attribute__((visibility("default")))
#endif

#define LC_DEVICE_ID_BYTES 512   // raw ma_device_id, large enough on every platform
#define LC_NAME_BYTES      256

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lc_device_info {
    char          name[LC_NAME_BYTES];        // UTF-8
    unsigned char id[LC_DEVICE_ID_BYTES];
    int32_t       is_default;
} lc_device_info;

typedef struct lc_audio_params {
    int32_t              kind;           // 0 = capture (microphone), 1 = system audio (Windows loopback, macOS process tap)
    const unsigned char *device_id;      // NULL = the system default device (ignored by the macOS tap)
    int32_t              channel_mode;   // 0 = mix to mono, 1 = left, 2 = right
    int32_t              period_ms;      // device callback period; longer = fewer wake-ups
    int32_t              ring_ms;        // capacity of the capture ring buffer
    int32_t              lpf_order;      // resampler low-pass filter order, 0..8
    const char          *tap_apps;       // macOS tap: comma-separated bundle IDs to capture; NULL or "" = all audio
} lc_audio_params;

typedef struct lc_decode_params {
    int32_t        n_threads;
    int32_t        translate;           // 1 = translate to English
    int32_t        no_context;
    int32_t        single_segment;
    int32_t        token_timestamps;
    int32_t        max_len;             // max segment length in characters, 0 = no limit
    int32_t        split_on_word;
    int32_t        max_tokens;          // max tokens per segment, 0 = no limit
    int32_t        audio_ctx;           // 0 = full 30 s encoder window
    int32_t        beam_size;           // <= 1 = greedy
    int32_t        best_of;             // greedy candidates when the temperature is above 0
    int32_t        suppress_blank;
    int32_t        suppress_nst;        // suppress non-speech tokens such as "*" or music notes
    float          temperature;
    float          temperature_inc;     // 0 = no fallback
    float          entropy_thold;
    float          logprob_thold;
    float          no_speech_thold;
    float          length_penalty;
    float          max_initial_ts;
    float          thold_pt;
    float          thold_ptsum;
    const char    *language;            // "auto", "" or NULL = detect
    const int32_t *prompt_tokens;
    int32_t        prompt_n_tokens;
    const char    *suppress_regex;      // NULL = none
} lc_decode_params;

typedef struct lc_token_info {
    int32_t id;
    float   p;      // probability
    float   plog;   // log probability
    int64_t t0;     // centiseconds from the start of the audio, -1 if unknown
    int64_t t1;
} lc_token_info;

typedef struct lc_gpu_info {
    char    name[LC_NAME_BYTES];
    char    description[LC_NAME_BYTES];
    int32_t type;          // 1 = discrete GPU, 2 = integrated GPU
    int64_t memory_free;   // bytes, 0 if unknown
    int64_t memory_total;
} lc_gpu_info;

typedef struct lc_audio   lc_audio;
typedef struct lc_whisper lc_whisper;
typedef struct lc_vad     lc_vad;

// General
LC_API int32_t      lc_version(void);
LC_API void         lc_struct_sizes(int32_t *out, int32_t n);   // sizes of the structs above, in declaration order
LC_API void         lc_set_log_level(int32_t min_level);        // ggml levels: 1 debug .. 4 error, 5 = silent
LC_API const char * lc_system_info(void);
LC_API int32_t      lc_gpu_count(void);                         // GPU and iGPU devices, in whisper's gpu_device order
LC_API int32_t      lc_gpu_get(int32_t i, lc_gpu_info *out);

// Audio capture: the callback only copies into a ring buffer; lc_audio_read blocks without holding the GIL
LC_API int32_t    lc_system_audio_available(void);   // 1 on Windows, and on macOS 14.2+ (process tap)
LC_API int32_t    lc_audio_list(int32_t kind, lc_device_info *out, int32_t max);   // kind 0 = capture, 1 = playback
LC_API lc_audio * lc_audio_open(const lc_audio_params *params, char *err, int32_t err_len);
LC_API int32_t    lc_audio_start(lc_audio *a);
LC_API int32_t    lc_audio_stop(lc_audio *a);
LC_API int32_t    lc_audio_read(lc_audio *a, float *dst, int32_t max, int32_t timeout_ms);   // samples read, 0 on timeout
LC_API double     lc_audio_idle_seconds(lc_audio *a);   // time since the device last delivered data
LC_API int64_t    lc_audio_dropped(lc_audio *a);        // samples lost because the ring buffer was full
// Peak level per 10 ms column as the audio arrives (for a live waveform; takes nothing from the ring): copies the
// columns from `since` on, at most the newest `max`, returns how many; *total = columns produced so far.
LC_API int32_t    lc_audio_levels(lc_audio *a, uint64_t since, float *out, int32_t max, uint64_t *total);
// Capture position (samples since open) of the next sample lc_audio_read returns; overflow and stops count too.
LC_API uint64_t   lc_audio_position(lc_audio *a);
LC_API int32_t    lc_audio_flags(lc_audio *a);          // 1 = device stopped unexpectedly, 2 = rerouted
LC_API void       lc_audio_close(lc_audio *a);

// Whisper
LC_API lc_whisper *  lc_whisper_load(const char *path_utf8, int32_t use_gpu, int32_t gpu_device, int32_t flash_attn, char *err, int32_t err_len);
LC_API void          lc_whisper_free(lc_whisper *w);
LC_API int32_t       lc_whisper_full(lc_whisper *w, const lc_decode_params *p, const float *pcm, int32_t n);
LC_API void          lc_whisper_abort(lc_whisper *w, int32_t flag);   // 1 = stop the running decode as soon as possible
LC_API int32_t       lc_whisper_tokenize(lc_whisper *w, const char *text, int32_t *out, int32_t max);   // < 0: -needed
LC_API int32_t       lc_whisper_token_eot(lc_whisper *w);
LC_API int32_t       lc_whisper_is_multilingual(lc_whisper *w);
LC_API int32_t       lc_whisper_n_text_ctx(lc_whisper *w);
LC_API const char *  lc_whisper_model_type(lc_whisper *w);
LC_API int32_t       lc_whisper_lang_id(lc_whisper *w);   // language of the last run
LC_API const char *  lc_lang_str(int32_t id);
LC_API int32_t       lc_whisper_n_segments(lc_whisper *w);
LC_API int64_t       lc_whisper_segment_t0(lc_whisper *w, int32_t i);   // centiseconds
LC_API int64_t       lc_whisper_segment_t1(lc_whisper *w, int32_t i);
LC_API float         lc_whisper_segment_no_speech_prob(lc_whisper *w, int32_t i);
LC_API int32_t       lc_whisper_n_tokens(lc_whisper *w, int32_t i);
LC_API int32_t       lc_whisper_token(lc_whisper *w, int32_t i, int32_t j, lc_token_info *out);
LC_API const char *  lc_whisper_token_text(lc_whisper *w, int32_t i, int32_t j);   // raw bytes, may be part of a UTF-8 character

// Silero VAD (whisper.cpp's ggml port), streaming: the LSTM state is kept between calls
LC_API lc_vad * lc_vad_load(const char *path_utf8, int32_t n_threads, char *err, int32_t err_len);
LC_API int32_t  lc_vad_window(lc_vad *v);   // samples per probability (512 at 16 kHz)
LC_API int32_t  lc_vad_process(lc_vad *v, const float *samples, int32_t n, float *probs, int32_t max_probs);   // n: whole windows
LC_API void     lc_vad_reset(lc_vad *v);
LC_API void     lc_vad_free(lc_vad *v);

#ifdef __cplusplus
}
#endif
