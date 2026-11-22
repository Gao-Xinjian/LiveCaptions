// lc_native: miniaudio capture, whisper.cpp and Silero VAD behind a small C API for ctypes (see lc_native.h).
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define MINIAUDIO_IMPLEMENTATION   // miniaudio's code is compiled here; lc_internal.h sets its configuration
#include "lc_internal.h"

#include "ggml-backend.h"
#include "whisper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static_assert(sizeof(ma_device_id) <= LC_DEVICE_ID_BYTES, "LC_DEVICE_ID_BYTES is too small for ma_device_id");

namespace {

constexpr int kRate = WHISPER_SAMPLE_RATE;
static_assert(kRate == kLcRate, "capture and whisper.cpp must use the same sample rate");

std::atomic<int> g_log_min{GGML_LOG_LEVEL_WARN};
int              g_log_last = GGML_LOG_LEVEL_INFO;   // level of the line a CONT message continues
std::once_flag   g_log_once;

void LogCallback(ggml_log_level level, const char * text, void * /*user*/) {
    const int lv = level == GGML_LOG_LEVEL_CONT ? g_log_last : (int) level;
    g_log_last = lv;
    if (lv >= g_log_min.load()) fputs(text, stderr);
}

void EnsureLog() {
    std::call_once(g_log_once, [] { whisper_log_set(LogCallback, nullptr); });
}

void SetError(char * err, int32_t len, const std::string & msg) {
    if (err && len > 0) snprintf(err, (size_t) len, "%s", msg.c_str());
}

int64_t NowNs() { return LcNowNs(); }

// Opens a UTF-8 path for reading; whisper.cpp itself would use the ANSI code page on Windows.
FILE * OpenUtf8(const char * path) {
#ifdef _WIN32
    const int n = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    if (n <= 0) return nullptr;
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path, -1, w.data(), n);
    return _wfopen(w.c_str(), L"rb");
#else
    return fopen(path, "rb");
#endif
}

struct FileLoader {
    FILE *               file = nullptr;
    whisper_model_loader loader{};

    explicit FileLoader(FILE * f) : file(f) {
        loader.context = this;
        loader.read    = [](void * ctx, void * out, size_t n) { return fread(out, 1, n, static_cast<FileLoader *>(ctx)->file); };
        loader.eof     = [](void * ctx) { return feof(static_cast<FileLoader *>(ctx)->file) != 0; };
        loader.close   = [](void * ctx) { static_cast<FileLoader *>(ctx)->Close(); };
    }
    ~FileLoader() { Close(); }
    void Close() {
        if (file) fclose(file);
        file = nullptr;
    }
};

} // namespace

struct lc_whisper {
    whisper_context * ctx = nullptr;
    whisper_state *   st  = nullptr;
    std::atomic<int>  abort{0};
};

struct lc_vad {
    whisper_vad_context * ctx    = nullptr;
    int                   window = 512;
};

namespace {

void OnAudioData(ma_device * device, void * /*output*/, const void * input, ma_uint32 frames) {
    lc_audio * a = static_cast<lc_audio *>(device->pUserData);
    LcPushSamples(a, static_cast<const float *>(input), frames, a->channel_mode == 0 ? 1 : 2, a->channel_mode == 2 ? 1 : 0);
}

void OnAudioNotification(const ma_device_notification * n) {
    lc_audio * a = static_cast<lc_audio *>(n->pDevice->pUserData);
    if (n->type == ma_device_notification_type_stopped && !a->stopping) a->flags |= 1;
    if (n->type == ma_device_notification_type_rerouted) a->flags |= 2;
    a->cv.notify_one();
}

} // namespace

extern "C" {

int32_t lc_version(void) { return 2; }

int32_t lc_system_audio_available(void) {
#ifdef _WIN32
    return 1;   // WASAPI loopback
#elif defined(__APPLE__)
    return LcTapAvailable() ? 1 : 0;
#else
    return 0;
#endif
}

void lc_struct_sizes(int32_t * out, int32_t n) {
    const int32_t sizes[] = {
        (int32_t) sizeof(lc_device_info), (int32_t) sizeof(lc_audio_params), (int32_t) sizeof(lc_decode_params),
        (int32_t) sizeof(lc_token_info),  (int32_t) sizeof(lc_gpu_info),
    };
    for (int32_t i = 0; i < n && i < (int32_t) (sizeof(sizes) / sizeof(sizes[0])); i++) out[i] = sizes[i];
}

void lc_set_log_level(int32_t min_level) {
    EnsureLog();
    g_log_min = min_level;
}

const char * lc_system_info(void) {
    EnsureLog();
    return whisper_print_system_info();
}

int32_t lc_gpu_count(void) {
    EnsureLog();
    int32_t cnt = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) cnt++;
    }
    return cnt;
}

int32_t lc_gpu_get(int32_t index, lc_gpu_info * out) {
    EnsureLog();
    int32_t cnt = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev  = ggml_backend_dev_get(i);
        const auto         type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (cnt++ != index) continue;
        memset(out, 0, sizeof(*out));
        snprintf(out->name, sizeof(out->name), "%s", ggml_backend_dev_name(dev));
        snprintf(out->description, sizeof(out->description), "%s", ggml_backend_dev_description(dev));
        out->type = type == GGML_BACKEND_DEVICE_TYPE_GPU ? 1 : 2;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        out->memory_free  = (int64_t) free;
        out->memory_total = (int64_t) total;
        return 1;
    }
    return 0;
}

int32_t lc_audio_list(int32_t kind, lc_device_info * out, int32_t max) {
    int32_t count = 0;
    // On its own thread: WASAPI wants a multithreaded COM apartment, and the Qt thread is single-threaded.
    std::thread([&] {
        ma_context ctx;
        if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) return;
        ma_device_info * playback = nullptr;
        ma_device_info * capture  = nullptr;
        ma_uint32        n_play = 0, n_cap = 0;
        if (ma_context_get_devices(&ctx, &playback, &n_play, &capture, &n_cap) == MA_SUCCESS) {
            ma_device_info * list = kind == 0 ? capture : playback;
            const ma_uint32  n    = kind == 0 ? n_cap : n_play;
            for (ma_uint32 i = 0; i < n && count < max; i++, count++) {
                memset(&out[count], 0, sizeof(out[count]));
                snprintf(out[count].name, sizeof(out[count].name), "%s", list[i].name);
                memcpy(out[count].id, &list[i].id, sizeof(ma_device_id));
                out[count].is_default = list[i].isDefault ? 1 : 0;
            }
        }
        ma_context_uninit(&ctx);
    }).join();
    return count;
}

lc_audio * lc_audio_open(const lc_audio_params * p, char * err, int32_t err_len) {
    auto * a         = new lc_audio();
    a->channel_mode  = std::clamp((int) p->channel_mode, 0, 2);
    a->ring.assign((size_t) std::max(1000, (int) p->ring_ms) * kRate / 1000, 0.0f);
    a->last_ns       = NowNs();
#ifdef __APPLE__
    if (p->kind == 1) {   // system audio: miniaudio has no loopback on macOS, a process tap does it
        std::string msg;
        if (!LcTapOpen(a, p, msg)) {
            SetError(err, err_len, msg);
            delete a;
            return nullptr;
        }
        return a;
    }
#endif
    if (ma_context_init(nullptr, 0, nullptr, &a->ctx) != MA_SUCCESS) {
        SetError(err, err_len, "could not initialise the audio system");
        delete a;
        return nullptr;
    }
    a->ctx_ok = true;

    ma_device_config cfg = ma_device_config_init(p->kind == 1 ? ma_device_type_loopback : ma_device_type_capture);
    ma_device_id     id;
    if (p->device_id) {
        memcpy(&id, p->device_id, sizeof(id));
        cfg.capture.pDeviceID = &id;   // for loopback this is the playback device to listen to
    }
    cfg.capture.format                = ma_format_f32;
    cfg.capture.channels              = a->channel_mode == 0 ? 1 : 2;
    cfg.sampleRate                    = kRate;
    cfg.periodSizeInMilliseconds      = (ma_uint32) std::clamp((int) p->period_ms, 10, 500);
    cfg.resampling.linear.lpfOrder    = (ma_uint32) std::clamp((int) p->lpf_order, 0, (int) MA_MAX_FILTER_ORDER);
    cfg.dataCallback                  = OnAudioData;
    cfg.notificationCallback          = OnAudioNotification;
    cfg.pUserData                     = a;
    if (ma_device_init(&a->ctx, &cfg, &a->dev) != MA_SUCCESS) {
        SetError(err, err_len, p->kind == 1 ? "could not open the playback device for loopback capture" : "could not open the capture device");
        lc_audio_close(a);
        return nullptr;
    }
    a->dev_ok = true;
    return a;
}

int32_t lc_audio_start(lc_audio * a) {
    a->stopping = false;
    a->flags    = 0;
    a->last_ns  = NowNs();
#ifdef __APPLE__
    if (a->tap) return LcTapStart(a) ? 1 : 0;
#endif
    return ma_device_start(&a->dev) == MA_SUCCESS ? 1 : 0;
}

int32_t lc_audio_stop(lc_audio * a) {
    a->stopping = true;
    bool ok = true;
#ifdef __APPLE__
    if (a->tap) LcTapStop(a);
    else
#endif
        ok = ma_device_stop(&a->dev) == MA_SUCCESS;
    std::lock_guard<std::mutex> lock(a->mu);
    a->head = 0;
    a->size = 0;
    return ok ? 1 : 0;
}

int32_t lc_audio_read(lc_audio * a, float * dst, int32_t max, int32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(a->mu);
    a->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return a->size > 0 || a->flags.load() != 0; });
    const size_t n   = std::min(a->size, (size_t) std::max(0, (int) max));
    const size_t cap = a->ring.size();
    for (size_t i = 0; i < n; i++) dst[i] = a->ring[(a->head + i) % cap];
    a->head = (a->head + n) % cap;
    a->size -= n;
    return (int32_t) n;
}

double lc_audio_idle_seconds(lc_audio * a) { return (double) (NowNs() - a->last_ns.load()) * 1e-9; }

int64_t lc_audio_dropped(lc_audio * a) { return a->dropped.load(); }

int32_t lc_audio_flags(lc_audio * a) { return a->flags.load(); }

void lc_audio_close(lc_audio * a) {
    if (!a) return;
    a->stopping = true;
#ifdef __APPLE__
    if (a->tap) LcTapClose(a);
#endif
    if (a->dev_ok) ma_device_uninit(&a->dev);
    if (a->ctx_ok) ma_context_uninit(&a->ctx);
    delete a;
}

lc_whisper * lc_whisper_load(const char * path, int32_t use_gpu, int32_t gpu_device, int32_t flash_attn, char * err, int32_t err_len) {
    EnsureLog();
    FILE * f = OpenUtf8(path);
    if (!f) {
        SetError(err, err_len, std::string("could not open ") + path);
        return nullptr;
    }
    FileLoader             fl(f);
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu                = use_gpu != 0;
    cp.gpu_device             = gpu_device;
    cp.flash_attn             = flash_attn != 0;
    whisper_context * ctx     = whisper_init_with_params_no_state(&fl.loader, cp);
    if (!ctx) {
        SetError(err, err_len, "whisper.cpp could not load the model (wrong format or out of memory)");
        return nullptr;
    }
    whisper_state * st = whisper_init_state(ctx);
    if (!st) {
        whisper_free(ctx);
        SetError(err, err_len, "whisper.cpp could not allocate its state (out of memory?)");
        return nullptr;
    }
    auto * w = new lc_whisper();
    w->ctx   = ctx;
    w->st    = st;
    return w;
}

void lc_whisper_free(lc_whisper * w) {
    if (!w) return;
    if (w->st) whisper_free_state(w->st);
    if (w->ctx) whisper_free(w->ctx);
    delete w;
}

int32_t lc_whisper_full(lc_whisper * w, const lc_decode_params * d, const float * pcm, int32_t n) {
    const bool          beam = d->beam_size > 1;
    whisper_full_params p    = whisper_full_default_params(beam ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    p.n_threads              = std::max(1, (int) d->n_threads);
    p.translate              = d->translate != 0;
    p.no_context             = d->no_context != 0;
    p.single_segment         = d->single_segment != 0;
    p.token_timestamps       = d->token_timestamps != 0;
    p.max_len                = d->max_len;
    p.split_on_word          = d->split_on_word != 0;
    p.max_tokens             = d->max_tokens;
    p.audio_ctx              = d->audio_ctx;
    p.suppress_blank         = d->suppress_blank != 0;
    p.suppress_nst           = d->suppress_nst != 0;
    p.temperature            = d->temperature;
    p.temperature_inc        = d->temperature_inc;
    p.entropy_thold          = d->entropy_thold;
    p.logprob_thold          = d->logprob_thold;
    p.no_speech_thold        = d->no_speech_thold;
    p.length_penalty         = d->length_penalty;
    p.max_initial_ts         = d->max_initial_ts;
    p.thold_pt               = d->thold_pt;
    p.thold_ptsum            = d->thold_ptsum;
    p.suppress_regex         = d->suppress_regex;
    p.language               = (d->language && d->language[0]) ? d->language : "auto";
    p.detect_language        = false;
    p.prompt_tokens          = d->prompt_n_tokens > 0 ? (const whisper_token *) d->prompt_tokens : nullptr;
    p.prompt_n_tokens        = d->prompt_n_tokens;
    p.print_progress         = false;
    p.print_realtime         = false;
    p.print_special          = false;
    p.print_timestamps       = false;
    if (beam) p.beam_search.beam_size = d->beam_size;
    if (d->best_of > 0) p.greedy.best_of = d->best_of;
    p.abort_callback                   = [](void * ud) { return static_cast<lc_whisper *>(ud)->abort.load() != 0; };
    p.abort_callback_user_data         = w;
    p.encoder_begin_callback           = [](whisper_context *, whisper_state *, void * ud) { return static_cast<lc_whisper *>(ud)->abort.load() == 0; };
    p.encoder_begin_callback_user_data = w;
    return whisper_full_with_state(w->ctx, w->st, p, pcm, n);
}

void lc_whisper_abort(lc_whisper * w, int32_t flag) { w->abort = flag; }

int32_t lc_whisper_tokenize(lc_whisper * w, const char * text, int32_t * out, int32_t max) {
    return whisper_tokenize(w->ctx, text, (whisper_token *) out, max);
}

int32_t      lc_whisper_token_eot(lc_whisper * w) { return whisper_token_eot(w->ctx); }
int32_t      lc_whisper_is_multilingual(lc_whisper * w) { return whisper_is_multilingual(w->ctx); }
int32_t      lc_whisper_n_text_ctx(lc_whisper * w) { return whisper_model_n_text_ctx(w->ctx); }
const char * lc_whisper_model_type(lc_whisper * w) { return whisper_model_type_readable(w->ctx); }
int32_t      lc_whisper_lang_id(lc_whisper * w) { return whisper_full_lang_id_from_state(w->st); }
const char * lc_lang_str(int32_t id) { return whisper_lang_str(id); }
int32_t      lc_whisper_n_segments(lc_whisper * w) { return whisper_full_n_segments_from_state(w->st); }
int64_t      lc_whisper_segment_t0(lc_whisper * w, int32_t i) { return whisper_full_get_segment_t0_from_state(w->st, i); }
int64_t      lc_whisper_segment_t1(lc_whisper * w, int32_t i) { return whisper_full_get_segment_t1_from_state(w->st, i); }
float        lc_whisper_segment_no_speech_prob(lc_whisper * w, int32_t i) { return whisper_full_get_segment_no_speech_prob_from_state(w->st, i); }
int32_t      lc_whisper_n_tokens(lc_whisper * w, int32_t i) { return whisper_full_n_tokens_from_state(w->st, i); }

int32_t lc_whisper_token(lc_whisper * w, int32_t i, int32_t j, lc_token_info * out) {
    const whisper_token_data d = whisper_full_get_token_data_from_state(w->st, i, j);
    out->id   = d.id;
    out->p    = d.p;
    out->plog = d.plog;
    out->t0   = d.t0;
    out->t1   = d.t1;
    return 1;
}

const char * lc_whisper_token_text(lc_whisper * w, int32_t i, int32_t j) {
    return whisper_full_get_token_text_from_state(w->ctx, w->st, i, j);
}

lc_vad * lc_vad_load(const char * path, int32_t n_threads, char * err, int32_t err_len) {
    EnsureLog();
    FILE * f = OpenUtf8(path);
    if (!f) {
        SetError(err, err_len, std::string("could not open ") + path);
        return nullptr;
    }
    FileLoader                 fl(f);
    whisper_vad_context_params vp = whisper_vad_default_context_params();
    vp.n_threads                  = std::max(1, (int) n_threads);
    vp.use_gpu                    = false;   // a tiny LSTM: the CPU is cheaper than waking a GPU
    whisper_vad_context * ctx     = whisper_vad_init_with_params(&fl.loader, vp);
    if (!ctx) {
        SetError(err, err_len, "could not load the Silero VAD model (expected a ggml-silero-*.bin file)");
        return nullptr;
    }
    auto * v = new lc_vad();
    v->ctx   = ctx;
    // the API does not expose the window size: measure it, then forget the test audio
    std::vector<float> zeros(4096, 0.0f);
    if (whisper_vad_detect_speech_no_reset(ctx, zeros.data(), (int) zeros.size())) {
        const int n = whisper_vad_n_probs(ctx);
        if (n > 0 && 4096 % n == 0) v->window = 4096 / n;
    }
    whisper_vad_reset_state(ctx);
    return v;
}

int32_t lc_vad_window(lc_vad * v) { return v->window; }

int32_t lc_vad_process(lc_vad * v, const float * samples, int32_t n, float * probs, int32_t max_probs) {
    if (n <= 0 || !whisper_vad_detect_speech_no_reset(v->ctx, samples, n)) return 0;
    const int     np  = std::min(whisper_vad_n_probs(v->ctx), (int) max_probs);
    const float * src = whisper_vad_probs(v->ctx);
    std::copy(src, src + np, probs);
    return np;
}

void lc_vad_reset(lc_vad * v) { whisper_vad_reset_state(v->ctx); }

void lc_vad_free(lc_vad * v) {
    if (!v) return;
    if (v->ctx) whisper_vad_free(v->ctx);
    delete v;
}

} // extern "C"
