// System audio on macOS 14.2+: a Core Audio process tap inside a private aggregate device (PLAN.md §10).
// The IO block converts the tap's stream to 16 kHz with miniaudio's converter and fills the same ring buffer as
// miniaudio capture, so Python reads both the same way. Compiled with ARC (-fobjc-arc).
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <CoreAudio/CoreAudio.h>
#import <Foundation/Foundation.h>

#include "lc_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct LcTap {
    AudioObjectID       tap   = kAudioObjectUnknown;
    AudioObjectID       agg   = kAudioObjectUnknown;
    AudioDeviceIOProcID proc  = nullptr;
    dispatch_queue_t    queue = nil;
    ma_data_converter   conv{};
    bool                conv_ok     = false;
    bool                running     = false;
    UInt32              channels    = 2;      // of the tap stream
    bool                interleaved = true;
    int                 out_channels = 1;     // 1 = mixed to mono, 2 = keep both and pick one
    int                 pick         = 0;
    std::vector<float>  in_buf;               // interleaving buffer for non-interleaved streams
    std::vector<float>  out_buf;              // converted 16 kHz audio
    int64_t             start_ns  = 0;
    bool                got_audio = false;    // the first IO cycle has been logged
};

AudioObjectPropertyAddress MakeAddr(AudioObjectPropertySelector sel) {
    return {sel, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
}

std::string MakeStatusText(const char * what, OSStatus st) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s failed (OSStatus %d)", what, (int) st);
    return buf;
}

NSString * GetStringProperty(AudioObjectID obj, AudioObjectPropertySelector sel) {
    AudioObjectPropertyAddress addr = MakeAddr(sel);
    CFStringRef                s    = nullptr;
    UInt32                     size = sizeof(s);
    if (AudioObjectGetPropertyData(obj, &addr, 0, nullptr, &size, &s) != noErr || !s) return nil;
    return (__bridge_transfer NSString *) s;
}

// Splits "com.google.Chrome, us.zoom.xos" into bundle IDs.
std::vector<std::string> SplitApps(const char * text) {
    std::vector<std::string> out;
    std::string              cur;
    for (const char * c = text ? text : ""; ; c++) {
        if (*c == '\0' || *c == ',' || *c == ';' || std::isspace((unsigned char) *c)) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*c == '\0') break;
        } else {
            cur += *c;
        }
    }
    return out;
}

// Returns the Core Audio process objects of running apps whose bundle ID is listed.
API_AVAILABLE(macos(14.2)) NSArray<NSNumber *> * FindProcesses(const std::vector<std::string> & ids) {
    NSMutableArray<NSNumber *> * out  = [NSMutableArray array];
    AudioObjectPropertyAddress   addr = MakeAddr(kAudioHardwarePropertyProcessObjectList);
    UInt32                       size = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, nullptr, &size) != noErr) return out;
    std::vector<AudioObjectID> procs(size / sizeof(AudioObjectID));
    if (procs.empty() || AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr, &size, procs.data()) != noErr) {
        return out;
    }
    for (AudioObjectID p : procs) {
        NSString * bundle = GetStringProperty(p, kAudioProcessPropertyBundleID);
        if (!bundle) continue;
        for (const std::string & id : ids) {
            if ([bundle caseInsensitiveCompare:[NSString stringWithUTF8String:id.c_str()]] == NSOrderedSame) {
                [out addObject:@(p)];
                break;
            }
        }
    }
    return out;
}

// Called on the tap's queue for every IO cycle: convert to 16 kHz and push into the ring buffer.
void OnTapData(lc_audio * a, LcTap * t, const AudioBufferList * list) {
    if (!t->got_audio) {   // once: shows whether audio reaches us at all
        t->got_audio = true;
        LcLog(kLcLogInfo, "lc_tap: first IO cycle %.0f ms after start, %u buffers, %u bytes\n", GetMsSince(t->start_ns),
              list ? (unsigned) list->mNumberBuffers : 0u, list && list->mNumberBuffers ? (unsigned) list->mBuffers[0].mDataByteSize : 0u);
    }
    if (!list || list->mNumberBuffers == 0 || !list->mBuffers[0].mData) return;
    const UInt32  ch = t->channels;
    UInt32        frames;
    const float * src;
    if (t->interleaved) {
        frames = list->mBuffers[0].mDataByteSize / (UInt32) (sizeof(float) * ch);
        src    = static_cast<const float *>(list->mBuffers[0].mData);
    } else {
        frames = list->mBuffers[0].mDataByteSize / (UInt32) sizeof(float);
        if (t->in_buf.size() < (size_t) frames * ch) t->in_buf.resize((size_t) frames * ch);
        for (UInt32 c = 0; c < ch; c++) {
            const float * b = c < list->mNumberBuffers ? static_cast<const float *>(list->mBuffers[c].mData) : nullptr;
            for (UInt32 i = 0; i < frames; i++) t->in_buf[(size_t) i * ch + c] = b ? b[i] : 0.0f;
        }
        src = t->in_buf.data();
    }
    ma_uint64       left    = frames;
    const ma_uint64 out_cap = t->out_buf.size() / (size_t) t->out_channels;
    while (left > 0) {
        ma_uint64 n_in = left, n_out = out_cap;
        if (ma_data_converter_process_pcm_frames(&t->conv, src, &n_in, t->out_buf.data(), &n_out) != MA_SUCCESS) break;
        LcPushSamples(a, t->out_buf.data(), n_out, t->out_channels, t->pick);
        src += n_in * ch;
        left -= n_in;
        if (n_in == 0 && n_out == 0) break;
    }
}

} // namespace

bool LcTapAvailable() {
    if (@available(macOS 14.2, *)) return true;
    return false;
}

bool LcTapOpen(lc_audio * a, const lc_audio_params * p, std::string & err) {
    if (@available(macOS 14.2, *)) {
        @autoreleasepool {
            int64_t t0      = LcNowNs();   // each step is timed: on CI one of them takes minutes
            auto *  t       = new LcTap();
            a->tap          = t;
            t->out_channels = a->channel_mode == 0 ? 1 : 2;
            t->pick         = a->channel_mode == 2 ? 1 : 0;

            // The tap: all system audio, or only the listed apps (those running now).
            const std::vector<std::string> apps = SplitApps(p->tap_apps);
            CATapDescription *             desc = nil;
            if (apps.empty()) {
                desc = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:@[]];
            } else {
                NSArray<NSNumber *> * procs = FindProcesses(apps);
                LcLog(kLcLogInfo, "lc_tap: %u of the listed apps running (%.0f ms)\n", (unsigned) procs.count, GetMsSince(t0));
                t0 = LcNowNs();
                if (procs.count == 0) {
                    err = "none of the listed apps is running";
                    LcTapClose(a);
                    return false;
                }
                desc = [[CATapDescription alloc] initStereoMixdownOfProcesses:procs];
            }
            desc.name   = @"LiveCaptions";
            OSStatus st = AudioHardwareCreateProcessTap(desc, &t->tap);
            LcLog(kLcLogInfo, "lc_tap: process tap created, status %d (%.0f ms)\n", (int) st, GetMsSince(t0));
            t0 = LcNowNs();
            if (st != noErr) {
                err = MakeStatusText("Creating the process tap (allow audio recording in System Settings > Privacy & Security)", st);
                LcTapClose(a);
                return false;
            }

            // The tap's stream format: Float32 at the output device's rate.
            AudioStreamBasicDescription asbd{};
            AudioObjectPropertyAddress  faddr = MakeAddr(kAudioTapPropertyFormat);
            UInt32                      size  = sizeof(asbd);
            st = AudioObjectGetPropertyData(t->tap, &faddr, 0, nullptr, &size, &asbd);
            if (st != noErr || asbd.mFormatID != kAudioFormatLinearPCM || !(asbd.mFormatFlags & kAudioFormatFlagIsFloat) ||
                asbd.mBitsPerChannel != 32 || asbd.mSampleRate <= 0) {
                err = st != noErr ? MakeStatusText("Reading the tap format", st) : "the tap delivers an unexpected audio format";
                LcTapClose(a);
                return false;
            }
            t->channels    = std::max<UInt32>(1, asbd.mChannelsPerFrame);
            t->interleaved = !(asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved);
            LcLog(kLcLogInfo, "lc_tap: tap format %.0f Hz, %u channels, %s (%.0f ms)\n", asbd.mSampleRate, (unsigned) t->channels,
                  t->interleaved ? "interleaved" : "non-interleaved", GetMsSince(t0));
            t0 = LcNowNs();

            // A private aggregate device holding the tap, clocked by the default output device.
            AudioObjectID              out_dev = kAudioObjectUnknown;
            AudioObjectPropertyAddress daddr   = MakeAddr(kAudioHardwarePropertyDefaultSystemOutputDevice);
            size = sizeof(out_dev);
            st   = AudioObjectGetPropertyData(kAudioObjectSystemObject, &daddr, 0, nullptr, &size, &out_dev);
            NSString * out_uid = st == noErr ? GetStringProperty(out_dev, kAudioDevicePropertyDeviceUID) : nil;
            if (!out_uid) {
                err = "no default output device";
                LcTapClose(a);
                return false;
            }
            NSDictionary * agg_desc = @{
                @kAudioAggregateDeviceNameKey : @"LiveCaptions tap",
                @kAudioAggregateDeviceUIDKey : [NSUUID UUID].UUIDString,
                @kAudioAggregateDeviceMainSubDeviceKey : out_uid,
                @kAudioAggregateDeviceIsPrivateKey : @YES,
                @kAudioAggregateDeviceIsStackedKey : @NO,
                @kAudioAggregateDeviceTapAutoStartKey : @YES,
                @kAudioAggregateDeviceSubDeviceListKey : @[ @{@kAudioSubDeviceUIDKey : out_uid} ],
                @kAudioAggregateDeviceTapListKey : @[ @{@kAudioSubTapDriftCompensationKey : @YES, @kAudioSubTapUIDKey : desc.UUID.UUIDString} ],
            };
            st = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef) agg_desc, &t->agg);
            LcLog(kLcLogInfo, "lc_tap: aggregate device created, status %d (%.0f ms)\n", (int) st, GetMsSince(t0));
            t0 = LcNowNs();
            if (st != noErr) {
                err = MakeStatusText("Creating the aggregate device for the tap", st);
                LcTapClose(a);
                return false;
            }

            // Conversion to 16 kHz (mono, or both channels when one is picked).
            ma_data_converter_config cc = ma_data_converter_config_init(
                ma_format_f32, ma_format_f32, t->channels, (ma_uint32) t->out_channels, (ma_uint32) asbd.mSampleRate, kLcRate);
            cc.resampling.linear.lpfOrder = (ma_uint32) std::clamp((int) p->lpf_order, 0, (int) MA_MAX_FILTER_ORDER);
            if (ma_data_converter_init(&cc, nullptr, &t->conv) != MA_SUCCESS) {
                err = "could not set up the conversion to 16 kHz";
                LcTapClose(a);
                return false;
            }
            t->conv_ok = true;
            t->in_buf.resize((size_t) 8192 * t->channels);
            t->out_buf.resize((size_t) 8192 * t->out_channels);

            // The IO block runs on a serial queue and only converts and copies.
            t->queue = dispatch_queue_create("livecaptions.tap",
                                             dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0));
            st = AudioDeviceCreateIOProcIDWithBlock(&t->proc, t->agg, t->queue,
                                                    ^(const AudioTimeStamp * now, const AudioBufferList * in,
                                                      const AudioTimeStamp * in_time, AudioBufferList * out,
                                                      const AudioTimeStamp * out_time) {
                                                        (void) now, (void) in_time, (void) out, (void) out_time;
                                                        OnTapData(a, t, in);
                                                    });
            LcLog(kLcLogInfo, "lc_tap: IO callback set up, status %d (%.0f ms)\n", (int) st, GetMsSince(t0));
            if (st != noErr) {
                err = MakeStatusText("Creating the tap's IO callback", st);
                LcTapClose(a);
                return false;
            }
            return true;
        }
    }
    err = "system audio capture needs macOS 14.2 or later";
    return false;
}

bool LcTapStart(lc_audio * a) {
    auto * t = static_cast<LcTap *>(a->tap);
    if (!t || !t->proc) return false;
    t->start_ns       = LcNowNs();
    const OSStatus st = AudioDeviceStart(t->agg, t->proc);
    LcLog(kLcLogInfo, "lc_tap: started, status %d (%.0f ms)\n", (int) st, GetMsSince(t->start_ns));
    t->running = st == noErr;
    return t->running;
}

void LcTapStop(lc_audio * a) {
    auto * t = static_cast<LcTap *>(a->tap);
    if (t && t->running) {
        const int64_t t0 = LcNowNs();
        AudioDeviceStop(t->agg, t->proc);
        t->running = false;
        LcLog(kLcLogInfo, "lc_tap: stopped (%.0f ms)\n", GetMsSince(t0));
    }
}

void LcTapClose(lc_audio * a) {
    auto * t = static_cast<LcTap *>(a->tap);
    if (!t) return;
    LcTapStop(a);
    if (@available(macOS 14.2, *)) {
        const int64_t t0 = LcNowNs();
        if (t->proc) AudioDeviceDestroyIOProcID(t->agg, t->proc);
        const double ms_proc = GetMsSince(t0);
        if (t->agg != kAudioObjectUnknown) AudioHardwareDestroyAggregateDevice(t->agg);
        const double ms_agg = GetMsSince(t0) - ms_proc;
        if (t->tap != kAudioObjectUnknown) AudioHardwareDestroyProcessTap(t->tap);
        LcLog(kLcLogInfo, "lc_tap: closed (IO callback %.0f ms, aggregate device %.0f ms, tap %.0f ms)\n", ms_proc, ms_agg,
              GetMsSince(t0) - ms_proc - ms_agg);
    }
    if (t->conv_ok) ma_data_converter_uninit(&t->conv, nullptr);
    t->queue = nil;
    delete t;
    a->tap = nullptr;
}
