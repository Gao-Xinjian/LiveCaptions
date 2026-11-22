"""whisper.cpp backend through lc_native."""
import ctypes as C
import time

import numpy as np

from livecaptions import SAMPLE_RATE, native
from livecaptions.engines.base import ASREngine, EngineCaps, Piece, Segment, TranscribeResult, guess_can_translate


def get_audio_ctx(recog_cfg, n_samples):
    """Return the encoder window (frames) for a buffer: 0 = full 30 s, or a smaller window if configured."""
    if recog_cfg.audio_ctx_mode == 'fixed':
        return int(recog_cfg.audio_ctx)
    if recog_cfg.audio_ctx_mode == 'auto':
        # 50 frames per second plus a margin, rounded up to steps of 256 frames (~5 s): every new size costs a
        # one-time setup on the GPU, so few sizes are cheaper overall
        frames = int(np.ceil(n_samples / SAMPLE_RATE * 50)) + 32
        return int(min(1500, (frames + 255) // 256 * 256))
    return 0


class WhisperCppEngine(ASREngine):
    name = 'whispercpp'

    def __init__(self):
        self.lib = native.load_lib()
        self.handle = None
        self.model_path = ''
        self.load_key = None
        self.gpu_name = ''
        self.caps = EngineCaps()
        self._eot = 0

    def load(self, model_path, recog_cfg):
        use_gpu = recog_cfg.device == 'gpu'
        key = (model_path, use_gpu, recog_cfg.gpu_device, recog_cfg.flash_attn)
        if self.handle and key == self.load_key:
            return
        self.unload()
        err = native.make_err_buf()
        h = self.lib.lc_whisper_load(model_path.encode('utf-8'), int(use_gpu), recog_cfg.gpu_device,
                                     int(recog_cfg.flash_attn), err, len(err))
        if not h:
            raise native.NativeError(native.err2str(err))
        self.handle, self.model_path, self.load_key = h, model_path, key
        self._eot = self.lib.lc_whisper_token_eot(h)
        gpus = dict((i, d) for i, d, _ in native.list_gpus()) if use_gpu else {}
        self.gpu_name = gpus.get(recog_cfg.gpu_device, '')
        multi = bool(self.lib.lc_whisper_is_multilingual(h))
        self.caps = EngineCaps(gpu=bool(gpus), can_translate=guess_can_translate(model_path, multi),
                               multilingual=multi, word_times=False, hotwords=False)
        if use_gpu and gpus:  # the first GPU run sets up its pipelines; do it now, not on the first sentence
            self.transcribe(np.zeros(SAMPLE_RATE, np.float32), 'en', 'transcribe', [], recog_cfg)

    def unload(self):
        if self.handle:
            self.lib.lc_whisper_free(self.handle)
        self.handle = None
        self.load_key = None

    def is_loaded(self):
        return self.handle is not None

    def describe(self):
        return f'whisper.cpp · {"GPU: " + self.gpu_name if self.gpu_name else "CPU"}'

    def tokenize(self, text):
        if not text:
            return []
        data = text.encode('utf-8')
        buf = (C.c_int32 * 1024)()
        n = self.lib.lc_whisper_tokenize(self.handle, data, buf, 1024)
        if n < 0:
            buf = (C.c_int32 * -n)()
            n = self.lib.lc_whisper_tokenize(self.handle, data, buf, -n)
        return list(buf[:max(0, n)])

    def abort(self):
        if self.handle:
            self.lib.lc_whisper_abort(self.handle, 1)

    def transcribe(self, audio, language, task, prompt_tokens, recog_cfg, hotwords=''):
        lib, h, rc = self.lib, self.handle, recog_cfg
        lib.lc_whisper_abort(h, 0)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        prompt = (C.c_int32 * max(1, len(prompt_tokens)))(*prompt_tokens)
        p = native.DecodeParams(
            n_threads=rc.threads, translate=int(task == 'translate'), no_context=1, single_segment=int(rc.single_segment),
            token_timestamps=int(rc.token_timestamps), max_len=rc.max_len, split_on_word=int(rc.split_on_word),
            max_tokens=rc.max_tokens, audio_ctx=get_audio_ctx(rc, len(audio)), beam_size=rc.beam_size, best_of=rc.best_of,
            suppress_blank=int(rc.suppress_blank), suppress_nst=int(rc.suppress_nst), temperature=rc.temperature,
            temperature_inc=rc.temperature_inc, entropy_thold=rc.entropy_thold, logprob_thold=rc.logprob_thold,
            no_speech_thold=rc.no_speech_thold, length_penalty=rc.length_penalty, max_initial_ts=rc.max_initial_ts,
            thold_pt=rc.thold_pt, thold_ptsum=rc.thold_ptsum, language=(language or 'auto').encode('ascii'),
            prompt_tokens=C.cast(prompt, C.POINTER(C.c_int32)), prompt_n_tokens=len(prompt_tokens),
            suppress_regex=rc.suppress_regex.encode('utf-8') if rc.suppress_regex else None)
        t_start = time.perf_counter()
        ret = lib.lc_whisper_full(h, C.byref(p), audio.ctypes.data_as(C.POINTER(C.c_float)), len(audio))
        seconds = time.perf_counter() - t_start
        if ret != 0:
            return TranscribeResult([], language or '', seconds)
        # Collect segments and their text tokens (timestamps and other special tokens have ids >= eot).
        segments = []
        tok = native.TokenInfo()
        for i in range(lib.lc_whisper_n_segments(h)):
            s0, s1 = lib.lc_whisper_segment_t0(h, i) * 0.01, lib.lc_whisper_segment_t1(h, i) * 0.01
            seg = Segment(s0, s1, no_speech_prob=lib.lc_whisper_segment_no_speech_prob(h, i))
            plogs = []
            for j in range(lib.lc_whisper_n_tokens(h, i)):
                lib.lc_whisper_token(h, i, j, C.byref(tok))
                if tok.id >= self._eot:
                    continue
                t0, t1 = tok.t0 * 0.01, tok.t1 * 0.01
                if tok.t0 < 0 or tok.t1 < tok.t0:  # no token timestamp: use the segment's
                    t0, t1 = s0, s1
                seg.pieces.append(Piece(lib.lc_whisper_token_text(h, i, j) or b'', t0, t1, tok.p))
                plogs.append(tok.plog)
            seg.avg_logprob = float(np.mean(plogs)) if plogs else 0.0
            segments.append(seg)
        lang = lib.lc_lang_str(lib.lc_whisper_lang_id(h))
        return TranscribeResult(segments, lang.decode('ascii') if lang else (language or ''), seconds)
