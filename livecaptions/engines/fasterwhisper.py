"""faster-whisper (CTranslate2) backend; imported only when selected."""
import time

import numpy as np

from livecaptions.engines.base import ASREngine, EngineCaps, Piece, Segment, TranscribeResult, guess_can_translate


def get_cuda_count():
    """Return the number of CUDA devices CTranslate2 can use (0 without CUDA)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


class FasterWhisperEngine(ASREngine):
    name = 'fasterwhisper'

    def __init__(self):
        self.model = None
        self.model_path = ''
        self.load_key = None
        self.device = 'cpu'
        self.caps = EngineCaps()

    def load(self, model_path, recog_cfg):
        from faster_whisper import WhisperModel
        device = 'cuda' if recog_cfg.device == 'gpu' and get_cuda_count() > 0 else 'cpu'
        key = (model_path, device, recog_cfg.gpu_device, recog_cfg.compute_type, recog_cfg.threads)
        if self.model is not None and key == self.load_key:
            return
        self.unload()
        self.model = WhisperModel(model_path, device=device, device_index=recog_cfg.gpu_device if device == 'cuda' else 0,
                                  compute_type=recog_cfg.compute_type, cpu_threads=recog_cfg.threads, num_workers=1,
                                  local_files_only=True)
        self.model_path, self.load_key, self.device = model_path, key, device
        multi = bool(self.model.model.is_multilingual)
        self.caps = EngineCaps(gpu=device == 'cuda', can_translate=guess_can_translate(model_path, multi),
                               multilingual=multi, word_times=True, hotwords=True)

    def unload(self):
        if self.model is not None:
            del self.model
        self.model = None
        self.load_key = None

    def is_loaded(self):
        return self.model is not None

    def describe(self):
        return f'faster-whisper · {"CUDA" if self.device == "cuda" else "CPU"}'

    def tokenize(self, text):
        if not text:
            return []
        return list(self.model.hf_tokenizer.encode(text, add_special_tokens=False).ids)

    def transcribe(self, audio, language, task, prompt_tokens, recog_cfg, hotwords=''):
        rc = recog_cfg
        temps = [rc.temperature]
        if rc.temperature_inc > 0:
            temps = list(np.arange(rc.temperature, 1.0 + 1e-6, rc.temperature_inc))
        t_start = time.perf_counter()
        segs, info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32), language=language, task=task, beam_size=rc.beam_size, best_of=rc.best_of,
            patience=rc.patience, length_penalty=1.0 if rc.length_penalty < 0 else rc.length_penalty, temperature=temps,
            compression_ratio_threshold=rc.entropy_thold, log_prob_threshold=rc.logprob_thold,
            no_speech_threshold=rc.no_speech_thold, condition_on_previous_text=False,
            initial_prompt=list(prompt_tokens) or None, suppress_blank=rc.suppress_blank, word_timestamps=True,
            vad_filter=False, max_initial_timestamp=rc.max_initial_ts, hotwords=hotwords or None)
        segments = []
        for s in segs:  # the generator runs the decoder
            seg = Segment(s.start, s.end, no_speech_prob=s.no_speech_prob, avg_logprob=s.avg_logprob)
            for w in s.words or []:
                seg.pieces.append(Piece(w.word.encode('utf-8'), w.start, w.end, w.probability))
            if not seg.pieces and s.text:
                seg.pieces.append(Piece(s.text.encode('utf-8'), s.start, s.end))
            segments.append(seg)
        return TranscribeResult(segments, info.language or (language or ''), time.perf_counter() - t_start)
