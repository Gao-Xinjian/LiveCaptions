"""Backend-neutral ASR engine interface (PLAN.md §4.2)."""
import os
from dataclasses import dataclass, field


@dataclass
class Piece:
    data: bytes          # UTF-8 bytes; a whisper.cpp token may hold only part of a character
    t0: float            # seconds from the start of the transcribed audio
    t1: float
    p: float = 1.0       # probability


@dataclass
class Segment:
    t0: float
    t1: float
    pieces: list = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    @property
    def text(self):
        return b''.join(p.data for p in self.pieces).decode('utf-8', errors='ignore')


@dataclass
class TranscribeResult:
    segments: list
    language: str = ''   # detected or given language code
    seconds: float = 0.0  # wall time the run took


@dataclass
class EngineCaps:
    gpu: bool = False            # can run on a GPU in this build
    can_translate: bool = False  # the loaded model can translate to English
    multilingual: bool = True
    word_times: bool = False     # pieces are words with good timestamps
    hotwords: bool = False


class ASREngine:
    """Base class of the backends; transcribe() is called from the pipeline thread only."""
    name = ''
    caps = EngineCaps()

    def load(self, model_path, recog_cfg):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def is_loaded(self):
        raise NotImplementedError

    def tokenize(self, text):
        """Return the token ids of text (for the prompt)."""
        raise NotImplementedError

    def transcribe(self, audio, language, task, prompt_tokens, recog_cfg, hotwords=''):
        """
        Transcribe one buffer.

        Args:
            audio: float32 16 kHz mono, at most 30 s
            language: language code, or None to detect
            task: 'transcribe' or 'translate'
            prompt_tokens: token ids given to the decoder as earlier text
            recog_cfg: RecogCfg with the decoding options
            hotwords: faster-whisper hotwords ('' = none)

        Returns:
            TranscribeResult
        """
        raise NotImplementedError

    def abort(self):
        """Ask a running transcribe() to stop early (called from another thread)."""

    def describe(self):
        """Return a short text for the status line, e.g. the device in use."""
        return self.name


def get_threads(n_threads):
    """Return the CPU threads to use: the setting, at most the logical cores (extra threads only wait on the others)."""
    n_threads = max(1, int(n_threads))
    return min(n_threads, os.cpu_count() or n_threads)


def guess_can_translate(model_path, multilingual=True):
    """Return whether a model can translate, from its name: turbo, distil and English-only models cannot."""
    name = os.path.basename(os.path.normpath(model_path)).lower()
    if not multilingual or '.en' in name or name.endswith('-en') or 'distil' in name or 'turbo' in name:
        return False
    return True
