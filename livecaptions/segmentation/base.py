"""Shared types of the streaming policies."""
from dataclasses import dataclass, field


@dataclass
class Line:
    """A finished caption line (D4): what writers save and SRT cues are made of."""
    text: str
    t0: float  # seconds into the session
    t1: float


@dataclass
class PolicyOutput:
    committed: str = ''              # newly final text; '\n' ends a line
    lines: list = field(default_factory=list)       # Lines that ended
    tentative: str | None = None     # the replaceable unfinished tail; None = unchanged
    runs: int = 0
    busy_s: float = 0.0              # inference time spent
    audio_s: float = 0.0             # audio transcribed by those runs
    latencies: list = field(default_factory=list)   # s from the end of a unit's audio to its commit
    language: str = ''               # language of the last run


class StreamingPolicy:
    """Feeds audio and VAD results; decides when to run the engine, what to commit and when to cut."""

    def process(self, chunk, vres, idle_s):
        """Take new audio (may be empty) and return a PolicyOutput."""
        raise NotImplementedError

    def flush(self, commit=True):
        """Pause or stop: transcribe what is left and make it final (or drop it if commit is False)."""
        raise NotImplementedError

    def set_terms(self, text):
        """Change the course terms; used from the next run on."""
        raise NotImplementedError
