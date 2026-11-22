"""ASR backends behind one interface: whisper.cpp and faster-whisper."""
from livecaptions.engines.base import ASREngine, EngineCaps, Piece, Segment, TranscribeResult  # noqa: F401


def create_engine(backend):
    """Create the engine for a backend name ('whispercpp' or 'fasterwhisper')."""
    if backend == 'fasterwhisper':
        from livecaptions.engines.fasterwhisper import FasterWhisperEngine
        return FasterWhisperEngine()
    from livecaptions.engines.whispercpp import WhisperCppEngine
    return WhisperCppEngine()


def is_backend_available(backend):
    """Return whether a backend's library can be imported/loaded."""
    if backend == 'fasterwhisper':
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False
    from livecaptions import native
    return native.is_available()
