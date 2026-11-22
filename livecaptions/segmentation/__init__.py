"""Segmentation and streaming: endpointing, LocalAgreement-n, prompt building, text units."""
from livecaptions.segmentation.base import Line, PolicyOutput, StreamingPolicy


def create_policy(engine, cfg, terms_text=''):
    """Create the streaming policy selected in the settings."""
    if cfg.stream.policy == 'vad_chunk':
        from livecaptions.segmentation.vad_chunk import VadChunkPolicy
        return VadChunkPolicy(engine, cfg, terms_text)
    if cfg.stream.policy == 'sliding_window':
        from livecaptions.segmentation.sliding_window import SlidingWindowPolicy
        return SlidingWindowPolicy(engine, cfg, terms_text)
    from livecaptions.segmentation.local_agreement import LocalAgreementPolicy
    return LocalAgreementPolicy(engine, cfg, terms_text)
