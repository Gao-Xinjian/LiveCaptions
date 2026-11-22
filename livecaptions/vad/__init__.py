"""Voice activity detection: adaptive energy, Silero (via whisper.cpp), TEN VAD, or off."""
from livecaptions.vad.base import NoVAD, VadResult  # noqa: F401
from livecaptions.vad.energy import EnergyVAD
from livecaptions.vad.silero import CascadeVAD, SileroVAD, find_silero_model
from livecaptions.vad.ten import TenVAD, find_ten_library


def create_vad(vad_cfg, models_dir):
    """Create the VAD selected in the settings; raises NativeError or FileNotFoundError if a model is missing."""
    if vad_cfg.method == 'off':
        return NoVAD()
    if vad_cfg.method == 'energy':
        return EnergyVAD(vad_cfg)
    if vad_cfg.method == 'ten':
        path = find_ten_library(vad_cfg, models_dir)
        if not path:
            raise FileNotFoundError('TEN VAD library not found: download it in Model, or choose another VAD')
        neural = TenVAD(path, int(vad_cfg.ten_hop), vad_cfg)
    else:
        path = find_silero_model(vad_cfg, models_dir)
        if not path:
            raise FileNotFoundError('Silero VAD model not found: download it in Model, or choose another VAD')
        neural = SileroVAD(path, vad_cfg.silero_threads, vad_cfg)
    return CascadeVAD(neural, vad_cfg) if vad_cfg.cascade else neural
