"""
Known downloadable models (registry.json). Hugging Face entries were generated from the repos' file lists;
GitHub entries (TEN VAD) are pinned to a commit, with one file per OS and its Git blob hash for verification.
"""
import json
import os
import sys
from dataclasses import dataclass, field

REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'registry.json')


@dataclass
class RegistryEntry:
    id: str
    backend: str        # 'whispercpp', 'fasterwhisper' or 'vad'
    repo: str           # Hugging Face or GitHub repo
    files: list = field(default_factory=list)  # files to fetch; empty = the whole repo (faster-whisper)
    size: int = 0       # bytes
    quant: str = ''
    multilingual: bool = True
    can_translate: bool = True
    family: str = ''
    source: str = 'hf'  # 'hf' (huggingface_hub) or 'github' (one raw file at a pinned commit)
    commit: str = ''
    platforms: dict = field(default_factory=dict)  # GitHub: OS -> {path, name, size, blob}
    license: str = ''
    github: dict = field(default_factory=dict)     # GitHub: the file of this OS

    def get_ram_estimate(self):
        """Return a rough RAM/VRAM need in bytes: weights plus compute buffers."""
        return int(self.size * 1.15 + 150e6)


def get_platform_key():
    """Return 'win32', 'darwin' or 'linux'."""
    return sys.platform if sys.platform in ('win32', 'darwin') else 'linux'


def load_registry():
    """Return the registry entries usable on this OS."""
    with open(REGISTRY_FILE, encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for m in data['models']:
        e = RegistryEntry(**m)
        if e.source == 'github':
            plat = e.platforms.get(get_platform_key())
            if not plat:
                continue  # no build for this OS
            e.github, e.files, e.size = plat, [plat['name']], plat['size']
        out.append(e)
    return out


def find_entry(model_id):
    """Return the registry entry with this id, or None."""
    return next((e for e in load_registry() if e.id == model_id), None)
