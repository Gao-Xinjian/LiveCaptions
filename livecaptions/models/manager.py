"""Find, download (huggingface_hub, or a pinned GitHub file) and delete models in the models folder."""
import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass

from livecaptions.models.registry import load_registry

GGML_MAGIC = b'lmgg'  # 0x67676d6c little-endian, used by whisper.cpp and its Silero VAD files
CT2_WEIGHTS = 'model.bin'
TEN_LIB_NAMES = ('ten_vad.dll', 'libten_vad.dylib', 'libten_vad.so')


class DownloadCancelled(Exception):
    pass


@dataclass
class InstalledModel:
    path: str
    backend: str   # 'whispercpp', 'fasterwhisper' or 'vad'
    name: str
    size: int      # bytes on disk


def detect_format(path):
    """
    Detect what a model file or folder is.

    Returns:
        'whispercpp', 'vad', 'fasterwhisper', 'pt' (openai-whisper, not supported yet) or None
    """
    if os.path.isdir(path):
        names = set(os.listdir(path))
        has_vocab = 'tokenizer.json' in names or any(n.startswith('vocabulary') for n in names)
        if CT2_WEIGHTS in names and 'config.json' in names and has_vocab:
            return 'fasterwhisper'
        return None
    if not os.path.isfile(path):
        return None
    if path.lower().endswith('.pt'):
        return 'pt'
    if os.path.basename(path).lower() in TEN_LIB_NAMES:
        return 'vad'  # the TEN VAD library
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
    except OSError:
        return None
    if magic != GGML_MAGIC:
        return None
    return 'vad' if 'silero' in os.path.basename(path).lower() else 'whispercpp'


def get_size(path):
    """Return the size of a file or a folder in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def list_installed(models_dir):
    """Return the models found in the models folder (top level files and folders)."""
    if not os.path.isdir(models_dir):
        return []
    out = []
    for name in sorted(os.listdir(models_dir)):
        if name.startswith('.'):
            continue  # .cache of huggingface_hub
        path = os.path.join(models_dir, name)
        fmt = detect_format(path)
        if fmt in ('whispercpp', 'fasterwhisper', 'vad'):
            out.append(InstalledModel(path, fmt, name, get_size(path)))
    return out


def get_target_path(entry, models_dir):
    """Return where a registry entry is (or will be) installed."""
    if entry.files and len(entry.files) == 1:
        return os.path.join(models_dir, entry.files[0])
    return os.path.join(models_dir, entry.id)


def is_installed(entry, models_dir):
    path = get_target_path(entry, models_dir)
    return detect_format(path) is not None


def _make_tqdm(on_bytes, cancel_event):
    """Return a tqdm class that reports downloaded bytes and raises DownloadCancelled when asked to stop."""
    from tqdm.auto import tqdm

    class ProgressTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs['disable'] = True  # no console output; update() below still runs
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled()
            if n:
                on_bytes(int(n))
            return super().update(n)

    return ProgressTqdm


def download(entry, models_dir, endpoint='', proxy='', on_progress=None, cancel_event=None):
    """
    Download a registry entry into the models folder. Partial downloads resume next time.

    Args:
        entry: RegistryEntry
        models_dir: models folder
        endpoint: Hugging Face endpoint or mirror ('' = default)
        proxy: http(s) proxy URL ('' = system settings)
        on_progress: called with (bytes done, bytes total)
        cancel_event: threading.Event; set it to stop (raises DownloadCancelled)

    Returns:
        the installed path
    """
    os.makedirs(models_dir, exist_ok=True)
    if proxy:
        os.environ['HTTPS_PROXY'] = os.environ['HTTP_PROXY'] = proxy
    done = [0]

    def on_bytes(n):
        done[0] += n
        if on_progress:
            on_progress(done[0], entry.size)

    if entry.source == 'github':
        return _download_github(entry, models_dir, on_bytes, cancel_event)
    from huggingface_hub import HfApi, hf_hub_download
    tqdm_class = _make_tqdm(on_bytes, cancel_event)
    target = get_target_path(entry, models_dir)
    files = entry.files
    local_dir = models_dir
    if not files:  # a whole repo (faster-whisper) goes into its own folder
        files = [f for f in HfApi(endpoint=endpoint or None).list_repo_files(entry.repo)
                 if not f.startswith('.') and f.lower() != 'readme.md']
        local_dir = target
    for fn in files:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled()
        hf_hub_download(entry.repo, fn, local_dir=local_dir, endpoint=endpoint or None, tqdm_class=tqdm_class)
    if detect_format(target) is None:
        raise RuntimeError(f'downloaded files in {target} are not a usable model')
    return target


def calculate_git_blob_sha1(data):
    """Return the Git blob hash of file contents (what GitHub lists for each file)."""
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def _download_github(entry, models_dir, on_bytes, cancel_event):
    """Download one file of a GitHub repo at the pinned commit and check it against its Git blob hash."""
    plat = entry.github
    url = f'https://raw.githubusercontent.com/{entry.repo}/{entry.commit}/{plat["path"]}'
    target = os.path.join(models_dir, plat['name'])
    part = target + '.part'
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(part, 'wb') as f:  # proxy from HTTPS_PROXY
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled()
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                on_bytes(len(chunk))
        with open(part, 'rb') as f:
            blob = calculate_git_blob_sha1(f.read())
        if blob != plat['blob']:
            raise RuntimeError(f'{plat["name"]} does not match the expected file (hash {blob})')
        os.replace(part, target)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return target


def delete_model(path):
    """Delete an installed model file or folder."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def get_registry(backend=None):
    """Return the registry entries, optionally of one backend."""
    return [e for e in load_registry() if backend is None or e.backend == backend]
