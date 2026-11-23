"""
`LiveCaptions --self-check`: check an install (above all a packaged one) without using it: the native library, both
backends, the bundled data, optionally one transcription and the windows built offscreen. Exit code 0 when all pass.
"""
import argparse
import os
import platform
import sys
import tempfile
import time
import traceback

from livecaptions import APP_NAME, __version__


class Report:
    """Collects check lines; printed as they come (if there is a console) and written to a file at the end."""

    def __init__(self, path):
        self.path = path
        self.lines = []
        self.failed = 0

    def add(self, name, ok, detail=''):
        line = f'{"ok  " if ok else "FAIL"} {name}' + (f': {detail}' if detail else '')
        self.lines.append(line)
        self.failed += 0 if ok else 1
        if sys.stdout is not None:
            print(line, flush=True)

    def run(self, name, fn):
        """Run fn() -> detail text; a raised exception marks the check as failed."""
        try:
            self.add(name, True, fn())
        except Exception as e:
            self.add(name, False, f'{type(e).__name__}: {e}')
            self.lines.append(traceback.format_exc().rstrip())

    def save(self):
        if self.path:
            with open(self.path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.lines) + '\n')


def check_native():
    from livecaptions import native
    lib = native.load_lib()   # also checks the struct layouts
    gpus = ', '.join(name for _, name, _ in native.list_gpus()) or 'none'
    return f'version {lib.lc_version()}, GPUs: {gpus}'


def check_faster_whisper():
    try:
        import ctranslate2
        import faster_whisper
    except ModuleNotFoundError as e:
        if e.name in ('faster_whisper', 'ctranslate2'):   # left out of this build (build_app.py --no-faster-whisper)
            return 'not included'
        raise
    return f'faster-whisper {getattr(faster_whisper, "__version__", "?")}, CTranslate2 {ctranslate2.__version__}'


def check_data():
    from livecaptions.filters.postprocess import _get_converter, to_simplified
    from livecaptions.models.registry import load_registry
    n = len(load_registry())
    text = to_simplified('語音辨識')
    if text != '语音辨识':
        raise RuntimeError(f'traditional to simplified gave {text!r}')
    return f'{n} registry entries, Chinese conversion through {"OpenCC" if _get_converter() else "Windows"}'


def check_transcription(model, wav, cpu):
    from livecaptions.audio.capture import read_wav
    from livecaptions.config.schema import Config
    from livecaptions.engines import create_engine
    from livecaptions.models.manager import detect_format
    backend = detect_format(model)
    if backend not in ('whispercpp', 'fasterwhisper'):
        raise RuntimeError(f'not a model: {model}')
    cfg = Config()
    cfg.model.backend = backend
    cfg.recog.device = 'cpu' if cpu else cfg.recog.device
    engine = create_engine(backend)
    t0 = time.perf_counter()
    engine.load(model, cfg.recog)
    t_load = time.perf_counter() - t0
    audio = read_wav(wav)[:30 * 16000]
    result = engine.transcribe(audio, None, 'transcribe', [], cfg.recog)
    text = ' '.join(s.text.strip() for s in result.segments)
    engine.unload()
    if not text:
        raise RuntimeError('no text')
    return f'{engine.describe()}, load {t_load:.1f} s, {len(audio) / 16000:.0f} s audio in {result.seconds:.1f} s: {text[:120]}'


def check_ui():
    from PySide6.QtWidgets import QApplication
    import livecaptions.config.store as store
    from livecaptions.config.schema import Config
    from livecaptions.ui.i18n import install_language, tr
    cfg_dir = tempfile.mkdtemp()
    store.get_config_dir = lambda: cfg_dir   # leave the real settings alone
    app = QApplication.instance() or QApplication([sys.argv[0]])
    from livecaptions.ui.app import Controller
    from livecaptions.ui.first_run import FirstRunDialog
    cfg = Config()
    cfg.general.start_minimized = False
    cfg.model.models_dir = os.path.join(cfg_dir, 'models')
    ctl = Controller(app, cfg)
    app.processEvents()   # the first-run dialog is due after 300 ms; this check builds one itself instead
    size = ctl.main.size()
    dlg = FirstRunDialog(ctl)
    n_choices = len(dlg.radios)
    install_language(app, 'zh_CN')
    zh = tr('Ready.')
    install_language(app, 'en')
    dlg.deleteLater()
    ctl.quit()
    if zh == 'Ready.':
        raise RuntimeError('the Chinese interface text did not load')
    if n_choices == 0:
        raise RuntimeError('the first-run dialog found no models in the registry')
    return (f'Qt platform {app.platformName()}, main window {size.width()}x{size.height()}, first-run dialog with '
            f'{n_choices} models, Chinese text loads')


def run_self_check(argv):
    """Run the checks; argv are the command-line arguments after the program name. Returns the exit code."""
    parser = argparse.ArgumentParser(prog=f'{APP_NAME} --self-check', description=__doc__)
    parser.add_argument('--self-check', action='store_true', help='run these checks instead of the app')
    parser.add_argument('--model', default='', help='ggml file or CTranslate2 folder to transcribe with (default: none)')
    parser.add_argument('--wav', default='', help='WAV file to transcribe, first 30 s (default: none)')
    parser.add_argument('--cpu', action='store_true', help='transcribe on the CPU (default: the settings default)')
    parser.add_argument('--ui', action='store_true', help='also build the windows; offscreen with QT_QPA_PLATFORM=offscreen (default: off)')
    parser.add_argument('--report', default='', help='also write the report to this file; packaged Windows builds have no console (default: none)')
    args, _ = parser.parse_known_args(argv)

    rep = Report(args.report)
    rep.add('app', True, f'{APP_NAME} {__version__}, Python {platform.python_version()}, {platform.platform()}, '
                         f'{"packaged" if getattr(sys, "frozen", False) else "from source"}')
    rep.run('lc_native', check_native)
    rep.run('faster-whisper', check_faster_whisper)
    rep.run('data', check_data)
    if args.model and args.wav:
        rep.run(f'transcribe {os.path.basename(os.path.normpath(args.model))}', lambda: check_transcription(args.model, args.wav, args.cpu))
    if args.ui:
        rep.run('windows', check_ui)
    rep.add('result', rep.failed == 0, f'{rep.failed} failed')
    rep.save()
    return 0 if rep.failed == 0 else 1
