# PyInstaller spec: a one-folder app, dist/LiveCaptions/LiveCaptions.exe on Windows and dist/LiveCaptions.app on macOS.
# One folder, not one file: it starts without unpacking, and Qt's LGPL libraries stay replaceable.
# scripts/build_app.py runs it after making the icon and the licence folder:
#   pyinstaller packaging/LiveCaptions.spec -- --icon build/pkg/icon.png --licenses build/pkg/licenses
import argparse
import os
import sys

from PyInstaller.utils.hooks import collect_data_files

parser = argparse.ArgumentParser()
parser.add_argument('--icon', default='', help='PNG icon; Pillow converts it to .ico / .icns (default: none)')
parser.add_argument('--licenses', default='', help='folder of licence texts copied into the app (default: none)')
parser.add_argument('--no-faster-whisper', action='store_true', help='leave faster-whisper out, ~200 MB less (default: off)')
opts = parser.parse_args()

ROOT = os.path.dirname(SPECPATH)
sys.path.insert(0, ROOT)
from livecaptions import APP_NAME, __version__  # noqa: E402

lib_name = {'win32': 'lc_native.dll', 'darwin': 'lc_native.dylib'}.get(sys.platform, 'lc_native.so')
lib_path = os.path.join(ROOT, 'livecaptions', 'bin', lib_name)
if not os.path.exists(lib_path):
    raise SystemExit(f'{lib_path} is missing: run python scripts/build_native.py first')

datas = [(os.path.join(ROOT, 'livecaptions', 'models', 'registry.json'), 'livecaptions/models')]
datas += collect_data_files('opencc', includes=['clib/share/**'])  # OpenCC's conversion dictionaries
if opts.licenses:
    datas.append((opts.licenses, 'licenses'))
excludes = ['tkinter', 'pytest', 'PyInstaller', 'PIL']   # Pillow: only optional imports of other libraries
# A Jupyter kernel in the build environment (e.g. installed by an editor) would come along through optional imports.
excludes += ['IPython', 'ipykernel', 'jupyter_client', 'jupyter_core', 'jedi', 'parso', 'prompt_toolkit', 'zmq',
             'tornado', 'comm', 'debugpy', 'matplotlib_inline', 'traitlets']
if opts.no_faster_whisper:
    excludes += ['faster_whisper', 'ctranslate2', 'onnxruntime', 'av', 'tokenizers']

a = Analysis(
    [os.path.join(SPECPATH, 'launch.py')],
    pathex=[ROOT],
    binaries=[(lib_path, 'livecaptions/bin')],   # native.py looks in <bundle>/livecaptions/bin
    datas=datas,
    hiddenimports=['livecaptions.ui.i18n.zh_CN'],
    excludes=excludes,
    noarchive=False,
)
# OpenCC's hook takes its whole C library folder; the conversion needs only the module and clib/share.
unused = tuple(os.path.normpath(p) + os.sep for p in ('opencc/clib/bin', 'opencc/clib/include', 'opencc/clib/lib'))
a.datas = [t for t in a.datas if not os.path.normpath(t[0]).startswith(unused)]
a.binaries = [t for t in a.binaries if not os.path.normpath(t[0]).startswith(unused)]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,
    icon=opts.icon or None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP_NAME, upx=False)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name=f'{APP_NAME}.app',
        icon=opts.icon or None,
        bundle_identifier='io.github.gao-xinjian.livecaptions',
        version=__version__,
        info_plist={
            'CFBundleDisplayName': APP_NAME,
            'CFBundleShortVersionString': __version__,
            'LSMinimumSystemVersion': '13.3',
            'NSHighResolutionCapable': True,
            # macOS shows these when it asks for the permissions; without them it refuses silently
            'NSMicrophoneUsageDescription': 'LiveCaptions turns speech from the microphone into captions on this Mac.',
            'NSAudioCaptureUsageDescription': 'LiveCaptions turns what this Mac plays (system audio) into captions on this Mac.',
        },
    )
