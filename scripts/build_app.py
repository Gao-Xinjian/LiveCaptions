"""
Package LiveCaptions with PyInstaller into dist/: a folder with LiveCaptions.exe (Windows) or LiveCaptions.app (macOS),
plus a zip of it. Build lc_native first (scripts/build_native.py; --portable for a package to share).
"""
import argparse
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys

from packaging.requirements import Requirement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from livecaptions import APP_NAME, __version__  # noqa: E402

LICENSE_FILE = re.compile(r'^(LICEN[CS]E|COPYING|NOTICE|AUTHORS|THIRD.?PARTY)', re.IGNORECASE)


def find_license_files(dist):
    """Return a distribution's licence files: in its .dist-info folder, or else in its package folders."""
    files = [f for f in dist.files or [] if LICENSE_FILE.match(f.name) and not f.name.endswith(('.py', '.pyc', '.pyi'))]
    in_meta = [f for f in files if '.dist-info' in str(f)]
    return in_meta or files


def find_python_license():
    """Return the licence file of the Python that runs this (python.org, setup-python or conda layouts)."""
    ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    for rel in ('LICENSE.txt', 'LICENSE_PYTHON.txt', os.path.join('Lib', 'LICENSE.txt'), os.path.join('lib', ver, 'LICENSE.txt')):
        path = os.path.join(sys.base_prefix, rel)
        if os.path.exists(path):
            return path
    return ''


def make_icon_png(path):
    """Draw the app icon at 1024 px into a PNG (PyInstaller converts it to .ico / .icns)."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtGui import QGuiApplication
    from livecaptions.ui.tray import draw_icon
    app = QGuiApplication.instance() or QGuiApplication([])
    if not draw_icon(1024).save(path):
        raise SystemExit(f'could not write {path}')
    del app


def get_dists(extras):
    """
    Return the installed distributions the app bundles: its requirements and theirs, recursively.

    Args:
        extras: optional groups of livecaptions to include, e.g. ['chinese', 'faster-whisper']

    Returns:
        {normalised name: Distribution}
    """
    found = {}
    todo = [(r, extras) for r in md.requires('livecaptions') or []]
    while todo:
        text, active = todo.pop()
        req = Requirement(text)
        if req.marker and not any(req.marker.evaluate({'extra': e}) for e in (active or [''])):
            continue
        key = re.sub(r'[-_.]+', '-', req.name).lower()
        if key in found:
            continue
        try:
            dist = md.distribution(req.name)
        except md.PackageNotFoundError:
            continue
        found[key] = dist
        todo += [(r, list(req.extras)) for r in dist.requires or []]
    return found


def get_license_name(dist):
    """Return a distribution's licence as its metadata states it."""
    meta = dist.metadata
    text = meta.get('License-Expression') or meta.get('License') or ''
    if not text or len(text) > 80:   # some packages put the whole text here
        classifiers = [c.split(' :: ')[-1] for c in meta.get_all('Classifier') or [] if c.startswith('License ::')]
        text = ', '.join(classifiers) or (text[:80] + '...' if text else 'see the files')
    return text


def find_whisper_src():
    """Return the whisper.cpp source folder used for lc_native (fetched by CMake, or the sibling checkout)."""
    for cand in (os.path.join(ROOT, 'build', '_deps', 'whisper-src'), os.path.join(ROOT, '..', 'whisper.cpp')):
        if os.path.exists(os.path.join(cand, 'LICENSE')):
            return os.path.abspath(cand)
    return ''


def write_licenses(out_dir, extras):
    """Copy the licence texts of everything in the package into out_dir, with an index file."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    lines = [f'{APP_NAME} {__version__}: MIT licence (LiveCaptions-LICENSE).', '',
             'Bundled components and their licences (texts in the folders next to this file):', '']
    shutil.copy2(os.path.join(ROOT, 'LICENSE'), os.path.join(out_dir, 'LiveCaptions-LICENSE'))
    # The native library: whisper.cpp with ggml, and miniaudio (its licence is at the end of miniaudio.h).
    src = find_whisper_src()
    if src:
        os.makedirs(os.path.join(out_dir, 'whisper.cpp'))
        shutil.copy2(os.path.join(src, 'LICENSE'), os.path.join(out_dir, 'whisper.cpp', 'LICENSE'))
        with open(os.path.join(src, 'examples', 'miniaudio.h'), encoding='utf-8', errors='replace') as f:
            header = f.read()
        start = header.rfind('This software is available as a choice of the following licenses')
        if start >= 0:
            os.makedirs(os.path.join(out_dir, 'miniaudio'))
            with open(os.path.join(out_dir, 'miniaudio', 'LICENSE'), 'w', encoding='utf-8') as f:
                f.write(header[start:].rstrip().removesuffix('*/').rstrip() + '\n')
        lines += ['whisper.cpp and ggml (in lc_native) - MIT - whisper.cpp/',
                  'miniaudio (in lc_native) - public domain or MIT-0 - miniaudio/']
    else:
        print('warning: whisper.cpp sources not found; its licence is missing from the package')
    # Python itself; its licence also covers what it bundles (OpenSSL, libffi, zlib, ...).
    py_license = find_python_license()
    if py_license:
        os.makedirs(os.path.join(out_dir, 'Python'))
        shutil.copy2(py_license, os.path.join(out_dir, 'Python', 'LICENSE.txt'))
        lines.append(f'Python {platform.python_version()} - PSF-2.0 - Python/')
    else:
        print('warning: the Python licence was not found; it is missing from the package')
    # Python packages: their licence files, or a note with the stated licence when the package ships none.
    for key, dist in sorted(get_dists(extras).items()):
        name = dist.metadata['Name']
        os.makedirs(os.path.join(out_dir, name))
        files = find_license_files(dist)
        for f in files:
            shutil.copy2(f.locate(), os.path.join(out_dir, name, f.name))
        if not files:
            urls = [u.split(', ')[-1] for u in dist.metadata.get_all('Project-URL') or []] + [dist.metadata.get('Home-page') or '']
            with open(os.path.join(out_dir, name, 'LICENSE-NOTE.txt'), 'w', encoding='utf-8') as f:
                f.write(f'{name} {dist.version} ships no licence text. Its licence: {get_license_name(dist)}.\n'
                        f'Full text: {next((u for u in urls if u), "see the project")}\n')
        lines.append(f'{name} {dist.version} - {get_license_name(dist)} - {name}/')
    lines += ['', 'Models, the Silero VAD model and the TEN VAD library are not included; they are downloaded on request',
              'and keep their own licences (shown in the download dialog).']
    with open(os.path.join(out_dir, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def get_zip_name():
    """Return the zip name, e.g. LiveCaptions-0.1.0-windows-x64.zip."""
    system = {'win32': 'windows', 'darwin': 'macos'}.get(sys.platform, sys.platform)
    arch = {'amd64': 'x64', 'x86_64': 'x64', 'aarch64': 'arm64'}.get(platform.machine().lower(), platform.machine().lower())
    return f'{APP_NAME}-{__version__}-{system}-{arch}.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-faster-whisper', action='store_true',
                        help='leave faster-whisper and its libraries out, about 200 MB less (default: off)')
    parser.add_argument('--no-zip', action='store_true', help='only build the folder / .app (default: off)')
    parser.add_argument('--dist-dir', default=os.path.join(ROOT, 'dist'), help='output folder (default: dist)')
    args = parser.parse_args()

    pkg_dir = os.path.join(ROOT, 'build', 'pkg')
    os.makedirs(pkg_dir, exist_ok=True)
    extras = ['chinese', 'audio-formats'] + ([] if args.no_faster_whisper else ['faster-whisper'])
    icon = os.path.join(pkg_dir, 'icon.png')
    make_icon_png(icon)
    licenses = os.path.join(pkg_dir, 'licenses')
    write_licenses(licenses, extras)

    # PyInstaller.
    cmd = [sys.executable, '-m', 'PyInstaller', os.path.join(ROOT, 'packaging', 'LiveCaptions.spec'), '--noconfirm',
           '--clean', '--distpath', args.dist_dir, '--workpath', os.path.join(ROOT, 'build', 'pyinstaller'),
           '--', '--icon', icon, '--licenses', licenses]
    if args.no_faster_whisper:
        cmd.append('--no-faster-whisper')
    subprocess.check_call(cmd)

    # Zip it; ditto keeps the .app's symlinks and signature.
    if sys.platform == 'darwin':
        target, run = os.path.join(args.dist_dir, f'{APP_NAME}.app'), f'{APP_NAME}.app/Contents/MacOS/{APP_NAME}'
    else:
        target, run = os.path.join(args.dist_dir, APP_NAME), f'{APP_NAME}/{APP_NAME}.exe'
    print(f'built {target} (run {os.path.join(args.dist_dir, run)})')
    if not args.no_zip:
        zip_path = os.path.join(args.dist_dir, get_zip_name())
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if sys.platform == 'darwin':
            subprocess.check_call(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', target, zip_path])
        else:
            shutil.make_archive(zip_path[:-4], 'zip', root_dir=args.dist_dir, base_dir=APP_NAME)
        print(f'zipped {zip_path} ({os.path.getsize(zip_path) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
