"""Build native/lc_native with CMake and copy the library into livecaptions/bin."""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_default_gpu():
    """Return the GPU backend used when none is given."""
    if sys.platform == 'darwin':
        return 'metal'
    if sys.platform == 'win32':
        return 'vulkan'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default=get_default_gpu(), choices=['vulkan', 'metal', 'cuda', 'cpu'],
                        help=f'GPU backend of whisper.cpp (default: {get_default_gpu()})')
    parser.add_argument('--build-dir', default=os.path.join(ROOT, 'build'),
                        help='CMake build folder; keep it short on Windows (default: build)')
    parser.add_argument('--whisper-dir', default='',
                        help='whisper.cpp source folder; empty = sibling checkout or fetch the pinned commit (default: empty)')
    parser.add_argument('--portable', action='store_true',
                        help='do not optimise for this CPU, for builds to share (default: off)')
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 4, help='parallel build jobs (default: CPU count)')
    args = parser.parse_args()

    # Configure.
    cfg_cmd = ['cmake', '-S', os.path.join(ROOT, 'native', 'lc_native'), '-B', args.build_dir,
               '-DCMAKE_BUILD_TYPE=Release', f'-DLC_GPU={args.gpu}',
               f'-DLC_NATIVE_CPU={"OFF" if args.portable else "ON"}']
    if args.whisper_dir:
        cfg_cmd.append(f'-DWHISPER_DIR={os.path.abspath(args.whisper_dir)}')
    subprocess.check_call(cfg_cmd)

    # Build.
    subprocess.check_call(['cmake', '--build', args.build_dir, '--config', 'Release', '--target', 'lc_native',
                           '--parallel', str(args.jobs)])

    # Copy the library next to the Python package.
    patterns = ['lc_native.dll', 'lc_native.dylib', 'lc_native.so']
    found = [f for p in patterns for f in glob.glob(os.path.join(args.build_dir, '**', p), recursive=True)]
    if not found:
        sys.exit('lc_native library not found in the build folder')
    bin_dir = os.path.join(ROOT, 'livecaptions', 'bin')
    os.makedirs(bin_dir, exist_ok=True)
    newest = max(found, key=os.path.getmtime)
    shutil.copy2(newest, bin_dir)
    print(f'Copied {newest} -> {bin_dir}')


if __name__ == '__main__':
    main()
