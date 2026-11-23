"""
Replay a WAV file through the pipeline as if it were captured live, and report latency, load and error rate.

    python tools/replay.py speech.wav --model ggml-large-v3-turbo-q5_0.bin --ref speech.txt --terms "突触 递质 抑制"
"""
import argparse
import os
import sys
import threading
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecaptions import SAMPLE_RATE  # noqa: E402
from livecaptions.audio.capture import WavFileSource, read_wav  # noqa: E402
from livecaptions.config.store import load_config  # noqa: E402
from livecaptions.pipeline import EngineHolder, Pipeline  # noqa: E402
from livecaptions.segmentation.units import is_cjk  # noqa: E402


def calculate_edit_distance(ref, hyp):
    """Return the Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def text2tokens(text):
    """Split text for error rates: CJK characters one by one, other words by spaces, no punctuation."""
    tokens, word = [], ''
    for c in text.lower():
        if is_cjk(c):
            if word:
                tokens.append(word)
                word = ''
            tokens.append(c)
        elif c.isalnum() or c == "'":
            word += c
        elif word:
            tokens.append(word)
            word = ''
    if word:
        tokens.append(word)
    return tokens


def calculate_error_rate(ref_text, hyp_text):
    """Return the error rate (CER for CJK, WER otherwise) of hyp against ref."""
    ref, hyp = text2tokens(ref_text), text2tokens(hyp_text)
    return calculate_edit_distance(ref, hyp) / max(1, len(ref))


def set_option(cfg, item):
    """Apply 'section.name=value' to cfg, converting value to the option's type."""
    key, _, value = item.partition('=')
    section, _, name = key.partition('.')
    obj = getattr(cfg, section)
    old = getattr(obj, name)
    if isinstance(old, bool):
        new = value.lower() in ('1', 'true', 'yes', 'on')
    elif isinstance(old, list):
        new = [v for v in value.split(',') if v]
    else:
        new = type(old)(value)
    setattr(obj, name, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('wav', help='16 kHz mono WAV (other formats are converted)')
    parser.add_argument('--model', default='', help='model file or folder (default: from config.toml)')
    parser.add_argument('--backend', default='', choices=['', 'whispercpp', 'fasterwhisper'], help='backend (default: from config)')
    parser.add_argument('--config', default='', help='settings file (default: the app settings, or defaults)')
    parser.add_argument('--lang', default='', help='language code or auto (default: from config)')
    parser.add_argument('--terms', default='', help='course terms, separated by spaces (default: none)')
    parser.add_argument('--ref', default='', help='reference transcript to compute the error rate (default: none)')
    parser.add_argument('--speed', type=float, default=1.0, help='1 = real time, 0 = as fast as possible (default: 1)')
    parser.add_argument('--vad', default='', choices=['', 'silero', 'ten', 'energy', 'off'], help='VAD method (default: from config)')
    parser.add_argument('--silero', default='', help='Silero ggml model file (default: from config / models folder)')
    parser.add_argument('--cpu', action='store_true', help='run on the CPU (default: config)')
    parser.add_argument('--quiet', action='store_true', help='print only the summary (default: off)')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.NAME=VALUE',
                        help='override any setting, e.g. --set recog.audio_ctx_mode=auto (default: none)')
    args = parser.parse_args()

    # Settings.
    cfg = load_config(args.config or None)
    if args.model:
        cfg.model.path = args.model
    if args.backend:
        cfg.model.backend = args.backend
    if args.lang:
        cfg.recog.language = args.lang
    if args.vad:
        cfg.vad.method = args.vad
    if args.silero:
        cfg.vad.silero_model = args.silero
    if args.cpu:
        cfg.recog.device = 'cpu'
    for item in args.set:
        set_option(cfg, item)

    # Replay.
    samples = read_wav(args.wav)
    source = WavFileSource(samples, speed=args.speed)
    final = []
    commits = []  # (wall time, audio time, text)
    done = threading.Event()
    proc = psutil.Process()
    peak_rss = [proc.memory_info().rss]
    t_start = [time.perf_counter()]
    pipe = None

    def emit(kind, payload):
        if kind == 'commit':
            final.append(payload)
            commits.append((time.perf_counter() - t_start[0], pipe.audio_s if pipe else 0.0, payload))
            if not args.quiet:
                lat = pipe.latencies[-1] if pipe and pipe.latencies else 0.0
                print(f'[{pipe.audio_s:7.2f}s +{lat:4.1f}] {payload!r}', flush=True)
        elif kind == 'started':
            t_start[0] = time.perf_counter()
            print(f'started: {payload["engine"]}', flush=True)
        elif kind in ('error', 'warning'):
            print(f'{kind}: {payload}', flush=True)
        elif kind == 'stopped':
            done.set()

    pipe = Pipeline(cfg, EngineHolder(), emit, terms_text=args.terms, source=source, save=False)
    pipe.start()
    while not done.wait(0.25):
        peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
    wall = time.perf_counter() - t_start[0]

    # Summary.
    text = ''.join(final).replace('\n', ' ')
    audio_s = len(samples) / SAMPLE_RATE
    lats = pipe.latencies
    print('\n' + text.strip())
    print(f'\naudio {audio_s:.1f} s, wall {wall:.1f} s, runs {pipe.runs}, inference {pipe.busy_s:.1f} s '
          f'({100 * pipe.busy_s / max(audio_s, 1e-9):.0f}% of the audio), transcribed {pipe.run_audio_s:.0f} s of buffers')
    if lats:
        print(f'finalised {sum(lats) / len(lats):.2f} s after speech on average (max {max(lats):.1f} s, last 200 units)')
    if args.speed > 0 and commits:   # in real time: how far processing lags behind the audio coming in
        lags = [max(0.0, w * args.speed - a) for w, a, _ in commits]
        print(f'behind the live audio at commits: {sum(lags) / len(lags):.1f} s on average, {max(lags):.1f} s at most'
              + (f'; text appears {sum(lats) / len(lats) + sum(lags) / len(lags):.1f} s after speech in all' if lats else ''))
    if cfg.stream.adapt_on:
        print(f'adaptive step: longest {pipe.step_max_s:.1f} s, at the end {pipe.step_s:.1f} s')
    print(f'peak RSS {peak_rss[0] / 2**20:.0f} MB')
    if args.ref:
        with open(args.ref, encoding='utf-8-sig') as f:
            ref = f.read()
        print(f'error rate {100 * calculate_error_rate(ref, text):.1f}%')


if __name__ == '__main__':
    main()
