# LiveCaptions

Live lecture captions on a laptop running on battery, from the microphone or from what the laptop plays.
Everything runs locally, with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (GPU through Vulkan or Metal)
or [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

- **Sources**: microphone (Windows, macOS) or system audio (Windows WASAPI loopback; macOS 14.2+ Core Audio process
  tap, all audio or chosen apps — built on CI, not yet used on a real Mac)
- **Streaming**, three policies:
  - *LocalAgreement-n*: text turns final (white) about 1.4 s after it is spoken; the unfinished tail is grey.
    The buffer is cut at the end of final Whisper segments, at pauses (T2), and at its quietest point when long (T3).
  - *VAD-chunked*: each sentence is transcribed once, after the pause that ends it (optional grey previews).
    About a tenth of the inference, cleaner text, but the text comes at the end of each sentence.
  - *Sliding window* (whisper.cpp `stream` style): the current window is re-run every step and becomes final when full.
- **Power**: no inference while the VAD hears no speech (Silero through whisper.cpp, or TEN VAD, both with automatic
  gain for quiet, distant speakers; or an adaptive energy threshold),
  a warmed-up GPU, priority and Windows EcoQoS options, a monitor strip with RAM / CPU / inference / GPU load and battery.
  Optional automatic adaptation (Streaming page, off by default): when runs take longer than the step allows, the step
  grows (up to a limit), so text comes later but the delay stops growing; VAD-chunked spaces out its previews.
- **Captions**: floating window (always on top, click-through, colours, opacity, font, lines) or docked in the main window,
  scroll back and copy. Start/pause, stop, show/hide and click-through are buttons in the main window and the tray
  menu; optional hotkeys too (global on Windows, only while the app is active on macOS).
- **Saving**: per session folder with audio (WAV, FLAC, Opus) and transcripts (txt, srt, vtt, jsonl, md), written as it goes.
- **Terms**: per-course term profiles in Whisper's prompt (also as faster-whisper hotwords), editable while running.
- **Options**: every setting is in the settings window, with tooltips; no built-in modes. English and 简体中文 interface.

## Setup (Windows)

Needs Python 3.11+, CMake 3.21+, Visual Studio C++ build tools and, for the GPU, the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
whisper.cpp is taken from `../whisper.cpp` if it is there, otherwise CMake fetches the pinned commit.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,chinese,audio-formats,faster-whisper]"
.venv\Scripts\python scripts\build_native.py            # Vulkan; --gpu cpu without the Vulkan SDK
.venv\Scripts\python -m livecaptions
```

## Setup and first test on a Mac

Needs macOS 13.3+ (system audio: 14.2+), Python 3.11+, CMake 3.28+ and the Xcode command line tools
(`xcode-select --install`). CMake fetches whisper.cpp at the pinned commit unless `../whisper.cpp` exists.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,chinese,audio-formats,faster-whisper]"
.venv/bin/python scripts/build_native.py            # Metal and the Core Audio process tap
.venv/bin/python -m livecaptions
```

The macOS capture code has been built on CI but never run with real audio. To check it first:

```bash
LC_TEST_TAP=1 LC_TEST_MIC=1 .venv/bin/python -m pytest tests/test_vad_audio.py -rP
```

The tap test plays a short tone and records system audio. macOS asks for permission to record system audio and to
use the microphone; from a terminal the permission goes to the terminal app (System Settings → Privacy & Security),
and the packaged app asks under its own name. In the output, `system: … read N samples …, peak …`
should show samples with a peak near 0.3 (the tone), the `lc_tap:` lines give each Core Audio call's time (none
should take seconds), and `mic:` should show samples from the microphone. Then in the app choose system audio as the
source (or only some apps, by bundle ID), play a video and watch the captions.

## Packaged app

`scripts/build_app.py` makes a folder app with PyInstaller and zips it into `dist/`:
`dist/LiveCaptions/LiveCaptions.exe` on Windows (about 370 MB, 145 MB zipped) and `dist/LiveCaptions.app` on macOS.
It uses the `lc_native` in `livecaptions/bin`, so build that first (`--portable` for a package to share; on Windows
with Vulkan, which the CI package lacks). `--no-faster-whisper` leaves faster-whisper out, about 200 MB less.

```powershell
.venv\Scripts\python -m pip install -e ".[package]"
.venv\Scripts\python scripts\build_app.py
```

Models are not included: while none is installed, a first-run dialog offers a model for the GPU or the CPU (and the
Silero VAD model), optionally through hf-mirror.com, or takes a model file you have. The licences of everything
bundled are in the app's `licenses` folder (`_internal/licenses` on Windows, `Contents/Resources/licenses` on macOS).
A packaged app has no console: it logs to `%LOCALAPPDATA%\LiveCaptions\Logs\livecaptions.log` (Windows) or
`~/Library/Logs/LiveCaptions/livecaptions.log` (macOS), keeping the previous run's log next to it.
`LiveCaptions --self-check [--model M --wav W] [--ui] [--report FILE]` checks an install without starting the app:
the native library, both backends, the bundled data, one transcription and the windows (CI runs it on both packages).

CI uploads both zips as artifacts. The macOS app is not signed by Apple: after unzipping, open it with right-click →
Open (or System Settings → Privacy & Security → Open Anyway), or run `xattr -dr com.apple.quarantine LiveCaptions.app`.
As an app it asks for the microphone and system audio permissions under its own name.

## Models

Model page → **Download…** lists whisper.cpp and faster-whisper models from Hugging Face (with sizes and approximate
memory), the Silero VAD model, which the default VAD needs, and the TEN VAD library. TEN VAD comes as a prebuilt
library from [its repository](https://github.com/TEN-framework/ten-vad) at a pinned commit, checked against its Git
hash; its licence is Apache 2.0 with Agora's conditions (fine for personal use). **Browse…** takes any `ggml-*.bin` file or
CTranslate2 model folder. A mirror (e.g. `https://hf-mirror.com`) and a proxy can be set in the advanced options.
`large-v3-turbo`, `distil-*` and `*.en` models cannot translate.

## Replay tool

Feeds a WAV file through the pipeline as if it were live and prints when each piece of text became final, the load,
peak memory and, with a reference text, the error rate. `--set section.option=value` changes any setting.

```powershell
.venv\Scripts\python tools\replay.py lecture.wav --model ggml-large-v3-turbo-q5_0.bin --speed 0 --ref lecture.txt
```

Measured on the Intel Arc B390 (Vulkan) with `large-v3-turbo-q5_0`, Silero VAD, `--speed 0`; inference time as a
share of the audio time, and how long after it was spoken text became final (mean). Peak RSS was about 1.0–1.1 GB.

| Recording | LocalAgreement-2 | VAD-chunked | Sliding window (10 s) |
|---|---|---|---|
| English clip, 62 s, almost no pauses | 76 %, 1.4 s | 6 %, 12 s (7 s with adaptive pauses) | 74 %, 4.0 s |
| Chinese clip, 57 s, pauses | 25 %, 1.4 s | 9 %, 1.9 s | 24 %, 1.7 s |
| Quiet lecture (−46 dBFS), 189 s | 79 %, 3.0 s | 7 %, 8.8 s (5.6 s adaptive) | 81 %, 2.8 s |

On the lecture, VAD-chunked gave the cleanest text, the sliding window garbled words at window edges, and
LocalAgreement fell into one repetition loop. Inference shares vary by ±15 % between identical runs (GPU clocks);
the run counts are stable.

Share of the audio each VAD took for speech, without and with automatic gain:

| Recording | Silero | TEN VAD |
|---|---|---|
| English clip | 85 % / 85 % | 85 % / 86 % |
| Quiet lecture (−46 dBFS) | 1 % / 61 % | 36 % / 52 % |
| Louder lecture (−37 dBFS) | 77 % / 79 % | 76 % / 76 % |

An encoder window fixed at 15 s (Recognition → advanced) cut the English inference to 52 % but dropped and invented
words; "fit to the audio" was slower, because whisper.cpp sets up the GPU again for every new window size.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

Unit tests need nothing else. More tests run when these point to files: `LC_TEST_MODEL` (ggml model), `LC_TEST_WAV`,
`LC_TEST_FW_MODEL` (CTranslate2 folder), `LC_TEST_TEN_VAD` (TEN VAD library; or `LC_TEST_TEN_DOWNLOAD=1` to fetch it).
`LC_TEST_MIC=1` reads the default microphone for 1 s; `LC_TEST_TAP=1` records system audio on macOS while a tone plays.
`.github/workflows/ci.yml` builds `lc_native` (CPU on Windows; Metal and the process tap on the newest macOS, arm64) and
runs the tests with both backends' tiny models and TEN VAD. CI cannot test audio capture (the macOS runner is a VM
whose system audio permission nobody can grant), the GPU, the caption window or power use.

## Layout

```
native/lc_native/     C++ library for ctypes: miniaudio capture, whisper.cpp, Silero VAD, macOS process tap (one DLL / dylib)
scripts/              build_native.py (lc_native), build_app.py (PyInstaller package)
packaging/            PyInstaller spec and entry script
livecaptions/
  native.py           ctypes bindings (struct sizes are checked against the library)
  pipeline.py         capture -> VAD -> streaming policy -> engine -> filters -> writers, on its own thread
  config/             settings schema with UI metadata, TOML storage, term profiles
  audio/ vad/ engines/ segmentation/ filters/ output/ models/ stats/
  ui/                 Qt windows generated from the schema; i18n/zh_CN.py holds the Chinese text
tools/replay.py
tests/
```

## Not done yet

Trying it on a real Mac (CI only builds and unit-tests it).

## Licence

MIT, see [LICENSE](LICENSE). `lc_native` links in whisper.cpp and ggml (MIT) and miniaudio (public domain or MIT-0);
packages ship their notices (and those of all bundled Python packages) in `licenses`. The TEN VAD library is not part of this repository: it is downloaded on
request and keeps its own licence.
