# LiveCaptions

Live lecture captions on a laptop running on battery, from the microphone or from what the laptop plays.
Everything runs locally, with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (GPU through Vulkan or Metal)
or [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

- **Sources**: microphone (Windows, macOS) or system audio (Windows WASAPI loopback; macOS 14.2+ Core Audio process
  tap, all audio or chosen apps — built and smoke-tested on CI, not yet used on a real Mac)
- **Streaming**, three policies:
  - *LocalAgreement-n*: text turns final (white) about 1.4 s after it is spoken; the unfinished tail is grey.
    The buffer is cut at the end of final Whisper segments, at pauses (T2), and at its quietest point when long (T3).
  - *VAD-chunked*: each sentence is transcribed once, after the pause that ends it (optional grey previews).
    About a tenth of the inference, cleaner text, but the text comes at the end of each sentence.
  - *Sliding window* (whisper.cpp `stream` style): the current window is re-run every step and becomes final when full.
- **Power**: no inference while the VAD hears no speech (Silero through whisper.cpp, or TEN VAD, both with automatic
  gain for quiet, distant speakers; or an adaptive energy threshold),
  a warmed-up GPU, priority and Windows EcoQoS options, a monitor strip with RAM / CPU / inference / GPU load and battery.
- **Captions**: floating window (always on top, click-through, colours, opacity, font, lines) or docked in the main window,
  scroll back and copy. Hotkeys work globally on Windows.
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

On macOS (13.3 or later): `python scripts/build_native.py` builds with Metal and the process tap. CI builds it on
macOS 26 arm64 and opens the tap once; nobody has used it with real audio yet. macOS asks for permission to use the microphone and
to record system audio; when the app runs from a terminal, the permission goes to the terminal app, and a packaged
`.app` (M4) will carry the usage strings macOS expects.

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
`LC_TEST_MIC=1` also reads the default microphone for 1 s.
`.github/workflows/ci.yml` builds `lc_native` (CPU on Windows; Metal and the process tap on the newest macOS, arm64), runs the
tests with both backends' tiny models and TEN VAD, and on macOS reads the runner's virtual microphone and opens the tap
while a tone plays (after granting the system audio recording permission). CI cannot test real audio hardware, the GPU,
the caption window or power use.

## Layout

```
native/lc_native/     C++ library for ctypes: miniaudio capture, whisper.cpp, Silero VAD, macOS process tap (one DLL / dylib)
scripts/              build_native.py
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

Trying it on a real Mac (CI only builds it and opens the tap once); packaging (PyInstaller, with the macOS
`Info.plist` permission strings) and a first-run dialog; optional automatic adaptation when inference falls behind.

## Licence

MIT, see [LICENSE](LICENSE). `lc_native` links in whisper.cpp and ggml (MIT) and miniaudio (public domain or MIT-0);
a packaged build has to ship their notices too. The TEN VAD library is not part of this repository: it is downloaded on
request and keeps its own licence.
