"""
Settings schema. Every option carries its UI metadata, so the settings window is generated from it.

Metadata keys: label, tip, choices [(value, label)], lo/hi/step, unit, kind, advanced, backends, policies, live.
kind: None (from the type), 'device', 'model', 'gpu', 'dir', 'file', 'lines', 'multi', 'color', 'font', 'hotkey',
'lang', 'profile', 'hidden'. backends / policies: the backends or streaming policies an option applies to
(others grey it out). live: the option takes effect while captions run; otherwise on the next Start.
"""
from dataclasses import dataclass, field, fields

CONFIG_VERSION = 1


def opt(default, label='', *, tip='', choices=None, lo=None, hi=None, step=None, unit='', kind=None,
        advanced=False, backends=None, policies=None, live=False, decimals=None):
    """Declare one option with its UI metadata."""
    meta = dict(label=label, tip=tip, choices=choices, lo=lo, hi=hi, step=step, unit=unit, kind=kind,
                advanced=advanced, backends=backends, policies=policies, live=live, decimals=decimals)
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: type(default)(default), metadata=meta)
    return field(default=default, metadata=meta)


WCPP = ('whispercpp',)
FW = ('fasterwhisper',)
LA = ('local_agreement',)
VC = ('vad_chunk',)
SW = ('sliding_window',)

LANGUAGES = [('auto', 'Auto detect'), ('en', 'English'), ('zh', 'Chinese'), ('de', 'German'), ('fr', 'French'),
             ('es', 'Spanish'), ('it', 'Italian'), ('ja', 'Japanese'), ('ko', 'Korean'), ('ru', 'Russian'),
             ('pt', 'Portuguese'), ('nl', 'Dutch')]

DEFAULT_BLACKLIST = [
    '请不吝点赞', '订阅 转发 打赏', '明镜与点点', '字幕由', 'Amara.org', '优优独播剧场', 'YoYo Television',
    'Thanks for watching', 'Thank you for watching', 'Please subscribe', 'Subtitles by',
]
DEFAULT_EXACT_HALLUCINATIONS = ['谢谢观看', '谢谢观看。', 'Thank you.', '.', 'you']


@dataclass
class AudioCfg:
    source: str = opt('mic', 'Audio source', choices=[('mic', 'Microphone'), ('system', 'System audio (what the speakers play)')],
                      tip='System audio uses WASAPI loopback on Windows and a Core Audio process tap on macOS 14.2+.')
    device_id: str = opt('', 'Device', kind='device', tip='Empty = the system default device (follows device changes).')
    device_name: str = opt('', kind='hidden')
    tap_apps: str = opt('', 'Only these apps (macOS)',
                        tip='System audio on macOS: bundle IDs of the apps to capture, e.g. com.google.Chrome, us.zoom.xos. '
                            'Empty = all system audio. The apps must be running when captions start.')
    channel_mode: str = opt('mix', 'Channels', choices=[('mix', 'Mix to mono'), ('left', 'Left only'), ('right', 'Right only')])
    gain_db: float = opt(0.0, 'Input gain', lo=-20.0, hi=30.0, step=1.0, unit='dB', live=True, decimals=0)
    period_ms: int = opt(50, 'Device period', lo=10, hi=500, unit='ms', advanced=True,
                         tip='How often the audio device wakes the app. Longer saves power and adds a little latency.')
    ring_s: int = opt(30, 'Capture buffer', lo=5, hi=120, unit='s', advanced=True,
                      tip='Audio waiting to be processed. When full, the oldest audio is dropped.')
    lpf_order: int = opt(4, 'Resampler filter order', lo=0, hi=8, advanced=True,
                         tip='Low-pass filter of the conversion to 16 kHz. Higher is cleaner and costs a little CPU.')
    pause_close: bool = opt(False, 'Close the device while paused', tip='Saves a little power; resuming takes a moment longer.')


@dataclass
class ModelCfg:
    backend: str = opt('whispercpp', 'Backend', choices=[('whispercpp', 'whisper.cpp'), ('fasterwhisper', 'faster-whisper')],
                       tip='whisper.cpp: CPU, Vulkan (Windows) or Metal (macOS). faster-whisper: CPU (int8) or NVIDIA CUDA.')
    path: str = opt('', 'Model', kind='model', tip='A ggml-*.bin file (whisper.cpp) or a CTranslate2 model folder (faster-whisper).')
    models_dir: str = opt('', 'Models folder', kind='dir', tip='Where downloads go and where installed models are listed. Empty = default.')
    hf_endpoint: str = opt('', 'Download server', advanced=True,
                           tip='Empty = https://huggingface.co. A mirror such as https://hf-mirror.com also works.')
    proxy: str = opt('', 'Download proxy', advanced=True, tip='e.g. http://127.0.0.1:7890. Empty = system settings.')
    idle_unload_min: int = opt(0, 'Unload the model when idle for', lo=0, hi=240, unit='min',
                               tip='0 = keep it loaded. Loading again takes a few seconds; a loaded model costs no power.')


@dataclass
class RecogCfg:
    language: str = opt('auto', 'Language', kind='lang', choices=LANGUAGES, live=False,
                        tip='Auto detection lets whisper.cpp run the encoder twice per run until the language is locked.')
    lang_lock_runs: int = opt(3, 'Lock the detected language after', lo=0, hi=50, unit='runs',
                              tip='With Auto: after this many runs, keep the most frequent language. 0 = detect every run '
                                  '(handles mixed languages, costs more).')
    translate: bool = opt(False, 'Output English (Whisper translate)',
                          tip='The text becomes English instead of the spoken language. Not supported by turbo, distil and .en models.')
    to_simplified: bool = opt(True, 'Chinese output in Simplified characters', live=True)
    device: str = opt('gpu', 'Compute device', choices=[('gpu', 'GPU'), ('cpu', 'CPU')],
                      tip='On battery, a discrete GPU can cost more power than an integrated GPU or the CPU.')
    gpu_device: int = opt(0, 'GPU', kind='gpu')
    flash_attn: bool = opt(True, 'Flash attention', backends=WCPP, advanced=True)
    threads: int = opt(4, 'CPU threads', lo=1, hi=64,
                       tip='More threads finish sooner but wake more cores. Never more than the logical cores.')
    compute_type: str = opt('int8', 'Compute type', backends=FW,
                            choices=[('default', 'Model default'), ('int8', 'int8'), ('int8_float16', 'int8_float16'),
                                     ('int8_float32', 'int8_float32'), ('float16', 'float16'), ('float32', 'float32')])
    beam_size: int = opt(1, 'Beam size', lo=1, hi=10, tip='1 = greedy decoding (cheapest). Larger beams are slower and a little more accurate.')
    best_of: int = opt(5, 'Best of (sampling)', lo=1, hi=10, advanced=True, tip='Candidates when the temperature falls back above 0.')
    patience: float = opt(1.0, 'Beam patience', lo=0.5, hi=3.0, step=0.1, backends=FW, advanced=True, decimals=1)
    temperature: float = opt(0.0, 'Temperature', lo=0.0, hi=1.0, step=0.1, advanced=True, decimals=1)
    temperature_inc: float = opt(0.2, 'Temperature fallback step', lo=0.0, hi=1.0, step=0.1, advanced=True, decimals=1,
                                 tip='Decode again at a higher temperature when a result looks bad. 0 = never (cheaper).')
    entropy_thold: float = opt(2.4, 'Entropy / compression ratio threshold', lo=0.0, hi=10.0, step=0.1, advanced=True, decimals=1)
    logprob_thold: float = opt(-1.0, 'Log probability threshold', lo=-10.0, hi=0.0, step=0.1, advanced=True, decimals=1)
    no_speech_thold: float = opt(0.6, 'No-speech threshold', lo=0.0, hi=1.0, step=0.05, advanced=True, decimals=2)
    length_penalty: float = opt(-1.0, 'Length penalty', lo=-1.0, hi=2.0, step=0.1, advanced=True, decimals=1, tip='-1 = default.')
    max_initial_ts: float = opt(1.0, 'Max initial timestamp', lo=0.0, hi=30.0, step=0.5, unit='s', advanced=True, decimals=1)
    suppress_blank: bool = opt(True, 'Suppress blank output', advanced=True)
    suppress_nst: bool = opt(True, 'Suppress non-speech tokens', backends=WCPP, advanced=True, tip='No "*", music notes or brackets.')
    audio_ctx_mode: str = opt('full', 'Encoder window', backends=WCPP, advanced=True,
                              choices=[('full', 'Full 30 s'), ('auto', 'Fit to the audio'), ('fixed', 'Fixed size')],
                              tip='A smaller encoder window is cheaper but loses or invents words (fixed 15 s: about 25% '
                                  'less inference in a test). "Fit to the audio" changes the size every run, which '
                                  'is slower on GPUs with the current whisper.cpp.')
    audio_ctx: int = opt(768, 'Encoder window size', lo=64, hi=1500, unit='frames', backends=WCPP, advanced=True,
                         tip='Used with "Fixed size". 50 frames = 1 s; the longest buffer is capped to fit.')
    token_timestamps: bool = opt(True, 'Token timestamps', backends=WCPP, advanced=True)
    thold_pt: float = opt(0.01, 'Timestamp probability threshold', lo=0.0, hi=1.0, step=0.01, backends=WCPP, advanced=True, decimals=2)
    thold_ptsum: float = opt(0.01, 'Timestamp sum probability threshold', lo=0.0, hi=1.0, step=0.01, backends=WCPP, advanced=True, decimals=2)
    single_segment: bool = opt(False, 'Single segment per run', backends=WCPP, advanced=True,
                               tip='Fewer cut points for the buffer (see Segmentation).')
    max_len: int = opt(0, 'Max segment length', lo=0, hi=500, unit='chars', backends=WCPP, advanced=True, tip='0 = no limit.')
    split_on_word: bool = opt(False, 'Split segments on words', backends=WCPP, advanced=True)
    max_tokens: int = opt(0, 'Max tokens per segment', lo=0, hi=224, backends=WCPP, advanced=True, tip='0 = no limit.')
    suppress_regex: str = opt('', 'Suppress tokens matching', backends=WCPP, advanced=True, tip='A regular expression; empty = none.')
    hotwords: bool = opt(False, 'Also pass the terms as hotwords', backends=FW)


@dataclass
class StreamCfg:
    policy: str = opt('local_agreement', 'Streaming policy',
                      choices=[('local_agreement', 'LocalAgreement-n (steady text, ~1–2 s)'),
                               ('vad_chunk', 'VAD-chunked (one run per sentence, cheapest)'),
                               ('sliding_window', 'Sliding window (whisper.cpp stream style)')],
                      tip='LocalAgreement-n: text becomes final when n runs in a row agree on it; re-runs the buffer often. '
                          'VAD-chunked: each sentence is transcribed once, after the pause that ends it; needs a VAD. '
                          'Sliding window: the current window is re-run every step and becomes final when it is full.')
    agree_n: int = opt(2, 'Runs that must agree (n)', lo=2, hi=5, policies=LA, tip='Higher = steadier text, later commits.')
    step_s: float = opt(1.0, 'Run every', lo=0.2, hi=10.0, step=0.1, unit='s', decimals=1, policies=LA + SW,
                        tip='New speech needed before the next run. Longer saves power, delays text.')
    preview_on: bool = opt(False, 'Show a preview while speaking', policies=VC,
                           tip='Transcribe the sentence so far and show it in grey. Costs extra runs.')
    preview_s: float = opt(3.0, 'Preview every', lo=0.5, hi=20.0, step=0.5, unit='s', decimals=1, policies=VC)
    adapt_on: bool = opt(False, 'Run less often when inference falls behind', live=True,
                         tip='When runs take longer than the step allows, the step grows, so each run covers more audio: '
                             'text comes later, but the delay stops growing. VAD-chunked spaces out its previews.')
    adapt_load: int = opt(80, 'Inference share that lengthens the step', lo=30, hi=100, step=5, unit='%', advanced=True,
                          live=True, tip='The step grows when a run takes longer than this share of it (recent average).')
    adapt_max_s: float = opt(6.0, 'Longest adapted step', lo=1.0, hi=20.0, step=0.5, unit='s', decimals=1, advanced=True,
                             live=True)
    window_s: float = opt(10.0, 'Window length', lo=2.0, hi=29.0, step=1.0, unit='s', decimals=0, policies=SW,
                          tip='When the window is this long, its text becomes final and a new window starts.')
    keep_s: float = opt(1.0, 'Overlap kept for the next window', lo=0.0, hi=5.0, step=0.1, unit='s', decimals=1,
                        policies=SW, tip='Audio from the end of a window given to the next one, so words at the edge '
                                     'are heard whole; their final text is not repeated.')
    pause_s: float = opt(1.0, 'Pause that ends a sentence (T2)', lo=0.2, hi=5.0, step=0.1, unit='s', decimals=1,
                         tip='Silence after recognised speech: everything heard so far becomes final and the buffer is cleared.')
    adaptive_pause: bool = opt(False, 'Shorter pauses end long sentences',
                               tip='The pause needed shrinks from T2 to the minimum as the sentence grows.')
    pause_min_s: float = opt(0.3, 'Shortest pause', lo=0.1, hi=2.0, step=0.1, unit='s', decimals=1)
    adaptive_from_s: float = opt(5.0, 'Start shrinking at', lo=0.0, hi=30.0, step=1.0, unit='s', decimals=0)
    adaptive_to_s: float = opt(20.0, 'Reach the minimum at', lo=1.0, hi=30.0, step=1.0, unit='s', decimals=0)
    noise_reset_s: float = opt(2.0, 'Discard unrecognised sound after (T1)', lo=0.5, hi=10.0, step=0.5, unit='s', decimals=1,
                               tip='The VAD heard speech but Whisper recognised nothing in this much audio: drop it.')
    max_buffer_s: float = opt(25.0, 'Longest buffer (T3)', lo=5.0, hi=29.0, step=1.0, unit='s', decimals=0,
                              tip='Then the buffer is cut at its quietest point and the text before it becomes final.')
    smart_cut_s: float = opt(3.0, 'Look for the quietest point within', lo=0.5, hi=10.0, step=0.5, unit='s', decimals=1,
                             policies=LA + VC)
    trim_after_s: float = opt(15.0, 'Trim the buffer once longer than', lo=2.0, hi=28.0, step=1.0, unit='s', decimals=0,
                              policies=LA, tip='Audio whose text is final leaves the buffer, which keeps every run short.')
    cut_at: str = opt('segment', 'Trim at', policies=LA, choices=[('segment', 'End of a final Whisper segment'),
                                                                 ('sentence', 'End of a final sentence')])
    preroll_s: float = opt(0.3, 'Audio kept before speech starts', lo=0.0, hi=2.0, step=0.1, unit='s', decimals=1)
    idle_end_s: float = opt(1.0, 'End a sentence when the device sends nothing for', lo=0.2, hi=10.0, step=0.1, unit='s',
                            decimals=1, tip='Loopback capture delivers no data while nothing plays.')
    commit_on_pause: bool = opt(True, 'Pause makes all text final')
    max_line_chars: int = opt(80, 'Longest caption line', lo=10, hi=400, unit='chars', live=True)
    break_on_sentence: bool = opt(True, 'New line after each sentence', live=True)


@dataclass
class VadCfg:
    method: str = opt('silero', 'Speech detection (VAD)',
                      choices=[('silero', 'Silero VAD'), ('ten', 'TEN VAD'), ('energy', 'Adaptive energy threshold'),
                               ('off', 'Off (everything is speech)')],
                      tip='No inference runs while the VAD hears no speech: the biggest power saving. TEN VAD is a '
                          'separate download under Apache 2.0 with Agora\'s conditions (personal use is fine).')
    cascade: bool = opt(False, 'Energy gate before the neural VAD',
                        tip='Silero or TEN VAD runs only when the energy detector hears something.')
    threshold: float = opt(0.5, 'Speech threshold', lo=0.05, hi=0.95, step=0.05, decimals=2, live=True)
    neg_threshold: float = opt(0.35, 'Silence threshold', lo=0.0, hi=0.9, step=0.05, decimals=2, live=True,
                               tip='Below this is silence; between the two thresholds the previous state holds.')
    min_speech_ms: int = opt(150, 'Shortest speech', lo=0, hi=2000, step=10, unit='ms', live=True)
    silero_model: str = opt('', 'Silero model', kind='file', tip='A ggml-silero-*.bin file. Empty = the one in the models folder.')
    silero_threads: int = opt(1, 'Silero threads', lo=1, hi=4, advanced=True)
    ten_library: str = opt('', 'TEN VAD library', kind='file',
                           tip='ten_vad.dll (Windows) or libten_vad.dylib (macOS). Empty = the one in the models folder.')
    ten_hop: str = opt('256', 'TEN VAD frame', choices=[('256', '16 ms (256 samples)'), ('160', '10 ms (160 samples)')])
    agc_on: bool = opt(True, 'Automatic gain for speech detection', live=True,
                       tip='Silero misses quiet speech (below about -30 dBFS, e.g. a lecturer far from the laptop). '
                           'This raises the level Silero or TEN VAD hears; the recording and Whisper are not affected.')
    agc_max_db: float = opt(30.0, 'Most gain', lo=0.0, hi=50.0, step=1.0, unit='dB', decimals=0, live=True, advanced=True)
    floor_window_s: float = opt(10.0, 'Noise floor window', lo=1.0, hi=60.0, step=1.0, unit='s', decimals=0, advanced=True)
    floor_percentile: float = opt(10.0, 'Noise floor percentile', lo=1.0, hi=50.0, step=1.0, decimals=0, advanced=True)
    margin_db: float = opt(10.0, 'Speech above the noise floor by', lo=1.0, hi=40.0, step=1.0, unit='dB', decimals=0, live=True)
    min_level_db: float = opt(-55.0, 'Ignore sound quieter than', lo=-90.0, hi=-10.0, step=1.0, unit='dBFS', decimals=0, live=True)
    band_limit: bool = opt(True, 'Only speech frequencies (100–4000 Hz)', advanced=True)


@dataclass
class FilterCfg:
    blacklist_on: bool = opt(True, 'Drop known hallucinations', live=True)
    blacklist: list = opt(DEFAULT_BLACKLIST, 'Phrases (one per line)', kind='lines', live=True,
                          tip='A run containing one of these is dropped.')
    exact: list = opt(DEFAULT_EXACT_HALLUCINATIONS, 'Whole-text matches (one per line)', kind='lines', live=True)
    blacklist_regex: bool = opt(False, 'Phrases are regular expressions', live=True)
    no_speech_on: bool = opt(True, 'Drop segments that are probably not speech', live=True)
    no_speech_prob: float = opt(0.6, 'No-speech probability above', lo=0.0, hi=1.0, step=0.05, decimals=2, live=True)
    avg_logprob: float = opt(-1.0, 'and average log probability below', lo=-5.0, hi=0.0, step=0.1, decimals=1, live=True)
    repetition_on: bool = opt(True, 'Cut repeated words', live=True)
    repetition_max: int = opt(4, 'Repeats allowed', lo=1, hi=20, live=True)
    rate_on: bool = opt(True, 'Drop runs with too much text for the audio', live=True)
    max_units_per_s: float = opt(12.0, 'Max words / characters per second', lo=2.0, hi=50.0, step=1.0, decimals=0, live=True)
    terms_only: bool = opt(True, 'Drop new lines made only of terms', live=True,
                           tip='Whisper tends to "hear" the prompt terms in noise.')
    kana_in_zh: bool = opt(True, 'Drop Japanese kana in Chinese', live=True)


@dataclass
class TermsCfg:
    profile: str = opt('default', 'Profile', kind='profile', live=True)
    use_terms: bool = opt(True, 'Give the terms to Whisper', live=True)
    use_context: bool = opt(True, 'Give recent text to Whisper', live=True)
    context_chars: int = opt(200, 'Recent text', lo=0, hi=1000, unit='chars', live=True)
    prompt_tokens: int = opt(200, 'Prompt size', lo=0, hi=224, unit='tokens', live=True, tip='Whisper reads at most 224.')
    term_tokens: int = opt(100, 'Terms may use', lo=0, hi=224, unit='tokens', live=True)


@dataclass
class SaveCfg:
    dir: str = opt('', 'Save folder', kind='dir', tip='Empty = Documents/LiveCaptions.')
    save_audio: bool = opt(True, 'Save audio')
    audio_format: str = opt('wav', 'Audio format', choices=[('wav', 'WAV (16-bit)'), ('flac', 'FLAC'), ('opus', 'Opus (small)')],
                            tip='FLAC and Opus need the soundfile package.')
    save_text: bool = opt(True, 'Save transcript')
    formats: list = opt(['txt'], 'Transcript formats', kind='multi',
                        choices=[('txt', 'Text (.txt)'), ('srt', 'Subtitles (.srt)'), ('vtt', 'WebVTT (.vtt)'),
                                 ('jsonl', 'JSON lines (.jsonl)'), ('md', 'Markdown (.md)')])
    name_pattern: str = opt('%Y-%m-%d %H-%M-%S {profile}', 'Session folder name',
                            tip='strftime codes plus {profile} for the terms profile.')
    flush_s: float = opt(2.0, 'Write to disk every', lo=0.5, hi=60.0, step=0.5, unit='s', decimals=1,
                         tip='How much a crash or a flat battery can lose, vs. how often the disk wakes up.')


@dataclass
class CaptionCfg:
    docked: bool = opt(False, 'Show captions inside the main window', live=True)
    scheme: str = opt('dark', 'Colours', live=True, choices=[('dark', 'White on black'), ('light', 'Black on white'), ('custom', 'Custom')])
    fg: str = opt('#FFFFFF', 'Text colour', kind='color', live=True)
    bg: str = opt('#000000', 'Background colour', kind='color', live=True)
    tentative_fg: str = opt('#9A9A9A', 'Unfinished text colour', kind='color', live=True)
    bg_opacity: float = opt(0.65, 'Background opacity', lo=0.0, hi=1.0, step=0.05, decimals=2, live=True)
    text_opacity: float = opt(1.0, 'Text opacity', lo=0.1, hi=1.0, step=0.05, decimals=2, live=True)
    font_family: str = opt('', 'Font', kind='font', live=True)
    font_size: int = opt(24, 'Font size', lo=8, hi=96, unit='pt', live=True)
    bold: bool = opt(False, 'Bold', live=True)
    line_spacing: float = opt(1.15, 'Line spacing', lo=0.8, hi=3.0, step=0.05, decimals=2, live=True)
    max_lines: int = opt(3, 'Visible lines', lo=1, hi=30, live=True)
    align: str = opt('left', 'Alignment', choices=[('left', 'Left'), ('center', 'Centre')], live=True)
    tentative_italic: bool = opt(False, 'Unfinished text in italics', live=True)
    on_top: bool = opt(True, 'Always on top', live=True)
    click_through: bool = opt(False, 'Click-through', live=True, tip='Clicks go to the windows below. Toggle back with the button in the main window or the tray menu.')
    scrollback: int = opt(1000, 'Lines kept for scrolling back', lo=10, hi=100000, live=True)
    geometry: list = opt([], kind='hidden')
    hk_start: str = opt('Ctrl+Alt+S', 'Hotkey: start / pause', kind='hotkey', live=True)
    hk_stop: str = opt('Ctrl+Alt+X', 'Hotkey: stop', kind='hotkey', live=True)
    hk_click: str = opt('Ctrl+Alt+C', 'Hotkey: click-through', kind='hotkey', live=True)
    hk_show: str = opt('Ctrl+Alt+H', 'Hotkey: show / hide captions', kind='hotkey', live=True)


@dataclass
class ShareCfg:
    enabled: bool = opt(False, 'Share the captions in a web page', live=True,
                        tip='A read-only page with the live captions (white final text, grey unfinished text) for browsers '
                            'on this computer and, if allowed, on the local network. Programs, such as a chatbot later, '
                            'can read /events and /transcript.txt.')
    lan: bool = opt(False, 'Allow devices on the local network', live=True,
                    tip='Phones and computers on the same network can open the page. The first time, Windows asks to let '
                        'the app through its firewall.')
    port: int = opt(8765, 'Port', lo=1024, hi=65535, live=True)
    use_code: bool = opt(True, 'Local network needs the access code', live=True,
                         tip='The links below contain it. This computer never needs it.')
    code: str = opt('', kind='hidden')
    send_partial: bool = opt(True, 'Also send the grey unfinished text', live=True)
    partial_interval_s: float = opt(0.5, 'Send unfinished text at most every', lo=0.1, hi=5.0, step=0.1, unit='s',
                                    decimals=1, advanced=True, live=True)


@dataclass
class MonitorCfg:
    enabled: bool = opt(True, 'Show the monitor strip', live=True)
    show_wave: bool = opt(True, 'Waveform', live=True)
    show_vad: bool = opt(True, 'Speech detection', live=True)
    show_ram: bool = opt(True, 'Memory', live=True)
    show_cpu: bool = opt(True, 'CPU', live=True)
    show_busy: bool = opt(True, 'Inference load', live=True)
    show_gpu: bool = opt(True, 'GPU (Windows)', live=True)
    wave_s: float = opt(6.0, 'Waveform length', lo=1.0, hi=60.0, step=1.0, unit='s', decimals=0, live=True)
    fps: int = opt(25, 'Waveform frame rate', lo=5, hi=30, unit='fps', live=True)
    sample_s: float = opt(1.0, 'Resource sampling interval', lo=0.5, hi=10.0, step=0.5, unit='s', decimals=1, live=True)
    cpu_per_core: bool = opt(False, 'CPU % per core (100 % = one core)', live=True)
    silent_warn_s: int = opt(10, 'Warn when the input is silent for', lo=0, hi=600, unit='s', live=True, tip='0 = never.')


@dataclass
class GeneralCfg:
    ui_lang: str = opt('en', 'Interface language', choices=[('en', 'English'), ('zh_CN', '简体中文')], live=True)
    start_minimized: bool = opt(False, 'Start minimised to the tray')
    low_priority: bool = opt(False, 'Run at low priority', tip='Other programs come first; captions may lag under load.')
    eco_qos: bool = opt(False, 'Windows efficiency mode (EcoQoS)',
                        tip='Windows 11 runs the app on efficient cores at low clock speed. Saves power; check that captions keep up.')
    log_level: int = opt(3, 'Native log level', lo=1, hi=5, advanced=True, tip='1 debug, 2 info, 3 warnings, 4 errors, 5 silent.')


@dataclass
class Config:
    version: int = CONFIG_VERSION
    audio: AudioCfg = field(default_factory=AudioCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    recog: RecogCfg = field(default_factory=RecogCfg)
    stream: StreamCfg = field(default_factory=StreamCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    filters: FilterCfg = field(default_factory=FilterCfg)
    terms: TermsCfg = field(default_factory=TermsCfg)
    save: SaveCfg = field(default_factory=SaveCfg)
    caption: CaptionCfg = field(default_factory=CaptionCfg)
    share: ShareCfg = field(default_factory=ShareCfg)
    monitor: MonitorCfg = field(default_factory=MonitorCfg)
    general: GeneralCfg = field(default_factory=GeneralCfg)


SECTIONS = [f.name for f in fields(Config) if f.name != 'version']
