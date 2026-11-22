"""Transcript writers: each appends finished lines, so a crash loses at most the last flush interval."""
import json


def sec2clock(t, sep=',', with_ms=True):
    """Convert seconds to hh:mm:ss[,mmm]."""
    ms = int(round(max(0.0, t) * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f'{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}' if with_ms else f'{h:02d}:{m:02d}:{s:02d}'


class TextWriter:
    ext = '.txt'
    encoding = 'utf-8-sig'  # a BOM, so every Windows editor shows Chinese correctly

    def __init__(self, base_path):
        self.path = base_path + self.ext
        self.f = open(self.path, 'w', encoding=self.encoding, newline='\n')
        self.n = 0
        self.write_header()

    def write_header(self):
        pass

    def write_line(self, line):
        self.n += 1
        self.f.write(self.format_line(line))

    def format_line(self, line):
        return f'[{sec2clock(line.t0, with_ms=False)}] {line.text}\n'

    def flush(self):
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


class SrtWriter(TextWriter):
    ext = '.srt'
    encoding = 'utf-8'

    def format_line(self, line):
        t1 = max(line.t1, line.t0 + 0.5)
        return f'{self.n}\n{sec2clock(line.t0)} --> {sec2clock(t1)}\n{line.text}\n\n'


class VttWriter(TextWriter):
    ext = '.vtt'
    encoding = 'utf-8'

    def write_header(self):
        self.f.write('WEBVTT\n\n')

    def format_line(self, line):
        t1 = max(line.t1, line.t0 + 0.5)
        return f'{sec2clock(line.t0, ".")} --> {sec2clock(t1, ".")}\n{line.text}\n\n'


class JsonlWriter(TextWriter):
    ext = '.jsonl'
    encoding = 'utf-8'

    def format_line(self, line):
        return json.dumps({'t0': round(line.t0, 3), 't1': round(line.t1, 3), 'text': line.text}, ensure_ascii=False) + '\n'


class MarkdownWriter(TextWriter):
    ext = '.md'
    encoding = 'utf-8'

    def write_header(self):
        self.f.write('# Transcript\n\n')

    def format_line(self, line):
        return f'- **{sec2clock(line.t0, with_ms=False)}** {line.text}\n'


WRITERS = {'txt': TextWriter, 'srt': SrtWriter, 'vtt': VttWriter, 'jsonl': JsonlWriter, 'md': MarkdownWriter}
