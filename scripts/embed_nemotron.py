#!/usr/bin/env python3
"""Embed only verified Nemotron result text without changing other dashboard data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / 'results/dashboard.html'
RESULTS = ROOT / 'results/corpus'
BEGIN = '\n// ---- nemotron results ----'
END = '// ---- end nemotron results ----'
ANCHOR = '// ---- end corpus ----'
HUMAN = {'brook': 'brook.wav', 'silly': 'silly.m4a', 'sing': 'sing.mp3'}


def main():
    values = {}
    for path in sorted(RESULTS.glob('*.nemotron.json')):
        stem = path.name[:-len('.nemotron.json')]
        payload = json.loads(path.read_text())
        text = payload.get('text')
        if payload.get('status') == 'ok' and isinstance(text, str) and text.strip():
            values[HUMAN.get(stem, stem + '.wav')] = text.strip()
    encoded = json.dumps(values, ensure_ascii=True).replace('</', '<\\/')
    block = (BEGIN + '\nconst NEMOTRON_RESULTS = ' + encoded
             + ';\nfor (const [file, text] of Object.entries(NEMOTRON_RESULTS)) {\n'
             + '  if (CORPUS[file]) CORPUS[file].nemotron = text;\n}\n' + END)
    html = HTML.read_text()
    if BEGIN in html and END in html:
        before, remaining = html.split(BEGIN, 1)
        _, after = remaining.split(END, 1)
        html = before + block + after
    elif BEGIN in html or END in html:
        raise ValueError('incomplete Nemotron results markers')
    elif html.count(ANCHOR) == 1:
        html = html.replace(ANCHOR, ANCHOR + block, 1)
    else:
        raise ValueError('expected one corpus anchor')
    HTML.write_text(html)
    print(f'embedded {len(values)} Nemotron transcripts')


if __name__ == '__main__':
    main()
