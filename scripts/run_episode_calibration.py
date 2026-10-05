#!/usr/bin/env python3
"""Run the PCB-010 v0.1 contract over an episode folder without production writes."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from speech_candidate_evaluator import evaluate, normalise_text

MODEL = 'mlx-community/parakeet-tdt-0.6b-v3'

def prompt_text(path: Path) -> str:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format_tags=prompt', '-of', 'default=noprint_wrappers=1', str(path)]
    raw = subprocess.check_output(cmd, text=True).strip()
    payload = json.loads(raw.removeprefix('TAG:prompt='))
    return payload['2']['inputs']['text']

def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit('usage: run_episode_calibration.py EPISODE_DIR OUTPUT_JSONL')
    episode, output = Path(sys.argv[1]), Path(sys.argv[2])
    from parakeet_mlx import from_pretrained
    model = from_pretrained(MODEL)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        for path in sorted(episode.glob('audio_*.mp3'), key=lambda p: int(p.stem.split('_')[1])):
            expected = prompt_text(path)
            started = time.time()
            result = model.transcribe(str(path))
            observed = ' '.join(s.text.strip() for s in (getattr(result, 'sentences', None) or []))
            record = evaluate(path, expected, observed)
            record.update({'segmentId': path.stem, 'expectedText': expected, 'observedText': observed,
                           'asrModel': MODEL, 'wallSeconds': round(time.time() - started, 3),
                           'normalisedExpected': normalise_text(expected),
                           'normalisedObserved': normalise_text(observed)})
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            print(path.name, record['passed'], record['wallSeconds'], flush=True)

if __name__ == '__main__':
    main()
