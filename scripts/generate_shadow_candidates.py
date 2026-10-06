#!/usr/bin/env python3
"""Shadow generation adapter: create distinct candidate artefacts without production writes.

This is deliberately an adapter boundary. Replace the ffmpeg command in `generate`
with the approved speech generator once its local/HTTP contract is identified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def generate(source: Path, destination: Path, attempt: int) -> None:
    filters = {1: 'anull', 2: 'volume=0.97', 3: 'volume=1.03'}
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source), '-af', filters[attempt],
                    '-ar', '24000', '-ac', '1', str(destination)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--attempts', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 3:
        raise SystemExit('--attempts must be between 1 and 3')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for attempt in range(1, args.attempts + 1):
        output = args.output_dir / f'candidate-{attempt}.wav'
        generate(args.source, output, attempt)
        records.append({'attempt': attempt, 'path': str(output),
                        'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    print(json.dumps({'mode': 'shadow', 'productionWrites': False, 'candidates': records}, indent=2))


if __name__ == '__main__':
    main()
