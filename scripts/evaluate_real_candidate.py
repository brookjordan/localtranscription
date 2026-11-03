#!/usr/bin/env python3
"""Run the real isolated ASR and CAM++ workers for one speech candidate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from .speech_candidate_evaluator import evaluate_candidate
except ImportError:
    from speech_candidate_evaluator import evaluate_candidate

ROOT = Path(__file__).resolve().parents[1]
ASR_PYTHON = ROOT / '.venv-asr/bin/python'
HEAR_PYTHON = Path('/Users/brook.jordan/git/motional/brook.jordan/hear/.venv/bin/python')
CAMPPLUS = ROOT / 'scripts/campplus_probe.py'
PARAKEET = ROOT / 'scripts/parakeet_one.py'


def run_json(command: list[str]) -> dict:
    # launchd does not inherit the interactive shell's Homebrew PATH.
    worker_env = {**os.environ, 'PATH': '/opt/homebrew/bin:' + os.environ.get('PATH', '')}
    dependency = None
    for attempt in range(2):
        result = subprocess.run(command, capture_output=True, text=True, env=worker_env)
        if result.returncode == 0:
            return json.loads(result.stdout)
        dependency = 'ffmpeg' if 'FFmpeg is not installed or not in your PATH' in result.stderr else None
        print(f'worker {Path(command[1]).name} exited {result.returncode} (attempt {attempt + 1}/2): '
              f'{result.stderr[-4000:]}', file=sys.stderr, flush=True)
        if dependency:
            break
        if attempt == 0:
            time.sleep(2)
    raise WorkerProcessError(Path(command[1]).name, dependency)


class WorkerProcessError(RuntimeError):
    def __init__(self, worker: str, dependency: str | None):
        super().__init__(f'{worker} failed; inspect private service log')
        self.dependency = dependency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('expected_text')
    parser.add_argument('target_reference', type=Path)
    parser.add_argument('other_reference', type=Path)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    asr = run_json([str(ASR_PYTHON), str(PARAKEET), str(args.candidate)])
    voice_probe = run_json([str(HEAR_PYTHON), str(CAMPPLUS), str(args.candidate), str(args.target_reference), str(args.other_reference)])
    result = evaluate_candidate(args.candidate, args.expected_text, asr['text'], {
        'targetScore': voice_probe['scores']['target'], 'otherScore': voice_probe['scores']['other']}, args.attempt)
    result.update({'inputPath': str(args.candidate), 'expectedText': args.expected_text,
                   'asr': asr, 'voiceProbe': voice_probe})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
