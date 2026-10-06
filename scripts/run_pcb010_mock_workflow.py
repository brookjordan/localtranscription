#!/usr/bin/env python3
"""Exercise PCB-010 selection with generated known-good/bad mock candidates."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from speech_candidate_evaluator import choose_candidate, evaluate_candidate


def audio(path: Path, effect: str = 'volume=0.2') -> None:
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=1',
                    '-af', effect, '-ar', '24000', '-ac', '1', str(path)], check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='pcb010-mock-') as directory:
        root = Path(directory)
        paths = [root / f'attempt-{number}.wav' for number in (1, 2, 3)]
        for path in paths:
            audio(path)
        attempts = [
            evaluate_candidate(paths[0], 'hello world', 'nonsense', {'targetScore': 0.8, 'otherScore': 0.1}, 1),
            evaluate_candidate(paths[1], 'hello world', 'hello world', {'targetScore': 0.8, 'otherScore': 0.1}, 2),
            evaluate_candidate(paths[2], 'hello world', 'hello world', {'targetScore': 0.8, 'otherScore': 0.1}, 3),
        ]
        selected = choose_candidate(attempts)
        report = {'workflow': 'pcb-010-three-attempt-mock', 'maxAttempts': 3,
                  'attempts': attempts, 'selectedAttempt': selected['attempt'],
                  'degraded': selected['degraded']}
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
