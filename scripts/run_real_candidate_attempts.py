#!/usr/bin/env python3
"""Evaluate up to three real candidates and persist immutable attempt evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv-asr/bin/python'
EVALUATOR = ROOT / 'scripts/evaluate_real_candidate.py'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('expected_text')
    parser.add_argument('target_reference', type=Path)
    parser.add_argument('other_reference', type=Path)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('candidates', nargs='+', type=Path)
    args = parser.parse_args()
    if len(args.candidates) > 3:
        raise SystemExit('maximum three candidates')
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    selected = None
    records = []
    for attempt, candidate in enumerate(args.candidates, 1):
        command = [str(PYTHON), str(EVALUATOR), str(candidate), args.expected_text,
                   str(args.target_reference), str(args.other_reference), '--attempt', str(attempt)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            record = {'attempt': attempt, 'inputPath': str(candidate), 'passed': False,
                      'degraded': False, 'rejectionReasons': ['worker_error'],
                      'workerStderr': completed.stderr[-2000:]}
        else:
            record = json.loads(completed.stdout)
        records.append(record)
        with args.evidence.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        if record.get('passed'):
            selected = {**record, 'degraded': False, 'selectionReason': 'first_passing_candidate'}
            break
    if selected is None:
        selected = max(records, key=lambda item: tuple(item.get('rankVector', [-1])))
        selected = {**selected, 'degraded': True,
                    'selectionReason': 'deterministic_best_after_three_failures'}
    print(json.dumps({'attemptsEvaluated': len(records), 'selectedAttempt': selected['attempt'],
                      'degraded': selected['degraded'], 'selectionReason': selected['selectionReason']}, indent=2))


if __name__ == '__main__':
    main()
