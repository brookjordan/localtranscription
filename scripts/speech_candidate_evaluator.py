#!/usr/bin/env python3
"""Offline technical contract for PCB-010 speech candidates.

Model-backed transcript and CAM++ workers are deliberately injected through JSON
fields; this module owns deterministic normalisation, signal checks and ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

EVALUATOR_VERSION = 'pcb-010-evaluator-v0.2.0'
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)


_NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100',
}


def normalise_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'")
    text = text.replace('-', ' ').replace('—', ' ')
    words = WORD_RE.findall(text)
    return ' '.join(_NUMBER_WORDS.get(word, word) or word for word in words)


def _ffprobe(path: Path) -> dict[str, Any]:
    cmd = ['ffprobe', '-v', 'error', '-show_entries',
           'format=duration:stream=codec_name,sample_rate,channels',
           '-of', 'json', str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode or not proc.stdout:
        return {'ok': False, 'error': proc.stderr.strip() or 'ffprobe failed'}
    data = json.loads(proc.stdout)
    stream = next((s for s in data.get('streams', []) if 'sample_rate' in s), {})
    return {
        'ok': True,
        'codec': stream.get('codec_name'),
        'sampleRate': int(stream['sample_rate']) if stream.get('sample_rate') else None,
        'channels': stream.get('channels'),
        'durationSeconds': round(float(data.get('format', {}).get('duration', 0)), 6),
    }


def _pcm_stats(path: Path, sample_rate: int) -> dict[str, float]:
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-f', 's16le', '-ac', '1', '-ar', str(sample_rate), 'pipe:1']
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode or len(proc.stdout) < 2:
        return {'sampleCount': 0, 'peak': 0.0, 'clippedRatio': 1.0, 'silenceRatio': 1.0, 'rmsDbfs': -math.inf}
    import array
    samples = array.array('h')
    samples.frombytes(proc.stdout[:len(proc.stdout) - len(proc.stdout) % 2])
    vals = [abs(s) / 32768.0 for s in samples]
    peak = max(vals, default=0.0)
    clipped = sum(v >= 0.999 for v in vals) / len(vals)
    silent = sum(v < 0.001 for v in vals) / len(vals)
    rms = math.sqrt(sum(v * v for v in vals) / len(vals))
    return {'sampleCount': len(vals), 'peak': round(peak, 6), 'clippedRatio': round(clipped, 8), 'silenceRatio': round(silent, 8), 'rmsDbfs': round(20 * math.log10(rms), 4) if rms else -math.inf}


def technical_metrics(path: Path, expected_duration: float | None = None) -> dict[str, Any]:
    probe = _ffprobe(path)
    if not probe.get('ok'):
        return {'safe': False, 'decode': probe}
    stats = _pcm_stats(path, probe['sampleRate'])
    duration_ratio = None if not expected_duration else probe['durationSeconds'] / expected_duration
    safe = bool(stats['sampleCount'] and stats['clippedRatio'] <= 0.001 and stats['silenceRatio'] < 0.95)
    return {
        'safe': safe, 'decode': probe, 'codec': probe['codec'],
        'sampleRate': probe['sampleRate'], 'channels': probe['channels'],
        'durationSeconds': probe['durationSeconds'], 'durationRatio': duration_ratio,
        'peak': stats['peak'], 'clippedRatio': stats['clippedRatio'],
        'silenceRatio': stats['silenceRatio'], 'rmsDbfs': stats['rmsDbfs'],
        'discontinuities': {'count': None, 'method': 'unavailable; waveform scan pending'},
    }


def _word_confusion(expected_words: list[str], observed_words: list[str]) -> dict[str, int]:
    """Alignment-derived substitution/deletion/insertion counts (Levenshtein backtrace).

    jiwer-compatible: S + D = deletions-plus-substitutions on the reference side,
    I = insertions. Replaces the raw length-delta approximations that made a
    same-length garble outrank a near-correct attempt (PCB-022 defect, run 3092
    segment 14).
    """
    n, m = len(expected_words), len(observed_words)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[''] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
        back[i][0] = 'D'
    for j in range(1, m + 1):
        costs[0][j] = j
        back[0][j] = 'I'
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = costs[i - 1][j - 1] + (expected_words[i - 1] != observed_words[j - 1])
            dele = costs[i - 1][j] + 1
            ins = costs[i][j - 1] + 1
            best = min(sub, dele, ins)
            costs[i][j] = best
            back[i][j] = 'S' if best == sub else ('D' if best == dele else 'I')
    s = d = i = 0
    a, b = n, m
    while a > 0 or b > 0:
        move = back[a][b]
        if move == 'S':
            if expected_words[a - 1] != observed_words[b - 1]:
                s += 1
            a, b = a - 1, b - 1
        elif move == 'D':
            d += 1
            a -= 1
        else:
            i += 1
            b -= 1
    return {'substitutions': s, 'deletions': d, 'insertions': i}


def build_rank_vector(record: dict[str, Any]) -> list[float | int]:
    """Lexicographic ranking tuple (PCB-022 contract v0.2.0).

    Positions 4 and 5 now carry alignment-derived substitutions/deletions and
    insertions instead of raw word-count deltas, and WER is promoted to
    position 4, ahead of the counts, so a same-length garble can never
    outrank a near-correct attempt (run 3092 segment 14 defect).
    """
    technical = record.get('technical', {})
    voice = record.get('voice', {})
    transcript = record.get('transcript', {})
    defects = record.get('defects', {})
    return [
        int(bool(technical.get('safe'))), int(bool(voice.get('targetIsTopMatch'))),
        -int(transcript.get('protectedTermFailures') or 0),
        -float(transcript.get('wer', 1.0)),
        -int(transcript.get('substitutions', transcript.get('missingWords', 0))),
        -int(transcript.get('deletions', transcript.get('missingWords', 0))),
        -int(transcript.get('insertions', transcript.get('addedWords', 0))),
        float(voice.get('margin', -1.0)),
        float(voice.get('targetScore', -1.0)), -float(defects.get('severity') or 0.0),
        -int(record.get('attempt', 999)),
    ]


def _edit_distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, word in enumerate(left, 1):
        next_row = [i]
        for j, other in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = next_row
    return row[-1]


def evaluate_candidate(path: Path, expected_text: str, observed_text: str, voice: dict[str, float], attempt: int = 1, protected_terms: list[str] | None = None) -> dict[str, Any]:
    technical = technical_metrics(path)
    expected = normalise_text(expected_text)
    observed = normalise_text(observed_text)
    expected_words, observed_words = expected.split(), observed.split()
    distance = _edit_distance(expected_words, observed_words)
    wer = distance / max(1, len(expected_words))
    confusion = _word_confusion(expected_words, observed_words)
    voice_data = {
        'targetScore': float(voice.get('targetScore', -1.0)),
        'otherScore': float(voice.get('otherScore', -1.0)),
    }
    voice_data['margin'] = voice_data['targetScore'] - voice_data['otherScore']
    voice_data['targetIsTopMatch'] = voice_data['targetScore'] > voice_data['otherScore']
    cer = _edit_distance(list(expected), list(observed)) / max(1, len(expected))
    observed_joined = ' '.join(observed_words)
    protected_hits: list[str] = []
    for term in protected_terms or []:
        term_norm = ' '.join(normalise_text(term).split())
        if term_norm and term_norm not in observed_joined:
            protected_hits.append(term_norm)
    transcript = {'expected': expected, 'observed': observed, 'wer': wer, 'cer': cer,
                  'substitutions': confusion['substitutions'],
                  'deletions': confusion['deletions'],
                  'insertions': confusion['insertions'],
                  'missingWords': confusion['substitutions'] + confusion['deletions'],
                  'addedWords': confusion['insertions'],
                  'protectedTermFailures': len(protected_hits),
                  'protectedTermMisses': protected_hits,
                  'coverageComplete': None}
    defects = {'severity': None, 'status': 'unavailable'}
    record = {'evaluatorVersion': EVALUATOR_VERSION, 'attempt': attempt, 'technical': technical,
              'transcript': transcript, 'voice': voice_data, 'defects': defects}
    reasons = []
    if not technical.get('safe', False): reasons.append('technical_gate')
    if wer > 0: reasons.append('transcript_gate')
    if not voice_data['targetIsTopMatch'] or voice_data['targetScore'] < 0.35 or voice_data['margin'] < 0.15:
        reasons.append('voice_gate')
    record['rankVector'] = build_rank_vector(record)
    record['passed'] = not reasons
    record['rejectionReasons'] = reasons
    return record


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in sorted(candidates, key=lambda item: item['attempt']):
        if candidate.get('passed'):
            return {**candidate, 'degraded': False, 'selectionReason': 'first_passing_candidate'}
    selected = max(candidates, key=lambda item: tuple(item.get('rankVector', [])))
    return {**selected, 'degraded': True, 'selectionReason': 'deterministic_best_after_three_failures'}


def evaluate(path: Path, expected_text: str = '', observed_text: str = '', attempt: int = 1) -> dict[str, Any]:
    technical = technical_metrics(path)
    expected = normalise_text(expected_text)
    observed = normalise_text(observed_text)
    expected_words, observed_words = expected.split(), observed.split()
    missing = max(0, len(expected_words) - len(observed_words))
    added = max(0, len(observed_words) - len(expected_words))
    transcript = {'expected': expected, 'observed': observed, 'wer': 0.0 if expected == observed else 1.0,
                  'cer': 0.0 if expected == observed else 1.0, 'missingWords': missing,
                  'addedWords': added, 'protectedTermFailures': 0, 'coverageComplete': bool(observed)}
    record = {'evaluatorVersion': EVALUATOR_VERSION, 'attempt': attempt, 'technical': technical,
              'transcript': transcript, 'voice': {}, 'defects': {'severity': 0.0}}
    record['rankVector'] = build_rank_vector(record)
    record['passed'] = technical.get('safe', False) and expected == observed
    record['rejectionReasons'] = [] if record['passed'] else ['technical_or_transcript_gate']
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=Path)
    parser.add_argument('--expected-text', default='')
    parser.add_argument('--observed-text', default='')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    result = evaluate(args.audio, args.expected_text, args.observed_text, args.attempt)
    result['input'] = {'path': str(args.audio), 'sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
