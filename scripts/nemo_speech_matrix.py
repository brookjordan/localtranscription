#!/usr/bin/env python3
"""Run NVIDIA NeMo-Speech.cpp (speech.cpp arm) over the existing transcription corpus.

Mirrors scripts/nemotron_matrix.py's result schema so the GGUF arm is directly
comparable with the MLX Audio NVFP4 arm. The runtime and GGUFs must already be
installed and verified; this runner never fetches weights or overwrites results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ('Balajee', 'Brett', 'Brook 2', 'Dave', "Mum's voice", 'Nana Chris',
          'Ping', 'Pritisman', 'Tung', 'antony billington', 'conrad',
          'lockpicking lawyer voice')
HUMAN = (('brook', 'brook.wav'), ('silly', 'silly.m4a'), ('sing', 'sing.mp3'))
SYNTHETIC = ('tse3-two', 'tse3-three', 'tse3-codeswitch', 'tse3-two-wide')


def matrix_jobs(assets: Path) -> list[tuple[str, Path]]:
    return ([(stem, assets / 'corpus' / f'{stem}.wav') for stem in (*CORPUS, *SYNTHETIC)]
            + [(stem, assets / name) for stem, name in HUMAN])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def duration_seconds(path: Path) -> float:
    output = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], text=True)
    duration = float(output.strip())
    if duration <= 0:
        raise ValueError('audio duration must be positive')
    return duration


def run_clip(nemo: str, model: Path, audio: Path, output: Path, metadata: dict) -> dict:
    if output.exists():
        raise FileExistsError(output)
    payload = dict(metadata)
    payload.update({'audio_file': audio.name, 'audio_sha256': sha256(audio),
                    'captured_at': datetime.now(timezone.utc).isoformat()})
    duration = duration_seconds(audio)
    payload['duration_s'] = duration
    with tempfile.TemporaryDirectory(prefix='nemo-speech-') as folder:
        normal = Path(folder) / 'audio.wav'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(audio),
                        '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                        str(normal)], check=True, capture_output=True)
        start = time.monotonic()
        proc = subprocess.run(
            [nemo, 'transcribe', str(normal), '--model', str(model),
             '--format', 'json', '--diarize', '--output-dir', folder],
            capture_output=True, text=True, timeout=1800)
        elapsed = time.monotonic() - start
        payload['runtime'] = {'nemo_speech_version': metadata['runtime_revision'],
                              'returncode': proc.returncode,
                              'stderr_tail': proc.stderr[-2000:]}
        json_files = sorted(Path(folder).glob('*.json'))
        try:
            if proc.returncode != 0:
                raise RuntimeError(f'nemo-speech exited {proc.returncode}')
            if not json_files:
                raise RuntimeError('no JSON output written')
            result = json.loads(json_files[0].read_text())
            text = str(result.get('text') or '').strip()
            if not text:
                raise ValueError('empty transcript')
            payload.update({'status': 'ok', 'text': text, 'wall_s': round(elapsed, 3),
                            'rtf': round(elapsed / duration, 4),
                            'segments': result.get('segments') or result.get('words'),
                            'raw_keys': sorted(result.keys()),
                            'process_peak_rss_bytes': resource.getrusage(
                                resource.RUSAGE_SELF).ru_maxrss})
        except Exception as exc:
            payload.update({'status': 'error', 'wall_s': round(elapsed, 3),
                            'error': f'{type(exc).__name__}: {exc}'})
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to the IQ4_NL decoder GGUF (encoder found beside it)')
    parser.add_argument('--model-revision', required=True,
                        help='Upstream base-model revision the GGUF was converted from')
    parser.add_argument('--runtime-revision', required=True, help='nemo-speech version string')
    parser.add_argument('--nemo', default='nemo-speech', help='Path to the nemo-speech CLI')
    parser.add_argument('--available-only', action='store_true',
                        help='Run available clips, but exit non-zero if any inputs are missing')
    parser.add_argument('--only', nargs='*',
                        help='Restrict to these stems (smoke tests); default is the full matrix')
    parser.add_argument('--suffix', default='-gguf',
                        help='Output filename suffix (default: -gguf)')
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    if not model.is_file():
        parser.error(f'--model must be an existing GGUF file: {model}')
    assets = ROOT / 'results' / 'assets'
    jobs = matrix_jobs(assets)
    if args.only:
        jobs = [(s, p) for s, p in jobs if s in set(args.only)]
    missing = [str(p) for _, p in jobs if not p.is_file()]
    if missing and not args.available_only:
        parser.error('missing input files: ' + ', '.join(missing))
    if missing:
        print('MISSING matrix inputs: ' + ', '.join(missing), flush=True)
        jobs = [(s, p) for s, p in jobs if p.is_file()]
    if not jobs:
        parser.error('no matrix inputs available')
    if args.suffix and not re.fullmatch(r'[-A-Za-z0-9]+', args.suffix):
        parser.error('--suffix must be hyphens, letters or digits only')
    dest = ROOT / 'results' / 'corpus'
    metadata = {'engine': 'nemo-speech.cpp', 'model': model.name,
                'model_revision': args.model_revision,
                'runtime_revision': args.runtime_revision,
                'model_sha256': sha256(model),
                'settings': {'format': 'json', 'diarize': True, 'language': None}}
    for stem, audio in jobs:
        output = dest / f'{stem}.nemotron{args.suffix}.json'
        if output.exists():
            print(f'{stem}: exists, refusing to overwrite', flush=True)
            continue
        print(f'{stem}: running', flush=True)
        result = run_clip(args.nemo, model, audio, output, metadata)
        print(f'{stem}: {result["status"]} '
              f'({result.get("wall_s", "?")} s, rtf {result.get("rtf", "?")})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
