#!/usr/bin/env python3
"""Run the official SenseVoice llama.cpp runtime over the transcription corpus.

Mirrors nemotron_matrix.py's schema. The binary, GGUFs and VAD model must be
present and verified; this runner never fetches weights or overwrites results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
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


def run_clip(binary: str, model: Path, vad: Path, audio: Path, output: Path,
             metadata: dict, threads: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    payload = dict(metadata)
    payload.update({'audio_file': audio.name, 'audio_sha256': sha256(audio),
                    'captured_at': datetime.now(timezone.utc).isoformat()})
    duration = duration_seconds(audio)
    payload['duration_s'] = duration
    normal = Path('/tmp') / f'svofficial-{audio.stem}.wav'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio),
                    '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(normal)],
                   check=True, capture_output=True)
    start = time.monotonic()
    proc = subprocess.run(
        [binary, '-m', str(model), '-a', str(normal), '--vad', str(vad)],
        capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - start
    payload['runtime'] = {'returncode': proc.returncode, 'stderr_tail': proc.stderr[-1500:]}
    try:
        if proc.returncode != 0:
            raise RuntimeError(f'binary exited {proc.returncode}')
        text = ' '.join(proc.stdout.split())
        if not text:
            raise ValueError('empty transcript')
        payload.update({'status': 'ok', 'text': text, 'wall_s': round(elapsed, 3),
                        'rtf': round(elapsed / duration, 4),
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
    parser.add_argument('--binary', required=True, help='llama-funasr-sensevoice binary')
    parser.add_argument('--model', type=Path, required=True, help='sensevoice GGUF')
    parser.add_argument('--vad', type=Path, required=True, help='fsmn-vad GGUF')
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--runtime-revision', required=True,
                        help='runtime/llama.cpp commit of the built binary')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--suffix', default='-svofficial',
                        help='result filename suffix (default: -svofficial)')
    args = parser.parse_args()

    if args.suffix and not re.fullmatch(r'[-A-Za-z0-9]+', args.suffix):
        parser.error('--suffix must be hyphens, letters or digits only')
    assets = ROOT / 'results' / 'assets'
    dest = ROOT / 'results' / 'corpus'
    metadata = {'engine': 'sensevoice-official-llamacpp',
                'model': Path(args.model).name,
                'model_sha256': sha256(Path(args.model)),
                'model_revision': args.model_revision,
                'runtime_revision': args.runtime_revision,
                'settings': {'vad': Path(args.vad).name, 'threads': args.threads}}
    for stem, audio in matrix_jobs(assets):
        if not audio.is_file():
            print(f'{stem}: MISSING input {audio}', flush=True)
            continue
        output = dest / f'{stem}.sensevoice{args.suffix}.json'
        if output.exists():
            print(f'{stem}: exists, refusing to overwrite', flush=True)
            continue
        result = run_clip(args.binary, Path(args.model), Path(args.vad), audio,
                          output, metadata, args.threads)
        print(f'{stem}: {result["status"]} ({result.get("wall_s")} s, '
              f'rtf {result.get("rtf")})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
