#!/usr/bin/env python3
"""Run the Nemotron MLX baseline over the existing local transcription corpus.

The model must be a local, previously verified checkpoint. This runner never
fetches weights or overwrites an existing result. Model acquisition, runtime
installation and source revision verification are separate operations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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


def select_jobs(jobs, dest: Path, model_revision: str, runtime_revision: str,
                resume: bool, suffix: str = ''):
    pending, skipped = [], []
    for stem, audio in jobs:
        output = dest / f'{stem}.nemotron{suffix}.json'
        if not output.exists():
            pending.append((stem, audio))
            continue
        if not resume:
            raise FileExistsError(output)
        existing = json.loads(output.read_text())
        if (existing.get('status') != 'ok' or existing.get('model_revision') != model_revision
                or existing.get('runtime_revision') != runtime_revision
                or existing.get('audio_sha256') != sha256(audio)):
            raise ValueError(f'cannot resume with non-matching result: {output}')
        skipped.append(stem)
    return pending, skipped


def duration_seconds(path: Path) -> float:
    output = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], text=True)
    duration = float(output.strip())
    if duration <= 0:
        raise ValueError('audio duration must be positive')
    return duration


def run_clip(model, audio: Path, output: Path, metadata: dict, settings: dict) -> dict:
    """Transcribe a clip once and save its actual result without clobbering."""
    if output.exists():
        raise FileExistsError(output)
    payload = dict(metadata)
    payload.update({'audio_file': audio.name, 'audio_sha256': sha256(audio),
                    'captured_at': datetime.now(timezone.utc).isoformat(),
                    'settings': settings})
    duration = duration_seconds(audio)
    payload['duration_s'] = duration
    try:
        with tempfile.TemporaryDirectory(prefix='nemotron-audio-') as folder:
            normal = Path(folder) / 'audio.wav'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(audio),
                            '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                            str(normal)], check=True, capture_output=True)
            start = time.monotonic()
            result = model.generate(str(normal), **payload['settings'])
            elapsed = time.monotonic() - start
            text = str(result.text or '').strip()
            if not text:
                raise ValueError('empty transcript')
            payload.update({'status': 'ok', 'text': text, 'wall_s': elapsed,
                            'rtf': elapsed / duration,
                            'segments': getattr(result, 'segments', None),
                            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    except Exception as exc:
        payload.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Verified local checkpoint directory')
    parser.add_argument('--model-revision', required=True, help='Verified upstream model revision')
    parser.add_argument('--runtime-revision', required=True, help='Installed MLX Audio source revision')
    parser.add_argument('--available-only', action='store_true',
                        help='Run available clips, but exit non-zero if any matrix inputs are missing')
    parser.add_argument('--resume', action='store_true',
                        help='Skip only existing results with matching revisions and audio hashes')
    parser.add_argument('--task', default='asr',
                        choices=('asr', 'transcribe', 'translate', 'saa', 'rich', 'qa'),
                        help='Generation task (default: asr)')
    parser.add_argument('--word-timestamps', action='store_true',
                        help='Request CTC word segments with speaker IDs')
    parser.add_argument('--suffix', default='',
                        help='Output filename suffix, e.g. "-saa"; required unless the default name is free')
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir() or not (model_path / 'config.json').is_file():
        parser.error('--model must identify an existing local checkpoint with config.json')
    assets = ROOT / 'results' / 'assets'
    jobs = matrix_jobs(assets)
    missing = [str(path) for _, path in jobs if not path.is_file()]
    if missing and not args.available_only:
        parser.error('missing input files: ' + ', '.join(missing))
    if missing:
        print('MISSING matrix inputs: ' + ', '.join(missing), flush=True)
        jobs = [(stem, path) for stem, path in jobs if path.is_file()]
    if not jobs:
        parser.error('no matrix inputs available')
    dest = ROOT / 'results' / 'corpus'
    if args.suffix and not re.fullmatch(r'[-A-Za-z0-9]+', args.suffix):
        parser.error('--suffix must be hyphens, letters or digits only')
    settings = {'task': args.task, 'temperature': 0.0,
                'word_timestamps': args.word_timestamps, 'speculative': False,
                'max_tokens': 8192}
    try:
        jobs, skipped = select_jobs(jobs, dest, args.model_revision,
                                    args.runtime_revision, args.resume,
                                    suffix=args.suffix)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f'{len(skipped)} existing matching results retained', flush=True)
    if not jobs:
        print(f'no pending available inputs; {len(missing)} inputs missing', flush=True)
        return 1 if missing else 0
    from mlx_audio.stt import load
    started = time.monotonic()
    model = load(model_path, strict=True)
    load_s = time.monotonic() - started
    metadata = {'model_path': str(model_path), 'model_revision': args.model_revision,
                'runtime_revision': args.runtime_revision,
                'mlx_audio_version': importlib.metadata.version('mlx-audio'),
                'host': platform.platform(), 'machine': platform.machine(),
                'model_load_s': load_s}
    failed = 0
    for stem, audio in jobs:
        try:
            result = run_clip(model, audio,
                              dest / f'{stem}.nemotron{args.suffix}.json',
                              metadata, settings)
            failed += result['status'] != 'ok'
            print(f"{stem}: {result['status']}", flush=True)
        except Exception as exc:
            failed += 1
            print(f'{stem}: error before result saved: {type(exc).__name__}: {exc}', flush=True)
    print(f'finished {len(jobs)} clips; {failed} failed; {len(missing)} inputs missing', flush=True)
    return 1 if failed or missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
