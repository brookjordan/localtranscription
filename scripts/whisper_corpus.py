#!/usr/bin/env python3
"""MLX Whisper large-v3-turbo across the whole corpus, with RTF.

Usage: .venv-asr/bin/python scripts/whisper_corpus.py
Writes results/corpus/<stem>.whisper.json: { file, wall_s, rtf, text }
"""
import json
import time
import wave
from pathlib import Path

import mlx_whisper

CORPUS = Path(__file__).resolve().parent.parent / "results" / "assets" / "corpus"
OUTDIR = Path(__file__).resolve().parent.parent / "results" / "corpus"
REPO = "mlx-community/whisper-large-v3-turbo"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CORPUS.glob("*.wav"))
    print(f"{len(files)} corpus files", flush=True)
    for f in files:
        out = OUTDIR / (f.stem + ".whisper.json")
        if out.exists():
            print(f"skip {f.name} (done)", flush=True)
            continue
        dur = wav_duration(f)
        t0 = time.perf_counter()
        result = mlx_whisper.transcribe(str(f), path_or_hf_repo=REPO, verbose=None, language="en")
        elapsed = time.perf_counter() - t0
        text = result["text"].strip()
        payload = {"file": f.name, "wall_s": round(elapsed, 2), "rtf": round(elapsed / dur, 3), "text": text}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"done {f.name}: RTF {payload['rtf']}", flush=True)


if __name__ == "__main__":
    main()
