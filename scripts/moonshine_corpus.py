#!/usr/bin/env python3
"""Moonshine base (ONNX runtime) over the corpus.

Writes results/corpus/<stem>.moonshine.txt (plain text; no timestamps/speaker
in this API). Resumable. Wall-clock timing per file.
"""
import json
import time
from pathlib import Path

import moonshine_onnx

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "assets" / "corpus"
OUTDIR = ROOT / "results" / "corpus"


def transcribe_chunked(path: Path, chunk_s: float = 30.0) -> str:
    """Moonshine hard-limits a single call to <64s audio; split longer files
    into fixed chunks on silence-seeking boundaries (simple: fixed windows)."""
    import numpy as np
    from moonshine_onnx import load_audio

    audio = load_audio(str(path))
    if audio.ndim == 2:
        audio = audio[0]  # [batch, samples] -> [samples]
    sr = 16000
    n = int(chunk_s * sr)
    pieces = []
    for i in range(0, len(audio), n):
        seg = audio[i : i + n]
        if len(seg) < 0.1 * sr:
            break
        t = moonshine_onnx.transcribe(seg, model="moonshine/base")
        if isinstance(t, list):
            t = " ".join(str(x) for x in t)
        pieces.append(str(t).strip())
    return " ".join(p for p in pieces if p)


def main() -> None:
    print("moonshine corpus: model=moonshine/base (ONNX)", flush=True)
    for wav in sorted(ASSETS.glob("*.wav")):
        out = OUTDIR / f"{wav.stem}.moonshine.txt"
        if out.exists() and out.stat().st_size > 0:
            print(f"skip {wav.stem} (exists)", flush=True)
            continue
        t0 = time.time()
        try:
            text = transcribe_chunked(wav)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL moonshine: {wav.stem}: {exc}", flush=True)
            out.write_text(f"ERROR: {exc}")
            continue
        wall = time.time() - t0
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        out.write_text(str(text).strip())
        print(f"done moonshine: {wav.stem} wall={wall:.2f}s", flush=True)
    print("moonshine pass complete", flush=True)


if __name__ == "__main__":
    main()
