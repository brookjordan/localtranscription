#!/usr/bin/env python3
"""Qwen3-ASR 1.7B (MLX, 8-bit) across the whole corpus, with RTF.

Usage: .venv-asr/bin/python scripts/qwen3_corpus.py
Writes results/corpus/<stem>.qwen3.json: { file, wall_s, rtf, text }
"""
import json
import time
import wave
from pathlib import Path

from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model

CORPUS = Path(__file__).resolve().parent.parent / "results" / "assets" / "corpus"
OUTDIR = Path(__file__).resolve().parent.parent / "results" / "corpus"
REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CORPUS.glob("*.wav"))
    print(f"{len(files)} corpus files", flush=True)
    model = load_model(REPO)
    for f in files:
        out = OUTDIR / (f.stem + ".qwen3.json")
        if out.exists():
            print(f"skip {f.name} (done)", flush=True)
            continue
        dur = wav_duration(f)
        t0 = time.perf_counter()
        result = generate_transcription(model=model, audio=str(f), format="txt")
        elapsed = time.perf_counter() - t0
        text = result.text.strip() if hasattr(result, "text") else str(result).strip()
        payload = {"file": f.name, "wall_s": round(elapsed, 2), "rtf": round(elapsed / dur, 3), "text": text}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"done {f.name}: RTF {payload['rtf']}", flush=True)


if __name__ == "__main__":
    main()
