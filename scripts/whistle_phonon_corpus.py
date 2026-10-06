#!/usr/bin/env python3
"""Run Whistle and Phonon-2 over the shared WAV corpus."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results/assets/corpus"
OUT = ROOT / "results/corpus"
WHISTLE_WEIGHTS = ROOT / "whistle.cact"
HUMAN_AUDIO = {
    "brook": ROOT / "results/assets/voices/brook.wav",
    "silly": ROOT / "results/assets/wav/silly.wav",
    "sing": ROOT / "results/assets/wav/sing.wav",
}


def run_whistle(wav: Path) -> dict:
    import needle
    import soundfile as sf
    started = time.perf_counter()
    try:
        samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        chunk = int(29.5 * rate)
        parts = []
        for start in range(0, len(samples), chunk):
            result = needle.transcribe(samples[start:start + chunk], language="en", weights=str(WHISTLE_WEIGHTS), word_timestamps=True)
            offset = start / rate
            for word in result.get("words", []):
                word = dict(word)
                word["start"] += offset
                word["end"] += offset
                parts.append(word)
        text = " ".join(w["word"] for w in parts).strip()
        return {"engine": "whistle", "model": "Cactus-Compute/whistle", "text": text,
                "language": "en", "words": parts,
                "wall_s": round(time.perf_counter() - started, 3), "status": "ok"}
    except Exception as exc:
        return {"engine": "whistle", "model": "Cactus-Compute/whistle", "wall_s": round(time.perf_counter() - started, 3),
                "status": "error", "error": str(exc)}


def run_phonon(model, wav: Path) -> dict:
    started = time.perf_counter()
    try:
        result = model.transcribe(str(wav))
        if isinstance(result, tuple):
            text, _, decode_s = result
        elif isinstance(result, str):
            text, decode_s = result, None
        else:
            text, decode_s = result.get("text", str(result)), result.get("decode_s")
        record = {"engine": "phonon-2", "model": "FermionResearch/Phonon-2", "text": text,
                  "wall_s": round(time.perf_counter() - started, 3), "status": "ok"}
        if decode_s is not None:
            record["decode_s"] = decode_s
        return record
    except Exception as exc:
        return {"engine": "phonon-2", "model": "FermionResearch/Phonon-2", "wall_s": round(time.perf_counter() - started, 3),
                "status": "error", "error": str(exc)}


def _duration(wav: Path) -> float:
    import soundfile as sf
    return float(sf.info(str(wav)).duration)


def main() -> int:
    waves = sorted(CORPUS.glob("*.wav"))
    OUT.mkdir(parents=True, exist_ok=True)
    import needle
    whistle = None
    try:
        whistle = needle.Whistle(weights=str(WHISTLE_WEIGHTS))
    except Exception as exc:
        print(f"Whistle load failed: {exc}", file=sys.stderr)
    phonon = None
    try:
        from fermion._speech.engine_phonon2_cpu import load
        model_dir = Path.home() / ".cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6"
        phonon = load(model_dir, profile="five-value", backend="cpu", quiet=False)
    except Exception as exc:
        print(f"Phonon-2 load failed: {exc}", file=sys.stderr)
    for wav in waves:
        if whistle is not None:
            record = run_whistle(wav)
            record["duration_s"] = _duration(wav)
            record["rtf"] = round(record["wall_s"] / record["duration_s"], 4) if record["duration_s"] else None
            (OUT / f"{wav.stem}.whistle.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"whistle {wav.name}: {record['status']} {record['wall_s']}s", flush=True)
        if phonon is not None:
            record = run_phonon(phonon, wav)
            record["duration_s"] = _duration(wav)
            record["rtf"] = round(record["wall_s"] / record["duration_s"], 4) if record["duration_s"] else None
            (OUT / f"{wav.stem}.phonon2.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"phonon-2 {wav.name}: {record['status']} {record['wall_s']}s", flush=True)
    for stem, wav in HUMAN_AUDIO.items():
        if whistle is not None:
            record = run_whistle(wav); record["duration_s"] = _duration(wav); record["rtf"] = round(record["wall_s"] / record["duration_s"], 4)
            (OUT / f"{stem}.whistle.json").write_text(json.dumps(record, indent=2) + "\n")
        if phonon is not None:
            record = run_phonon(phonon, wav); record["duration_s"] = _duration(wav); record["rtf"] = round(record["wall_s"] / record["duration_s"], 4)
            (OUT / f"{stem}.phonon2.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
