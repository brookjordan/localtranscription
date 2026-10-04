#!/usr/bin/env python3
"""Run diarization-capable engines over the new TSE-03 clips.

Follows vibevoice_corpus.py conventions: one JSON per (clip, engine) in
results/corpus/, resumable (skips existing), Metal GPU-timeout guard.
VibeVoice is the corpus baseline convention (<stem>.json).
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "assets" / "corpus"
OUTDIR = ROOT / "results" / "corpus"
CLIPS = ["tse3-two", "tse3-three", "tse3-codeswitch"]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def run_vibevoice() -> None:
    from mlx_audio.stt.utils import load_model

    model = load_model("microsoft/VibeVoice-ASR")
    for stem in CLIPS:
        f = CORPUS / f"{stem}.wav"
        out = OUTDIR / f"{stem}.json"
        if out.exists():
            print(f"skip {stem} (done)", flush=True)
            continue
        dur = 0.0
        wall = 0.0
        result = None
        # Metal GPU-timeout guard: on failure, drop the model, wait, reload, retry.
        for attempt in range(1, 4):
            try:
                dur = wav_duration(f)
                t0 = time.perf_counter()
                result = model.generate(str(f))
                wall = time.perf_counter() - t0
                break
            except RuntimeError as e:
                print(f"attempt {attempt} failed on {f.name}: {e}", flush=True)
                if attempt == 3:
                    raise
                model = None
                time.sleep(30 * attempt)
                model = load_model("microsoft/VibeVoice-ASR")
        segs = [
            {
                "start": s.get("start"),
                "end": s.get("end"),
                "speaker_id": s.get("speaker_id"),
                "text": (s.get("text") or "").strip(),
            }
            for s in (result.segments or [])
        ]
        payload = {
            "file": f.name,
            "duration_s": round(dur, 3),
            "rtf": round(wall / dur, 3) if dur else None,
            "segments": segs,
            "text": result.text.strip(),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"done vibevoice: {stem} wall={wall:.1f}s rtf={wall/dur:.2f}", flush=True)


if __name__ == "__main__":
    run_vibevoice()
