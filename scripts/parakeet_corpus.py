#!/usr/bin/env python3
"""Parakeet TDT 0.6B v3 (mlx-community/parakeet-tdt-0.6b-v3) over the corpus.

Outputs one .parakeet.json per clip in results/corpus/:
  {"start": float, "end": float, "speaker": str|null, "text": str} segments
Skips clips whose output already exists (resumable). Timing: wall + RTF.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "assets" / "corpus"
OUTDIR = ROOT / "results" / "corpus"
MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def main() -> None:
    from parakeet_mlx import from_pretrained

    model = from_pretrained(MODEL)
    waves = sorted(ASSETS.glob("*.wav"))
    print(f"parakeet corpus: {len(waves)} files, model={MODEL}", flush=True)
    total_audio = 0.0
    total_wall = 0.0
    for wav in waves:
        out = OUTDIR / f"{wav.stem}.parakeet.json"
        if out.exists() and out.stat().st_size > 0:
            print(f"skip {wav.stem} (exists)", flush=True)
            continue
        t0 = time.time()
        try:
            result = model.transcribe(str(wav))
        except Exception as exc:  # noqa: BLE001 — record failure as a finding
            print(f"FAIL parakeet: {wav.stem}: {exc}", flush=True)
            out.write_text(json.dumps({"error": str(exc)}))
            continue
        wall = time.time() - t0
        segs = getattr(result, "sentences", None) or []
        segments = [
            {
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "speaker": None,
                "text": s.text.strip(),
            }
            for s in segs
        ]
        payload = {
            "model": "parakeet-tdt-0.6b-v3",
            "audio_file": wav.name,
            "duration_s": wall and None,  # filled below
            "segments": segments,
            "text": " ".join(s["text"] for s in segments).strip(),
        }
        # duration from audio file via result metadata if exposed
        dur = getattr(result, "audio_duration", None) or getattr(result, "duration", None)
        payload["duration_s"] = round(float(dur), 2) if dur else None
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        total_audio += float(dur) if dur else 0.0
        total_wall += wall
        rtf = f" rtf={wall / float(dur):.3f}" if dur else ""
        print(f"done parakeet: {wav.stem} wall={wall:.2f}s{rtf}", flush=True)
    if total_audio:
        print(
            f"parakeet pass complete: total audio={total_audio:.0f}s wall={total_wall:.0f}s rtf={total_wall / total_audio:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
