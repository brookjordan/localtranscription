#!/usr/bin/env python3
"""Whisper large-v3 / distil-large-v3 (MLX) over the corpus.

Usage: python3 scripts/whisper_variants_corpus.py <large-v3|distil-large-v3>
Writes results/corpus/<stem>.<variant>.json with {start,end,speaker,text}
segments. Resumable. Wall-clock timing per file.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "assets" / "corpus"
OUTDIR = ROOT / "results" / "corpus"

VARIANTS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def transcribe(variant: str, hint: str, wav: Path):
    import mlx_whisper

    # Pin English: these are English-only evaluation targets, and unpinned
    # language detection misfired to Welsh on `antony billington.wav`
    # (whole clip came back fluent Welsh). English-only variants ignore this.
    return mlx_whisper.transcribe(str(wav), path_or_hf_repo=hint, language="en")


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    if variant not in VARIANTS:
        print(f"usage: {sys.argv[0]} <{'|'.join(VARIANTS)}>")
        raise SystemExit(2)
    hint = VARIANTS[variant]
    print(f"{variant} corpus: model={hint}", flush=True)
    for wav in sorted(ASSETS.glob("*.wav")):
        out = OUTDIR / f"{wav.stem}.{variant}.json"
        if out.exists() and out.stat().st_size > 0:
            print(f"skip {wav.stem} (exists)", flush=True)
            continue
        t0 = time.time()
        try:
            result = transcribe(variant, hint, wav)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {variant}: {wav.stem}: {exc}", flush=True)
            out.write_text(json.dumps({"error": str(exc)}))
            continue
        wall = time.time() - t0
        segments = []
        for s in result.get("segments", []):
            segments.append({
                "start": round(float(s["start"]), 3),
                "end": round(float(s["end"]), 3),
                "speaker": None,
                "text": s["text"].strip(),
            })
        payload = {
            "model": variant,
            "audio_file": wav.name,
            "segments": segments,
            "text": result.get("text", "").strip(),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"done {variant}: {wav.stem} wall={wall:.2f}s segs={len(segments)}", flush=True)
    print(f"{variant} pass complete", flush=True)


if __name__ == "__main__":
    main()
