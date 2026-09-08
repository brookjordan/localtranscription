#!/usr/bin/env python3
"""Transcribe all corpus WAVs with VibeVoice-ASR (baseline transcripts).

Usage: .venv-asr/bin/python scripts/vibevoice_corpus.py
Writes one JSON per file into results/corpus/:
  { "file", "duration_s", "rtf", "segments": [{start,end,speaker_id,text}], "text" }
"""
import json
import time
import wave
from pathlib import Path

from mlx_audio.stt.utils import load_model

CORPUS = Path(__file__).resolve().parent.parent / "results" / "assets" / "corpus"
OUTDIR = Path(__file__).resolve().parent.parent / "results" / "corpus"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CORPUS.glob("*.wav"))
    print(f"{len(files)} corpus files", flush=True)
    model = load_model("microsoft/VibeVoice-ASR")  # load once
    for f in files:
        out = OUTDIR / (f.stem + ".json")
        if out.exists():
            print(f"skip {f.name} (done)", flush=True)
            continue
        dur = 0.0
        t0 = time.perf_counter()
        result = None
        # Metal GPU-timeout guard: on failure, drop the model, wait, reload, retry.
        for attempt in range(1, 4):
            try:
                dur = wav_duration(f)
                t0 = time.perf_counter()
                result = model.generate(str(f))
                break
            except RuntimeError as e:
                print(f"attempt {attempt} failed on {f.name}: {e}", flush=True)
                if attempt == 3:
                    raise
                model = None
                time.sleep(30 * attempt)
                model = load_model("microsoft/VibeVoice-ASR")
        if result is None:
            raise RuntimeError(f"{f.name}: transcription failed after 3 attempts")
        elapsed = result.total_time or (time.perf_counter() - t0)
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
            "rtf": round(elapsed / dur, 3) if dur else None,
            "segments": segs,
            "text": result.text.strip(),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"done {f.name}: {len(segs)} segs, {dur:.1f}s audio, RTF {payload['rtf']}", flush=True)


if __name__ == "__main__":
    main()
