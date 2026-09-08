#!/usr/bin/env python3
"""Run the new engines (parakeet, distil-large-v3, large-v3, moonshine, kyutai)
over the three human-scored clips (brook.wav, silly, sing) so they get real WER.

Writes results/smoke/<engine>.<name>.txt|json matching the corpus naming so the
dashboard embedder can pick them up with a small change. Resumable.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "assets"
SMOKE = [
    ("brook", ASSETS / "brook.wav"),
    ("silly", ASSETS / "silly.m4a"),
    ("sing", ASSETS / "sing.mp3"),
]
OUTDIR = ROOT / "results" / "smoke"


def run_parakeet():
    from parakeet_mlx import from_pretrained

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    for stem, path in SMOKE:
        out = OUTDIR / f"parakeet.{stem}.json"
        if out.exists():
            continue
        r = model.transcribe(str(path))
        segs = [
            {"start": round(float(s.start), 3), "end": round(float(s.end), 3), "speaker": None, "text": s.text.strip()}
            for s in (getattr(r, "sentences", None) or [])
        ]
        out.write_text(json.dumps({"model": "parakeet-tdt-0.6b-v3", "segments": segs, "text": " ".join(s["text"] for s in segs)}))
        print(f"parakeet {stem}: {len(segs)} segs", flush=True)


def run_whisper_variant(variant: str, repo: str):
    import mlx_whisper

    for stem, path in SMOKE:
        out = OUTDIR / f"{variant}.{stem}.json"
        if out.exists():
            continue
        r = mlx_whisper.transcribe(str(path), path_or_hf_repo=repo)
        segs = [
            {"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3), "speaker": None, "text": s["text"].strip()}
            for s in r.get("segments", [])
        ]
        out.write_text(json.dumps({"model": variant, "segments": segs, "text": r.get("text", "").strip()}))
        print(f"{variant} {stem}: {len(segs)} segs", flush=True)


def run_moonshine():
    import numpy as np
    import moonshine_onnx
    from moonshine_onnx import load_audio

    for stem, path in SMOKE:
        out = OUTDIR / f"moonshine.{stem}.txt"
        if out.exists():
            continue
        audio = load_audio(str(path))
        if audio.ndim == 2:
            audio = audio[0]
        n = int(30 * 16000)
        pieces = []
        for i in range(0, len(audio), n):
            seg = audio[i : i + n]
            if len(seg) < 1600:
                break
            t = moonshine_onnx.transcribe(seg, model="moonshine/base")
            if isinstance(t, list):
                t = " ".join(str(x) for x in t)
            pieces.append(str(t).strip())
        out.write_text(" ".join(p for p in pieces if p))
        print(f"moonshine {stem}: done", flush=True)


def run_kyutai():
    sys.path.insert(0, str(ROOT / "scripts"))
    import kyutai_corpus as K

    model, text_tokenizer, audio_tok, lc, stt_config, ct = K.load_model()
    for stem, path in SMOKE:
        out = OUTDIR / f"kyutai.{stem}.txt"
        if out.exists():
            continue
        text = K.transcribe(model, text_tokenizer, audio_tok, lc, stt_config, ct, path)
        out.write_text(text)
        print(f"kyutai {stem}: done", flush=True)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    engines = sys.argv[1:] or ["parakeet", "distil", "largev3", "moonshine", "kyutai"]
    if "parakeet" in engines:
        run_parakeet()
    if "distil" in engines:
        run_whisper_variant("distil-large-v3", "mlx-community/distil-whisper-large-v3")
    if "largev3" in engines:
        run_whisper_variant("large-v3", "mlx-community/whisper-large-v3-mlx")
    if "moonshine" in engines:
        run_moonshine()
    if "kyutai" in engines:
        run_kyutai()
    print("smoke pass complete", flush=True)


if __name__ == "__main__":
    main()
