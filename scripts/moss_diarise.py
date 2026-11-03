#!/usr/bin/env python3
"""Transcribe and diarise the synthetic clips with MOSS-Transcribe-Diarize.

Purpose: Nemotron collapses two of the four synthetic clips to one speaker. This asks whether a
model built for transcription with diarisation does better on the same audio, with the same
ground truth, on the same machine.

    .venv-nemotron/bin/python scripts/moss_diarise.py --model ~/models/incoming/moss-transcribe-diarize-8bit \
        --out results/moss-diarise.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import string
import sys
import time

from mlx_audio.stt import load

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results/assets/corpus"
VOICES = ROOT / "results/assets/voices"
CLIPS = ("tse3-two", "tse3-three", "tse3-codeswitch", "tse3-two-wide")
CORPUS_CLIPS = ('Balajee', 'Brett', 'Brook 2', 'Dave', "Mum's voice", 'Nana Chris',
                'Ping', 'Pritisman', 'Tung', 'antony billington', 'conrad',
                'lockpicking lawyer voice')
HUMAN = (('brook', 'brook.wav'), ('silly', 'silly.m4a'), ('sing', 'sing.mp3'))
SEG = re.compile(r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]", re.DOTALL)


def norm(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w]


def wer(ref: list[str], hyp: list[str]) -> float:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="also run the 12 corpus clips and 3 human clips")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model = load(args.model)
    rows = []
    jobs = [(c, CORPUS / f"{c}.wav") for c in CLIPS]
    if args.all:
        jobs += [(c, CORPUS / f"{c}.wav") for c in CORPUS_CLIPS]
        jobs += [(c, VOICES / name) for c, name in HUMAN]
    for clip, audio in jobs:
        truth_path = VOICES / f"{clip}.segments.json"
        if not audio.exists():
            continue
        if not truth_path.exists():
            out = model.generate(str(audio), max_tokens=4096, temperature=args.temperature)
            text = getattr(out, "text", "") or ""
            segs = [m.groupdict() for m in SEG.finditer(text)]
            rows.append({"clip": clip, "wall_s": None, "truth_speakers": None,
                         "speakers": len({s["speaker"] for s in segs}), "segments": len(segs),
                         "wer": None, "text": text[:4000]})
            print(f"{clip}: speakers {len({s['speaker'] for s in segs})} segments {len(segs)}", flush=True)
            continue
        truth = json.loads((VOICES / f"{clip}.segments.json").read_text())
        truth_speakers = len({s["speaker"] for s in truth})
        truth_text = " ".join(s["text"] for s in truth)
        t = time.perf_counter()
        out = model.generate(str(audio), max_tokens=4096, temperature=0.0)
        wall = round(time.perf_counter() - t, 2)
        text = getattr(out, "text", "") or ""
        segments = [m.groupdict() for m in SEG.finditer(text)]
        speakers = {s["speaker"] for s in segments} or set(re.findall(r"\[S(\d+)\]", text))
        row = {
            "clip": clip, "wall_s": wall,
            "truth_speakers": truth_speakers,
            "speakers": len(speakers),
            "segments": len(segments),
            "wer": round(wer(norm(truth_text), norm(text)), 3),
            "text": text[:1500],
        }
        rows.append(row)
        print(f"{clip}: speakers {row['speakers']}/{truth_speakers} segments {row['segments']} WER {row['wer']} {wall}s", flush=True)

    pathlib.Path(args.out).write_text(json.dumps({"model": args.model, "rows": rows}, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
