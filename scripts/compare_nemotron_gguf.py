#!/usr/bin/env python3
"""Score the NeMo-Speech.cpp GGUF arm against ground truth and the MLX arm.

For each synthetic clip: per-segment WER vs results/assets/voices/<stem>.segments.json
(matching by speaker), predicted speaker count vs truth, and whole-text WER vs the
MLX NVFP4 arm's text. Prints a compact table; writes round1-style JSON on request.
"""
from __future__ import annotations

import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ("tse3-two", "tse3-three", "tse3-codeswitch", "tse3-two-wide")
SUFFIX = "-gguf"


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


def load_gt(stem: str) -> list[dict]:
    return json.loads((ROOT / f"results/assets/voices/{stem}.segments.json").read_text())


def seg_text(result: dict, want_speaker: int | None) -> str:
    """Concatenate the result's segments for one speaker (or all when None)."""
    segs = result.get("segments") or []
    out = []
    for seg in segs:
        speaker = seg.get("speaker_id", seg.get("speaker"))
        text = seg.get("text") or ""
        if want_speaker is None or speaker == want_speaker:
            out.append(text)
    return " ".join(out)


def main() -> int:
    rows = []
    for stem in CLIPS:
        result_file = ROOT / f"results/corpus/{stem}.nemotron{SUFFIX}.json"
        if not result_file.is_file():
            print(f"{stem}: no result yet")
            continue
        result = json.loads(result_file.read_text())
        if result.get("status") != "ok":
            print(f"{stem}: {result.get('status')}: {result.get('error', '')[:80]}")
            continue
        truth = load_gt(stem)
        truth_speakers = sorted({seg["speaker"] for seg in truth})
        pred_segments = result.get("segments") or []
        pred_speakers = sorted({s.get("speaker_id", s.get("speaker"))
                                for s in pred_segments})
        row = {"stem": stem, "truth_speakers": len(truth_speakers),
               "pred_speakers": len(pred_speakers), "segments": []}
        whole_ref, whole_hyp = [], []
        for seg in truth:
            ref = norm(seg["text"])
            hyp = norm(seg_text(result, seg["speaker"]))
            whole_ref.extend(ref)
            whole_hyp.extend(hyp)
            row["segments"].append({"speaker": seg["speaker"],
                                    "wer": round(wer(ref, hyp), 3) if hyp else None})
        row["whole_wer_vs_gt"] = round(wer(whole_ref, whole_hyp), 3)
        mlx_file = ROOT / f"results/corpus/{stem}.nemotron-saa.json"
        if mlx_file.is_file():
            mlx = json.loads(mlx_file.read_text())
            if mlx.get("status") == "ok":
                row["wer_vs_mlx_text"] = round(
                    wer(norm(mlx["text"]), norm(result["text"])), 3)
        row["rtf"] = result.get("rtf")
        row["wall_s"] = result.get("wall_s")
        rows.append(row)
        print(f"{stem}: speakers {row['truth_speakers']}->{row['pred_speakers']} "
              f"whole WER vs GT {row['whole_wer_vs_gt']} "
              f"vs MLX {row.get('wer_vs_mlx_text', 'n/a')} rtf {row['rtf']}")
        for seg in row["segments"]:
            print(f"    spk {seg['speaker']}: WER {seg['wer']}")
    if len(sys.argv) > 1 and rows:
        out = ROOT / f"results/corpus/nemotron{SUFFIX}-scores.json"
        out.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
