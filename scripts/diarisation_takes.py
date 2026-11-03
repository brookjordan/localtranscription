#!/usr/bin/env python3
"""Score diarisation over several TTS takes of the same script.

The completion run showed speaker labels flipping between takes of one clip, which makes a
single take an anecdote rather than a measurement. This regenerates the synthetic clips three
times, runs the Nemotron matrix on each take, and reports per-clip, per-take speaker counts.

    .venv-nemotron/bin/python scripts/diarisation_takes.py --takes 3

Heavy: each take loads the 20 GB checkpoint once. Clips are preserved under
results/assets/voice-takes/take<N>/ so a result can be traced back to the exact audio.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results/assets/corpus"
TAKES = ROOT / "results/assets/voice-takes"
RESULTS = ROOT / "results/corpus"
CLIPS = ("tse3-two", "tse3-three", "tse3-codeswitch", "tse3-two-wide")
PY_GEN = "/Users/brook.jordan/git/brookjordan/localaudiogen/.venv/bin/python"
MODEL = "/Users/brook.jordan/models/incoming/nemotron/nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
REVISION = "66c6adc4020179567074d0399cda2a4eeb2bb347"
RUNTIME = "5364ef911e2000f0469afd3f4ac534909665953f"


def speakers(path: pathlib.Path) -> dict:
    d = json.loads(path.read_text())
    segs = d.get("segments") or d.get("words") or []
    counts: dict = {}
    for s in segs:
        if isinstance(s, dict):
            k = s.get("speaker", s.get("speaker_id"))
            counts[k] = counts.get(k, 0) + 1
    return {"status": d.get("status"), "speakers": len(counts), "segments": len(segs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=3)
    args = ap.parse_args()

    summary: dict = {"takes": {}, "model_revision": REVISION, "runtime_revision": RUNTIME}
    for take in range(1, args.takes + 1):
        print(f"== take {take}: regenerating clips", flush=True)
        r = subprocess.run([PY_GEN, str(ROOT / "scripts/gen_tse3_clips.py")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-500:], file=sys.stderr)
            return 1
        keep = TAKES / f"take{take}"
        keep.mkdir(parents=True, exist_ok=True)
        for clip in CLIPS:
            src = CORPUS / f"{clip}.wav"
            if src.exists():
                shutil.copy2(src, keep / f"{clip}.wav")
        suffix = f"-saa-take{take}"
        print(f"== take {take}: running the matrix", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/nemotron_matrix.py"),
                            "--model", MODEL, "--model-revision", REVISION,
                            "--runtime-revision", RUNTIME, "--task", "saa",
                            "--word-timestamps", f"--suffix={suffix}", "--available-only"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-800:], r.stderr[-800:], file=sys.stderr)
            return 1
        counts = {}
        for clip in CLIPS:
            p = RESULTS / f"{clip}.nemotron{suffix}.json"
            counts[clip] = speakers(p) if p.exists() else {"status": "missing", "speakers": None}
        summary["takes"][f"take{take}"] = counts
        print(f"take {take}: " + ", ".join(f"{c}={v['speakers']}" for c, v in counts.items()), flush=True)

    (ROOT / "results/diarisation-takes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nclip, speakers per take")
    for clip in CLIPS:
        row = [summary["takes"][t].get(clip, {}).get("speakers") for t in summary["takes"]]
        print(f"{clip}: {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
