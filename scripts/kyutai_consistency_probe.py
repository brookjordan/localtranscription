#!/usr/bin/env python3
"""Consistency probe: same segment decoded N times — collapse rate vs length.

Inputs: results/corpus/segments/vad/antony billington/antony billington.vad01.wav
(30.25s, the segment that collapsed as 'icultidualiculticultidualst') and a
25.0s crop of the same audio. Each is decoded 10x with the production greedy
path (best_of_n NOT used — we want the raw per-draw collapse rate).
Output: /tmp/kyutai_consistency_probe.json + console summary.
"""
import glob, json, pathlib, sys, time

import numpy as np
import sphn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kyutai_corpus import SR, load_model, transcribe  # noqa: E402
from kyutai_mitigations import looks_degenerate  # noqa: E402

SEG = ROOT / "results/corpus/segments/vad/antony billington/antony billington.vad01.wav"
CROP = ROOT / "results/corpus/segments/vad/antony billington/_probe25s.wav"
N = 10

# Build the 25s crop (same start, shorter window)
pcm0, _ = sphn.read(str(SEG), sample_rate=SR)
x0 = np.ascontiguousarray(pcm0[0], dtype=np.float32)
crop = x0[: int(25.0 * SR)]
import wave
wf = wave.open(str(CROP), "wb")
wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
wf.writeframes((np.clip(crop, -1, 1) * 32767).astype(np.int16).tobytes())
wf.close()

model, text_tokenizer, ct, stt_config, utils = load_model()
mimi_path = glob.glob(str(pathlib.Path.home() /
    ".cache/huggingface/hub/models--kyutai--stt-2.6b-en-mlx/snapshots/*/mimi-pytorch-*.safetensors"))[0]

results = {}
for name, pcm in (("30.25s", x0), ("25.0s", crop)):
    draws, degenerate = [], 0
    t0 = time.perf_counter()
    for i in range(N):
        txt, _, _ = transcribe(pcm[None, :], model, text_tokenizer, ct,
                               stt_config, utils, mimi_path, temp=0.0)
        deg = looks_degenerate(txt)
        degenerate += deg
        draws.append({"i": i + 1, "chars": len(txt), "degenerate": deg, "head": txt[:60]})
        print(f"{name} draw {i+1}: {len(txt)}ch {'DEGEN' if deg else 'ok'} :: {txt[:50]!r}", flush=True)
    wall = time.perf_counter() - t0
    results[name] = {"n": N, "degenerate": degenerate,
                     "collapse_rate": round(degenerate / N, 2),
                     "wall_s": round(wall, 1), "draws": draws}
    print(f"== {name}: {degenerate}/{N} collapsed ({100*degenerate/N:.0f}%) in {wall:.0f}s\n", flush=True)

out = pathlib.Path("/tmp/kyutai_consistency_probe.json")
out.write_text(json.dumps(results, indent=1))
print("WROTE", out, flush=True)
