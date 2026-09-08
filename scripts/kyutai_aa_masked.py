#!/usr/bin/env python3
"""A/A probe: token-0 masked sampler, Tung + conrad baselines x3 in-process."""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sphn
from kyutai_corpus import SR, load_model, transcribe

ROOT = Path(__file__).resolve().parents[1]
model, text_tokenizer, ct, stt_config, utils = load_model()
mimi = glob.glob(str(Path.home() /
    ".cache/huggingface/hub/models--kyutai--stt-2.6b-en-mlx/snapshots/*/mimi-pytorch-*.safetensors"))[0]

for stem in ("Tung", "conrad"):
    pcms, _ = sphn.read(str(ROOT / "results/assets/corpus" / f"{stem}.wav"), sample_rate=SR)
    for i in range(3):
        txt, wall, _ = transcribe(pcms, model, text_tokenizer, ct, stt_config, utils, mimi, temp=0.0)
        print(f"{stem} run{i+1}: {len(txt)} chars | {txt[:90]!r}", flush=True)
