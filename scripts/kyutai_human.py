#!/usr/bin/env python3
"""Kyutai STT 2.6B en on the 3 human-scored clips (one model load)."""
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kyutai_corpus import load_model, transcribe, SR

import sphn

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [ROOT / "results/assets/brook.wav",
         ROOT / "results/assets/wav/silly.wav",
         ROOT / "results/assets/wav/sing.wav"]  # 24 kHz mono WAVs (sphn can't read m4a/mp3)

model, text_tokenizer, ct, stt_config, utils = load_model()
mimi = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--kyutai--stt-2.6b-en-mlx/snapshots/*/mimi-pytorch-*.safetensors"))[0]

for wav in CLIPS:
    out = ROOT / "results/corpus" / f"{wav.stem}.kyutai.human.json"
    pcms, _ = sphn.read(str(wav), sample_rate=SR)
    text, wall, audio_s = transcribe(pcms, model, text_tokenizer, ct, stt_config, utils, mimi)
    out.write_text(json.dumps({
        "segments": [{"start": None, "end": None, "speaker": None, "text": text}],
        "wall_s": round(wall, 2), "audio_s": round(audio_s, 2),
        "rtf": round(wall / audio_s, 3) if audio_s else None,
    }, ensure_ascii=False, indent=1))
    print(f"{wav.stem}: rtf={wall/audio_s:.3f} text[:80]={text[:80]!r}")
