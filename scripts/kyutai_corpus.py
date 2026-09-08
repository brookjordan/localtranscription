#!/usr/bin/env python3
"""Kyutai STT 2.6B en via moshi-mlx over the corpus.

Uses the dedicated MLX checkpoint kyutai/stt-2.6b-en-mlx (the PyTorch
checkpoint kyutai/stt-2.6b-en is NOT loadable by moshi-mlx — 177 unmapped
tensors). Loads the model once, transcribes all 12 corpus files, and writes
results/corpus/<stem>.kyutai.json ({segments, wall_s, audio_s, rtf}).
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "assets" / "corpus"
OUTDIR = ROOT / "results" / "corpus"

FRAME = 1920  # samples per 80 ms step at 24 kHz
SR = 24000


def load_model(repo: str = "kyutai/stt-2.6b-en-mlx"):
    global mimi_weights
    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece
    from huggingface_hub import hf_hub_download

    from moshi_mlx import models, utils

    with open(hf_hub_download(repo, "config.json")) as fobj:
        cfg = json.load(fobj)
    stt_config = cfg.get("stt_config", {})
    mimi_weights = hf_hub_download(repo, cfg["mimi_name"])
    moshi_weights = hf_hub_download(repo, cfg.get("moshi_name", "model.safetensors"))
    tokenizer = hf_hub_download(repo, cfg["tokenizer_name"])

    lm_config = models.LmConfig.from_config_dict(cfg)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if str(moshi_weights).endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif str(moshi_weights).endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)
    model.load_weights(moshi_weights, strict=True)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)

    ct = None
    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    model.warmup(ct)
    return model, text_tokenizer, ct, stt_config, utils


def transcribe(pcms, model, text_tokenizer, ct, stt_config, utils, mimi_weights, temp: float = 0.0):
    import numpy as np
    import rustymimi
    import mlx.core as mx

    from moshi_mlx import models

    pad_left = int(stt_config.get("audio_silence_prefix_seconds", 0.0) * SR)
    pad_right = int((stt_config.get("audio_delay_seconds", 0.0) + 1.0) * SR)
    in_pcms = np.pad(pcms, pad_width=[(0, 0), (pad_left, pad_right)], mode="constant")

    cfg = model.cfg
    generated_codebooks = cfg.generated_codebooks
    other_codebooks = cfg.other_codebooks
    audio_tokenizer = rustymimi.Tokenizer(
        mimi_weights, num_codebooks=max(generated_codebooks, other_codebooks)
    )

    steps = in_pcms.shape[-1] // FRAME
    gen = models.LmGen(
        model=model,
        max_steps=steps,
        text_sampler=utils.Sampler(top_k=25, temp=temp),
        audio_sampler=utils.Sampler(top_k=250, temp=temp),
        cfg_coef=1.0,
        check=False,
    )
    words = []
    t0 = time.perf_counter()
    for idx in range(steps):
        pcm = in_pcms[:, idx * FRAME:(idx + 1) * FRAME]
        other = audio_tokenizer.encode_step(pcm[None, 0:1])
        other = mx.array(other).transpose(0, 2, 1)[:, :, :other_codebooks]
        tok = gen.step(other[0], ct)[0].item()
        if tok not in (0, 3):
            w = text_tokenizer.id_to_piece(tok).replace("▁", " ")
            words.append(w)
    wall = time.perf_counter() - t0
    return "".join(words).strip(), wall, steps * FRAME / SR


if __name__ == "__main__":
    import sphn

    model, text_tokenizer, ct, stt_config, utils = load_model()
    import glob
    mimi_path = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--kyutai--stt-2.6b-en-mlx/snapshots/*/mimi-pytorch-*.safetensors"))[0]

    for wav in sorted(ASSETS.glob("*.wav")):
        stem = wav.stem
        out = OUTDIR / f"{stem}.kyutai.json"
        if out.exists():
            print(f"skip {stem} (exists)")
            continue
        try:
            pcms, _ = sphn.read(str(wav), sample_rate=SR)
            text, wall, audio_s = transcribe(pcms, model, text_tokenizer, ct, stt_config, utils, mimi_path)
            out.write_text(json.dumps({
                "segments": [{"start": None, "end": None, "speaker": None, "text": text}],
                "engine": "kyutai/stt-2.6b-en-mlx (moshi-mlx 0.3.0)",
                "wall_s": round(wall, 2), "audio_s": round(audio_s, 2),
                "rtf": round(wall / audio_s, 3) if audio_s else None,
            }, ensure_ascii=False, indent=1))
            print(f"done {stem}: wall={wall:.1f}s audio={audio_s:.1f}s rtf={wall/audio_s:.3f}")
        except Exception as e:
            out.write_text(json.dumps({"segments": [], "error": str(e)}))
            print(f"FAIL {stem}: {e}")
