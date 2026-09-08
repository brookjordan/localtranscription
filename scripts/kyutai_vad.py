#!/usr/bin/env python3
"""Kyutai STT 2.6B en (moshi-mlx) with Silero-VAD segmentation.

Per WhisperX (Bain et al. 2023), cutting at speech boundaries in silences
beats blind fixed windows: no words severed at chunk edges, and each chunk
caps how far a repetition loop can run (delayed-streams-modeling#175).

VAD runs via the official silero-vad pip package (torch), which is ABI-safe
here after pinning torch/torchaudio to 2.9.0 — the raw ONNX export mis-ranks
its state tensor across If-node branches under plain onnxruntime.

For each clip: Silero-VAD finds speech regions, merges them into groups with
at most MAX_SEG seconds of speech plus PAD seconds of surrounding silence,
transcribes each group with the same greedy moshi-mlx path as the corpus
runner, and writes results/corpus/kyutai_vad.json.
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kyutai_corpus import SR, load_model, transcribe  # noqa: E402
from kyutai_mitigations import BEST_OF_N, accuracy, best_of_n, ref_text  # noqa: E402

OUT = ROOT / "results" / "corpus" / "kyutai_vad.json"
CORPUS_STEMS = [
    "Balajee", "Brett", "Brook 2", "Dave", "Mum's voice", "Nana Chris",
    "Ping", "Pritisman", "Tung", "antony billington", "conrad",
    "lockpicking lawyer voice",
]
CLIPS = {  # stem -> wav path (relative to results/)
    # human-scored clips
    "brook": "assets/brook.wav",
    "silly": "assets/wav/silly.wav",
    "sing": "assets/wav/sing.wav",
    # corpus clips
    **{stem: f"assets/corpus/{stem}.wav" for stem in CORPUS_STEMS},
}
MAX_SEG = 30.0   # max seconds of speech per transcribed group
PAD = 0.25       # silence padding kept around each group



def _bounded(s: int, e: int, pad: int, vsr: int) -> tuple[int, int]:
    """Pad a group then enforce the total span (incl. pad) stays <= MAX_SEG."""
    s2 = max(0, s - pad)
    e2 = min(e + pad, s2 + int(MAX_SEG * vsr))
    return s2, e2

def vad_groups(pcm, vad_model):
    """Silero-VAD speech regions -> transcription groups (sample indices)."""
    from silero_vad import get_speech_timestamps
    speech = get_speech_timestamps(
        pcm, vad_model, sampling_rate=16000,
        min_speech_duration_ms=250, min_silence_duration_ms=350,
    )
    vsr = 16000  # vad_groups receives 16 kHz audio; all spans below are in 16k samples
    pad = int(PAD * vsr)
    total = pcm.shape[-1]
    groups = []
    cur_start = cur_end = None
    for seg in speech:
        s, e = seg["start"], seg["end"]
        if cur_start is None:
            cur_start, cur_end = s, e
        elif (e - cur_start) / vsr <= MAX_SEG:  # keep growing while span stays under cap
            cur_end = e
        elif (e - s) / vsr > MAX_SEG:  # single region longer than cap: hard-split it
            gs = cur_start
            while (e - gs) / vsr > MAX_SEG:
                groups.append(_bounded(gs, gs + int(MAX_SEG * vsr), pad, vsr))
                gs += int(MAX_SEG * vsr)
            cur_start, cur_end = gs, e
        else:
            groups.append(_bounded(cur_start, cur_end, pad, vsr))
            cur_start, cur_end = s, e
    if cur_start is not None:
        groups.append(_bounded(cur_start, cur_end, pad, vsr))
    # post-pass: hard-split any group still over cap (dense speech with no
    # silences to cut at — e.g. singing — still gets windowed at MAX_SEG)
    win = int(MAX_SEG * vsr)
    split = []
    for s, e in groups:
        gs = s
        while (e - gs) / vsr > MAX_SEG:
            split.append(_bounded(gs, gs + win, pad, vsr))
            gs += win
        split.append(_bounded(gs, e, pad, vsr))
    return split


def main():
    import glob
    import sphn
    from silero_vad import load_silero_vad
    model, text_tokenizer, ct, stt_config, utils = load_model()
    mimi_path = glob.glob(str(pathlib.Path.home() /
        ".cache/huggingface/hub/models--kyutai--stt-2.6b-en-mlx/snapshots/*/mimi-pytorch-*.safetensors"))[0]
    vad = load_silero_vad()

    runs = json.loads(OUT.read_text())["runs"] if OUT.exists() else {}
    for stem, rel in CLIPS.items():
        if "vad" in runs.get(stem, {}):
            print(f"skip {stem} (cached)", flush=True)
            continue
        wav = ROOT / "results" / rel
        pcm, _ = sphn.read(str(wav), sample_rate=SR)          # 24 kHz — Kyutai native
        x = np.ascontiguousarray(pcm[0], dtype=np.float32)
        # Silero-VAD only accepts 16 kHz streams; resample for VAD and map
        # regions back into 24 kHz sample space.
        vad_sr = 16000
        pcm16, _ = sphn.read(str(wav), sample_rate=vad_sr)
        x16 = np.ascontiguousarray(pcm16[0], dtype=np.float32)
        groups16 = vad_groups(x16, vad)
        scale = SR / vad_sr
        groups = [(round(s * scale), round(e * scale)) for s, e in groups16]
        print(f"{stem}: {len(groups)} VAD groups ({[round((e-s)/SR,1) for s,e in groups]})", flush=True)
        t0 = time.perf_counter()
        texts = []
        for gs, ge in groups:
            # best-of-N per group: VAD restarts the stream per boundary just
            # like chunked, so each group carries the same collapse risk.
            txt, _, _ = best_of_n(x[None, gs:ge], model, text_tokenizer, ct,
                                  stt_config, utils, mimi_path,
                                  ref=ref_text(stem))
            texts.append(txt)
        wall = time.perf_counter() - t0
        joined = " ".join(t for t in texts if t).strip()
        runs.setdefault(stem, {})["vad"] = {
            "text": joined, "wall_s": round(wall, 1),
            "accuracy": accuracy(ref_text(stem), joined),
            "deterministic": False,
            "note": f"temp=0, token-0 masked, Silero-VAD boundaries, {MAX_SEG:.0f}s max speech/group, {len(groups)} groups, best-of-{BEST_OF_N} per group",
        }
        OUT.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1))
        print(f"done {stem}: wall={wall:.1f}s chars={len(joined)}", flush=True)
    print("ALL DONE ->", OUT)


if __name__ == "__main__":
    main()
