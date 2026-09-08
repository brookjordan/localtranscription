#!/usr/bin/env python3
"""Materialize Kyutai chunked/VAD segments as real WAV files + transcribe each.

Answers "are the windows the models actually saw valid?": instead of slicing
in memory, every 30 s chunked window and every Silero-VAD group is written to
results/corpus/segments/{chunked,vad}/<stem>/<stem>.{chunk,vad}NN.wav, then
transcribed FROM DISK with the same greedy best-of-3 path as the mitigation
runner. Per-segment text, char counts, and roll logs are attached to the
variant entries in kyutai_mitigations.json / kyutai_vad.json so the dashboard
can play each window and show its transcript.

Idempotent: re-runs redo everything (fresh lottery draw per window).
"""
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kyutai_corpus import SR, load_model, transcribe  # noqa: E402
from kyutai_mitigations import BEST_OF_N, accuracy, best_of_n, ref_text  # noqa: E402
from kyutai_vad import MAX_SEG, PAD, vad_groups  # noqa: E402

CORPUS = ROOT / "results" / "corpus"
SEGDIR = CORPUS / "segments"
MIT_OUT = CORPUS / "kyutai_mitigations.json"
VAD_OUT = CORPUS / "kyutai_vad.json"
CHUNK_S = 30
CORPUS_STEMS = [
    "Balajee", "Brett", "Brook 2", "Dave", "Mum's voice", "Nana Chris",
    "Ping", "Pritisman", "Tung", "antony billington", "conrad",
    "lockpicking lawyer voice",
]
CLIPS = {
    "brook": "assets/brook.wav",
    "silly": "assets/wav/silly.wav",
    "sing": "assets/wav/sing.wav",
    **{stem: f"assets/corpus/{stem}.wav" for stem in CORPUS_STEMS},
}


def write_wav(path: Path, pcm: np.ndarray) -> None:
    x = np.clip(pcm[0], -1.0, 1.0)
    data = (x * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)


def read_wav(path: Path) -> np.ndarray:
    import sphn
    pcms, _ = sphn.read(str(path), sample_rate=SR)
    return pcms


def slice_chunked(stem: str, wav: Path) -> list[dict]:
    outdir = SEGDIR / "chunked" / stem
    outdir.mkdir(parents=True, exist_ok=True)
    pcms = read_wav(wav)
    chunk = SR * CHUNK_S
    segs = []
    for i, off in enumerate(range(0, pcms.shape[-1], chunk)):
        part = pcms[:, off:off + chunk]
        if part.shape[-1] < SR // 2:  # same drop-tiny-tail rule as the runner
            break
        fn = f"{stem}.chunk{i:02d}.wav"
        write_wav(outdir / fn, part)
        segs.append({"file": fn, "start": round(off / SR, 2),
                     "end": round((off + part.shape[-1]) / SR, 2)})
    return segs


def slice_vad(stem: str, wav: Path, vad_model) -> list[dict]:
    import sphn
    outdir = SEGDIR / "vad" / stem
    outdir.mkdir(parents=True, exist_ok=True)
    pcm16, _ = sphn.read(str(wav), sample_rate=16000)
    x16 = np.ascontiguousarray(pcm16[0], dtype=np.float32)
    groups16 = vad_groups(x16, vad_model)
    scale = SR / 16000
    groups = [(round(s * scale), round(e * scale)) for s, e in groups16]
    pcm24 = read_wav(wav)
    x24 = np.ascontiguousarray(pcm24[0], dtype=np.float32)
    segs = []
    for i, (gs, ge) in enumerate(groups):
        fn = f"{stem}.vad{i:02d}.wav"
        write_wav(outdir / fn, x24[None, gs:ge])
        segs.append({"file": fn, "start": round(gs / SR, 2),
                     "end": round(ge / SR, 2)})
    return segs


def transcribe_segments(stem: str, variant: str, segs: list[dict],
                        model, text_tokenizer, ct, stt_config, utils,
                        mimi_path) -> tuple[str, float, list[dict]]:
    """Transcribe each written segment file from disk (best-of-N hedged)."""
    texts = []
    detail = []
    t0 = time.perf_counter()
    for s in segs:
        path = SEGDIR / variant / stem / s["file"]
        pcms = read_wav(path)
        txt, _, _ = best_of_n(pcms, model, text_tokenizer, ct,
                              stt_config, utils, mimi_path)
        texts.append(txt)
        detail.append({"file": s["file"], "start": s["start"], "end": s["end"],
                       "text": txt, "chars": len(txt.strip()),
                       "segment_wav": f"segments/{variant}/{stem}/{s['file']}"})
        print(f"  {stem}/{variant} {s['file']}: {len(txt.strip())} chars", flush=True)
    wall = time.perf_counter() - t0
    joined = " ".join(texts).strip()
    return joined, wall, detail


def main() -> None:
    import glob
    from silero_vad import load_silero_vad

    model, text_tokenizer, ct, stt_config, utils = load_model()
    mimi_path = glob.glob(str(Path.home() /
        ".cache/huggingface/hub/models--kyutai--stt-2.6b-en-mlx/snapshots/*/mimi-pytorch-*.safetensors"))[0]
    vad = load_silero_vad()

    mit = json.loads(MIT_OUT.read_text()) if MIT_OUT.exists() else {"runs": {}}
    vadj = json.loads(VAD_OUT.read_text()) if VAD_OUT.exists() else {"runs": {}}

    for stem, rel in CLIPS.items():
        wav = ROOT / "results" / rel
        ref = ref_text(stem)
        prev_c = mit["runs"].get(stem, {}).get("chunked", {})
        prev_v = vadj["runs"].get(stem, {}).get("vad", {})

        # chunked (skip when this run already recorded segments for every window)
        csegs = slice_chunked(stem, wav)
        have_c = prev_c.get("segments") and len(prev_c["segments"]) == len(csegs)
        if have_c:
            print(f"{stem}: chunked complete ({len(csegs)} windows) — skipping", flush=True)
        else:
            print(f"{stem}: {len(csegs)} chunked windows", flush=True)
            text, wall, detail = transcribe_segments(
                stem, "chunked", csegs, model, text_tokenizer, ct,
                stt_config, utils, mimi_path)
            mit.setdefault("runs", {}).setdefault(stem, {})["chunked"] = {
                "text": text, "wall_s": round(wall, 1),
                "accuracy": accuracy(ref, text), "deterministic": False,
                "note": f"temp=0, {CHUNK_S}s windows written to disk, transcribed from files, best-of-{BEST_OF_N} per window",
                "segments": detail,
            }
            MIT_OUT.write_text(json.dumps(mit, ensure_ascii=False, indent=1))

        # vad (same skip rule)
        vsegs = slice_vad(stem, wav, vad)
        have_v = prev_v.get("segments") and len(prev_v["segments"]) == len(vsegs)
        if have_v:
            print(f"{stem}: vad complete ({len(vsegs)} groups) — skipping", flush=True)
        else:
            print(f"{stem}: {len(vsegs)} VAD groups", flush=True)
            text, wall, detail = transcribe_segments(
                stem, "vad", vsegs, model, text_tokenizer, ct,
                stt_config, utils, mimi_path)
            vadj.setdefault("runs", {}).setdefault(stem, {})["vad"] = {
                "text": text, "wall_s": round(wall, 1),
                "accuracy": accuracy(ref, text), "deterministic": False,
                "note": f"temp=0, Silero-VAD boundaries ({MAX_SEG:.0f}s max, {PAD}s pad), groups written to disk, transcribed from files, best-of-{BEST_OF_N} per group",
                "segments": detail,
            }
            VAD_OUT.write_text(json.dumps(vadj, ensure_ascii=False, indent=1))
        c_acc = mit["runs"][stem]["chunked"]["accuracy"]
        v_acc = vadj["runs"][stem]["vad"]["accuracy"]
        print(f"{stem} DONE: chunked={c_acc}% vad={v_acc}%", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
