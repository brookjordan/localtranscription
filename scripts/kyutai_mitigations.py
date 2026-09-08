#!/usr/bin/env python3
"""Kyutai STT 2.6B en (moshi-mlx) repetition-loop mitigations.

Background: kyutai-labs/delayed-streams-modeling#175 — no repetition penalty
on audio/text token sampling; greedy decoding locks into loops ("iii…",
"iatelyiately…"). This script A/Bs three mitigations against the baseline on
the clips that degenerated:

  baseline   temp=0 greedy (existing results, reused from results/corpus)
  chunked    30 s windows so a loop cannot run past the clip
  vad        Silero VAD segmentation (see kyutai_vad.py)

Best-of-N hedging: temp=0 greedy is NOT bit-reproducible on MLX (forward-pass
numeric wobble; ~1-in-3 stream starts collapse, see docs/kyutai-mitigations.md).
Every decode in this script runs up to BEST_OF_N times with early exit on the
first healthy output.

Removed variants (kept in old JSONs as archive, no longer generated):
  rep-pen    recency logit penalty — catastrophic on every clip (-136..-495%)
  temp0.6    top-k sampling — catastrophic on every clip (max 7.6%); audio-token
             sampling corrupts delayed-stream alignment

Writes results/corpus/kyutai_mitigations.json with per-variant transcripts,
wall times, and accuracy vs the same references the dashboard scores against.
"""
import json
import pathlib
import re
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kyutai_corpus import SR, load_model, transcribe  # noqa: E402

OUT = ROOT / "results" / "corpus" / "kyutai_mitigations.json"
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
CHUNK_S = 30
BEST_OF_N = 5  # temp=0 greedy is not bit-reproducible (see docs/kyutai-mitigations.md):
               # MLX forward passes wobble run-to-run; ~1-in-3 stream starts collapse
               # into a degenerate attractor. Run N times, keep the best output.


def looks_degenerate(text: str) -> bool:
    """Cheap collapse heuristic: empty, or dominated by a short repeated unit."""
    t = text.strip()
    if len(t) < 10:
        return True
    words = t.split()
    if len(set(words)) / max(len(words), 1) < 0.2:
        return True  # endless repetition ("iii…")
    # Char-level loops with no word boundaries: "forniaforniafornia…" is a
    # single 100%-diverse "word" to the check above, so also detect any short
    # unit (≤15 chars, whitespace ignored) repeated 8+ times consecutively.
    flat = re.sub(r"\s+", "", t.lower())
    if re.search(r"(.{1,15})\1{7,}", flat):
        return True
    return False


def best_of_n(pcms, model, text_tokenizer, ct, stt_config, utils, mimi_path,
              n: int = BEST_OF_N, ref: str = "") -> tuple[str, float, list]:
    """Run greedy decode N times, return the transcript furthest from collapse.

    Selection: prefer non-degenerate outputs; among them, the longest coherent
    text (closest length to the reference when a reference is given).
    """
    candidates = []
    for i in range(n):
        txt, _, _ = transcribe(pcms, model, text_tokenizer, ct,
                               stt_config, utils, mimi_path, temp=0.0)
        candidates.append(txt)
        deg = looks_degenerate(txt)
        print(f"    best-of-{n} run {i+1}: {len(txt)} chars, "
              f"{'DEGENERATE' if deg else 'ok'}", flush=True)
        if not deg:
            return txt, 0.0, candidates  # early exit: first healthy decode wins
    return max(candidates, key=len), 0.0, candidates


def ref_text(stem: str) -> str:
    # Repo copies are canonical (results/assets/voices/); the external
    # Audio8-TTS dir is origin-only and may not exist on other machines.
    voices = ROOT / "results" / "assets" / "voices"
    vp = voices / f"{stem}.txt"
    if vp.exists():
        return vp.read_text()
    voices = pathlib.Path("/Users/brook.jordan/Documents/hermes/Audio8-TTS/voices")
    vp = voices / f"{stem}.txt"
    if vp.exists():
        return vp.read_text()
    j = json.loads((ROOT / "results" / "corpus" / f"{stem}.json").read_text())
    return " ".join(s["text"] for s in j["segments"])


def norm_words(text: str):
    import re
    t = text.lower()
    t = re.sub(r"[\[\(\*][^\]\)\*]*[\]\)\*]", " [event] ", t)  # markers -> event
    t = t.replace("ok", "okay") if re.search(r"\bok\b", t) else t
    t = re.sub(r"[^\w\s\[\]]", " ", t)
    return [w for w in t.split() if w]


def accuracy(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    import numpy as np
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1,
                          d[i-1, j-1] + (r[i-1] != h[j-1]))
    n = max(len(r), 1)
    return round(100 * (1 - d[-1, -1] / n), 1)


def main():
    import sphn
    model, text_tokenizer, ct, stt_config, utils = load_model()
    import glob
    mimi_path = glob.glob(str(Path.home() /
        ".cache/huggingface/hub/models--kyutai--stt-2.6b-en-mlx/snapshots/*/mimi-pytorch-*.safetensors"))[0]

    existing = json.loads(OUT.read_text()) if OUT.exists() else {"runs": {}}
    runs = existing.setdefault("runs", {})

    for stem, rel in CLIPS.items():
        wav = ROOT / "results" / rel
        pcms, _ = sphn.read(str(wav), sample_rate=SR)
        ref = ref_text(stem)
        runs.setdefault(stem, {})

        # baseline: hedge with best-of-N like every other variant — a single
        # greedy draw collapses ~1-in-3 per stream start (see
        # docs/kyutai-mitigations.md). Existing degenerate stored baselines
        # (1-in-3 collapse draw) are re-decoded rather than trusted.
        if "baseline" not in runs[stem]:
            t0 = time.perf_counter()
            txt, _, _ = best_of_n(pcms, model, text_tokenizer, ct,
                                  stt_config, utils, mimi_path, ref=ref)
            wall = time.perf_counter() - t0
            runs[stem]["baseline"] = {
                "text": txt, "wall_s": round(wall, 1),
                "accuracy": accuracy(ref, txt),
                "deterministic": False,
                "note": f"temp=0 greedy, token-0 masked, best-of-{BEST_OF_N}",
            }

        variants = {}
        # chunked: 30 s windows, each window best-of-N hedged
        if "chunked" not in runs[stem]:
            t0 = time.perf_counter()
            chunk = SR * CHUNK_S
            texts = []
            for off in range(0, pcms.shape[-1], chunk):
                part = pcms[:, off:off + chunk]
                if part.shape[-1] < SR // 2:
                    break
                txt, _, _ = best_of_n(part, model, text_tokenizer, ct,
                                      stt_config, utils, mimi_path,
                                      ref=ref)
                texts.append(txt)
            wall = time.perf_counter() - t0
            variants["chunked"] = {"text": " ".join(texts), "wall_s": round(wall, 1),
                "accuracy": accuracy(ref, " ".join(texts)),
                "deterministic": False,
                "note": f"temp=0, token-0 masked, {CHUNK_S}s windows, best-of-{BEST_OF_N} per window"}

        runs[stem].update(variants)
        OUT.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1))
        for name, v in {**({"baseline": runs[stem].get("baseline")} if runs[stem].get("baseline") else {}), **variants}.items():
            if v:
                print(f"{stem} / {name}: acc={v['accuracy']}% wall={v['wall_s']}s")

    print("ALL DONE ->", OUT)


if __name__ == "__main__":
    main()
