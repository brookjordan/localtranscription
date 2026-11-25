#!/usr/bin/env python3
"""Compare SenseVoice arms on the corpus: official llama.cpp f16 vs funasr cpu/mps vs q3_k."""
from __future__ import annotations

import json
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEMS = ('Balajee', 'Brett', 'Brook 2', 'Dave', "Mum's voice", 'Nana Chris',
         'Ping', 'Pritisman', 'Tung', 'antony billington', 'conrad',
         'lockpicking lawyer voice', 'tse3-two', 'tse3-three', 'tse3-codeswitch',
         'tse3-two-wide', 'brook', 'silly', 'sing')


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


def load(stem: str, arm: str):
    path = ROOT / f"results/corpus/{stem}.{arm}.json"
    if not path.is_file():
        text_path = ROOT / f"results/corpus/{stem}.{arm}.txt"
        if text_path.is_file():
            clean = text_path.read_text(errors="replace")
        else:
            return None
    else:
        data = json.loads(path.read_text())
        clean = data.get("text") or ""
    if not clean.strip():
        return None
    for tag in ("<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>", "<|NEUTRAL|>",
                "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|Speech|>", "<|woitn|>",
                "<|witn|>", "<|BG|>"):
        clean = clean.replace(tag, " ")
    return norm(clean)


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def main() -> int:
    pairs = (("sensevoice-svofficial", "sensevoice-upstream-cpu"),
             ("sensevoice-svofficial", "sensevoice"),
             ("sensevoice-upstream-cpu", "sensevoice"))
    for a, b in pairs:
        per_stem = []
        for stem in STEMS:
            ta, tb = load(stem, a), load(stem, b)
            per_stem.append((stem, round(wer(ta, tb), 4) if ta and tb else None))
        vals = [v for _, v in per_stem]
        print(f"{a} vs {b}: mean {mean(vals)} over {len([v for v in vals if v is not None])} clips")
        for stem, v in per_stem:
            if v is not None and v > 0.15:
                print(f"    {stem}: {v}")
    for arm in ("sensevoice-svofficial", "sensevoice-upstream-cpu",
                "sensevoice-upstream-mps", "sensevoice"):
        rtfs = []
        for stem in STEMS:
            path = ROOT / f"results/corpus/{stem}.{arm}.json"
            if path.is_file():
                data = json.loads(path.read_text())
                if data.get("rtf"):
                    rtfs.append(data["rtf"])
        if rtfs:
            print(f"{arm}: rtf mean {round(sum(rtfs)/len(rtfs), 4)} min {min(rtfs)} max {max(rtfs)} over {len(rtfs)}")
    return 0


if __name__ == "__main__":
    main()
