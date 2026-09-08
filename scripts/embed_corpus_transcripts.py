#!/usr/bin/env python3
"""Embed corpus transcripts from results/corpus/*.json into results/dashboard.html.

Generates the `const CORPUS = {...}` JS block (audio-first, VibeVoice segments as
structured data; models without corpus transcripts yet are null). Re-runnable:
replaces the block between the CORPUS markers in place.

Usage: .venv-asr/bin/python scripts/embed_corpus_transcripts.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "results" / "corpus"
HTML = ROOT / "results" / "dashboard.html"
BASELINE_SUFFIX = ".json"  # VibeVoice baseline; other models use <stem>.<model>.json/.txt
MODEL_SUFFIXES = {
    ".qwen3.json": "qwen3",
    ".whisper.json": "whisper",
    ".parakeet.json": "parakeet",
    ".large-v3.json": "largev3",
    ".distil-large-v3.json": "distil",
}
TXT_ENGINES = {
    ".whispercpp.txt": "whispercpp",
    ".whisperkit.txt": "whisperkit",
    ".server.txt": "server",
    ".sensevoice.txt": "sensevoice",
    ".moonshine.txt": "moonshine",
}
JSON_ENGINES = {
    ".kyutai.json": "kyutai",  # structured {segments, wall_s, rtf}
}
RTFS = {}  # engine key -> RTF measured on corpus, carried into the MODELS block

BEGIN = "// ---- corpus (embedded from results/corpus/*.json by scripts/embed_corpus_transcripts.py) ----"
END = "// ---- end corpus ----"
MIT_BEGIN = "// ---- kyutai mitigations (embedded from kyutai_mitigations.json + kyutai_vad.json by scripts/embed_corpus_transcripts.py) ----"
MIT_END = "// ---- end kyutai mitigations ----"
# human-scored clips use different run stems than their FILES keys
HUMAN_STEM_TO_FILE = {"brook": "brook.wav", "silly": "silly.m4a", "sing": "sing.mp3"}
VARIANT_FIELDS = ("text", "wall_s", "accuracy", "deterministic", "note", "segments")


def js_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def js_val(v):
    return js_str(v) if isinstance(v, str) else json.dumps(v)


def js_variant(v: dict) -> str:
    fields = ", ".join(f"{k}: {js_val(v[k])}" for k in VARIANT_FIELDS if k in v)
    return "{ " + fields + " }"


def build_mitigations_block() -> str:
    mit_path = CORPUS_DIR / "kyutai_mitigations.json"
    vad_path = CORPUS_DIR / "kyutai_vad.json"
    runs: dict[str, dict] = {}
    if mit_path.exists():
        runs.update(json.loads(mit_path.read_text())["runs"])
    if vad_path.exists():
        for stem, vr in json.loads(vad_path.read_text())["runs"].items():
            runs.setdefault(stem, {}).update(vr)
    lines = [MIT_BEGIN, "const KYUTAI_MITIGATIONS = {"]
    for stem, variants in runs.items():
        file_key = HUMAN_STEM_TO_FILE.get(stem, f"{stem}.wav")
        lines.append(f"  {js_str(file_key)}: {{")
        for k, v in variants.items():
            lines.append(f"    {js_str(k)}: {js_variant(v)},")
        lines.append("  },")
    lines += ["};", MIT_END, ""]
    return "\n".join(lines)


def main() -> None:
    corpus: dict[str, dict] = {}
    for baseline in sorted(CORPUS_DIR.glob("*.json")):
        if baseline.name in ("kyutai_mitigations.json", "kyutai_vad.json") or baseline.name.endswith(".kyutai.human.json"):
            continue  # mitigation/VAD/human-kyutai runs; not VibeVoice baselines
        if any(baseline.name.endswith(s) for s in MODEL_SUFFIXES) or any(baseline.name.endswith(s) for s in JSON_ENGINES):
            continue  # qwen3/whisper/kyutai outputs, not VibeVoice baselines
        stem = baseline.name[: -len(BASELINE_SUFFIX)]
        meta = json.loads(baseline.read_text())
        entry: dict = {
            "meta": f"WAV · {meta['duration_s']:.1f} s · corpus",
            "audio": f"assets/corpus/{stem}.wav",
            "vvv": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker_id"],
                    "text": seg["text"],
                }
                for seg in meta["segments"]
            ],
        }
        for suffix, key in MODEL_SUFFIXES.items():
            p = CORPUS_DIR / f"{stem}{suffix}"
            entry[key] = json.loads(p.read_text())["text"] if p.exists() else None
        for suffix, key in TXT_ENGINES.items():
            p = CORPUS_DIR / f"{stem}{suffix}"
            entry[key] = p.read_text().strip() if p.exists() else None
        for suffix, key in JSON_ENGINES.items():
            p = CORPUS_DIR / f"{stem}{suffix}"
            if p.exists():
                meta = json.loads(p.read_text())
                if meta.get("error"):
                    entry[key] = f"FAILED: {meta['error']}"
                else:
                    entry[key] = " ".join(seg["text"] for seg in meta["segments"]).strip()
                if meta.get("rtf"):
                    RTFS.setdefault(key, []).append(meta["rtf"])
            else:
                entry[key] = None
        corpus[f"{stem}.wav"] = entry

    parts = [BEGIN + "\nconst CORPUS = {\n"]
    for name, e in corpus.items():
        segs = ",\n        ".join(
            f"{{ start: {s['start']}, end: {s['end']}, speaker: "
            f"{'null' if s['speaker'] is None else s['speaker']}, text: {js_str(s['text'])} }}"
            for s in e["vvv"]
        )
        extra = ",\n    ".join(
            f"{key}: {js_str(e[key]) if e[key] else 'null'}"
            for key in ("qwen3", "whisper", "parakeet", "largev3", "distil", *TXT_ENGINES.values(), *JSON_ENGINES.values())
        )
        parts.append(
            f"  {js_str(name)}: {{ meta: {js_str(e['meta'])}, audio: {js_str(e['audio'])},\n"
            f"    vvv: [\n        {segs},\n    ],\n"
            f"    {extra} }},\n"
        )
    parts.append("};\n" + END + "\n")
    block = "".join(parts)
    mit_block = build_mitigations_block()

    html = HTML.read_text()
    if BEGIN in html and END in html:
        pre, rest = html.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        html = pre + block + post
    else:
        anchor = "// ---- models ---"
        assert anchor in html, "models anchor not found"
        html = html.replace(anchor, block + "\n" + anchor, 1)
    if MIT_BEGIN in html and MIT_END in html:
        pre, rest = html.split(MIT_BEGIN, 1)
        _, post = rest.split(MIT_END, 1)
        html = pre + mit_block + post
    else:
        anchor = "// ---- models ---"
        assert anchor in html, "models anchor not found"
        html = html.replace(anchor, mit_block + "\n" + anchor, 1)
    HTML.write_text(html)
    kyutai_rtfs = RTFS.get("kyutai", [])
    if kyutai_rtfs:
        med = sorted(kyutai_rtfs)[len(kyutai_rtfs) // 2]
        import re as _re
        m = _re.search(r"(\{ name: ([\'\"])Kyutai STT 2\.6B en\2, failed: true,)", html)
        if m and "rtf:" not in html[m.start():m.start() + 300]:
            inj = f" rtf: {round(med, 3)}, rtfNote: \"median of {len(kyutai_rtfs)} corpus files, moshi-mlx batch\","
            html = html[: m.end(1)] + inj + html[m.end(1):]
            HTML.write_text(html)
        print(f"kyutai rtf median {med:.3f} from {len(kyutai_rtfs)} files")
    print(f"embedded {len(corpus)} corpus files into {HTML.name}")


if __name__ == "__main__":
    main()
