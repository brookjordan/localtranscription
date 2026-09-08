"""Qwen3-ASR 1.7B (MLX, 8-bit) on the 3 test files, with RTF.

Run with: .venv-asr/bin/python (needs python>=3.10 for modern mlx-audio).
"""
import json
import time
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model

ASSETS = "/Users/brook.jordan/git/brookjordan/localtranscription/results/assets"
FILES = [
    ("brook.wav", f"{ASSETS}/brook.wav", 12.032),
    ("silly", f"{ASSETS}/wav/silly.wav", 10.560),
    ("sing", f"{ASSETS}/wav/sing.wav", 31.680),
]
REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"

model = load_model(REPO)

results = []
for name, path, dur in FILES:
    t0 = time.perf_counter()
    out = generate_transcription(model=model, audio=path, format="txt")
    elapsed = time.perf_counter() - t0
    text = out.text.strip() if hasattr(out, "text") else str(out)
    print(f"=== {name}: {elapsed:.2f}s wall / {dur}s audio = RTF {elapsed/dur:.3f} ({dur/elapsed:.1f}x realtime)")
    print(text)
    print()
    results.append({"file": name, "wall_s": round(elapsed, 2), "rtf": round(elapsed / dur, 3), "text": text})

with open("/tmp/qwen3_asr_results.json", "w") as f:
    json.dump(results, f, indent=1)
print("saved /tmp/qwen3_asr_results.json")
