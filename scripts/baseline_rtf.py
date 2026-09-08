"""RTF baseline: MLX Whisper large-v3-turbo on the 3 test files."""
import time
import mlx_whisper

ASSETS = "/Users/brook.jordan/git/brookjordan/localtranscription/results/assets"
FILES = [
    ("brook.wav", f"{ASSETS}/brook.wav", 12.032),
    ("silly", f"{ASSETS}/wav/silly.wav", 10.560),
    ("sing", f"{ASSETS}/wav/sing.wav", 31.680),
]
MODEL = "mlx-community/whisper-large-v3-turbo"

# warm-up load excluded from timings: transcribe brook once and discard
mlx_whisper.transcribe(FILES[0][1], path_or_hf_repo=MODEL)

for name, path, dur in FILES:
    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(path, path_or_hf_repo=MODEL)
    elapsed = time.perf_counter() - t0
    rtf = elapsed / dur
    print(f"=== {name}: {elapsed:.2f}s wall / {dur}s audio = RTF {rtf:.3f} ({1/rtf:.1f}x realtime)")
    print(result["text"].strip())
    print()
