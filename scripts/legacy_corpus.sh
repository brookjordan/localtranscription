#!/usr/bin/env bash
# Legacy engines over the 12-file corpus: whisper.cpp tiny.en + SenseVoice q3_k.
# (WhisperKit and the MLX OpenAI server are driven separately — different runtimes.)
# Resumable: skips outputs that already exist. Usage: legacy_corpus.sh [cpp|all]
set -uo pipefail
cd "$(dirname "$0")/.."
CORPUS=results/assets/corpus
OUT=results/corpus
WHISPER_CLI=vendor/whisper.cpp/build/bin/whisper-cli
WHISPER_MODEL="$HOME/.cache/localtranscription/whisper.cpp/tiny.en/ggml-tiny.en.bin"
SENSE_BIN=vendor/sensevoice.cpp/build/bin/sense-voice-main
SENSE_MODEL="$HOME/.cache/localtranscription/sensevoice.cpp/q3_k/sense-voice-small-q3_k.gguf"

mode="${1:-all}"
mkdir -p "$OUT"
fail=0
for f in "$CORPUS"/*.wav; do
  stem=$(basename "$f" .wav)

  if [[ "$mode" == "all" || "$mode" == "cpp" ]]; then
    if [[ ! -s "$OUT/$stem.whispercpp.txt" ]]; then
      if ! "$WHISPER_CLI" -m "$WHISPER_MODEL" -f "$f" -otxt -of "$OUT/$stem.whispercpp" >/dev/null 2>&1; then
        echo "FAIL whisper.cpp: $stem"; fail=1
      else
        echo "done whisper.cpp: $stem"
      fi
    fi
    if [[ ! -s "$OUT/$stem.sensevoice.txt" ]]; then
      if ! "$SENSE_BIN" -m "$SENSE_MODEL" --language en -f "$f" > "$OUT/$stem.sensevoice.txt" 2>"$OUT/$stem.sensevoice.log"; then
        echo "FAIL sensevoice: $stem"; rm -f "$OUT/$stem.sensevoice.txt"; fail=1
      else
        echo "done sensevoice: $stem"
      fi
    fi
  fi
done
echo "legacy corpus pass complete (mode=$mode, fail=$fail)"
exit $fail
