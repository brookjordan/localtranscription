# localtranscription

A small, isolated speech-to-text evaluation harness for Tiny Agent Suite on Apple Silicon. It begins with a file-based `whisper.cpp` tiny-English baseline and keeps models, private audio, and generated results out of Git.

## First baseline

- Runtime: Homebrew `whisper-cpp`.
- Model: `ggerganov/whisper.cpp` `ggml-tiny.en.bin` (MIT model repository).
- Corpus: explicit local paths only; do not commit voice recordings without the speaker’s consent.
- Result: one JSON record per run containing source path, duration, model identifier, runtime version, elapsed time, transcript, and failure data.

## Status

The repository is a laboratory harness only. It does not expose a server, connect Telegram/n8n, or mutate Tiny Agent Suite production state. The first run will use the available 12-second `brook.wav` reference. Add a second authorised sample before comparing transcription consistency.

## Layout

```text
scripts/  Node/Python helpers: downloads, corpus runs, guards, dashboard embed
docs/     research notes and technical documentation
results/  corpus audio, ground truths, transcripts, segment WAVs, dashboard (tracked)
vendor/   compiled engine builds (whisper.cpp, sensevoice.cpp), ignored
          AI model weights live outside the repo in ~/.cache/localtranscription/
          (whisper.cpp ggml, sensevoice gguf) and ~/.cache/huggingface/ (MLX models)
```
