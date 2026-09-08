# localtranscription

A small, isolated speech-to-text evaluation harness for Tiny Agent Suite on Apple Silicon. It began with a file-based `whisper.cpp` tiny-English baseline and now compares several local engines while keeping model weights and source audio outside Git.

## First baseline

- Runtime: Homebrew `whisper-cpp`.
- Model: `ggerganov/whisper.cpp` `ggml-tiny.en.bin` (MIT model repository).
- Corpus: explicit local paths only; authorised sample audio is served separately at `https://assets.brook.dev/localtranscription/`; do not publish recordings without consent.
- Result: one JSON record per run containing source path, duration, model identifier, runtime version, elapsed time, transcript, and failure data.

## Status

The repository is a laboratory harness only. It does not expose a server, connect Telegram/n8n, or mutate Tiny Agent Suite production state. The first runs use the `brook.wav`, `silly.m4a`, and `sing.mp3` references; later corpus runs add authorised voices and longer reading samples.

## Devlog publishing

`site/` is the canonical publishable project page and devlog. It contains the
nine atomic dated posts, banner images, and the project index. When publishing,
copy `site/` to `brookjordan.github.io/projects/localtranscription/` and commit
that copy in the website repository. The audio itself is not copied into either
Git repository; posts link to the exact files on `assets.brook.dev`.

## Layout

```text
scripts/  Node/Python helpers: downloads, corpus runs, guards, dashboard embed
docs/     research notes and technical documentation
results/  corpus audio, ground truths, transcripts, segment WAVs, dashboard (tracked)
vendor/   compiled engine builds (whisper.cpp, sensevoice.cpp), ignored
          AI model weights live outside the repo in ~/.cache/localtranscription/
          (whisper.cpp ggml, sensevoice gguf) and ~/.cache/huggingface/ (MLX models)
```
