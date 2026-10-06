# Transcription evaluation matrix

Each candidate is measured on the same private corpus and emits one JSON record per `(runtime, model, input)` run. Do not commit private audio or transcript text; commit only the schema, scripts, and aggregate non-sensitive conclusions.

## Input and ingestion

- Source filename extension and container/codec.
- Byte size, duration, sample rate, channels, bit depth, and normalised WAV size.
- Decode/normalisation duration and whether native input succeeded without ffmpeg.
- Language actually spoken, expected transcript availability, background noise, music, laughter, overlaps, silence, and proper nouns.
- Rejection quality for malformed, empty, oversized, unsupported, and multi-channel files.

## Runtime and resource use

- Runtime family, version/revision, source URL, licence, and invocation/API mode.
- Model identifier, revision/hash, quantisation, model-artifact bytes, and total runtime/cache bytes.
- Cold launch/build/download time, model load time, first-result latency, total wall time, reported inference time, and real-time factor (`audio seconds / inference seconds`).
- Peak RSS, unified-memory/GPU allocation where observable, CPU/GPU utilisation, thread count, temperature/thermal-pressure state, energy/power when available, and memory after unload.
- Warm-run timing and variance over at least three runs.
- 8GB-M1 suitability: free memory before/after, peak delta, responsiveness of the production text gateway, and recovery after cancellation/OOM.

## Transcript and structured output

- Raw transcript and normalised transcript separately; word/character count.
- Word error rate or a documented qualitative comparison against known text.
- Language selection/detection and confidence if exposed.
- Segment and word timestamps; timestamp precision and monotonicity.
- Formatting: plain text, JSON, verbose JSON, SRT/VTT, punctuation/capitalisation/ITN, and diarisation/speaker labels.
- Non-speech signals: laughter, music, VAD/silence, audio events, emotion, and confidence—recorded only when the candidate actually emits them.

## Operational/API contract

- CLI exit code and structured error body.
- OpenAI endpoint compatibility: `/v1/audio/transcriptions`, multipart field names, `model` requirement, response formats, and authentication assumptions.
- Local bind address, TLS/auth requirements, queue depth, serialisation, concurrent-client behaviour, cancellation, timeout, retry safety, temporary-file retention, and output/log privacy.
- Server startup/health/unload semantics and whether one process can coexist with the production gateway.

## Current candidate additions

- **Whistle** (`Cactus-Compute/whistle`, via `cactus-needle`): runs locally on Apple Silicon with the bundled 16.92 MB `whistle.cact` weights. The harness uses 30-second chunks because the API rejects longer audio, preserves word timestamps, and records 12/12 corpus outputs. Median corpus RTF is 0.0071 (0.0058–0.0095). This is a corpus run, not yet a promotion: the chunk boundary strategy and lack of the 3 human-scored clips remain explicit limitations.
- **Phonon 2** (`FermionResearch/Phonon-2`, via `fermion-research` 0.2.7): runs locally through the CPU five-value profile on the M3 Max; the package fetched and verified the pinned model archive. It produced 12/12 corpus outputs. Median corpus RTF is 0.0087 (0.0079–0.0106). This is a CPU measurement and is not directly comparable with the warm MLX measurements without a matched protocol.

Both candidates are embedded in `results/dashboard.html`; raw non-sensitive result records are in `results/corpus/*.whistle.json` and `*.phonon2.json`. The shared runner is `scripts/whistle_phonon_corpus.py`. The decision gates above still apply: malformed-input behaviour, 8GB-M1 fit, retention/error semantics, and the OpenAI-compatible integration contract remain unverified for both.

## Decision gates

A candidate cannot be promoted on speed alone. It must transcribe both corpus files, survive malformed input, fit the 8GB M1 with the production gateway idle and loaded, have explicit retention/error semantics, and offer a verified integration contract. Extra metadata such as emotion or audio events is evaluated separately from ASR accuracy.