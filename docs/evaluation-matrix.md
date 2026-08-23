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

## Decision gates

A candidate cannot be promoted on speed alone. It must transcribe both corpus files, survive malformed input, fit the 8GB M1 with the production gateway idle and loaded, have explicit retention/error semantics, and offer a verified integration contract. Extra metadata such as emotion or audio events is evaluated separately from ASR accuracy.
