# PCB-010 Phase A calibration report — v0.1.2

## Scope

- Episode: `v2026.39.5-22.57`
- Corpus: 39 speech segments, `audio_0.mp3` through `audio_38.mp3`
- Input source: NAS assets share, `/tmp/nas-assets/podcast/v2026.39.5-22.57`
- Evaluator: `pcb-010-evaluator-v0.1.2`
- ASR: Parakeet TDT 0.6B v3, `mlx-community/parakeet-tdt-0.6b-v3`
- Production n8n: not executed or modified

## Measured result

- 30/39 segments matched after Unicode, case, punctuation, hyphen and approved number-word normalisation.
- 9/39 remained rejected by the transcript gate.
- Total Parakeet wall time: 12.488 seconds.
- Known suspect `audio_2.mp3`: passed. Expected and observed lexical content matched after normalisation.

## Remaining transcript rejects

- `audio_0`: severe unrelated transcript output.
- `audio_4`: expected “I'm Brook”; unrelated transcript output.
- `audio_7`: proper phrase “ratio legis” became “radiolegus”.
- `audio_17`: `$250 million` became “two hundred and fifty millionaires”.
- `audio_22`: “care settings” became “care setting”.
- `audio_25`: clause punctuation/segmentation differs; number normalisation now agrees on “59 days”. Requires phrase-level alignment before final classification.
- `audio_27`: “Dr. Hsu Chia-Da” became “doctor Sue Chiada”; protected-name handling is still pending.
- `audio_28`: “but” became “and”; lexical change must remain rejected.
- `audio_31`: “80-plus” and “60 per cent” became spoken equivalents; number and hyphen normalisation is not yet covering compound number phrases.

## Limitations

The current artifact has not yet connected the HEAR CAM++ worker, bounded Whisper adjudication, protected-term manifests, or waveform defect detectors. Therefore this is a working transcript/technical probe, not a production qualification result and not a Phase A completion claim.

Evidence JSONL: `/tmp/pcb010-calibration-v0.1.2.jsonl`
