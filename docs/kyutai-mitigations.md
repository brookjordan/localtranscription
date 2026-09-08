# Kyutai STT 2.6B mitigation study — two failure modes

Companion analysis for the mitigation tabs in `results/dashboard.html`.
Raw data: `results/corpus/kyutai_mitigations.json` (chunked / rep-pen / temp0.6)
and `results/corpus/kyutai_vad.json` (Silero-VAD segmentation), all 15 corpus
clips × 5 variants. Tracking and decision history live in the Obsidian
transcription-evaluation project (scratchpad packet
"2026-09-08 kyutai correction and rtf scores", addenda 4–7).

## The headline

Kyutai STT 2.6B en (MLX) exhibits **two distinct failure modes**, and the
mitigation tabs trade one against the other per clip:

1. **Repetition-loop degeneration** (`iii…` forever) — content-triggered,
   not universal. Upstream issue
   [#175](https://github.com/kyutai-labs/delayed-streams-modeling/issues/175).
   Hits Brook 2, Dave, sing (baseline 50–54%).
2. **Cold-start collapse under state resets** — every fresh decoder stream
   begins with low-quality prior-driven output: hallucinated warmup tokens
   (`body`, `fornia`/`lefornia`), BPE debris (`icultidual`), and an unflushed
   delayed-text buffer at truncation points.

Underlying both: **greedy decoding is not reproducible on this stack.**
An A/A test (2026-09-08, silly.wav, same process, same weights, temp=0,
three consecutive `transcribe()` calls) produced a near-perfect transcript,
the same transcript with a stray leading `i`, and a 4-char `body` collapse.
Temp=0 sampling is argmax, so the logits themselves vary run-to-run — most
likely non-deterministic float accumulation in Metal GPU kernels (mechanism
plausible, not yet verified). The model sits on a knife's edge: small logit
perturbations flip it into a degenerate attractor. Consequences: (a) single-run
baseline-vs-variant comparisons are one sample each — near-ties are noise;
(b) the `deterministic: true` flags in the result JSONs describe intent, not
observed behaviour; (c) the stored baseline entries (reused from earlier runs
in different processes) are not comparable to fresh runs byte-for-byte.

## Why chunked destroys clips that baseline handles fine

Chunking cuts a clip into blind fixed 30 s windows and **resets decoder state
at every boundary**. For clips ≤30 s the chunked path is code-identical to
baseline (one window = the whole clip, same `transcribe()` call), so any
difference there is pure run-to-run nondeterminism — e.g. silly.m4a:
stored baseline 96.0% vs fresh chunked 0.0% ("lefornia") on the same bytes.

On longer clips, each window is a fresh stream start, i.e. a fresh draw from
the collapse lottery (~1-in-3 observed on silly.wav). N windows ≈ N draws, so
windowed variants multiply collapse risk: Balajee (≈8 min → 16 windows)
emitted 116 chars of mostly debris, and Brett (≈60 s → 2 windows) 38 chars,
vs 93–95% single-draw baselines that happened to land good. The failure
signature across windows is the same warmup debris, confirming per-stream
cold starts as the compounding factor — with nondeterminism as the reason a
stream start can fail at all.

The reverse holds where baseline itself loops: Brook 2 baseline collapses to
`iii…` (53.9%) while chunked caps the loop and scores 87.2%.

**Chunked is a cure for exactly the clips where baseline is broken, and a
poison where baseline is fine.** It is a per-clip mitigation, never a
per-model configuration.

## Why VAD doesn't fix the corpus either

If blind mid-word cuts were the problem, silence-anchored VAD groups should
rescue Balajee/Brett. They don't (0.2%, 1.2%): each VAD group is still a
fresh stream start, another draw from the collapse lottery. VAD only wins
where the draw count is small and the audio is clean (brook 94.9%, silly
92.0%, Brook 2 83.9%). Hard corpus clips still produce empty or debris
output (Nana Chris, Ping, Pritisman, Tung: ~0 chars; conrad: ~2 kchars of
loop garbage).

rep-pen and temp0.6 fare no better: both also perturb decoding on every clip
and score catastrophically on corpus clips (−130% to −500%), while sampling at
temp 0.6 is additionally non-deterministic.

## Conclusion / next experiment

Boundary-based mitigations cannot fix a per-stream collapse lottery; they
multiply draws. The promising directions are:

- **State-carrying windows**: slide the audio window but keep decoder state
  across boundaries — no reset, no cold start, loop still capped by window
  cadence.
- **Upstream loop fix** at the sampler/decoder level (detect+break the
  repetition attractor), ideally contributed against
  [#175](https://github.com/kyutai-labs/delayed-streams-modeling/issues/175).
- **Reproducibility first**: establish whether the run-to-run logit variance
  can be eliminated (MX_METAL/MOJITO flags, consistent-kernel env vars,
  forced single-thread reduction) or quantified; until then any Kyutai A/B
  needs N-run averaging per variant.

## Why run-to-run nondeterminism? (probe experiments, 2026-09-08 late)

Question: temp=0 is pure `mx.argmax` (verified in `moshi_mlx/utils/sampling.py`:
`if self.temp == 0: token = mx.argmax(logits)`), no RNG anywhere in
`LmGen`/`models.py` — so why do identical runs differ?

Probes (same process, same weights, silly.wav, 400 steps, 4 runs):
- Run outcomes repeated: 2 perfect, 2 empty (collapse rate ~1-in-3 across
  all experiments today).
- **CPU is nondeterministic too** ("grass" vs "gras") — so it is not
  Metal-kernel scheduling alone; the MLX stack (both devices, presumably
  threaded/atomics reduction order) gives slightly different logits per run.
- First argmax flip is always a **TEXT token** (audio streams identical);
  e.g. step 54: token 3 (good run) vs token 0 (collapsed run). Every
  collapsing run flips into token 0 — the collapse signature.
- At the flip step the good run's top-2 logits were `[11.25, 8.94]`
  (margin 2.3 — a comfortable argmax), while the collapsed run's were
  `[14.5, 1.29]` (margin 13.2): the logit landscape itself had already
  drifted far apart through sub-threshold numeric noise accumulating in
  the recurrent state.

Mechanism (as far as evidenced): MLX forward passes are not bit-reproducible
run-to-run → sub-argmax-threshold logit noise accumulates recurrently →
eventually one text-token argmax flips (token 0) → the degenerate attractor
amplifies that single flip into empty/garbage output. The model does not
tolerate numeric noise; single-run comparisons sample a lottery.

Practical consequences (see "What temperature" below): best-of-N at temp 0
with a degeneration heuristic, or accept per-clip per-variant N-run medians.

## What temperature should we use? (researched 2026-09-08)

**Stick with temp 0.** Evidence:

- **Vendor default is greedy**: the standard `config-stt-en_fr-hf.toml` in
  [#175](https://github.com/kyutai-labs/delayed-streams-modeling/issues/175)
  runs `temperature = 0.0` in production. Kyutai ship temp 0; the loop bug
  happens *with* it, so it is not a temp-0 problem.
- **Our temp0.6 A/B is catastrophic**: 0–7.6% on corpus clips, with BPE
  debris (`idual`, `icult`) — sampling the *audio* tokens at temp>0 corrupts
  the audio-text alignment the delayed-stream architecture depends on.
  This is not like an LLM where a little temperature adds diversity.
- **Industry pattern (Whisper)**: greedy first, then *fallback* — escalate
  temperature (0.0 → 0.2 → 0.4 …) only when heuristics detect degeneration
  (compression ratio / log-prob checks). Never sample by default.

So the answer to "should we ever use temp 0" is: temp 0 always, plus a
degeneration detector with a bounded fallback (re-decode at temp 0.2–0.4 or
with a text-token repetition penalty). Given the Metal nondeterminism, a
cheaper equivalent is **best-of-N at temp 0**: run 2–3 times, keep the output
with the best compression-ratio/length heuristic — this also hedges the
collapse lottery without touching temperature.

## Variant cull + best-of-3 shipped (2026-09-08)

Implemented in `scripts/kyutai_mitigations.py`:

- **Dropped `rep-pen` and `temp0.6`** from the variant set and the corpus JSONs
  (they were catastrophic on every clip; archived in git history and in the
  addenda below). Dashboard now shows `baseline / chunked / vad` only; the
  `check_mitigations.mjs` guard asserts exactly these three and fails if a
  banned variant reappears.
- **Best-of-3 greedy hedge** (`best_of_n()`, `looks_degenerate()`): every
  chunked window decodes up to 3 times at temp 0; empty-output / BPE-prefix
  debris triggers a re-roll, and the longest non-degenerate candidate wins.
  First healthy decode early-exits, so clean clips pay almost no overhead —
  worst case is the ~1.5x of a 2-3 roll on collapsed windows.
- **Full 15-clip re-run** with best-of-3 confirms the mechanism: healthy
  windows pass on roll 1, collapsed windows recover on rolls 2-3 (e.g. Brett
  window recovered at 535 chars on roll 2, Brook 2 window at 675 chars on
  roll 3). Known limit: on the noisiest clips (`conrad`,
  `lockpicking lawyer voice`, some Balajee/Brett/antony windows) all 3 rolls
  collapse — best-of-3 is a hedge, not a cure; baseline stays near 0% there
  and VAD segmentation remains the only mitigation that helps those.

## Scoring caveat

Runner `accuracy` values in the JSONs and the dashboard's tab labels differ
slightly on some clips: the dashboard deliberately re-scores with its own
normalizer (kay/okay folding, event-marker equivalence) rather than trusting
runner values. Deltas of 1–3 points on noisy clips are expected.

## Universal hedging + failed sampler mask (2026-09-08, later)

Two follow-ups after Tung's dashboard card showed a fluent reference against a
2.6% baseline: the stored baseline was a collapsed single draw ("i"), and only
`chunked` had been hedged.

- **Baseline and VAD now hedge too.** `best_of_n()` wraps every variant
  (baseline whole-clip, chunked per window, vad per group) — no
  detect-and-rerun workflow, hedging is inside the decode itself. Roll count
  raised to **N=5**: at the measured ~1-in-3 per-stream-start collapse rate,
  all-5-collapse residual is ~0.4% per stream start (vs ~3.7% at N=3).
  Early-exit keeps the cost at exactly 1 roll whenever the first draw is
  healthy.
- **Token-0 logit mask: negative result, do not retry.** The collapse flip
  targets token 0, and `moshi_mlx.utils.Sampler` supports `logit_bias`, so we
  probed biasing token 0 to −1e9 (probe kept at
  `scripts/kyutai_aa_masked.py`, log at `results/corpus/aa_masked.log`).
  Result: **every decode returns 0 chars** — token 0 is the padding token the
  delayed-stream decoder consumes every step; starving it collapses generation
  entirely. The drift therefore cannot be patched out at the sampler level on
  this stack; structural hedging is the correct mechanism. (Probe: Tung ×3 +
  conrad ×3, all empty, in-process.)

## Consistency study + honest labelling (2026-09-08 evening)

Question: can we make Kyutai output *consistent* run-to-run, or must we
declare it non-deterministic and label collapsed outputs as not-consistent?

Findings:

- **VAD pad-after-cap bug fixed.** `vad_groups` applied the 0.25 s pad *after*
  the 30 s cap, so groups were up to 30.25 s on disk (the antony billington
  vad01 the user flagged was 30.25 s). Now both the hard-split and merge
  paths bound the *padded* span to `MAX_SEG`; unit-checked, all segments <=30.0 s.
- **Shorter windows are NOT safer — they are worse.** 10x re-decode probe
  (`scripts/kyutai_consistency_probe.py`, log
  `results/corpus/consistency_probe.log`) on antony billington vad01:
  at 30.25 s -> 9/10 fluent, 1/10 collapsed; same audio cropped to 25 s ->
  9/10 collapsed (`fornia`, `'siiiiiii`, `,,,,...`). Cutting mid-sentence
  removes closure the decode needs; more restarts = more collapse lotteries,
  not fewer. Do not chase a smaller window size.
- **Per-draw determinism is unattainable on this stack**: Metal batch-
  invariance float drift flips the argmax near ties, and the repetition-loop
  mode is a known open upstream bug (kyutai/delayed-streams-modeling #175 —
  their servers carry a repetition-penalty mechanism "completely missing"
  from the Python/MLX path).
- **Decision: keep hedging, label honestly.** N=5 early-exit hedging stays
  (measured 1/10 per-draw collapse -> all-5-collapse residual ~1e-5/group),
  and the dashboard now marks every variant honestly:
  `nondeterministic · hedged (N=5)` normally, and
  `collapsed — not consistent` (warning badge) when a record still landed
  degenerate after all 5 draws (e.g. silly.m4a chunked went 0-for-5 in the
  refresh — that is now visible as not-consistent rather than hidden).
- **Stale-record refresh gotcha (documented for future reruns):**
  `kyutai_mitigations.py`'s resume check is key-presence based
  (`if "baseline" not in runs[stem]`), so old-era records for stems that
  already had keys are silently kept. Invalidate records explicitly (delete
  the key) when the pipeline semantics change; key-presence resume is only
  safe for crash-resume of the same pipeline version.

## Whisper language misfire: Welsh/Malay transcripts (2026-09-08 evening)

User spotted `antony billington.wav` scoring -5.4% on Whisper large-v3 (full) with a
fluent **Welsh** transcript. Not gibberish: it was a semantically faithful Welsh rendering
of the same sermon — the signature of Whisper's automatic language detection firing on the
wrong language. `mlx_whisper.transcribe()` probes the first 30 s window when `language`
is unset; on this clip's intro it returned Welsh, and Whisper dutifully emitted coherent
Welsh text for English speech.

A non-English sweep (stopword-ratio) over all Whisper-family records found a second
misfire the Welsh scan missed: `Tung.large-v3.json` came back in **Malay** (Tung is
speaking English with a Malaysian-accented voice — detection latched onto the accent).

**Fix (structural, run-and-forget safe):** pin `language="en"` in both Whisper runners:

- `scripts/whisper_variants_corpus.py` — `mlx_whisper.transcribe(..., language="en")`
- `scripts/whisper_corpus.py` (large-v3-turbo) — same pin

English-only checkpoints ignore it; multilingual ones now cannot drift. Both affected
records were deleted and re-decoded under the pinned language:

| record | before | after |
|---|---|---|
| antony billington large-v3 | -5.4% (Welsh) | 96.1% |
| Tung large-v3 | 2.6% (Malay) | 89.7% |

Whisper large-v3 corpus average: 88.4% -> **94.2%**. Guards re-run: all 3 PASS.
Corrupted records kept for evidence at /tmp (not repo-tracked):
`antony.large-v3.welsh.json`, `tung.large-v3.malay.json`.
