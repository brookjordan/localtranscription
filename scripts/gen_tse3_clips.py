#!/usr/bin/env python3
"""Generate TSE-03 multi-speaker / code-switch test clips with Kokoro (local TTS).

Ground truth is script-verified by construction: every segment's exact text,
speaker and voice are fixed by this script. Writes:
  results/assets/corpus/<stem>.wav
  results/assets/voices/<stem>.segments.json   (ground truth: [{start,end,speaker,text,voice}])
  results/assets/voices/<stem>.txt             (reference text, one line per segment)

Clips (TSE-03 completion criterion: >=3 multi-speaker, distinct speaker counts,
>=1 code-switching):
  tse3-two      2 speakers (bf_emma, am_adam)        - English dialogue
  tse3-three    3 speakers (bm_george, af_bella, zm_yunyang) - English, Chinese guest line
  tse3-codeswitch 2 speakers alternating EN/FR mid-turn (am_fenrir, ff_siwis)

Re-runnable: regenerates deterministically from the same scripts.
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np
from mlx_audio.tts.generate import generate_audio

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "assets" / "corpus"
VOICES = ROOT / "results" / "assets" / "voices"
SR = 24000  # kokoro output sample rate

CLIPS: dict[str, list[dict]] = {
    "tse3-two": [
        {"speaker": 0, "voice": "bf_emma", "lang": "en",
         "text": "Welcome back to the show. Today we're looking at how speech recognisers handle more than one voice at a time, and why that is harder than it sounds."},
        {"speaker": 1, "voice": "am_adam", "lang": "en",
         "text": "Right, because a single speaker is predictable, but a conversation overlaps, interrupts and changes pace without warning."},
        {"speaker": 0, "voice": "bf_emma", "lang": "en",
         "text": "Exactly. So keep an ear on the speaker labels as we go. If the engine cannot tell us apart, that is a diarization failure, not a transcription failure."},
    ],
    "tse3-three": [
        {"speaker": 0, "voice": "bm_george", "lang": "en",
         "text": "Panel time. Three voices tonight, and a question for each of you in turn. Let us start with the newest member of the team."},
        {"speaker": 1, "voice": "af_bella", "lang": "en",
         "text": "Delighted to be here. The hardest part of multi-speaker audio for me is turn taking, because the model must decide when one person has finished."},
        {"speaker": 2, "voice": "zm_yunyang", "lang": "zh",
         "text": "大家好，很高兴参加这个节目。我认为最困难的是多人同时说话的时候，模型需要分辨每一个人的声音。"},
        {"speaker": 0, "voice": "bm_george", "lang": "en",
         "text": "And that is the three-voice problem in one clip: two accents of English and one full turn of Mandarin, with a real speaker rotation between every turn."},
    ],
    "tse3-codeswitch": [
        {"speaker": 0, "voice": "am_fenrir", "lang": "en",
         "text": "So, we said we'd try something different this episode. Let's mix it up a little. Prêt à essayer? Je vais continuer en français, si tu préfères."},
        {"speaker": 1, "voice": "ff_siwis", "lang": "fr",
         "text": "Bien sûr, pas de problème. Je pense que le passage d'une langue à l'autre au milieu d'une phrase est le vrai test pour un modèle de transcription."},
        {"speaker": 0, "voice": "am_fenrir", "lang": "en",
         "text": "Exactly — code-switching. Most engines pick one language per utterance and force everything into it, which is how you get fantasy French or nonsense English."},
    ],
}


KOKORO = str(Path.home() / ".cache/huggingface/hub/models--mlx-community--kokoro_mlx/snapshots/ca32fa7158a1a54e7aa6c56ebd544104cfeb16c8")
_model = None  # load once, reuse across segments


def synth(text: str, voice: str, lang: str, out: Path) -> None:
    """Generate one segment to `out`; generate_audio joins multi-line output
    itself when join_audio=True, writing a single file to output_path."""
    global _model
    if _model is None:
        # Point load_model at a subdirectory named "kokoro_mlx" (symlink to the
        # snapshot) so mlx_audio's name-parts matcher picks the kokoro
        # architecture — the snapshot config has no model_type field.
        link = VOICES / "kokoro_mlx"
        if not link.exists():
            link.symlink_to(KOKORO, target_is_directory=True)
        from mlx_audio.tts.utils import load_model as _load
        _model = _load(model_path=str(link))
    generate_audio(
        text=text,
        model=_model,
        voice=voice,
        lang_code=lang,
        speed=1.0,
        output_path=str(out.parent),
        file_prefix=out.stem,
        audio_format="wav",
        join_audio=True,
        play=False,
        verbose=False,
    )


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    VOICES.mkdir(parents=True, exist_ok=True)
    for stem, segs_spec in CLIPS.items():
        out_wav = CORPUS / f"{stem}.wav"
        truth: list[dict] = []
        chunks: list[np.ndarray] = []
        t = 0.0
        print(f"== {stem}: {len(segs_spec)} segments", flush=True)
        for i, spec in enumerate(segs_spec):
            part = VOICES / f".{stem}.seg{i}.wav"
            synth(spec["text"], spec["voice"], spec["lang"], part)
            with wave.open(str(part), "rb") as w:
                sr, n = w.getframerate(), w.getnframes()
                assert sr == SR, f"unexpected sr {sr}"
                audio = np.frombuffer(w.readframes(n), dtype=np.int16)
            dur = n / sr
            truth.append({
                "start": round(t, 3), "end": round(t + dur, 3),
                "speaker": spec["speaker"], "voice": spec["voice"],
                "lang": spec["lang"], "text": spec["text"],
            })
            chunks.append(audio)
            # 0.4 s silence gap between speakers, as in natural turn-taking
            chunks.append(np.zeros(int(0.4 * SR), dtype=np.int16))
            t += dur + 0.4
            print(f"  seg{i} spk{spec['speaker']} {spec['voice']} {dur:.2f}s", flush=True)
        joined = np.concatenate(chunks)
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(joined.tobytes())
        (VOICES / f"{stem}.segments.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))
        (VOICES / f"{stem}.txt").write_text("\n".join(s["text"] for s in truth) + "\n")
        for part in VOICES.glob(f".{stem}.seg*.wav"):
            part.unlink()
        print(f"  wrote {out_wav.name} {len(joined)/SR:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
