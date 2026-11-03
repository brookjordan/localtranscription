import json
import subprocess
from pathlib import Path

from scripts.speech_candidate_evaluator import (
    EVALUATOR_VERSION,
    build_rank_vector,
    normalise_text,
    technical_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIO = Path('/tmp/nas-assets/podcast/v2026.39.5-22.57/audio_2.mp3')


def test_normalise_text_ignores_punctuation_and_ellipsis():
    assert normalise_text("We're just... tumbling.") == normalise_text("we're just tumbling")


def test_rank_vector_is_lexicographic_and_attempt_is_last():
    record = {
        'technical': {'safe': True},
        'voice': {'targetIsTopMatch': True, 'margin': 0.7, 'targetScore': 0.8},
        'transcript': {'protectedTermFailures': 0, 'missingWords': 0, 'addedWords': 0, 'wer': 0.0},
        'defects': {'severity': 0.0},
        'attempt': 2,
    }
    assert build_rank_vector(record) == [1, 1, 0, 0, 0, 0.0, 0.7, 0.8, -0.0, -2]


def test_technical_metrics_reads_episode_audio():
    assert AUDIO.exists()
    metrics = technical_metrics(AUDIO)
    assert metrics['decode']['ok'] is True
    assert metrics['sampleRate'] == 24000
    assert metrics['channels'] == 1
    assert metrics['durationSeconds'] == 6.0
    assert metrics['safe'] is True


def test_contract_is_versioned():
    assert EVALUATOR_VERSION == 'pcb-010-evaluator-v0.1.2'
