import json
import subprocess
from pathlib import Path

from scripts.speech_candidate_evaluator import choose_candidate, evaluate_candidate


def make_tone(path: Path, *, silence=False, clip=False):
    source = 'anullsrc=r=24000:cl=mono' if silence else 'sine=frequency=220:duration=1'
    af = 'volume=10' if clip else 'volume=0.2'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', source,
                    '-t', '1', '-af', af, '-ar', '24000', '-ac', '1', str(path)], check=True)


def test_known_good_candidate_passes(tmp_path):
    audio = tmp_path / 'good.wav'
    make_tone(audio)
    result = evaluate_candidate(audio, 'hello world', 'hello world',
                                {'targetScore': 0.8, 'otherScore': 0.1})
    assert result['passed'] is True
    assert result['rejectionReasons'] == []


def test_known_bad_transcript_is_rejected(tmp_path):
    audio = tmp_path / 'bad.wav'
    make_tone(audio)
    result = evaluate_candidate(audio, 'hello world', 'nonsense',
                                {'targetScore': 0.8, 'otherScore': 0.1})
    assert result['passed'] is False
    assert 'transcript_gate' in result['rejectionReasons']


def test_three_attempt_workflow_stops_at_first_pass(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f'{index}.wav'
        make_tone(path)
        paths.append(path)
    candidates = [
        evaluate_candidate(paths[0], 'hello world', 'nonsense', {'targetScore': 0.8, 'otherScore': 0.1}, attempt=1),
        evaluate_candidate(paths[1], 'hello world', 'hello world', {'targetScore': 0.8, 'otherScore': 0.1}, attempt=2),
        evaluate_candidate(paths[2], 'hello world', 'hello world', {'targetScore': 0.8, 'otherScore': 0.1}, attempt=3),
    ]
    selected = choose_candidate(candidates)
    assert selected['attempt'] == 2
    assert selected['degraded'] is False


def test_three_failed_attempts_choose_deterministic_fallback(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f'{index}.wav'
        make_tone(path)
        paths.append(path)
    candidates = [
        evaluate_candidate(paths[0], 'hello world', 'hello', {'targetScore': 0.8, 'otherScore': 0.1}, attempt=1),
        evaluate_candidate(paths[1], 'hello world', 'nonsense', {'targetScore': 0.95, 'otherScore': 0.1}, attempt=2),
        evaluate_candidate(paths[2], 'hello world', 'hello', {'targetScore': 0.7, 'otherScore': 0.1}, attempt=3),
    ]
    selected = choose_candidate(candidates)
    assert selected['attempt'] == 1
    assert selected['degraded'] is True


def test_clipped_audio_is_rejected_by_technical_gate(tmp_path):
    audio = tmp_path / 'clipped.wav'
    make_tone(audio, clip=True)
    result = evaluate_candidate(audio, 'hello world', 'hello world',
                                {'targetScore': 0.8, 'otherScore': 0.1})
    assert result['passed'] is False
    assert 'technical_gate' in result['rejectionReasons']


def test_wrong_voice_is_rejected_even_with_correct_words(tmp_path):
    audio = tmp_path / 'wrong-voice.wav'
    make_tone(audio)
    result = evaluate_candidate(audio, 'hello world', 'hello world',
                                {'targetScore': 0.2, 'otherScore': 0.8})
    assert result['passed'] is False
    assert 'voice_gate' in result['rejectionReasons']


def test_malformed_audio_is_rejected(tmp_path):
    audio = tmp_path / 'malformed.wav'
    audio.write_bytes(b'not audio')
    result = evaluate_candidate(audio, 'hello world', 'hello world',
                                {'targetScore': 0.8, 'otherScore': 0.1})
    assert result['passed'] is False
    assert 'technical_gate' in result['rejectionReasons']
