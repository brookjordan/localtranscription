#!/usr/bin/env python3
"""Small JSON HTTP bridge for n8n; keeps ASR and HEAR in their own environments."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import threading
import uuid
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from .evaluate_real_candidate import HEAR_PYTHON, CAMPPLUS, ASR_PYTHON, PARAKEET, run_json, WorkerProcessError
    from .speech_candidate_evaluator import evaluate_candidate, EVALUATOR_VERSION
except ImportError:
    from evaluate_real_candidate import HEAR_PYTHON, CAMPPLUS, ASR_PYTHON, PARAKEET, run_json, WorkerProcessError
    from speech_candidate_evaluator import evaluate_candidate, EVALUATOR_VERSION

try:
    from .speech_attempt_store import AttemptStore, ManifestConflict
    from .pipeline_metrics import PipelineMetricsStore, capture_memory_snapshot
except ImportError:
    from speech_attempt_store import AttemptStore, ManifestConflict
    from pipeline_metrics import PipelineMetricsStore, capture_memory_snapshot

ROOT = Path(__file__).resolve().parents[1]
BROOK = Path('/Users/brook.jordan/ComfyUI-Shared/input/Brook 2.wav')
CONRAD = Path('/Users/brook.jordan/ComfyUI-Shared/input/conrad.wav')
REFERENCES = {
    'brook': BROOK,
    'conrad': CONRAD,
    'antony': Path('/Users/brook.jordan/ComfyUI-Shared/input/antony billington.mp3'),
    'brett': Path('/Users/brook.jordan/ComfyUI-Shared/input/Brett-campplus.wav'),
}
MAX_AUDIO_BYTES = 50 * 1024 * 1024
EVALUATION_SLOT = threading.BoundedSemaphore(value=1)
STORE = AttemptStore(Path(os.environ.get(
    'PCB010_ATTEMPT_ROOT',
    '/Users/brook.jordan/ComfyUI-Shared/output/pcb010-runs',
)))
METRICS = PipelineMetricsStore(Path(os.environ.get(
    'PODCAST_METRICS_ROOT',
    '/Users/brook.jordan/ComfyUI-Shared/output/podcast-metrics',
)))


class RequestError(ValueError):
    """A caller supplied an invalid evaluation request."""


class EvaluationStageError(RuntimeError):
    """Safe, structured classification; the underlying traceback remains private."""

    def __init__(self, stage: str, code: str, dependency: str | None = None):
        super().__init__(f'{stage}: {code}')
        self.stage, self.code, self.dependency = stage, code, dependency


def evaluator_health() -> dict:
    missing = [name for name in ('ffmpeg', 'ffprobe') if not shutil.which(name)]
    missing.extend(f'reference:{speaker}' for speaker, path in REFERENCES.items() if not path.is_file())
    return {
        'ok': not missing, 'busy': False,
        'evaluatorVersion': EVALUATOR_VERSION,
        'productionWrites': False,
        'missingDependencies': missing,
    }


def failure_envelope(exc: Exception, payload: object, diagnostic_id: str) -> dict:
    stage = exc.stage if isinstance(exc, EvaluationStageError) else 'evaluator'
    code = exc.code if isinstance(exc, EvaluationStageError) else 'worker_failure'
    dependency = exc.dependency if isinstance(exc, EvaluationStageError) else None
    context = payload if isinstance(payload, dict) else {}
    safe_id = lambda value: value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', value) else None
    return {
        'error': code, 'stage': stage, 'dependency': dependency,
        'message': f'{stage} failed ({code})' + (f': {dependency}' if dependency else ''),
        'mediaRunId': safe_id(context.get('mediaRunId')),
        'segmentId': safe_id(context.get('segmentId')),
        'attempt': context.get('attempt') if type(context.get('attempt')) is int and 1 <= context['attempt'] <= 3 else None,
        'diagnosticId': diagnostic_id,
    }


def _required_string(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f'{field} is required')
    return value.strip()


def validate_request(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RequestError('request body must be an object')
    media_run_id = _required_string(payload, 'mediaRunId')
    segment_id = _required_string(payload, 'segmentId')
    expected = _required_string(payload, 'expectedText')
    speaker_id = payload.get('speakerId')
    if not isinstance(speaker_id, str) or speaker_id not in REFERENCES:
        raise RequestError('speakerId must name a configured podcast voice')
    attempt = payload.get('attempt')
    if not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= 3:
        raise RequestError('attempt must be an integer from 1 to 3')
    audio_format = payload.get('format')
    if audio_format not in {'mp3', 'wav'}:
        raise RequestError('format must be mp3 or wav')
    encoded = payload.get('candidateBase64')
    if not isinstance(encoded, str) or not encoded:
        raise RequestError('candidateBase64 is required')
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RequestError('candidateBase64 is invalid') from exc
    if not audio:
        raise RequestError('audio payload is empty')
    if len(audio) > MAX_AUDIO_BYTES:
        raise RequestError('audio payload is too large')
    generation = payload.get('generation', {})
    if not isinstance(generation, dict):
        raise RequestError('generation must be an object')
    return {
        **payload,
        'mediaRunId': media_run_id,
        'segmentId': segment_id,
        'expectedText': expected,
        'speakerId': speaker_id,
        'attempt': attempt,
        'format': audio_format,
        'audio': audio,
        'generation': generation,
    }


def evaluate(payload: dict) -> dict:
    evaluation_started = time.monotonic()
    request = validate_request(payload)
    health = evaluator_health()
    if not health['ok']:
        dependency = health['missingDependencies'][0]
        raise EvaluationStageError('preflight', 'dependency_missing', dependency)
    expected = request['expectedText']
    attempt = request['attempt']
    target = REFERENCES[request['speakerId']]
    others = [path for speaker, path in REFERENCES.items() if speaker != request['speakerId']]
    if not all(path.is_file() for path in [target, *others]):
        raise RuntimeError('speaker reference audio is unavailable')
    reference_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    identity = {
        'expectedTextHash': hashlib.sha256(expected.encode('utf-8')).hexdigest(),
        'speakerId': request['speakerId'],
        'referenceHash': reference_hash,
        'generatorVersion': f"{request['generation'].get('engine', 'unknown')}:{request['generation'].get('model', 'unknown')}",
        'evaluatorVersion': EVALUATOR_VERSION,
    }
    candidate = STORE.persist_candidate(
        request['mediaRunId'],
        request['segmentId'],
        attempt,
        request['audio'],
        identity,
        request['generation'],
        request['format'],
    )
    try:
        try:
            asr = run_json([str(ASR_PYTHON), str(PARAKEET), str(candidate)])
        except WorkerProcessError as exc:
            raise EvaluationStageError('asr', 'dependency_missing' if exc.dependency else 'worker_failed', exc.dependency) from exc
        except Exception as exc:
            raise EvaluationStageError('asr', 'worker_failed') from exc
        try:
            voice = run_json([str(HEAR_PYTHON), str(CAMPPLUS), str(candidate), str(target), *(str(path) for path in others)])
        except WorkerProcessError as exc:
            raise EvaluationStageError('speaker_probe', 'dependency_missing' if exc.dependency else 'worker_failed', exc.dependency) from exc
        except Exception as exc:
            raise EvaluationStageError('speaker_probe', 'worker_failed') from exc
        try:
            protected_terms = request.get('protectedTerms')
            result = evaluate_candidate(candidate, expected, asr['text'], {
                'targetScore': voice['scores']['target'],
                'otherScore': voice['scores']['other'],
            }, attempt, protected_terms if isinstance(protected_terms, list) else None)
        except FileNotFoundError as exc:
            dependency = Path(exc.filename or '').name
            raise EvaluationStageError('technical_metrics', 'dependency_missing', dependency if dependency in {'ffmpeg', 'ffprobe'} else None) from exc
        except Exception as exc:
            raise EvaluationStageError('technical_metrics', 'evaluation_failed') from exc
        result.update({
            'evaluatorVersion': EVALUATOR_VERSION,
            'mediaRunId': request['mediaRunId'],
            'segmentId': request['segmentId'],
            'speakerId': request['speakerId'],
            'attempt': attempt,
            'generation': request['generation'],
            'asr': asr,
            'voiceProbe': voice,
            'productionWrites': False,
        })
        evaluation_path = STORE.persist_evaluation(
            request['mediaRunId'], request['segmentId'], attempt, result
        )
        METRICS.record(request['mediaRunId'], {
            'eventId': f"speech-attempt-{request['segmentId']}-{attempt}",
            'stage': 'speech.attempt', 'event': 'end',
            'memory': capture_memory_snapshot(),
            'details': {
                'segmentId': request['segmentId'], 'attempt': attempt,
                'passed': result['passed'],
                'rejectionReasons': result['rejectionReasons'],
                'evaluationMs': round((time.monotonic() - evaluation_started) * 1000),
            },
        })
        result['candidatePath'] = str(candidate)
        result['candidateSha256'] = hashlib.sha256(request['audio']).hexdigest()
        result['evaluationPath'] = str(evaluation_path)
        if result['passed']:
            result['selection'] = STORE.select(
                request['mediaRunId'],
                request['segmentId'],
                attempt,
                degraded=False,
                reason='first_passing_candidate',
            )
        return result
    except ManifestConflict:
        raise


class Handler(BaseHTTPRequestHandler):
    server_version = 'PCB010Evaluator/1.0'

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == '/healthz':
            health = evaluator_health()
            busy = not EVALUATION_SLOT.acquire(blocking=False)
            if not busy:
                EVALUATION_SLOT.release()
            health['busy'] = busy
            self._send(200 if health['ok'] else 503, health)
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self) -> None:
        if self.path not in {'/evaluate', '/select', '/metric', '/stats'}:
            self._send(404, {'error': 'not found'})
            return
        payload = None
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > MAX_AUDIO_BYTES * 2:
                raise RequestError('request too large')
            payload = json.loads(self.rfile.read(length))
            if self.path == '/select':
                media_run_id = _required_string(payload, 'mediaRunId')
                segment_id = _required_string(payload, 'segmentId')
                self._send(200, STORE.select_best_degraded(media_run_id, segment_id))
                return
            if self.path == '/metric':
                media_run_id = _required_string(payload, 'mediaRunId')
                metric = payload.get('metric')
                if not isinstance(metric, dict):
                    raise RequestError('metric must be an object')
                stored = METRICS.record(media_run_id, {
                    **metric,
                    'memory': capture_memory_snapshot(),
                })
                passthrough = payload.get('passthrough')
                if isinstance(passthrough, dict):
                    self._send(200, {**passthrough, 'metricRecorded': stored['eventId']})
                else:
                    self._send(200, {'ok': True, 'eventId': stored['eventId']})
                return
            if self.path == '/stats':
                media_run_id = _required_string(payload, 'mediaRunId')
                self._send(200, METRICS.build_public_stats(media_run_id))
                return
            if not EVALUATION_SLOT.acquire(blocking=False):
                self._send(503, {'error': 'worker_busy', 'message': 'the evaluator is processing another request'})
                return
            try:
                self._send(200, evaluate(payload))
            finally:
                EVALUATION_SLOT.release()
        except (RequestError, json.JSONDecodeError) as exc:
            self._send(400, {'error': 'validation_error', 'message': str(exc)})
        except ManifestConflict as exc:
            self._send(409, {'error': 'manifest_conflict', 'message': str(exc)})
        except Exception as exc:
            diagnostic_id = uuid.uuid4().hex[:12]
            print(f'evaluator diagnosticId={diagnostic_id}', file=sys.stderr, flush=True)
            traceback.print_exc()
            self._send(503 if isinstance(exc, EvaluationStageError) and exc.code == 'dependency_missing' else 500,
                       failure_envelope(exc, payload, diagnostic_id))

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    # launchd starts with a minimal PATH; all worker and technical probes need Homebrew ffmpeg/ffprobe.
    os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')
    host = os.environ.get('PCB010_API_HOST', '100.98.203.89')
    port = int(os.environ.get('PCB010_API_PORT', '8765'))
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == '__main__':
    main()
