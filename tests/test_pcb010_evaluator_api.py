import base64
import json
import threading
import urllib.error
import urllib.request
import unittest
from http.server import ThreadingHTTPServer

from scripts.pcb010_evaluator_api import Handler, RequestError, EvaluationStageError, evaluator_health, failure_envelope, validate_request
from unittest.mock import patch


VALID_AUDIO = base64.b64encode(b"candidate audio").decode("ascii")


class EvaluatorRequestValidationTests(unittest.TestCase):
    def valid_payload(self):
        return {
            "mediaRunId": "run-42",
            "segmentId": "segment-3",
            "attempt": 1,
            "expectedText": "Hello world",
            "speakerId": "brook",
            "candidateBase64": VALID_AUDIO,
            "format": "mp3",
            "generation": {
                "engine": "VibeVoice",
                "model": "VibeVoice-7",
                "seed": 123,
                "temperature": 0.45,
            },
        }

    def test_accepts_complete_request(self):
        request = validate_request(self.valid_payload())
        self.assertEqual(request["mediaRunId"], "run-42")
        self.assertEqual(request["segmentId"], "segment-3")
        self.assertEqual(request["speakerId"], "brook")
        self.assertEqual(request["audio"], b"candidate audio")

    def test_rejects_attempt_outside_bounded_range(self):
        payload = self.valid_payload()
        payload["attempt"] = 4
        with self.assertRaisesRegex(RequestError, "attempt must be an integer from 1 to 3"):
            validate_request(payload)

    def test_rejects_unknown_speaker(self):
        payload = self.valid_payload()
        payload["speakerId"] = "unknown"
        with self.assertRaisesRegex(RequestError, "speakerId must name a configured podcast voice"):
            validate_request(payload)

    def test_accepts_all_four_existing_podcast_voices(self):
        for speaker in ("brook", "conrad", "antony", "brett"):
            payload = self.valid_payload()
            payload["speakerId"] = speaker
            with self.subTest(speaker=speaker):
                self.assertEqual(validate_request(payload)["speakerId"], speaker)

    def test_rejects_unsupported_audio_format(self):
        payload = self.valid_payload()
        payload["format"] = "flac"
        with self.assertRaisesRegex(RequestError, "format must be mp3 or wav"):
            validate_request(payload)

    def test_rejects_invalid_base64(self):
        payload = self.valid_payload()
        payload["candidateBase64"] = "not base64"
        with self.assertRaisesRegex(RequestError, "candidateBase64 is invalid"):
            validate_request(payload)

    def test_rejects_missing_identity_fields(self):
        for field in ("mediaRunId", "segmentId", "expectedText"):
            payload = self.valid_payload()
            payload[field] = ""
            with self.subTest(field=field):
                with self.assertRaisesRegex(RequestError, f"{field} is required"):
                    validate_request(payload)
    def test_failure_envelope_reports_stage_without_leaking_audio_or_paths(self):
        payload = self.valid_payload()
        payload['expectedText'] = 'secret transcript'
        payload['candidateBase64'] = 'private audio bytes'
        error = EvaluationStageError('asr', 'dependency_missing', 'ffmpeg')
        result = failure_envelope(error, payload, 'abc123')
        self.assertEqual(result, {
            'error': 'dependency_missing', 'stage': 'asr', 'dependency': 'ffmpeg',
            'message': 'asr failed (dependency_missing): ffmpeg',
            'mediaRunId': 'run-42', 'segmentId': 'segment-3', 'attempt': 1,
            'diagnosticId': 'abc123',
        })
        self.assertNotIn('secret transcript', str(result))
        self.assertNotIn('private audio bytes', str(result))

    def test_health_flags_missing_ffmpeg_before_generation(self):
        with patch('scripts.pcb010_evaluator_api.shutil.which', side_effect=lambda name: None if name == 'ffmpeg' else '/opt/homebrew/bin/ffprobe'):
            result = evaluator_health()
        self.assertEqual(result['ok'], False)
        self.assertIn('ffmpeg', result['missingDependencies'])
    def test_http_failure_surfaces_safe_stage_and_diagnostic_id(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = self.valid_payload()
            payload['expectedText'] = 'private transcript'
            with patch('scripts.pcb010_evaluator_api.evaluate', side_effect=EvaluationStageError('asr', 'dependency_missing', 'ffmpeg')):
                request = urllib.request.Request(
                    f'http://127.0.0.1:{server.server_port}/evaluate',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code, 503)
                response = json.loads(caught.exception.read())
            self.assertEqual(response['error'], 'dependency_missing')
            self.assertEqual(response['stage'], 'asr')
            self.assertEqual(response['dependency'], 'ffmpeg')
            self.assertEqual(response['mediaRunId'], 'run-42')
            self.assertEqual(len(response['diagnosticId']), 12)
            self.assertNotIn('private transcript', str(response))
            self.assertNotIn('candidateBase64', str(response))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
