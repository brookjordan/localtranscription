import json
import tempfile
import unittest
from pathlib import Path

from scripts.pipeline_metrics import PipelineMetricsStore, PUBLIC_SCHEMA_VERSION


class PipelineMetricsStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PipelineMetricsStore(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_stage_duration_and_peak_memory(self):
        self.store.record("run-1", {
            "stage": "speech",
            "event": "start",
            "timestamp": "2026-09-26T00:00:00Z",
            "memory": {"hostUsedBytes": 100, "hostAvailableBytes": 900},
        })
        self.store.record("run-1", {
            "stage": "speech",
            "event": "end",
            "timestamp": "2026-09-26T00:00:05Z",
            "memory": {"hostUsedBytes": 180, "hostAvailableBytes": 820},
            "details": {"segments": 2},
        })
        stats = self.store.build_public_stats("run-1")
        self.assertEqual(stats["schemaVersion"], PUBLIC_SCHEMA_VERSION)
        self.assertEqual(stats["stages"][0]["durationMs"], 5000)
        self.assertEqual(stats["summary"]["peakHostUsedBytes"], 180)

    def test_redacts_private_fields_and_addresses(self):
        self.store.record("run-2", {
            "stage": "preflight",
            "event": "end",
            "timestamp": "2026-09-26T00:00:00Z",
            "details": {
                "token": "secret",
                "path": "/Users/brook/private/file.wav",
                "endpoint": "http://100.98.203.89:8188/system_stats",
                "route": "tailscale",
            },
        })
        encoded = json.dumps(self.store.build_public_stats("run-2"))
        self.assertNotIn("secret", encoded)
        self.assertNotIn("/Users/", encoded)
        self.assertNotIn("100.98.203.89", encoded)
        self.assertIn("tailscale", encoded)

    def test_counts_speech_attempts_and_rejections(self):
        self.store.record("run-3", {
            "stage": "speech.segment",
            "event": "end",
            "timestamp": "2026-09-26T00:00:00Z",
            "details": {
                "segmentId": "segment-1",
                "attempts": 3,
                "selectedAttempt": 2,
                "degraded": False,
                "rejectionReasons": ["wrong_voice", "missing_word"],
            },
        })
        stats = self.store.build_public_stats("run-3")
        speech = stats["speech"]
        self.assertEqual(speech["totalAttempts"], 3)
        self.assertEqual(speech["retries"], 2)
        self.assertEqual(speech["segments"][0]["selectedAttempt"], 2)

    def test_rejects_invalid_stage_and_event(self):
        with self.assertRaises(ValueError):
            self.store.record("run-4", {"stage": "../bad", "event": "start"})
        with self.assertRaises(ValueError):
            self.store.record("run-4", {"stage": "speech", "event": "maybe"})

    def test_repeated_event_is_idempotent(self):
        event = {
            "eventId": "speech-start-1",
            "stage": "speech",
            "event": "start",
            "timestamp": "2026-09-26T00:00:00Z",
        }
        self.store.record("run-5", event)
        self.store.record("run-5", event)
        stats = self.store.build_public_stats("run-5")
        self.assertEqual(len(stats["events"]), 1)

    def test_retry_with_same_event_id_does_not_conflict_on_new_memory_sample(self):
        event = {"eventId": "pipeline-start-run-8", "stage": "pipeline", "event": "start"}
        first = self.store.record("run-8", {**event, "memory": {"hostUsedBytes": 100}})
        again = self.store.record("run-8", {**event, "memory": {"hostUsedBytes": 200}})
        self.assertEqual(first, again)
        with self.assertRaises(ValueError):
            self.store.record("run-8", {**event, "stage": "speech"})

    def test_public_projection_never_reprints_untrusted_nested_text(self):
        self.store.record("run-6", {
            "stage": "speech.segment", "event": "end", "timestamp": "2026-09-26T00:00:00Z",
            "details": {
                "segmentId": "<img src=x onerror=alert(1)>",
                "attempts": 2, "selectedAttempt": 2,
                "rejectionReasons": ["wrong_voice", "Bearer private-value"],
                "nested": {"innocent": "ssh://host.internal/private?auth=private-value"},
                "model": "private-value", "unknown": "private-value",
            },
        })
        public = self.store.build_public_stats("run-6")
        encoded = json.dumps(public)
        self.assertNotIn("private-value", encoded)
        self.assertNotIn("<img", encoded)
        self.assertEqual(public["speech"]["segments"][0]["rejectionReasons"], ["wrong_voice"])

    def test_unfinished_pipeline_does_not_claim_completed_wall_clock(self):
        self.store.record("run-7", {"stage": "pipeline", "event": "start", "timestamp": "2026-09-26T00:00:00Z"})
        self.store.record("run-7", {"stage": "publication.assets", "event": "end", "timestamp": "2026-09-26T00:00:10Z"})
        stats = self.store.build_public_stats("run-7")
        self.assertIsNone(stats["summary"]["wallClockMs"])
        self.assertIsNone(stats["summary"]["endedAt"])
        self.assertIsNone(stats["speech"]["totalAttempts"])


if __name__ == "__main__":
    unittest.main()
