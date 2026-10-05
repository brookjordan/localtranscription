import json
import tempfile
import unittest
from pathlib import Path

from scripts.speech_attempt_store import AttemptStore, ManifestConflict


class AttemptStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = AttemptStore(Path(self.temp.name))
        self.identity = {
            "expectedTextHash": "text-hash",
            "speakerId": "brook",
            "referenceHash": "reference-hash",
            "generatorVersion": "vibevoice-7",
            "evaluatorVersion": "pcb-010-evaluator-v0.1.3",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_persists_candidate_evaluation_and_selection_atomically(self):
        candidate = self.store.persist_candidate(
            "run-42", "segment-3", 1, b"audio bytes", self.identity, {"seed": 123}
        )
        evaluation = {"attempt": 1, "passed": True, "rankVector": [1, 1, 0, 0]}
        self.store.persist_evaluation("run-42", "segment-3", 1, evaluation)
        selected = self.store.select("run-42", "segment-3", 1, degraded=False, reason="first_passing_candidate")

        self.assertEqual(candidate.read_bytes(), b"audio bytes")
        segment_dir = candidate.parent
        self.assertEqual(json.loads((segment_dir / "attempt-1.evaluation.json").read_text()), evaluation)
        self.assertEqual(json.loads((segment_dir / "selected.json").read_text()), selected)
        self.assertEqual(selected["candidateSha256"], "ef71589075ccf9332917b0d8d711d1a8d205560f96842f9221de70e6c29454e0")

    def test_repeated_same_attempt_is_idempotent(self):
        first = self.store.persist_candidate(
            "run-42", "segment-3", 1, b"audio bytes", self.identity, {"seed": 123}
        )
        second = self.store.persist_candidate(
            "run-42", "segment-3", 1, b"audio bytes", self.identity, {"seed": 123}
        )
        self.assertEqual(first, second)

    def test_changed_candidate_for_same_attempt_is_rejected(self):
        self.store.persist_candidate(
            "run-42", "segment-3", 1, b"audio bytes", self.identity, {"seed": 123}
        )
        with self.assertRaisesRegex(ManifestConflict, "attempt 1 already exists with different audio"):
            self.store.persist_candidate(
                "run-42", "segment-3", 1, b"different", self.identity, {"seed": 124}
            )

    def test_changed_segment_identity_invalidates_reuse(self):
        self.store.persist_candidate(
            "run-42", "segment-3", 1, b"audio bytes", self.identity, {"seed": 123}
        )
        changed = {**self.identity, "expectedTextHash": "changed"}
        with self.assertRaisesRegex(ManifestConflict, "segment identity does not match existing manifest"):
            self.store.persist_candidate(
                "run-42", "segment-3", 2, b"next candidate", changed, {"seed": 124}
            )

    def test_rejects_unsafe_path_components(self):
        with self.assertRaisesRegex(ValueError, "mediaRunId contains unsafe characters"):
            self.store.persist_candidate("../escape", "segment-3", 1, b"audio", self.identity, {})

    def test_selects_deterministic_degraded_fallback_from_persisted_evaluations(self):
        rankings = ([1, 1, 0, -1], [1, 1, 0, -2], [1, 0, 0, 0])
        for attempt, rank in enumerate(rankings, 1):
            self.store.persist_candidate(
                "run-42", "segment-3", attempt, f"audio {attempt}".encode(), self.identity, {"seed": attempt}
            )
            self.store.persist_evaluation(
                "run-42", "segment-3", attempt,
                {"attempt": attempt, "passed": False, "rankVector": list(rank), "rejectionReasons": ["transcript_gate"]},
            )

        selected = self.store.select_best_degraded("run-42", "segment-3")
        self.assertEqual(selected["selectedAttempt"], 1)
        self.assertTrue(selected["degraded"])
        self.assertEqual(selected["selectionReason"], "deterministic_best_after_three_failures")

    def test_degraded_fallback_requires_at_least_one_evaluation(self):
        with self.assertRaisesRegex(ManifestConflict, "at least one completed evaluation is required"):
            self.store.select_best_degraded("run-42", "segment-3")

    def test_degraded_fallback_selects_best_of_available_evaluations(self):
        self.store.persist_candidate(
            "run-42", "segment-3", 1, b"audio", self.identity, {"seed": 1}
        )
        self.store.persist_evaluation(
            "run-42", "segment-3", 1, {"attempt": 1, "passed": False, "rankVector": [1, 1]}
        )
        self.store.persist_candidate(
            "run-42", "segment-3", 3, b"audio3", self.identity, {"seed": 3}
        )
        self.store.persist_evaluation(
            "run-42", "segment-3", 3, {"attempt": 3, "passed": False, "rankVector": [2, 1]}
        )
        selected = self.store.select_best_degraded("run-42", "segment-3")
        self.assertEqual(selected["selectedAttempt"], 3)

    def test_degraded_fallback_still_rejects_when_an_attempt_passed(self):
        self.store.persist_candidate(
            "run-42", "segment-3", 1, b"audio", self.identity, {"seed": 1}
        )
        self.store.persist_evaluation(
            "run-42", "segment-3", 1, {"attempt": 1, "passed": True, "rankVector": [1]}
        )
        with self.assertRaisesRegex(ManifestConflict, "degraded fallback is invalid because an attempt passed"):
            self.store.select_best_degraded("run-42", "segment-3")


if __name__ == "__main__":
    unittest.main()
