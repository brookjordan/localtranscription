#!/usr/bin/env python3
"""Durable, idempotent storage for PCB-010 speech attempts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ManifestConflict(RuntimeError):
    """Stored attempt state conflicts with the requested operation."""


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class AttemptStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _component(field: str, value: str) -> str:
        if not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"{field} contains unsafe characters")
        return value

    def segment_dir(self, media_run_id: str, segment_id: str) -> Path:
        run = self._component("mediaRunId", media_run_id)
        segment = self._component("segmentId", segment_id)
        return self.root / run / "speech" / segment

    @staticmethod
    def _attempt(attempt: int) -> int:
        if not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= 3:
            raise ValueError("attempt must be an integer from 1 to 3")
        return attempt

    def persist_candidate(
        self,
        media_run_id: str,
        segment_id: str,
        attempt: int,
        audio: bytes,
        identity: dict,
        generation: dict,
        audio_format: str = "mp3",
    ) -> Path:
        attempt = self._attempt(attempt)
        if audio_format not in {"mp3", "wav"}:
            raise ValueError("audio_format must be mp3 or wav")
        directory = self.segment_dir(media_run_id, segment_id)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "segment.json"
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            if stored != identity:
                raise ManifestConflict("segment identity does not match existing manifest")
        else:
            _atomic_json(manifest_path, identity)

        candidate = directory / f"attempt-{attempt}.{audio_format}"
        metadata_path = directory / f"attempt-{attempt}.generation.json"
        digest = hashlib.sha256(audio).hexdigest()
        if candidate.exists():
            existing_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if existing_digest != digest:
                raise ManifestConflict(f"attempt {attempt} already exists with different audio")
            return candidate

        _atomic_bytes(candidate, audio)
        _atomic_json(metadata_path, {**generation, "attempt": attempt, "candidateSha256": digest})
        return candidate

    def persist_evaluation(
        self, media_run_id: str, segment_id: str, attempt: int, evaluation: dict
    ) -> Path:
        attempt = self._attempt(attempt)
        directory = self.segment_dir(media_run_id, segment_id)
        candidates = [
            path for path in sorted(directory.glob(f"attempt-{attempt}.*"))
            if not path.name.endswith(".json")
        ]
        if not candidates:
            raise ManifestConflict(f"attempt {attempt} candidate does not exist")
        path = directory / f"attempt-{attempt}.evaluation.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored != evaluation:
                raise ManifestConflict(f"attempt {attempt} evaluation already exists with different data")
            return path
        _atomic_json(path, evaluation)
        return path

    def select(
        self,
        media_run_id: str,
        segment_id: str,
        attempt: int,
        *,
        degraded: bool,
        reason: str,
    ) -> dict:
        attempt = self._attempt(attempt)
        directory = self.segment_dir(media_run_id, segment_id)
        candidates = [
            path for path in sorted(directory.glob(f"attempt-{attempt}.*"))
            if not path.name.endswith(".json")
        ]
        if not candidates:
            raise ManifestConflict(f"attempt {attempt} candidate does not exist")
        candidate = candidates[0]
        evaluation_path = directory / f"attempt-{attempt}.evaluation.json"
        if not evaluation_path.exists():
            raise ManifestConflict(f"attempt {attempt} evaluation does not exist")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        selected = {
            "selectedAttempt": attempt,
            "candidatePath": str(candidate),
            "candidateSha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "degraded": bool(degraded),
            "selectionReason": reason,
            "rejectionReasons": evaluation.get("rejectionReasons", []),
            "rankVector": evaluation.get("rankVector", []),
        }
        path = directory / "selected.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored != selected:
                raise ManifestConflict("selection already exists with different data")
            return stored
        _atomic_json(path, selected)
        return selected

    def select_best_degraded(self, media_run_id: str, segment_id: str) -> dict:
        directory = self.segment_dir(media_run_id, segment_id)
        evaluations = []
        for attempt in range(1, 4):
            path = directory / f"attempt-{attempt}.evaluation.json"
            if path.exists():
                evaluations.append(json.loads(path.read_text(encoding="utf-8")))
        if not evaluations:
            raise ManifestConflict("at least one completed evaluation is required for degraded fallback")
        if any(item.get("passed") for item in evaluations):
            raise ManifestConflict("degraded fallback is invalid because an attempt passed")
        selected = max(
            evaluations,
            key=lambda item: tuple(item.get("rankVector", [-1])),
        )
        return self.select(
            media_run_id,
            segment_id,
            int(selected["attempt"]),
            degraded=True,
            reason="deterministic_best_after_three_failures",
        )
