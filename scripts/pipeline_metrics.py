#!/usr/bin/env python3
"""Durable public-safe pipeline metrics for podcast runs."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PUBLIC_SCHEMA_VERSION = "podcast-stats-v0.2.0"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_STAGE = re.compile(r"^[a-z][a-z0-9._-]{0,79}$")
_EVENTS = {"start", "checkpoint", "end", "failure"}
_SECRET_KEYS = re.compile(r"token|secret|password|credential|authorization|cookie|api.?key", re.I)
_PRIVATE_KEYS = re.compile(r"path|endpoint|host|url|stdout|stderr|error", re.I)
_PRIVATE_TEXT = re.compile(
    r"(?:/Users/|/Volumes/|/volume\d+/|https?://(?:\d{1,3}\.){3}\d{1,3}|\b(?:\d{1,3}\.){3}\d{1,3}\b)"
)


def capture_memory_snapshot() -> dict[str, Any]:
    """Capture host and ComfyUI memory without exposing process arguments."""
    snapshot: dict[str, Any] = {}
    try:
        total = int(subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip())
        vm = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        page_match = re.search(r"page size of (\d+) bytes", vm)
        page_size = int(page_match.group(1)) if page_match else 4096
        pages = {}
        for line in vm.splitlines():
            match = re.match(r"([^:]+):\s+(\d+)\.?$", line.strip())
            if match:
                pages[match.group(1)] = int(match.group(2))
        available_pages = sum(pages.get(name, 0) for name in (
            "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"
        ))
        available = min(total, available_pages * page_size)
        snapshot.update({
            "hostTotalBytes": total,
            "hostAvailableBytes": available,
            "hostUsedBytes": max(0, total - available),
        })
    except (OSError, ValueError, subprocess.SubprocessError):
        snapshot["hostMemoryStatus"] = "unavailable"
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2) as response:
            body = json.load(response)
        system = body.get("system", {})
        devices = body.get("devices", [])
        snapshot["comfySystemRamTotalBytes"] = system.get("ram_total")
        snapshot["comfySystemRamFreeBytes"] = system.get("ram_free")
        snapshot["comfyDevices"] = [{
            "name": device.get("name"),
            "type": device.get("type"),
            "vramTotalBytes": device.get("vram_total"),
            "vramFreeBytes": device.get("vram_free"),
        } for device in devices]
    except (OSError, ValueError, TimeoutError):
        snapshot["comfyMemoryStatus"] = "unavailable"
    return snapshot


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_PUBLIC_REASONS = frozenset({
    "wrong_voice", "missing_word", "added_word", "transcript_mismatch",
    "low_similarity", "low_margin", "clipping", "silence", "dropout",
    "stutter", "duration_outlier", "decode_failure", "incomplete_coverage",
    "technical_gate", "transcript_gate", "voice_gate",
})
_PUBLIC_ROUTES = frozenset({"ethernet", "wlan", "tailscale", "local", "unavailable"})


def _public_details(stage: str, details: Any) -> dict[str, Any]:
    """Explicit projection: never copy arbitrary strings into a public sidecar."""
    if not isinstance(details, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("assetCount", "imageCount", "attempt", "attempts", "selectedAttempt", "seed",
                "bytes", "durationMs", "generationMs", "evaluationMs", "audioDurationMs"):
        value = details.get(key)
        if type(value) is int and value >= 0:
            result[key] = value
    if stage in {"speech.segment", "speech.attempt"}:
        if type(details.get("degraded")) is bool:
            result["degraded"] = details["degraded"]
        reasons = details.get("rejectionReasons")
        if isinstance(reasons, list):
            result["rejectionReasons"] = [r for r in reasons if type(r) is str and r in _PUBLIC_REASONS]
        if type(details.get("passed")) is bool:
            result["passed"] = details["passed"]
        identifier = details.get("segmentId")
        if isinstance(identifier, str) and _SAFE_COMPONENT.fullmatch(identifier):
            result["segmentId"] = identifier
    route = details.get("route")
    if isinstance(route, str) and route in _PUBLIC_ROUTES:
        result["route"] = route
    return result


def _clean(value: Any, key: str = "") -> Any:
    if _SECRET_KEYS.search(key):
        return None
    if _PRIVATE_KEYS.fullmatch(key):
        if key.lower() in {"route", "routeclass"}:
            return _clean(value)
        return None
    if isinstance(value, dict):
        return {k: cleaned for k, v in value.items() if (cleaned := _clean(v, k)) is not None}
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _clean(item)) is not None]
    if isinstance(value, str) and _PRIVATE_TEXT.search(value):
        return "[redacted]"
    return value


class PipelineMetricsStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _directory(self, run_id: str) -> Path:
        if not _SAFE_COMPONENT.fullmatch(str(run_id)):
            raise ValueError("invalid run ID")
        return self.root / str(run_id) / "metrics"

    def record(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        stage = str(event.get("stage", ""))
        kind = str(event.get("event", ""))
        if not _SAFE_STAGE.fullmatch(stage):
            raise ValueError("invalid stage")
        if kind not in _EVENTS:
            raise ValueError("invalid event")
        record = {
            "eventId": str(event.get("eventId") or f"{stage}:{kind}:{event.get('timestamp') or _now()}"),
            "stage": stage,
            "event": kind,
            "timestamp": str(event.get("timestamp") or _now()),
            "memory": event.get("memory") or {},
            "details": event.get("details") or {},
        }
        timestamp = _parse_time(record["timestamp"])
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "events.jsonl"
        with self._lock:
            existing = self._load_events(path)
            prior = next((item for item in existing if item["eventId"] == record["eventId"]), None)
            if prior:
                if (prior["stage"], prior["event"], prior["details"]) != (record["stage"], record["event"], record["details"]):
                    raise ValueError("event ID conflict")
                return prior
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record

    @staticmethod
    def _load_events(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def build_public_stats(self, run_id: str) -> dict[str, Any]:
        events = self._load_events(self._directory(run_id) / "events.jsonl")
        events.sort(key=lambda item: (item["timestamp"], item["eventId"]))
        by_stage: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            by_stage.setdefault(event["stage"], []).append(event)

        stages = []
        for stage, values in by_stage.items():
            starts = [item for item in values if item["event"] == "start"]
            finishes = [item for item in values if item["event"] in {"end", "failure"}]
            started = starts[0]["timestamp"] if starts else None
            ended = finishes[-1]["timestamp"] if finishes else None
            duration = None
            if started and ended:
                duration = round((_parse_time(ended) - _parse_time(started)).total_seconds() * 1000)
            stages.append({
                "stage": stage,
                "status": "failed" if any(item["event"] == "failure" for item in values) else (
                    "complete" if any(item["event"] == "end" for item in values) else "in_progress"
                ),
                "startedAt": started,
                "endedAt": ended,
                "durationMs": duration,
                "memory": [{k: v for k, v in item.get("memory", {}).items()
                            if k in {"hostTotalBytes", "hostAvailableBytes", "hostUsedBytes",
                                     "comfySystemRamTotalBytes", "comfySystemRamFreeBytes"}
                            and type(v) is int and v >= 0} for item in values if item.get("memory")],
                "details": [_public_details(stage, item.get("details", {})) for item in values if item.get("details")],
            })

        segments = []
        attempts = []
        for item in events:
            if item["stage"] == "speech.attempt" and item["event"] == "end":
                attempts.append(_public_details(item["stage"], item.get("details", {})))
            if item["stage"] != "speech.segment" or item["event"] != "end":
                continue
            details = _public_details(item["stage"], item.get("details", {}))
            segments.append(details)
        total_attempts = sum(item.get("attempts", 0) for item in segments) if segments else None
        retries = sum(max(0, item.get("attempts", 0) - 1) for item in segments) if segments else None
        used_values = [value for event in events
                       if type(value := event.get("memory", {}).get("hostUsedBytes")) is int and value >= 0]
        pipeline = by_stage.get("pipeline", [])
        started_at = next((item["timestamp"] for item in pipeline if item["event"] == "start"), None)
        ended_at = next((item["timestamp"] for item in reversed(pipeline)
                         if item["event"] in {"end", "failure"}), None)
        return {
            "schemaVersion": PUBLIC_SCHEMA_VERSION,
            "runId": str(run_id),
            "generatedAt": _now(),
            "summary": {
                "startedAt": started_at,
                "endedAt": ended_at,
                "wallClockMs": round((_parse_time(ended_at) - _parse_time(started_at)).total_seconds() * 1000)
                if started_at and ended_at else None,
                "stageCount": len(stages),
                "peakHostUsedBytes": max(used_values) if used_values else None,
            },
            "speech": {
                "segmentCount": len(segments),
                "totalAttempts": total_attempts,
                "retries": retries,
                "degradedSegments": sum(1 for item in segments if item.get("degraded") is True),
                "segments": segments,
                "attempts": attempts,
            },
            "stages": stages,
            "events": [{"stage": item["stage"], "event": item["event"],
                        "timestamp": item["timestamp"],
                        "details": _public_details(item["stage"], item.get("details", {}))}
                       for item in events],
        }

    def write_public_stats(self, run_id: str) -> Path:
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "stats.json"
        temporary = directory / ".stats.json.tmp"
        temporary.write_text(json.dumps(self.build_public_stats(run_id), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return target
