from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import threading
import uuid
from typing import Any, Literal

GenerationJobKind = Literal["single", "cycle"]
GenerationJobStatus = Literal["queued", "running", "succeeded", "failed"]

_JOB_TTL = timedelta(hours=24)
_MAX_JOBS = 120
_GENERATION_JOB_STORE_FILE = Path(__file__).resolve().parents[2] / ".generation_jobs_store.json"

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any, *, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return fallback or _utc_now()


def _normalize_kind(value: Any) -> GenerationJobKind:
    if value == "cycle":
        return "cycle"
    return "single"


def _normalize_status(value: Any) -> GenerationJobStatus:
    if value == "running":
        return "running"
    if value == "succeeded":
        return "succeeded"
    if value == "failed":
        return "failed"
    return "queued"


@dataclass
class _GenerationJobState:
    job_id: str
    kind: GenerationJobKind
    owner_user_id: str
    status: GenerationJobStatus = "queued"
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utc_now)
    progress_percent: float | None = 0.0
    stage: str | None = None
    message: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 1
    latest_generation: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None


class GenerationJobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _GenerationJobState] = {}
        with self._lock:
            self._load_from_disk_locked(replace=True)
            self._prune_locked()
            self._persist_locked()

    def create_job(self, *, kind: GenerationJobKind, owner_user_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            job_id = uuid.uuid4().hex
            state = _GenerationJobState(job_id=job_id, kind=kind, owner_user_id=owner_user_id)
            self._jobs[job_id] = state
            self._persist_locked()
            return self._snapshot_locked(job_id, since_event_id=0)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                self._load_from_disk_locked(replace=False)
                state = self._jobs.get(job_id)
                if state is None:
                    return
            now = _utc_now()
            state.status = "running"
            state.started_at = now
            state.updated_at = now
            self._persist_locked()

    def append_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                self._load_from_disk_locked(replace=False)
                state = self._jobs.get(job_id)
                if state is None:
                    return
            now = _utc_now()
            progress_percent = event.get("progress_percent")
            event_payload: dict[str, Any] = {
                "id": state.next_event_id,
                "at": now.isoformat(),
                "stage": str(event.get("stage") or "search"),
                "level": str(event.get("level") or "info"),
                "message": str(event.get("message") or "")[:500],
                "progress_percent": (
                    float(progress_percent)
                    if isinstance(progress_percent, (float, int))
                    else None
                ),
                "metrics": deepcopy(event.get("metrics") or {}),
            }
            latest_generation = event.get("latest_generation")
            if isinstance(latest_generation, dict):
                event_payload["latest_generation"] = deepcopy(latest_generation)
                state.latest_generation = deepcopy(latest_generation)
            else:
                event_payload["latest_generation"] = None

            state.next_event_id += 1
            state.events.append(event_payload)
            state.stage = event_payload["stage"]
            state.message = event_payload["message"]
            if event_payload["progress_percent"] is not None:
                state.progress_percent = event_payload["progress_percent"]
            state.updated_at = now

            if len(state.events) > 1000:
                state.events = state.events[-1000:]
                state.next_event_id = max((int(item.get("id", 0)) for item in state.events), default=0) + 1

            self._persist_locked()

    def mark_succeeded(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                self._load_from_disk_locked(replace=False)
                state = self._jobs.get(job_id)
                if state is None:
                    return
            now = _utc_now()
            state.status = "succeeded"
            state.result = deepcopy(result)
            state.progress_percent = 100.0
            state.finished_at = now
            state.updated_at = now
            self._persist_locked()

    def mark_failed(self, job_id: str, error_message: str) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                self._load_from_disk_locked(replace=False)
                state = self._jobs.get(job_id)
                if state is None:
                    return
            now = _utc_now()
            state.status = "failed"
            state.error_message = error_message[:1000]
            state.finished_at = now
            state.updated_at = now
            self._persist_locked()

    def snapshot(self, job_id: str, *, since_event_id: int = 0) -> dict[str, Any] | None:
        with self._lock:
            snapshot = self._snapshot_locked(job_id, since_event_id=since_event_id)
            if snapshot is not None:
                return snapshot
            # Cross-process fallback: load persisted state and retry.
            self._load_from_disk_locked(replace=False)
            return self._snapshot_locked(job_id, since_event_id=since_event_id)

    def _snapshot_locked(self, job_id: str, *, since_event_id: int = 0) -> dict[str, Any] | None:
        state = self._jobs.get(job_id)
        if state is None:
            return None
        events = [
            deepcopy(item)
            for item in state.events
            if int(item.get("id", 0)) > since_event_id
        ]
        return {
            "job_id": state.job_id,
            "kind": state.kind,
            "owner_user_id": state.owner_user_id,
            "status": state.status,
            "created_at": state.created_at,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "updated_at": state.updated_at,
            "progress_percent": state.progress_percent,
            "stage": state.stage,
            "message": state.message,
            "events": events,
            "last_event_id": max((item.get("id", 0) for item in state.events), default=0),
            "latest_generation": deepcopy(state.latest_generation),
            "result": deepcopy(state.result),
            "error_message": state.error_message,
        }

    def _prune_locked(self) -> None:
        now = _utc_now()
        expired_ids = [
            job_id
            for job_id, item in self._jobs.items()
            if item.finished_at is not None and (now - item.finished_at) > _JOB_TTL
        ]
        for job_id in expired_ids:
            self._jobs.pop(job_id, None)

        if len(self._jobs) <= _MAX_JOBS:
            return

        ordered = sorted(
            self._jobs.values(),
            key=lambda item: (item.updated_at, item.created_at),
        )
        for item in ordered:
            if len(self._jobs) <= _MAX_JOBS:
                break
            if item.status == "running":
                continue
            self._jobs.pop(item.job_id, None)

    @staticmethod
    def _serialize_state(state: _GenerationJobState) -> dict[str, Any]:
        return {
            "job_id": state.job_id,
            "kind": state.kind,
            "owner_user_id": state.owner_user_id,
            "status": state.status,
            "created_at": state.created_at.isoformat(),
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "finished_at": state.finished_at.isoformat() if state.finished_at else None,
            "updated_at": state.updated_at.isoformat(),
            "progress_percent": state.progress_percent,
            "stage": state.stage,
            "message": state.message,
            "events": deepcopy(state.events),
            "next_event_id": state.next_event_id,
            "latest_generation": deepcopy(state.latest_generation),
            "result": deepcopy(state.result),
            "error_message": state.error_message,
        }

    @staticmethod
    def _deserialize_state(raw: dict[str, Any]) -> _GenerationJobState | None:
        try:
            job_id = str(raw.get("job_id") or "").strip()
            if not job_id:
                return None
            created_at = _parse_datetime(raw.get("created_at"))
            updated_at = _parse_datetime(raw.get("updated_at"), fallback=created_at)
            started_at_raw = raw.get("started_at")
            finished_at_raw = raw.get("finished_at")
            started_at = _parse_datetime(started_at_raw) if started_at_raw else None
            finished_at = _parse_datetime(finished_at_raw) if finished_at_raw else None

            events_raw = raw.get("events")
            events: list[dict[str, Any]] = []
            if isinstance(events_raw, list):
                for item in events_raw[-1000:]:
                    if isinstance(item, dict):
                        events.append(deepcopy(item))

            max_event_id = max((int(item.get("id", 0)) for item in events), default=0)
            next_event_id = int(raw.get("next_event_id") or 0)
            if next_event_id <= max_event_id:
                next_event_id = max_event_id + 1
            if next_event_id <= 0:
                next_event_id = 1

            return _GenerationJobState(
                job_id=job_id,
                kind=_normalize_kind(raw.get("kind")),
                owner_user_id=str(raw.get("owner_user_id") or ""),
                status=_normalize_status(raw.get("status")),
                created_at=created_at,
                started_at=started_at,
                finished_at=finished_at,
                updated_at=updated_at,
                progress_percent=(
                    float(raw.get("progress_percent"))
                    if isinstance(raw.get("progress_percent"), (int, float))
                    else None
                ),
                stage=str(raw.get("stage")) if raw.get("stage") is not None else None,
                message=str(raw.get("message")) if raw.get("message") is not None else None,
                events=events,
                next_event_id=next_event_id,
                latest_generation=deepcopy(raw.get("latest_generation")) if isinstance(raw.get("latest_generation"), dict) else None,
                result=deepcopy(raw.get("result")) if isinstance(raw.get("result"), dict) else None,
                error_message=str(raw.get("error_message")) if raw.get("error_message") is not None else None,
            )
        except Exception:
            return None

    def _load_from_disk_locked(self, *, replace: bool) -> None:
        if not _GENERATION_JOB_STORE_FILE.exists():
            if replace:
                self._jobs = {}
            return
        try:
            raw = json.loads(_GENERATION_JOB_STORE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read generation job store file")
            return

        if isinstance(raw, dict) and isinstance(raw.get("jobs"), dict):
            raw_jobs = raw["jobs"]
        elif isinstance(raw, dict):
            raw_jobs = raw
        else:
            raw_jobs = {}

        loaded: dict[str, _GenerationJobState] = {}
        for _, payload in raw_jobs.items():
            if not isinstance(payload, dict):
                continue
            item = self._deserialize_state(payload)
            if item is None:
                continue
            loaded[item.job_id] = item

        if replace:
            self._jobs = loaded
            return

        for job_id, disk_state in loaded.items():
            mem_state = self._jobs.get(job_id)
            if mem_state is None or disk_state.updated_at > mem_state.updated_at:
                self._jobs[job_id] = disk_state

    def _persist_locked(self) -> None:
        try:
            payload = {
                "jobs": {
                    job_id: self._serialize_state(state)
                    for job_id, state in self._jobs.items()
                }
            }
            _GENERATION_JOB_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = _GENERATION_JOB_STORE_FILE.with_name(f"{_GENERATION_JOB_STORE_FILE.name}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(_GENERATION_JOB_STORE_FILE)
        except Exception:
            logger.exception("Failed to persist generation job store")


generation_job_store = GenerationJobStore()
