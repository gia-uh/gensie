"""Run-scoped artifact helpers for Slurm orchestration."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

_SLURM_EVENT_SCHEMA_VERSION = "gensie.slurm_event.v1"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def generate_run_id() -> str:
    """Generate a stable, filesystem-safe Slurm run id."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"slurm-{timestamp}-{suffix}"


def create_run_dir(run_id: str, *, root: Path | None = None) -> Path:
    """Create one Slurm run directory under the runtime root."""

    normalized = validate_run_id(run_id)
    target_root = root or (Path(".gensie") / "runs" / "slurm")
    run_dir = (target_root / normalized).resolve(strict=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def validate_run_id(run_id: str) -> str:
    """Validate a user-provided run id."""

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must be a non-empty string.")
    if normalized in {".", ".."}:
        raise ValueError("run_id must not be '.' or '..'.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must not contain path separators.")
    if not _RUN_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "run_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ "
            "(letters, digits, '_' and '-', up to 128 chars)."
        )
    return normalized


def write_submission_manifest(run_dir: Path, payload: Mapping[str, object]) -> Path:
    """Write `submission.json` for one rendered or submitted run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "submission.json"
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_submission_manifest(run_dir: Path) -> dict[str, object]:
    """Read `submission.json` from one run directory."""

    path = run_dir / "submission.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Submission manifest {path} must be a JSON object.")
    return data


def append_event(run_dir: Path, event: Mapping[str, object]) -> None:
    """Append one JSONL event to the run-level Slurm event log."""

    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("schema_version", _SLURM_EVENT_SCHEMA_VERSION)
    payload.setdefault("ts_utc", datetime.now(UTC).replace(microsecond=0).isoformat())

    events_path = run_dir / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
