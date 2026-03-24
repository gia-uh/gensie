"""Slurm scheduler operations (status, watch, cancel)."""

from __future__ import annotations

from collections.abc import Iterator
import subprocess
import time

_TERMINAL_STATES = {
    "CANCELLED",
    "COMPLETED",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


class SchedulerRuntimeError(RuntimeError):
    """Raised for scheduler command failures."""


def get_job_status(job_id: str) -> dict[str, str]:
    """Resolve job status via `squeue` (live) and fallback `sacct` (terminal)."""

    squeue_available = True
    squeue_result: subprocess.CompletedProcess[str] | None = None
    try:
        squeue_result = subprocess.run(
            [
                "squeue",
                "--jobs",
                job_id,
                "--noheader",
                "--format=%i|%T|%M|%l|%R",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        squeue_available = False

    if squeue_result is not None and squeue_result.returncode == 0:
        lines = [
            line.strip() for line in squeue_result.stdout.splitlines() if line.strip()
        ]
        if lines:
            first = lines[0].split("|", 4)
            if len(first) == 5:
                return {
                    "job_id": first[0],
                    "state": first[1],
                    "elapsed": first[2],
                    "time_limit": first[3],
                    "reason": first[4],
                    "source": "squeue",
                }

    sacct_available = True
    sacct_result: subprocess.CompletedProcess[str] | None = None
    try:
        sacct_result = subprocess.run(
            [
                "sacct",
                "--jobs",
                job_id,
                "--parsable2",
                "--noheader",
                "--format=JobIDRaw,State,Elapsed,ExitCode",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        sacct_available = False

    if sacct_result is not None and sacct_result.returncode == 0:
        lines = [
            line.strip() for line in sacct_result.stdout.splitlines() if line.strip()
        ]
        for line in lines:
            fields = line.split("|", 3)
            if len(fields) < 4:
                continue
            if fields[0] != job_id:
                continue
            return {
                "job_id": fields[0],
                "state": fields[1],
                "elapsed": fields[2],
                "exit_code": fields[3],
                "source": "sacct",
            }

    if not squeue_available and not sacct_available:
        raise SchedulerRuntimeError(
            "Slurm status tools are unavailable: both `squeue` and `sacct` are missing."
        )

    squeue_detail = (
        ""
        if squeue_result is None
        else (squeue_result.stderr.strip() or squeue_result.stdout.strip())
    )
    sacct_detail = (
        ""
        if sacct_result is None
        else (sacct_result.stderr.strip() or sacct_result.stdout.strip())
    )

    raise SchedulerRuntimeError(
        "Unable to resolve job status via Slurm commands. "
        f"squeue={squeue_detail!r}, sacct={sacct_detail!r}"
    )


def watch_job_status(job_id: str, interval_s: int) -> Iterator[dict[str, str]]:
    """Yield periodic status snapshots until terminal state."""

    if interval_s <= 0:
        raise SchedulerRuntimeError("Status watch interval must be a positive integer.")

    while True:
        snapshot = get_job_status(job_id)
        yield snapshot

        state = snapshot.get("state", "")
        normalized_state = state.split("+", 1)[0].strip().upper()
        if normalized_state in _TERMINAL_STATES:
            return

        time.sleep(interval_s)


def cancel_job(job_id: str, *, signal: str | None = None) -> None:
    """Cancel one Slurm job id via `scancel`."""

    tokens = ["scancel"]
    if signal is not None:
        signal_name = signal.strip()
        if not signal_name:
            raise SchedulerRuntimeError("Cancel signal must be a non-empty string.")
        tokens.extend(["--signal", signal_name])
    tokens.append(job_id)

    try:
        completed = subprocess.run(
            tokens,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SchedulerRuntimeError("Slurm command `scancel` is unavailable.") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise SchedulerRuntimeError(f"scancel failed: {detail}")
