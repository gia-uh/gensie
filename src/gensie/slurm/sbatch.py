"""`sbatch` command rendering and submission helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from gensie.slurm.models import SlurmSettings


class SbatchRuntimeError(RuntimeError):
    """Raised when building or submitting `sbatch` commands fails."""


def build_sbatch_command(
    settings: SlurmSettings,
    *,
    script_path: Path,
    working_directory: Path,
    output_path: str,
    error_path: str,
    job_name: str,
    array_spec: str | None = None,
) -> list[str]:
    """Build deterministic `sbatch` tokens for one launcher script."""

    if not script_path.exists():
        raise SbatchRuntimeError(f"Launcher script does not exist: {script_path}.")

    tokens = [
        "sbatch",
        "--parsable",
        "--chdir",
        str(working_directory),
        "--job-name",
        job_name,
        "--partition",
        settings.partition,
        "--time",
        settings.time,
        "--mem",
        settings.memory,
        "--gpus",
        str(settings.gpus),
        "--cpus-per-task",
        str(settings.cpus_per_task),
        "--nodes",
        str(settings.nodes),
        "--ntasks-per-node",
        str(settings.ntasks_per_node),
        "--output",
        output_path,
        "--error",
        error_path,
    ]
    if settings.qos is not None:
        tokens.extend(["--qos", settings.qos])
    if settings.account is not None:
        tokens.extend(["--account", settings.account])
    if settings.constraint is not None:
        tokens.extend(["--constraint", settings.constraint])
    if array_spec is not None:
        tokens.extend(["--array", array_spec])

    tokens.append(str(script_path))
    return tokens


def submit_sbatch(tokens: list[str]) -> str:
    """Submit one `sbatch` command and return the parsed job id."""

    try:
        completed = subprocess.run(tokens, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SbatchRuntimeError("Unable to execute `sbatch`; is Slurm installed?") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise SbatchRuntimeError(f"`sbatch` failed: {detail}")

    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise SbatchRuntimeError("`sbatch --parsable` returned an empty job id.")
    return job_id
