"""Conda and runtime preflight validation for Slurm submission."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from gensie.slurm.models import EvalSpec, SlurmSettings


class CondaPreflightError(RuntimeError):
    """Raised when pre-submission validation fails."""


def validate_runtime_preflight(settings: SlurmSettings, specs: tuple[EvalSpec, ...]) -> None:
    """Validate conda runtime access, environment requirements, and spec compatibility."""

    validate_conda_execution(settings)
    validate_required_env(settings)
    validate_specs(specs)
    validate_participant_paths(settings, specs)
    validate_optional_runtime_paths(settings)


def validate_conda_execution(settings: SlurmSettings) -> None:
    """Verify the conda executable, target env, and required imports."""

    conda_executable = Path(settings.conda_executable).expanduser()
    if not conda_executable.exists():
        raise CondaPreflightError(f"Conda executable does not exist: {conda_executable}.")
    if not conda_executable.is_file():
        raise CondaPreflightError(f"Conda executable path is not a file: {conda_executable}.")
    if not os.access(conda_executable, os.X_OK):
        raise CondaPreflightError(f"Conda executable is not executable: {conda_executable}.")

    target_args = _resolve_conda_target(settings=settings)
    for module_name in ("gensie", "vllm", "fastembed"):
        _validate_import(
            conda_executable=conda_executable,
            target_args=target_args,
            module_name=module_name,
        )


def validate_required_env(settings: SlurmSettings) -> None:
    """Ensure any declared required environment variables are present."""

    missing = [name for name in settings.required_env if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise CondaPreflightError(
            f"Missing required environment variable(s) for Slurm profile: {joined}."
        )


def validate_specs(specs: tuple[EvalSpec, ...]) -> None:
    """Validate that every evaluation spec points to existing runtime inputs."""

    if not specs:
        raise CondaPreflightError("At least one evaluation spec is required.")

    for spec in specs:
        if not spec.source_path.exists():
            raise CondaPreflightError(f"Evaluation spec does not exist: {spec.source_path}.")
        if not spec.data_path.exists():
            raise CondaPreflightError(f"Evaluation data path does not exist: {spec.data_path}.")
        if not spec.data_path.is_dir():
            raise CondaPreflightError(f"Evaluation data path is not a directory: {spec.data_path}.")


def validate_participant_paths(settings: SlurmSettings, specs: tuple[EvalSpec, ...]) -> None:
    """Ensure all participant import paths resolve inside the target environment."""

    conda_executable = Path(settings.conda_executable).expanduser()
    target_args = _resolve_conda_target(settings=settings)
    seen: set[str] = set()
    for spec in specs:
        if spec.participant_path in seen:
            continue
        seen.add(spec.participant_path)
        tokens = [
            str(conda_executable),
            "run",
            "--no-capture-output",
            *target_args,
            "python",
            "-c",
            (
                "import importlib, sys; "
                "module_name, class_name = sys.argv[1].rsplit('.', 1); "
                "module = importlib.import_module(module_name); "
                "getattr(module, class_name)"
            ),
            spec.participant_path,
        ]
        try:
            completed = subprocess.run(
                tokens,
                check=False,
                capture_output=True,
                text=True,
                env=dict(os.environ),
            )
        except FileNotFoundError as exc:
            raise CondaPreflightError(
                f"Unable to execute conda preflight for participant {spec.participant_path!r}."
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise CondaPreflightError(
                f"Participant path preflight failed for {spec.participant_path!r}: {detail}"
            )


def validate_optional_runtime_paths(settings: SlurmSettings) -> None:
    """Validate optional file paths that the launcher will reference directly."""

    if settings.vllm_chat_template is not None:
        path = Path(settings.vllm_chat_template).expanduser()
        if not path.exists():
            raise CondaPreflightError(f"vllm_chat_template does not exist: {path}.")
        if not path.is_file():
            raise CondaPreflightError(f"vllm_chat_template is not a file: {path}.")


def _validate_import(
    *,
    conda_executable: Path,
    target_args: tuple[str, str],
    module_name: str,
) -> None:
    tokens = [
        str(conda_executable),
        "run",
        "--no-capture-output",
        *target_args,
        "python",
        "-c",
        f"import {module_name}",
    ]
    try:
        completed = subprocess.run(
            tokens,
            check=False,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
    except FileNotFoundError as exc:
        raise CondaPreflightError(
            f"Unable to execute conda preflight for module {module_name!r}: {conda_executable}."
        ) from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise CondaPreflightError(f"Conda preflight failed for module {module_name!r}: {detail}")


def _resolve_conda_target(*, settings: SlurmSettings) -> tuple[str, str]:
    if settings.conda_env is not None:
        return ("-n", settings.conda_env)
    if settings.conda_prefix is not None:
        prefix = Path(settings.conda_prefix).expanduser().resolve(strict=False)
        if not prefix.exists():
            raise CondaPreflightError(f"Conda prefix does not exist: {prefix}.")
        return ("-p", str(prefix))
    raise CondaPreflightError(
        "Invalid Slurm profile: exactly one of 'conda_env' or 'conda_prefix' must be set."
    )
