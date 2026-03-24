"""Typer CLI for Slurm-backed GenSIE evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import glob
import json
from pathlib import Path
import time
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from gensie.slurm.conda import (
    CondaPreflightError,
    validate_conda_execution,
    validate_optional_runtime_paths,
    validate_required_env,
    validate_runtime_preflight,
)
from gensie.slurm.launcher import prepare_rendered_launch
from gensie.slurm.models import EvalSpec, RenderedLaunch, ResolvedSlurmSettings
from gensie.slurm.resolver import SlurmResolutionError, resolve_slurm_settings
from gensie.slurm.run_artifacts import (
    append_event,
    create_run_dir,
    generate_run_id,
    read_submission_manifest,
    validate_run_id,
    write_submission_manifest,
)
from gensie.slurm.sbatch import (
    SbatchRuntimeError,
    build_sbatch_command,
    submit_sbatch,
)
from gensie.slurm.scheduler import (
    SchedulerRuntimeError,
    cancel_job,
    get_job_status,
    watch_job_status,
)
from gensie.slurm.specs import EvalSpecError, load_eval_spec, load_manifest_specs

app = typer.Typer(help="Slurm-backed evaluation workflows for GenSIE.")
profiles_app = typer.Typer(help="List and inspect local Slurm profiles.")
eval_app = typer.Typer(help="Render or submit Slurm-backed evaluation jobs.")
app.add_typer(profiles_app, name="profiles")
app.add_typer(eval_app, name="eval")

console = Console()

_DEFAULT_PROFILE_NAME = "default"
_DEFAULT_PROFILE_DIR = Path(".gensie") / "slurm" / "profiles"
_DEFAULT_RUN_ROOT = Path(".gensie") / "runs" / "slurm"
_SUBMISSION_SCHEMA_VERSION = "gensie.slurm_submission.v1"


@dataclass(frozen=True, slots=True)
class PreparedEvalRun:
    """Resolved inputs for one render or submit command."""

    run_id: str
    run_dir: Path
    profile_dir: Path
    resolved: ResolvedSlurmSettings
    specs: tuple[EvalSpec, ...]
    selection_type: str
    manifest_source_path: Path | None
    original_spec_paths: tuple[Path, ...]
    launch: RenderedLaunch
    sbatch_tokens: list[str]


@profiles_app.command("list")
def list_profiles(
    profile_dir: Path = typer.Option(_DEFAULT_PROFILE_DIR, help="Directory with user Slurm profile TOML files."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """List local Slurm profile files."""

    profile_dir = profile_dir.resolve(strict=False)
    profile_names = sorted(path.stem for path in profile_dir.glob("*.toml")) if profile_dir.is_dir() else []

    payload = {
        "profile_dir": str(profile_dir),
        "profile_count": len(profile_names),
        "profiles": profile_names,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(title=f"Slurm Profiles ({profile_dir})")
    table.add_column("Profile", style="cyan")
    if profile_names:
        for name in profile_names:
            table.add_row(name)
    else:
        table.add_row("(none)")
    console.print(table)


@app.command()
def validate(
    profile: str = typer.Option(_DEFAULT_PROFILE_NAME, help="Profile name to resolve."),
    profile_dir: Path = typer.Option(_DEFAULT_PROFILE_DIR, help="Directory with user Slurm profile TOML files."),
    spec: Path | None = typer.Option(None, help="One evaluation spec YAML to validate."),
    manifest: Path | None = typer.Option(None, help="A manifest YAML to validate."),
    partition: str | None = typer.Option(None, help="Slurm partition override."),
    time_limit: str | None = typer.Option(None, "--time", help="Slurm wall-clock limit override."),
    memory: str | None = typer.Option(None, help="Slurm memory override."),
    gpus: int | None = typer.Option(None, help="GPU count override."),
    cpus_per_task: int | None = typer.Option(None, help="CPU count override."),
    nodes: int | None = typer.Option(None, help="Node count override."),
    ntasks_per_node: int | None = typer.Option(None, help="Task count per node override."),
    conda_executable: str | None = typer.Option(None, help="Shared conda executable path."),
    conda_env: str | None = typer.Option(None, help="Conda environment name."),
    conda_prefix: str | None = typer.Option(None, help="Conda environment prefix."),
    vllm_port: int | None = typer.Option(None, help="vLLM port override."),
    gensie_port: int | None = typer.Option(None, help="GenSIE server port override."),
    vllm_dtype: str | None = typer.Option(None, help="vLLM dtype override."),
    vllm_gpu_memory_utilization: float | None = typer.Option(
        None, help="vLLM GPU memory utilization override."
    ),
    vllm_max_model_len: int | None = typer.Option(None, help="vLLM max model length override."),
    vllm_chat_template: str | None = typer.Option(None, help="vLLM chat template path override."),
    startup_timeout_s: int | None = typer.Option(None, help="Launcher startup timeout override."),
    startup_poll_interval_s: int | None = typer.Option(None, help="Launcher probe interval override."),
    qos: str | None = typer.Option(None, help="Slurm QoS override."),
    account: str | None = typer.Option(None, help="Slurm account override."),
    constraint: str | None = typer.Option(None, help="Slurm constraint override."),
    output: str | None = typer.Option(None, help="Override sbatch stdout path."),
    error: str | None = typer.Option(None, help="Override sbatch stderr path."),
    status_interval_s: int | None = typer.Option(None, help="Status watch interval override."),
    hf_home: str | None = typer.Option(None, help="Override HF_HOME for vLLM jobs."),
    huggingface_hub_cache: str | None = typer.Option(
        None, help="Override HUGGINGFACE_HUB_CACHE for vLLM jobs."
    ),
    required_env: list[str] | None = typer.Option(
        None,
        "--required-env",
        help="Repeat to declare required exported environment variables.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Resolve one Slurm profile and run preflight validation."""

    try:
        selection = _resolve_spec_selection(spec=spec, manifest=manifest, allow_empty=True)
        resolved = _resolve_profile(
            profile=profile,
            profile_dir=profile_dir,
            cli_overrides=_build_cli_overrides(
                partition=partition,
                time_limit=time_limit,
                memory=memory,
                gpus=gpus,
                cpus_per_task=cpus_per_task,
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
                conda_executable=conda_executable,
                conda_env=conda_env,
                conda_prefix=conda_prefix,
                vllm_port=vllm_port,
                gensie_port=gensie_port,
                vllm_dtype=vllm_dtype,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                vllm_max_model_len=vllm_max_model_len,
                vllm_chat_template=vllm_chat_template,
                startup_timeout_s=startup_timeout_s,
                startup_poll_interval_s=startup_poll_interval_s,
                qos=qos,
                account=account,
                constraint=constraint,
                output=output,
                error=error,
                status_interval_s=status_interval_s,
                hf_home=hf_home,
                huggingface_hub_cache=huggingface_hub_cache,
                required_env=required_env,
            ),
        )

        if selection.specs:
            validate_runtime_preflight(resolved.settings, selection.specs)
        else:
            validate_conda_execution(resolved.settings)
            validate_required_env(resolved.settings)
            validate_optional_runtime_paths(resolved.settings)
    except (
        CondaPreflightError,
        EvalSpecError,
        SlurmResolutionError,
        ValueError,
    ) as exc:
        _fail(str(exc))

    payload = {
        "profile_name": resolved.profile_name,
        "profile_dir": str(profile_dir.resolve(strict=False)),
        "selection_type": selection.selection_type,
        "task_count": len(selection.specs),
        "settings": resolved.settings.to_dict(),
        "origins": dict(resolved.origins),
    }
    if selection.manifest_source_path is not None:
        payload["manifest_source_path"] = str(selection.manifest_source_path)
    if selection.original_spec_paths:
        payload["spec_paths"] = [str(path) for path in selection.original_spec_paths]

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(title=f"Validated Slurm Profile: {resolved.profile_name}")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_column("Origin", style="magenta")
    for key, value in resolved.settings.to_dict().items():
        table.add_row(key, str(value), resolved.origins.get(key, "n/a"))
    console.print(table)
    console.print(f"Validated {len(selection.specs)} evaluation spec(s).")


@eval_app.command("render")
def render_eval(
    profile: str = typer.Option(_DEFAULT_PROFILE_NAME, help="Profile name to resolve."),
    profile_dir: Path = typer.Option(_DEFAULT_PROFILE_DIR, help="Directory with user Slurm profile TOML files."),
    spec: Path | None = typer.Option(None, help="One evaluation spec YAML."),
    manifest: Path | None = typer.Option(None, help="A manifest YAML with multiple spec paths."),
    run_id: str | None = typer.Option(None, help="Optional explicit run id."),
    run_root: Path = typer.Option(_DEFAULT_RUN_ROOT, help="Root directory for rendered run artifacts."),
    partition: str | None = typer.Option(None, help="Slurm partition override."),
    time_limit: str | None = typer.Option(None, "--time", help="Slurm wall-clock limit override."),
    memory: str | None = typer.Option(None, help="Slurm memory override."),
    gpus: int | None = typer.Option(None, help="GPU count override."),
    cpus_per_task: int | None = typer.Option(None, help="CPU count override."),
    nodes: int | None = typer.Option(None, help="Node count override."),
    ntasks_per_node: int | None = typer.Option(None, help="Task count per node override."),
    conda_executable: str | None = typer.Option(None, help="Shared conda executable path."),
    conda_env: str | None = typer.Option(None, help="Conda environment name."),
    conda_prefix: str | None = typer.Option(None, help="Conda environment prefix."),
    vllm_port: int | None = typer.Option(None, help="vLLM port override."),
    gensie_port: int | None = typer.Option(None, help="GenSIE server port override."),
    vllm_dtype: str | None = typer.Option(None, help="vLLM dtype override."),
    vllm_gpu_memory_utilization: float | None = typer.Option(
        None, help="vLLM GPU memory utilization override."
    ),
    vllm_max_model_len: int | None = typer.Option(None, help="vLLM max model length override."),
    vllm_chat_template: str | None = typer.Option(None, help="vLLM chat template path override."),
    startup_timeout_s: int | None = typer.Option(None, help="Launcher startup timeout override."),
    startup_poll_interval_s: int | None = typer.Option(None, help="Launcher probe interval override."),
    qos: str | None = typer.Option(None, help="Slurm QoS override."),
    account: str | None = typer.Option(None, help="Slurm account override."),
    constraint: str | None = typer.Option(None, help="Slurm constraint override."),
    output: str | None = typer.Option(None, help="Override sbatch stdout path."),
    error: str | None = typer.Option(None, help="Override sbatch stderr path."),
    status_interval_s: int | None = typer.Option(None, help="Status watch interval override."),
    hf_home: str | None = typer.Option(None, help="Override HF_HOME for vLLM jobs."),
    huggingface_hub_cache: str | None = typer.Option(
        None, help="Override HUGGINGFACE_HUB_CACHE for vLLM jobs."
    ),
    required_env: list[str] | None = typer.Option(
        None,
        "--required-env",
        help="Repeat to declare required exported environment variables.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Render a Slurm launcher and artifact directory without submitting."""

    prepared = _prepare_eval_run(
        profile=profile,
        profile_dir=profile_dir,
        spec=spec,
        manifest=manifest,
        run_id=run_id,
        run_root=run_root,
        submit=False,
        cli_overrides=_build_cli_overrides(
            partition=partition,
            time_limit=time_limit,
            memory=memory,
            gpus=gpus,
            cpus_per_task=cpus_per_task,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            conda_executable=conda_executable,
            conda_env=conda_env,
            conda_prefix=conda_prefix,
            vllm_port=vllm_port,
            gensie_port=gensie_port,
            vllm_dtype=vllm_dtype,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_chat_template=vllm_chat_template,
            startup_timeout_s=startup_timeout_s,
            startup_poll_interval_s=startup_poll_interval_s,
            qos=qos,
            account=account,
            constraint=constraint,
            output=output,
            error=error,
            status_interval_s=status_interval_s,
            hf_home=hf_home,
            huggingface_hub_cache=huggingface_hub_cache,
            required_env=required_env,
        ),
    )

    payload = _submission_payload(prepared=prepared, job_id=None)
    write_submission_manifest(prepared.run_dir, payload)
    append_event(
        prepared.run_dir,
        {
            "event": "rendered",
            "run_id": prepared.run_id,
            "task_count": len(prepared.launch.tasks),
        },
    )

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    console.print(f"Rendered Slurm launch at {prepared.launch.launcher_path}")
    console.print(f"Run directory: {prepared.run_dir}")
    console.print(f"Task count: {len(prepared.launch.tasks)}")


@eval_app.command("submit")
def submit_eval(
    profile: str = typer.Option(_DEFAULT_PROFILE_NAME, help="Profile name to resolve."),
    profile_dir: Path = typer.Option(_DEFAULT_PROFILE_DIR, help="Directory with user Slurm profile TOML files."),
    spec: Path | None = typer.Option(None, help="One evaluation spec YAML."),
    manifest: Path | None = typer.Option(None, help="A manifest YAML with multiple spec paths."),
    run_id: str | None = typer.Option(None, help="Optional explicit run id."),
    run_root: Path = typer.Option(_DEFAULT_RUN_ROOT, help="Root directory for rendered run artifacts."),
    partition: str | None = typer.Option(None, help="Slurm partition override."),
    time_limit: str | None = typer.Option(None, "--time", help="Slurm wall-clock limit override."),
    memory: str | None = typer.Option(None, help="Slurm memory override."),
    gpus: int | None = typer.Option(None, help="GPU count override."),
    cpus_per_task: int | None = typer.Option(None, help="CPU count override."),
    nodes: int | None = typer.Option(None, help="Node count override."),
    ntasks_per_node: int | None = typer.Option(None, help="Task count per node override."),
    conda_executable: str | None = typer.Option(None, help="Shared conda executable path."),
    conda_env: str | None = typer.Option(None, help="Conda environment name."),
    conda_prefix: str | None = typer.Option(None, help="Conda environment prefix."),
    vllm_port: int | None = typer.Option(None, help="vLLM port override."),
    gensie_port: int | None = typer.Option(None, help="GenSIE server port override."),
    vllm_dtype: str | None = typer.Option(None, help="vLLM dtype override."),
    vllm_gpu_memory_utilization: float | None = typer.Option(
        None, help="vLLM GPU memory utilization override."
    ),
    vllm_max_model_len: int | None = typer.Option(None, help="vLLM max model length override."),
    vllm_chat_template: str | None = typer.Option(None, help="vLLM chat template path override."),
    startup_timeout_s: int | None = typer.Option(None, help="Launcher startup timeout override."),
    startup_poll_interval_s: int | None = typer.Option(None, help="Launcher probe interval override."),
    qos: str | None = typer.Option(None, help="Slurm QoS override."),
    account: str | None = typer.Option(None, help="Slurm account override."),
    constraint: str | None = typer.Option(None, help="Slurm constraint override."),
    output: str | None = typer.Option(None, help="Override sbatch stdout path."),
    error: str | None = typer.Option(None, help="Override sbatch stderr path."),
    status_interval_s: int | None = typer.Option(None, help="Status watch interval override."),
    hf_home: str | None = typer.Option(None, help="Override HF_HOME for vLLM jobs."),
    huggingface_hub_cache: str | None = typer.Option(
        None, help="Override HUGGINGFACE_HUB_CACHE for vLLM jobs."
    ),
    required_env: list[str] | None = typer.Option(
        None,
        "--required-env",
        help="Repeat to declare required exported environment variables.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Render and submit a Slurm-backed evaluation run."""

    prepared = _prepare_eval_run(
        profile=profile,
        profile_dir=profile_dir,
        spec=spec,
        manifest=manifest,
        run_id=run_id,
        run_root=run_root,
        submit=True,
        cli_overrides=_build_cli_overrides(
            partition=partition,
            time_limit=time_limit,
            memory=memory,
            gpus=gpus,
            cpus_per_task=cpus_per_task,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            conda_executable=conda_executable,
            conda_env=conda_env,
            conda_prefix=conda_prefix,
            vllm_port=vllm_port,
            gensie_port=gensie_port,
            vllm_dtype=vllm_dtype,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_chat_template=vllm_chat_template,
            startup_timeout_s=startup_timeout_s,
            startup_poll_interval_s=startup_poll_interval_s,
            qos=qos,
            account=account,
            constraint=constraint,
            output=output,
            error=error,
            status_interval_s=status_interval_s,
            hf_home=hf_home,
            huggingface_hub_cache=huggingface_hub_cache,
            required_env=required_env,
        ),
    )

    payload = _submission_payload(prepared=prepared, job_id=None)
    write_submission_manifest(prepared.run_dir, payload)
    append_event(
        prepared.run_dir,
        {
            "event": "submission_started",
            "run_id": prepared.run_id,
            "task_count": len(prepared.launch.tasks),
        },
    )

    try:
        job_id = submit_sbatch(prepared.sbatch_tokens)
    except SbatchRuntimeError as exc:
        _fail(str(exc))

    payload = _submission_payload(prepared=prepared, job_id=job_id)
    write_submission_manifest(prepared.run_dir, payload)
    append_event(
        prepared.run_dir,
        {
            "event": "submitted",
            "run_id": prepared.run_id,
            "job_id": job_id,
            "task_count": len(prepared.launch.tasks),
        },
    )

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    console.print(f"Submitted Slurm job {job_id} for run {prepared.run_id}")
    console.print(f"Run directory: {prepared.run_dir}")


@app.command()
def status(
    job_id: str | None = typer.Option(None, help="Slurm job id to inspect."),
    run_id: str | None = typer.Option(None, help="Existing GenSIE Slurm run id."),
    run_root: Path = typer.Option(_DEFAULT_RUN_ROOT, help="Root directory for run artifacts."),
    watch: bool = typer.Option(False, "--watch", help="Poll until the job reaches a terminal state."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Inspect live or terminal Slurm status for one GenSIE run."""

    run_dir: Path | None = None
    interval_s = 30
    if run_id is not None:
        try:
            run_dir, manifest = _load_existing_submission(run_id=run_id, run_root=run_root)
            if job_id is None:
                job_id = _required_manifest_string(manifest, "job_id")
            interval_s = _manifest_status_interval(manifest)
        except ValueError as exc:
            _fail(str(exc))

    if job_id is None:
        _fail("Either --job-id or --run-id is required.")

    try:
        if watch:
            snapshots = list(watch_job_status(job_id, interval_s=interval_s))
            payload: dict[str, Any] = {"job_id": job_id, "watch": True, "snapshots": snapshots}
        else:
            payload = get_job_status(job_id)
    except SchedulerRuntimeError as exc:
        _fail(str(exc))

    if run_dir is not None:
        append_event(
            run_dir,
            {
                "event": "status_checked",
                "run_id": validate_run_id(run_id),
                "job_id": job_id,
                "watch": watch,
            },
        )

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if watch:
        table = Table(title=f"Slurm Status Watch: {job_id}")
        table.add_column("State", style="cyan")
        table.add_column("Elapsed")
        table.add_column("Source")
        for snapshot in payload["snapshots"]:
            table.add_row(
                snapshot.get("state", ""),
                snapshot.get("elapsed", ""),
                snapshot.get("source", ""),
            )
        console.print(table)
    else:
        table = Table(title=f"Slurm Status: {job_id}")
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for key, value in payload.items():
            table.add_row(key, str(value))
        console.print(table)


@app.command()
def cancel(
    job_id: str | None = typer.Option(None, help="Slurm job id to cancel."),
    run_id: str | None = typer.Option(None, help="Existing GenSIE Slurm run id."),
    run_root: Path = typer.Option(_DEFAULT_RUN_ROOT, help="Root directory for run artifacts."),
):
    """Cancel one Slurm-backed GenSIE run."""

    run_dir: Path | None = None
    if run_id is not None:
        try:
            run_dir, manifest = _load_existing_submission(run_id=run_id, run_root=run_root)
            if job_id is None:
                job_id = _required_manifest_string(manifest, "job_id")
        except ValueError as exc:
            _fail(str(exc))

    if job_id is None:
        _fail("Either --job-id or --run-id is required.")

    try:
        cancel_job(job_id)
    except SchedulerRuntimeError as exc:
        _fail(str(exc))

    if run_dir is not None:
        append_event(
            run_dir,
            {
                "event": "cancelled",
                "run_id": validate_run_id(run_id),
                "job_id": job_id,
            },
        )

    console.print(f"Cancelled Slurm job {job_id}")


@app.command()
def logs(
    run_id: str = typer.Option(..., help="Existing GenSIE Slurm run id."),
    run_root: Path = typer.Option(_DEFAULT_RUN_ROOT, help="Root directory for run artifacts."),
    stream: str = typer.Option("both", help="Which stream to show: out, err, or both."),
    follow: bool = typer.Option(False, "--follow", help="Stream appended log output."),
):
    """Show stdout/stderr for one rendered or submitted GenSIE Slurm run."""

    normalized_stream = stream.strip().lower()
    if normalized_stream not in {"out", "err", "both"}:
        _fail("`--stream` must be one of: out, err, both.")

    try:
        run_dir, manifest = _load_existing_submission(run_id=run_id, run_root=run_root)
    except ValueError as exc:
        _fail(str(exc))
    job_id = manifest.get("job_id")

    patterns: list[tuple[str, str]] = []
    if normalized_stream in {"out", "both"}:
        patterns.append(("stdout", _required_manifest_string(manifest, "stdout_path")))
    if normalized_stream in {"err", "both"}:
        patterns.append(("stderr", _required_manifest_string(manifest, "stderr_path")))

    if follow:
        _follow_logs(run_dir=run_dir, job_id=str(job_id or ""), patterns=patterns)
        return

    matched_any = False
    for label, pattern in patterns:
        matches = _resolve_log_paths(run_dir=run_dir, job_id=str(job_id or ""), pattern=pattern)
        if not matches:
            continue
        matched_any = True
        for path in matches:
            console.print(f"== {label}: {path} ==")
            typer.echo(path.read_text(encoding="utf-8"))
    if not matched_any:
        _fail(f"No log files matched the rendered Slurm paths for run {run_id}.")


def _prepare_eval_run(
    *,
    profile: str,
    profile_dir: Path,
    spec: Path | None,
    manifest: Path | None,
    run_id: str | None,
    run_root: Path,
    submit: bool,
    cli_overrides: dict[str, object],
) -> PreparedEvalRun:
    try:
        selection = _resolve_spec_selection(spec=spec, manifest=manifest)
        resolved = _resolve_profile(
            profile=profile,
            profile_dir=profile_dir,
            cli_overrides=cli_overrides,
        )
        normalized_run_id = validate_run_id(run_id) if run_id is not None else generate_run_id()
        run_dir = create_run_dir(normalized_run_id, root=run_root)

        if submit:
            validate_runtime_preflight(resolved.settings, selection.specs)

        launch = prepare_rendered_launch(
            run_id=normalized_run_id,
            run_dir=run_dir,
            resolved=resolved,
            specs=selection.specs,
            normalized_manifest_path=selection.manifest_source_path,
        )
        sbatch_tokens = build_sbatch_command(
            resolved.settings,
            script_path=launch.launcher_path,
            working_directory=run_dir,
            output_path=launch.sbatch_output_path,
            error_path=launch.sbatch_error_path,
            job_name=f"gensie-{normalized_run_id}",
            array_spec=_array_spec(launch),
        )
        return PreparedEvalRun(
            run_id=normalized_run_id,
            run_dir=run_dir,
            profile_dir=profile_dir.resolve(strict=False),
            resolved=resolved,
            specs=selection.specs,
            selection_type=selection.selection_type,
            manifest_source_path=selection.manifest_source_path,
            original_spec_paths=selection.original_spec_paths,
            launch=launch,
            sbatch_tokens=sbatch_tokens,
        )
    except (
        CondaPreflightError,
        EvalSpecError,
        SlurmResolutionError,
        SbatchRuntimeError,
        ValueError,
    ) as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class _SpecSelection:
    selection_type: str
    specs: tuple[EvalSpec, ...]
    manifest_source_path: Path | None
    original_spec_paths: tuple[Path, ...]


def _resolve_spec_selection(
    *,
    spec: Path | None,
    manifest: Path | None,
    allow_empty: bool = False,
) -> _SpecSelection:
    if spec is None and manifest is None:
        if allow_empty:
            return _SpecSelection(
                selection_type="profile_only",
                specs=(),
                manifest_source_path=None,
                original_spec_paths=(),
            )
        raise ValueError("Exactly one of --spec or --manifest must be provided.")
    if spec is not None and manifest is not None:
        raise ValueError("Exactly one of --spec or --manifest must be provided.")

    if spec is not None:
        loaded = load_eval_spec(spec)
        return _SpecSelection(
            selection_type="spec",
            specs=(loaded,),
            manifest_source_path=None,
            original_spec_paths=(loaded.source_path,),
        )

    assert manifest is not None
    normalized_manifest = manifest.resolve()
    spec_paths = load_manifest_specs(normalized_manifest)
    specs = tuple(load_eval_spec(path) for path in spec_paths)
    return _SpecSelection(
        selection_type="manifest",
        specs=specs,
        manifest_source_path=normalized_manifest,
        original_spec_paths=spec_paths,
    )


def _resolve_profile(
    *,
    profile: str,
    profile_dir: Path,
    cli_overrides: dict[str, object],
) -> ResolvedSlurmSettings:
    try:
        return resolve_slurm_settings(
            profile_name=profile,
            profile_dir=profile_dir.resolve(strict=False),
            cli_overrides=cli_overrides,
        )
    except SlurmResolutionError as exc:
        # Surface a clearer message when the user has not created any local profile yet.
        normalized_dir = profile_dir.resolve(strict=False)
        if not normalized_dir.is_dir() or not any(normalized_dir.glob("*.toml")):
            raise SlurmResolutionError(
                f"No Slurm profiles found in {normalized_dir}. "
                "Create a profile such as `.gensie/slurm/profiles/default.toml` first."
            ) from exc
        raise


def _build_cli_overrides(
    *,
    partition: str | None,
    time_limit: str | None,
    memory: str | None,
    gpus: int | None,
    cpus_per_task: int | None,
    nodes: int | None,
    ntasks_per_node: int | None,
    conda_executable: str | None,
    conda_env: str | None,
    conda_prefix: str | None,
    vllm_port: int | None,
    gensie_port: int | None,
    vllm_dtype: str | None,
    vllm_gpu_memory_utilization: float | None,
    vllm_max_model_len: int | None,
    vllm_chat_template: str | None,
    startup_timeout_s: int | None,
    startup_poll_interval_s: int | None,
    qos: str | None,
    account: str | None,
    constraint: str | None,
    output: str | None,
    error: str | None,
    status_interval_s: int | None,
    hf_home: str | None,
    huggingface_hub_cache: str | None,
    required_env: list[str] | None,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    values = {
        "partition": partition,
        "time": time_limit,
        "memory": memory,
        "gpus": gpus,
        "cpus_per_task": cpus_per_task,
        "nodes": nodes,
        "ntasks_per_node": ntasks_per_node,
        "conda_executable": conda_executable,
        "conda_env": conda_env,
        "conda_prefix": conda_prefix,
        "vllm_port": vllm_port,
        "gensie_port": gensie_port,
        "vllm_dtype": vllm_dtype,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "vllm_max_model_len": vllm_max_model_len,
        "vllm_chat_template": vllm_chat_template,
        "startup_timeout_s": startup_timeout_s,
        "startup_poll_interval_s": startup_poll_interval_s,
        "qos": qos,
        "account": account,
        "constraint": constraint,
        "output": output,
        "error": error,
        "status_interval_s": status_interval_s,
        "hf_home": hf_home,
        "huggingface_hub_cache": huggingface_hub_cache,
    }
    for key, value in values.items():
        if value is not None:
            overrides[key] = value

    if required_env:
        overrides["required_env"] = required_env

    return overrides


def _submission_payload(*, prepared: PreparedEvalRun, job_id: str | None) -> dict[str, object]:
    return {
        "format": _SUBMISSION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_id": prepared.run_id,
        "run_dir": str(prepared.run_dir),
        "profile_name": prepared.resolved.profile_name,
        "profile_dir": str(prepared.profile_dir),
        "settings": prepared.resolved.settings.to_dict(),
        "origins": dict(prepared.resolved.origins),
        "selection_type": prepared.selection_type,
        "manifest_source_path": (
            str(prepared.manifest_source_path) if prepared.manifest_source_path is not None else None
        ),
        "normalized_manifest_path": (
            str(prepared.launch.normalized_manifest_path)
            if prepared.launch.normalized_manifest_path is not None
            else None
        ),
        "task_count": len(prepared.launch.tasks),
        "task_specs": [task.to_dict() for task in prepared.launch.tasks],
        "original_spec_paths": [str(path) for path in prepared.original_spec_paths],
        "resolved_spec_paths": [str(task.resolved_spec_path) for task in prepared.launch.tasks],
        "report_paths": [str(task.report_path) for task in prepared.launch.tasks],
        "launcher_path": str(prepared.launch.launcher_path),
        "sbatch_tokens": prepared.sbatch_tokens,
        "stdout_path": prepared.launch.sbatch_output_path,
        "stderr_path": prepared.launch.sbatch_error_path,
        "cwd": str(prepared.run_dir),
        "job_id": job_id,
    }


def _array_spec(launch: RenderedLaunch) -> str | None:
    if not launch.is_array:
        return None
    return f"0-{len(launch.tasks) - 1}"


def _required_manifest_string(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Submission manifest is missing `{key}`.")
    return value


def _manifest_status_interval(manifest: dict[str, object]) -> int:
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        return 30
    value = settings.get("status_interval_s")
    if isinstance(value, int) and value > 0:
        return value
    return 30


def _load_existing_submission(*, run_id: str, run_root: Path) -> tuple[Path, dict[str, object]]:
    normalized_run_id = validate_run_id(run_id)
    run_dir = (run_root / normalized_run_id).resolve(strict=False)
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist for run_id={normalized_run_id!r}: {run_dir}")
    try:
        manifest = read_submission_manifest(run_dir)
    except FileNotFoundError as exc:
        raise ValueError(f"Submission manifest does not exist for run {normalized_run_id!r}.") from exc
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return run_dir, manifest


def _resolve_log_paths(*, run_dir: Path, job_id: str, pattern: str) -> list[Path]:
    if not pattern:
        return []

    expanded = pattern
    if "%A" in expanded and job_id:
        expanded = expanded.replace("%A", job_id)
    glob_pattern = expanded.replace("%a", "*").replace("%A", "*")

    candidate = Path(glob_pattern)
    if not candidate.is_absolute():
        candidate = (run_dir / candidate)

    matches = [Path(path) for path in sorted(glob.glob(str(candidate)))]
    if matches:
        return matches

    concrete = Path(expanded)
    if not concrete.is_absolute():
        concrete = run_dir / concrete
    if concrete.exists():
        return [concrete]
    return []


def _follow_logs(*, run_dir: Path, job_id: str, patterns: list[tuple[str, str]]) -> None:
    offsets: dict[Path, int] = {}
    seen_headers: set[Path] = set()

    try:
        while True:
            had_output = False
            for label, pattern in patterns:
                for path in _resolve_log_paths(run_dir=run_dir, job_id=job_id, pattern=pattern):
                    if path not in seen_headers:
                        console.print(f"== {label}: {path} ==")
                        seen_headers.add(path)
                    offset = offsets.get(path, 0)
                    if not path.exists():
                        continue
                    with path.open("r", encoding="utf-8") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                        offsets[path] = handle.tell()
                    if chunk:
                        had_output = True
                        typer.echo(chunk, nl=False)
            if not had_output:
                time.sleep(1.0)
    except KeyboardInterrupt:
        raise typer.Exit(0)


def _fail(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(1)
