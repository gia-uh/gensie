"""Launcher rendering for Slurm-backed GenSIE evaluations."""

from __future__ import annotations

import hashlib
import secrets
import shlex
from pathlib import Path

from gensie.slurm.models import EvalSpec, RenderedLaunch, RenderedTask, ResolvedSlurmSettings
from gensie.slurm.specs import dump_resolved_eval_spec

_DEFAULT_VLLM_PORT_START = 20000
_DEFAULT_GENSIE_PORT_START = 30000
_DEFAULT_PORT_SPAN = 10000


def prepare_rendered_launch(
    *,
    run_id: str,
    run_dir: Path,
    resolved: ResolvedSlurmSettings,
    specs: tuple[EvalSpec, ...],
    normalized_manifest_path: Path | None,
) -> RenderedLaunch:
    """Render launch artifacts for one Slurm-backed GenSIE evaluation run."""

    if not specs:
        raise ValueError("At least one evaluation spec is required to render a launcher.")

    settings = resolved.settings
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[RenderedTask] = []
    width = max(3, len(str(len(specs) - 1)))
    for index, spec in enumerate(specs):
        task_dir = run_dir if len(specs) == 1 else run_dir / "tasks" / f"task_{index:0{width}d}"
        resolved_spec_path = task_dir / "resolved_spec.yaml"
        report_path = task_dir / "report.json"

        vllm_port, vllm_origin = _resolve_task_port(
            origin=resolved.origins.get("vllm_port", "unknown"),
            configured_value=settings.vllm_port,
            run_id=run_id,
            task_index=index,
            prefix="vllm",
            start=_DEFAULT_VLLM_PORT_START,
        )
        gensie_port, gensie_origin = _resolve_task_port(
            origin=resolved.origins.get("gensie_port", "unknown"),
            configured_value=settings.gensie_port,
            run_id=run_id,
            task_index=index,
            prefix="gensie",
            start=_DEFAULT_GENSIE_PORT_START,
        )

        dump_resolved_eval_spec(
            spec=spec,
            output_path=resolved_spec_path,
            report_path=report_path,
            vllm_port=vllm_port,
            gensie_port=gensie_port,
            vllm_port_origin=vllm_origin,
            gensie_port_origin=gensie_origin,
        )

        tasks.append(
            RenderedTask(
                index=index,
                spec=spec,
                resolved_spec_path=resolved_spec_path,
                report_path=report_path,
                api_key=f"sk-gensie-{secrets.token_urlsafe(24)}",
                vllm_port=vllm_port,
                gensie_port=gensie_port,
                vllm_port_origin=vllm_origin,
                gensie_port_origin=gensie_origin,
            )
        )

    default_output = logs_dir / ("job.out" if len(tasks) == 1 else "task-%a.out")
    default_error = logs_dir / ("job.err" if len(tasks) == 1 else "task-%a.err")
    sbatch_output_path = _normalize_log_path(run_dir=run_dir, raw_path=settings.output, default_path=default_output)
    sbatch_error_path = _normalize_log_path(run_dir=run_dir, raw_path=settings.error, default_path=default_error)

    launcher_path = run_dir / "launch.sh"
    launcher_path.write_text(
        _render_launcher_script(
            run_id=run_id,
            run_dir=run_dir,
            settings=resolved.settings,
            tasks=tuple(tasks),
        ),
        encoding="utf-8",
    )
    launcher_path.chmod(0o755)

    return RenderedLaunch(
        run_dir=run_dir,
        launcher_path=launcher_path,
        normalized_manifest_path=normalized_manifest_path,
        tasks=tuple(tasks),
        sbatch_output_path=sbatch_output_path,
        sbatch_error_path=sbatch_error_path,
    )


def _resolve_task_port(
    *,
    origin: str,
    configured_value: int,
    run_id: str,
    task_index: int,
    prefix: str,
    start: int,
) -> tuple[int, str]:
    if origin != "base":
        return configured_value, origin

    seed = f"{run_id}:{task_index}:{prefix}"
    digest = hashlib.blake2s(seed.encode("utf-8"), digest_size=4).digest()
    offset = int.from_bytes(digest, "big") % _DEFAULT_PORT_SPAN
    return start + offset, "derived:base"


def _normalize_log_path(*, run_dir: Path, raw_path: str | None, default_path: Path) -> str:
    if raw_path is None:
        chosen = default_path.resolve(strict=False)
    else:
        candidate = Path(raw_path)
        chosen = candidate if candidate.is_absolute() else (run_dir / candidate)

    chosen.parent.mkdir(parents=True, exist_ok=True)
    return str(chosen if raw_path is None else raw_path)


def _render_launcher_script(
    *,
    run_id: str,
    run_dir: Path,
    settings,
    tasks: tuple[RenderedTask, ...],
) -> str:
    is_array = len(tasks) > 1
    conda_flag, conda_value = settings.conda_target_args()

    required_env_checks = ""
    if settings.required_env:
        lines = []
        for env_name in settings.required_env:
            lines.append(
                f"""if [[ -z "${{{env_name}:-}}" ]]; then
  echo "Missing required environment variable: {env_name}" >&2
  exit 1
fi"""
            )
        required_env_checks = "\n".join(lines)

    hf_exports: list[str] = []
    if settings.hf_home is not None:
        hf_exports.append(f"export HF_HOME={_shell(settings.hf_home)}")
    if settings.huggingface_hub_cache is not None:
        hf_exports.append(
            f"export HUGGINGFACE_HUB_CACHE={_shell(settings.huggingface_hub_cache)}"
        )
    hf_exports_block = "".join(line + "\n" for line in hf_exports)

    task_binding = (
        _render_array_task_binding(tasks) if is_array else _render_single_task_binding(tasks[0])
    )

    max_model_len_line = ""
    if settings.vllm_max_model_len is not None:
        max_model_len_line = f'VLLM_ARGS+=(--max-model-len "{settings.vllm_max_model_len}")'

    chat_template_line = ""
    if settings.vllm_chat_template is not None:
        chat_template_line = f'VLLM_ARGS+=(--chat-template "{settings.vllm_chat_template}")'

    tensor_parallel_line = ""
    if settings.gpus > 1:
        tensor_parallel_line = f'VLLM_ARGS+=(--tensor-parallel-size "{settings.gpus}")'

    event_code = _shell(
        """
import os
import sys
from pathlib import Path

from gensie.slurm.run_artifacts import append_event

task_index = os.environ.get("TASK_INDEX", "")
payload = {
    "event": os.environ.get("EVENT_NAME", ""),
    "message": os.environ.get("EVENT_MESSAGE", ""),
    "run_id": os.environ.get("RUN_ID", ""),
    "spec_name": os.environ.get("SPEC_NAME", ""),
    "task_index": int(task_index) if task_index else None,
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
}
append_event(Path(sys.argv[1]), {k: v for k, v in payload.items() if v not in ("", None)})
""".strip()
    )
    vllm_probe_code = _shell(
        """
import json
import sys
import urllib.error
import urllib.request

base_url, api_key, model = sys.argv[1:4]
payload = json.dumps(
    {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
).encode("utf-8")
request = urllib.request.Request(
    base_url.rstrip("/") + "/chat/completions",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise SystemExit(1)
        response.read()
except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
    raise SystemExit(1)
""".strip()
    )
    gensie_probe_code = _shell(
        """
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/") + "/info"
request = urllib.request.Request(url, method="GET")
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise SystemExit(1)
        response.read()
except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
    raise SystemExit(1)
""".strip()
    )

    body = f"""#!/usr/bin/env bash
set -euo pipefail

RUN_ID={_shell(run_id)}
RUN_DIR={_shell(str(run_dir))}
CONDA_EXE={_shell(settings.conda_executable)}
CONDA_TARGET_FLAG={_shell(conda_flag)}
CONDA_TARGET_VALUE={_shell(conda_value)}
STARTUP_TIMEOUT_S={_shell(str(settings.startup_timeout_s))}
STARTUP_POLL_INTERVAL_S={_shell(str(settings.startup_poll_interval_s))}
VLLM_DTYPE={_shell(settings.vllm_dtype)}
VLLM_GPU_MEMORY_UTILIZATION={_shell(str(settings.vllm_gpu_memory_utilization))}
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
{hf_exports_block}{required_env_checks}

VLLM_PID=""
GENSIE_PID=""

conda_run() {{
  "$CONDA_EXE" run --no-capture-output "$CONDA_TARGET_FLAG" "$CONDA_TARGET_VALUE" "$@"
}}

emit_event() {{
  local event_name="$1"
  local event_message="${{2:-}}"
  EVENT_NAME="$event_name" \\
  EVENT_MESSAGE="$event_message" \\
  RUN_ID="$RUN_ID" \\
  TASK_INDEX="${{TASK_INDEX:-}}" \\
  SPEC_NAME="${{SPEC_NAME:-}}" \\
  SLURM_JOB_ID="${{SLURM_JOB_ID:-}}" \\
  SLURM_ARRAY_JOB_ID="${{SLURM_ARRAY_JOB_ID:-}}" \\
  SLURM_ARRAY_TASK_ID="${{SLURM_ARRAY_TASK_ID:-}}" \\
  conda_run python -c {event_code} "$RUN_DIR"
}}

probe_vllm_once() {{
  conda_run python -c {vllm_probe_code} "http://127.0.0.1:${{VLLM_PORT}}/v1" "$API_KEY" "$MODEL_NAME"
}}

probe_gensie_once() {{
  conda_run python -c {gensie_probe_code} "http://127.0.0.1:${{GENSIE_PORT}}"
}}

wait_for_vllm() {{
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  while true; do
    if probe_vllm_once; then
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited before becoming ready." >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM readiness." >&2
      return 1
    fi
    sleep "$STARTUP_POLL_INTERVAL_S"
  done
}}

wait_for_gensie() {{
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  while true; do
    if probe_gensie_once; then
      return 0
    fi
    if ! kill -0 "$GENSIE_PID" 2>/dev/null; then
      echo "GenSIE server exited before becoming ready." >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for GenSIE server readiness." >&2
      return 1
    fi
    sleep "$STARTUP_POLL_INTERVAL_S"
  done
}}

cleanup() {{
  local exit_code=$?
  trap - EXIT
  set +e
  if [[ -n "${{GENSIE_PID:-}}" ]]; then
    kill "$GENSIE_PID" 2>/dev/null || true
    wait "$GENSIE_PID" 2>/dev/null || true
  fi
  if [[ -n "${{VLLM_PID:-}}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  emit_event "task_finished" "exit_code=${{exit_code}}" || true
  exit "$exit_code"
}}

trap cleanup EXIT

emit_event "launcher_started" "host=${{HOSTNAME:-unknown}}"

{task_binding}

emit_event "task_selected" "model=${{MODEL_NAME}}"

VLLM_ARGS=(
  serve
  "$MODEL_SOURCE"
  --host
  127.0.0.1
  --port
  "$VLLM_PORT"
  --api-key
  "$API_KEY"
  --served-model-name
  "$MODEL_NAME"
  --dtype
  "$VLLM_DTYPE"
  --gpu-memory-utilization
  "$VLLM_GPU_MEMORY_UTILIZATION"
  --generation-config
  vllm
)
{max_model_len_line}
{chat_template_line}
{tensor_parallel_line}

emit_event "vllm_starting" "port=${{VLLM_PORT}}"
conda_run vllm "${{VLLM_ARGS[@]}}" &
VLLM_PID=$!
wait_for_vllm
emit_event "vllm_ready" "port=${{VLLM_PORT}}"

export OPENAI_BASE_URL="http://127.0.0.1:${{VLLM_PORT}}/v1"
export OPENAI_API_KEY="$API_KEY"
export PARTICIPANT_PATH="$PARTICIPANT_PATH_VALUE"

emit_event "gensie_starting" "port=${{GENSIE_PORT}}"
conda_run gensie serve --host 127.0.0.1 --port "$GENSIE_PORT" --no-reload &
GENSIE_PID=$!
wait_for_gensie
emit_event "gensie_ready" "port=${{GENSIE_PORT}}"

mkdir -p "$(dirname "$REPORT_PATH")"
EVAL_ARGS=(
  gensie
  eval
  --data
  "$DATA_PATH"
  --url
  "http://127.0.0.1:${{GENSIE_PORT}}"
  --pipeline
  "$PIPELINE"
  --model
  "$MODEL_NAME"
  --output
  "$REPORT_PATH"
)
if [[ -n "${{LIMIT:-}}" ]]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi

emit_event "eval_starting" "report_path=${{REPORT_PATH}}"
conda_run "${{EVAL_ARGS[@]}}"
emit_event "eval_completed" "report_path=${{REPORT_PATH}}"
"""
    return body


def _render_single_task_binding(task: RenderedTask) -> str:
    return _render_task_exports(task)


def _render_array_task_binding(tasks: tuple[RenderedTask, ...]) -> str:
    lines = [
        'TASK_SLOT="${SLURM_ARRAY_TASK_ID:-}"',
        'if [[ -z "${TASK_SLOT}" ]]; then',
        '  echo "SLURM_ARRAY_TASK_ID is required for array launches." >&2',
        "  exit 1",
        "fi",
        'case "$TASK_SLOT" in',
    ]
    for task in tasks:
        lines.append(f"  {task.index})")
        task_lines = _render_task_exports(task).splitlines()
        for line in task_lines:
            lines.append(f"    {line}")
        lines.append("    ;;")
    lines.extend(
        [
            "  *)",
            '    echo "Unsupported SLURM_ARRAY_TASK_ID: $TASK_SLOT" >&2',
            "    exit 1",
            "    ;;",
            "esac",
        ]
    )
    return "\n".join(lines)


def _render_task_exports(task: RenderedTask) -> str:
    limit = "" if task.spec.limit is None else str(task.spec.limit)
    return "\n".join(
        [
            f"TASK_INDEX={_shell(str(task.index))}",
            f"SPEC_NAME={_shell(task.spec.name)}",
            f"MODEL_SOURCE={_shell(task.spec.model_source)}",
            f"MODEL_NAME={_shell(task.spec.served_model_name)}",
            f"DATA_PATH={_shell(str(task.spec.data_path))}",
            f"PIPELINE={_shell(task.spec.pipeline)}",
            f"PARTICIPANT_PATH_VALUE={_shell(task.spec.participant_path)}",
            f"REPORT_PATH={_shell(str(task.report_path))}",
            f"RESOLVED_SPEC_PATH={_shell(str(task.resolved_spec_path))}",
            f"API_KEY={_shell(task.api_key)}",
            f"VLLM_PORT={_shell(str(task.vllm_port))}",
            f"GENSIE_PORT={_shell(str(task.gensie_port))}",
            f"LIMIT={_shell(limit)}",
        ]
    )


def _shell(value: str) -> str:
    return shlex.quote(value)
