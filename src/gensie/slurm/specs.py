"""Evaluation spec and manifest loading for Slurm runs."""

from __future__ import annotations

from pathlib import Path

import yaml

from gensie.slurm.models import EvalSpec


class EvalSpecError(ValueError):
    """Raised when evaluation specs or manifests are invalid."""


def load_eval_spec(path: Path) -> EvalSpec:
    """Load one evaluation spec YAML file."""

    resolved_path = path.resolve()
    try:
        data = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvalSpecError(f"Failed to read eval spec {resolved_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise EvalSpecError(f"Eval spec {resolved_path} must have a mapping root.")

    name = _required_string(data, "name", resolved_path)
    raw_data_path = _required_string(data, "data_path", resolved_path)
    data_path = _resolve_existing_path(
        raw_data_path,
        source_path=resolved_path,
        field_name="data_path",
    )
    if not data_path.is_dir():
        raise EvalSpecError(f"Eval spec {resolved_path} data_path is not a directory: {data_path}")

    pipeline = _optional_string(data, "pipeline") or "baseline"
    participant_path = _optional_string(data, "participant_path") or (
        "gensie.baseline.OfficialParticipant"
    )
    model_source = _required_string(data, "model_source", resolved_path)
    served_model_name = _optional_string(data, "served_model_name") or model_source
    limit = _optional_positive_int(data, "limit", resolved_path)

    return EvalSpec(
        name=name,
        source_path=resolved_path,
        data_path=data_path,
        pipeline=pipeline,
        participant_path=participant_path,
        model_source=model_source,
        served_model_name=served_model_name,
        limit=limit,
    )


def load_manifest_specs(manifest_path: Path) -> tuple[Path, ...]:
    """Load a manifest whose root shape is `{\"evaluations\": [...]}`."""

    resolved_manifest = manifest_path.resolve()
    try:
        data = yaml.safe_load(resolved_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvalSpecError(
            f"Failed to read eval manifest {resolved_manifest}: {exc}"
        ) from exc

    if not isinstance(data, dict) or set(data) != {"evaluations"}:
        raise EvalSpecError(
            f"Eval manifest {resolved_manifest} must have exactly one root key 'evaluations'."
        )

    raw_specs = data.get("evaluations")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise EvalSpecError(
            f"Eval manifest {resolved_manifest} must contain a non-empty 'evaluations' list."
        )

    normalized: list[Path] = []
    for index, value in enumerate(raw_specs):
        if not isinstance(value, str) or not value.strip():
            raise EvalSpecError(
                f"Eval manifest {resolved_manifest} entry {index} must be a non-empty string path."
            )
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (resolved_manifest.parent / candidate).resolve()
        normalized.append(candidate)
    return tuple(normalized)


def dump_resolved_eval_spec(
    *,
    spec: EvalSpec,
    output_path: Path,
    report_path: Path,
    vllm_port: int,
    gensie_port: int,
    vllm_port_origin: str,
    gensie_port_origin: str,
) -> Path:
    """Write a resolved eval spec for auditability."""

    payload = {
        "name": spec.name,
        "source_path": str(spec.source_path),
        "data_path": str(spec.data_path),
        "pipeline": spec.pipeline,
        "participant_path": spec.participant_path,
        "model_source": spec.model_source,
        "served_model_name": spec.served_model_name,
        "limit": spec.limit,
        "report_path": str(report_path),
        "vllm_port": vllm_port,
        "gensie_port": gensie_port,
        "vllm_port_origin": vllm_port_origin,
        "gensie_port_origin": gensie_port_origin,
        "openai_base_url": f"http://127.0.0.1:{vllm_port}/v1",
        "gensie_url": f"http://127.0.0.1:{gensie_port}",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return output_path


def _required_string(data: dict[str, object], key: str, source_path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalSpecError(
            f"Eval spec {source_path} must define a non-empty string `{key}`."
        )
    return value.strip()


def _optional_string(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvalSpecError(f"Eval spec field `{key}` must be a non-empty string when set.")
    return value.strip()


def _optional_positive_int(
    data: dict[str, object],
    key: str,
    source_path: Path,
) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvalSpecError(
            f"Eval spec {source_path} field `{key}` must be a positive integer."
        )
    return value


def _resolve_existing_path(
    raw_value: str,
    *,
    source_path: Path,
    field_name: str,
) -> Path:
    candidate = Path(raw_value)
    resolved = candidate if candidate.is_absolute() else (source_path.parent / candidate).resolve()
    if not resolved.exists():
        raise EvalSpecError(
            f"Eval spec {source_path} references missing `{field_name}`: {resolved}"
        )
    return resolved
