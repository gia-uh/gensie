"""Strict schema validation for GenSIE Slurm profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gensie.slurm.models import SlurmSettings


class SlurmSchemaError(ValueError):
    """Raised when a Slurm profile payload is invalid."""


_STRING_KEYS = {
    "partition",
    "time",
    "memory",
    "conda_executable",
    "conda_env",
    "conda_prefix",
    "vllm_dtype",
    "vllm_chat_template",
    "qos",
    "account",
    "constraint",
    "output",
    "error",
    "hf_home",
    "huggingface_hub_cache",
    "extends",
}
_INT_KEYS = {
    "gpus",
    "cpus_per_task",
    "nodes",
    "ntasks_per_node",
    "vllm_port",
    "gensie_port",
    "vllm_max_model_len",
    "startup_timeout_s",
    "startup_poll_interval_s",
    "status_interval_s",
}
_FLOAT_KEYS = {"vllm_gpu_memory_utilization"}
_SEQUENCE_KEYS = {"required_env"}
_REQUIRED_KEYS = {
    "partition",
    "time",
    "memory",
    "gpus",
    "cpus_per_task",
    "nodes",
    "ntasks_per_node",
    "conda_executable",
    "vllm_port",
    "gensie_port",
    "vllm_dtype",
    "vllm_gpu_memory_utilization",
    "startup_timeout_s",
    "startup_poll_interval_s",
}


def allowed_profile_keys() -> set[str]:
    """Return the complete allowed key set for profile files and CLI overrides."""

    return _STRING_KEYS | _INT_KEYS | _FLOAT_KEYS | _SEQUENCE_KEYS


def validate_profile_payload(
    payload: Mapping[str, object],
    *,
    source_name: str,
) -> dict[str, object]:
    """Validate one partial profile layer without requiring every mandatory key."""

    unknown = sorted(key for key in payload if key not in allowed_profile_keys())
    if unknown:
        joined = ", ".join(repr(key) for key in unknown)
        raise SlurmSchemaError(f"{source_name}: unknown profile key(s): {joined}.")

    normalized: dict[str, object] = {}
    for key, value in payload.items():
        if key in _STRING_KEYS:
            normalized[key] = _validate_string_value(
                key=key, value=value, source_name=source_name
            )
        elif key in _INT_KEYS:
            normalized[key] = _validate_int_value(
                key=key, value=value, source_name=source_name
            )
        elif key in _FLOAT_KEYS:
            normalized[key] = _validate_float_value(
                key=key, value=value, source_name=source_name
            )
        elif key in _SEQUENCE_KEYS:
            normalized[key] = _validate_required_env(
                value=value, source_name=source_name
            )
        else:  # pragma: no cover
            raise SlurmSchemaError(f"{source_name}: unsupported profile key {key!r}.")

    _validate_conda_fields(normalized=normalized, source_name=source_name)
    _validate_numeric_relationships(normalized=normalized, source_name=source_name)
    return normalized


def build_settings(*, merged: Mapping[str, object], source_name: str) -> SlurmSettings:
    """Build a fully resolved `SlurmSettings` object from merged layers."""

    normalized = validate_profile_payload(dict(merged), source_name=source_name)
    missing = sorted(key for key in _REQUIRED_KEYS if key not in normalized)
    if missing:
        joined = ", ".join(repr(key) for key in missing)
        raise SlurmSchemaError(f"{source_name}: missing required profile key(s): {joined}.")

    if ("conda_env" in normalized) == ("conda_prefix" in normalized):
        raise SlurmSchemaError(
            f"{source_name}: exactly one of 'conda_env' or 'conda_prefix' must be set."
        )

    status_interval_s = _optional_int(normalized, "status_interval_s")
    if status_interval_s is None:
        status_interval_s = 30

    return SlurmSettings(
        partition=_required_string(normalized, "partition"),
        time=_required_string(normalized, "time"),
        memory=_required_string(normalized, "memory"),
        gpus=_required_int(normalized, "gpus"),
        cpus_per_task=_required_int(normalized, "cpus_per_task"),
        nodes=_required_int(normalized, "nodes"),
        ntasks_per_node=_required_int(normalized, "ntasks_per_node"),
        conda_executable=_required_string(normalized, "conda_executable"),
        conda_env=_optional_string(normalized, "conda_env"),
        conda_prefix=_optional_string(normalized, "conda_prefix"),
        vllm_port=_required_int(normalized, "vllm_port"),
        gensie_port=_required_int(normalized, "gensie_port"),
        vllm_dtype=_required_string(normalized, "vllm_dtype"),
        vllm_gpu_memory_utilization=_required_float(
            normalized, "vllm_gpu_memory_utilization"
        ),
        startup_timeout_s=_required_int(normalized, "startup_timeout_s"),
        startup_poll_interval_s=_required_int(
            normalized, "startup_poll_interval_s"
        ),
        vllm_max_model_len=_optional_int(normalized, "vllm_max_model_len"),
        vllm_chat_template=_optional_string(normalized, "vllm_chat_template"),
        qos=_optional_string(normalized, "qos"),
        account=_optional_string(normalized, "account"),
        constraint=_optional_string(normalized, "constraint"),
        output=_optional_string(normalized, "output"),
        error=_optional_string(normalized, "error"),
        status_interval_s=status_interval_s,
        required_env=_optional_string_tuple(normalized, "required_env"),
        hf_home=_optional_string(normalized, "hf_home"),
        huggingface_hub_cache=_optional_string(
            normalized, "huggingface_hub_cache"
        ),
    )


def _validate_string_value(*, key: str, value: object, source_name: str) -> str:
    if not isinstance(value, str):
        raise SlurmSchemaError(f"{source_name}: {key!r} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise SlurmSchemaError(f"{source_name}: {key!r} must not be empty.")
    return normalized


def _validate_int_value(*, key: str, value: object, source_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SlurmSchemaError(f"{source_name}: {key!r} must be an integer.")
    if value <= 0:
        raise SlurmSchemaError(f"{source_name}: {key!r} must be greater than zero.")
    if key in {"vllm_port", "gensie_port"} and value > 65535:
        raise SlurmSchemaError(f"{source_name}: {key!r} must be <= 65535.")
    return value


def _validate_float_value(*, key: str, value: object, source_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SlurmSchemaError(f"{source_name}: {key!r} must be a float.")
    normalized = float(value)
    if key == "vllm_gpu_memory_utilization" and not (0.0 < normalized <= 1.0):
        raise SlurmSchemaError(
            f"{source_name}: 'vllm_gpu_memory_utilization' must be within (0.0, 1.0]."
        )
    return normalized


def _validate_required_env(*, value: object, source_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SlurmSchemaError(f"{source_name}: 'required_env' must be a list of strings.")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise SlurmSchemaError(f"{source_name}: 'required_env' entries must be strings.")
        token = item.strip()
        if not token:
            raise SlurmSchemaError(
                f"{source_name}: 'required_env' entries must not be empty strings."
            )
        if token not in seen:
            normalized.append(token)
            seen.add(token)
    return tuple(normalized)


def _validate_conda_fields(*, normalized: Mapping[str, object], source_name: str) -> None:
    has_conda_env = "conda_env" in normalized
    has_conda_prefix = "conda_prefix" in normalized
    if has_conda_env and has_conda_prefix:
        raise SlurmSchemaError(
            f"{source_name}: 'conda_env' and 'conda_prefix' are mutually exclusive."
        )


def _validate_numeric_relationships(
    *,
    normalized: Mapping[str, object],
    source_name: str,
) -> None:
    timeout = normalized.get("startup_timeout_s")
    poll = normalized.get("startup_poll_interval_s")
    if isinstance(timeout, int) and isinstance(poll, int) and poll > timeout:
        raise SlurmSchemaError(
            f"{source_name}: 'startup_poll_interval_s' must be <= 'startup_timeout_s'."
        )


def _required_string(normalized: Mapping[str, object], key: str) -> str:
    return str(normalized[key])


def _optional_string(normalized: Mapping[str, object], key: str) -> str | None:
    value = normalized.get(key)
    if value is None:
        return None
    return str(value)


def _required_int(normalized: Mapping[str, object], key: str) -> int:
    value = normalized[key]
    if not isinstance(value, int):
        raise SlurmSchemaError(f"{key!r} must resolve to an integer.")
    return value


def _optional_int(normalized: Mapping[str, object], key: str) -> int | None:
    value = normalized.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise SlurmSchemaError(f"{key!r} must resolve to an integer.")
    return value


def _required_float(normalized: Mapping[str, object], key: str) -> float:
    value = normalized[key]
    if not isinstance(value, float):
        raise SlurmSchemaError(f"{key!r} must resolve to a float.")
    return value


def _optional_string_tuple(normalized: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = normalized.get(key)
    if value is None:
        return ()
    if not isinstance(value, tuple):
        raise SlurmSchemaError(f"{key!r} must resolve to a tuple of strings.")
    return tuple(str(item) for item in value)
