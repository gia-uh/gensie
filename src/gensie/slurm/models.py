"""Typed models for Slurm-backed GenSIE evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SlurmSettings:
    """Resolved scheduler and runtime settings."""

    partition: str
    time: str
    memory: str
    gpus: int
    cpus_per_task: int
    nodes: int
    ntasks_per_node: int
    conda_executable: str
    conda_env: str | None
    conda_prefix: str | None
    vllm_port: int
    gensie_port: int
    vllm_dtype: str
    vllm_gpu_memory_utilization: float
    startup_timeout_s: int
    startup_poll_interval_s: int
    vllm_max_model_len: int | None = None
    vllm_chat_template: str | None = None
    qos: str | None = None
    account: str | None = None
    constraint: str | None = None
    output: str | None = None
    error: str | None = None
    status_interval_s: int = 30
    required_env: tuple[str, ...] = ()
    hf_home: str | None = None
    huggingface_hub_cache: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    def conda_target_args(self) -> tuple[str, str]:
        """Return the concrete `conda run` target arguments."""

        if self.conda_env is not None:
            return ("-n", self.conda_env)
        if self.conda_prefix is not None:
            return ("-p", self.conda_prefix)
        raise ValueError(
            "Invalid Slurm settings: exactly one of `conda_env` or `conda_prefix` must be set."
        )


@dataclass(frozen=True, slots=True)
class ResolvedSlurmSettings:
    """Resolved settings plus per-key provenance."""

    profile_name: str
    settings: SlurmSettings
    origins: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "profile_name": self.profile_name,
            "settings": self.settings.to_dict(),
            "origins": dict(self.origins),
        }


@dataclass(frozen=True, slots=True)
class EvalSpec:
    """One workload definition for a Slurm-backed GenSIE evaluation."""

    name: str
    source_path: Path
    data_path: Path
    pipeline: str
    participant_path: str
    model_source: str
    served_model_name: str
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "source_path": str(self.source_path),
            "data_path": str(self.data_path),
            "pipeline": self.pipeline,
            "participant_path": self.participant_path,
            "model_source": self.model_source,
            "served_model_name": self.served_model_name,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class RenderedTask:
    """One concrete rendered task in a Slurm run."""

    index: int
    spec: EvalSpec
    resolved_spec_path: Path
    report_path: Path
    api_key: str
    vllm_port: int
    gensie_port: int
    vllm_port_origin: str
    gensie_port_origin: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize paths and runtime metadata without leaking the raw API key."""

        return {
            "index": self.index,
            "spec": self.spec.to_dict(),
            "resolved_spec_path": str(self.resolved_spec_path),
            "report_path": str(self.report_path),
            "vllm_port": self.vllm_port,
            "gensie_port": self.gensie_port,
            "vllm_port_origin": self.vllm_port_origin,
            "gensie_port_origin": self.gensie_port_origin,
            "api_key": "<redacted>",
        }


@dataclass(frozen=True, slots=True)
class RenderedLaunch:
    """All generated files and metadata for one rendered Slurm run."""

    run_dir: Path
    launcher_path: Path
    normalized_manifest_path: Path | None
    tasks: tuple[RenderedTask, ...]
    sbatch_output_path: str
    sbatch_error_path: str

    @property
    def is_array(self) -> bool:
        """Whether the launch represents a Slurm array submission."""

        return len(self.tasks) > 1
