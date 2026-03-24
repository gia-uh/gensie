"""Profile file loading for Slurm settings."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from gensie.slurm.schema import validate_profile_payload


class SlurmProfileStoreError(ValueError):
    """Raised when Slurm profile files cannot be loaded."""


def load_base_profile(path: Path) -> Mapping[str, object]:
    """Load the repository base profile."""

    if not path.exists():
        raise SlurmProfileStoreError(f"Base Slurm profile does not exist: {path}.")
    payload = _load_toml(path)
    return validate_profile_payload(payload, source_name=f"base profile {path}")


def load_user_profiles(profile_dir: Path) -> dict[str, Mapping[str, object]]:
    """Load all user-local Slurm profiles from one directory."""

    if not profile_dir.exists():
        return {}
    if not profile_dir.is_dir():
        raise SlurmProfileStoreError(f"Profile directory is not a directory: {profile_dir}.")

    profiles: dict[str, Mapping[str, object]] = {}
    for path in sorted(profile_dir.glob("*.toml")):
        name = path.stem
        payload = _load_toml(path)
        profiles[name] = validate_profile_payload(payload, source_name=f"profile {path}")
    return profiles


def _load_toml(path: Path) -> dict[str, object]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover
        raise SlurmProfileStoreError(f"Slurm profile not found: {path}.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SlurmProfileStoreError(f"Invalid TOML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SlurmProfileStoreError(f"Slurm profile {path} must decode to a TOML table.")
    return data
