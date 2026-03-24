"""Layered profile resolution with per-key origin tracking."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from gensie.slurm.models import ResolvedSlurmSettings
from gensie.slurm.profile_store import (
    SlurmProfileStoreError,
    load_base_profile,
    load_user_profiles,
)
from gensie.slurm.schema import SlurmSchemaError, allowed_profile_keys, build_settings

_BASE_PROFILE_NAME = "base"
_DEFAULT_BASE_PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "base.toml"
_MAX_EXTENDS_DEPTH = 4


class SlurmResolutionError(ValueError):
    """Raised when profile resolution fails."""


def resolve_slurm_settings(
    *,
    profile_name: str,
    profile_dir: Path,
    cli_overrides: Mapping[str, object],
    base_profile_path: Path | None = None,
) -> ResolvedSlurmSettings:
    """Resolve a Slurm profile into concrete settings and key origins."""

    normalized_profile_name = profile_name.strip()
    if not normalized_profile_name:
        raise SlurmResolutionError("Profile name must be a non-empty string.")

    try:
        base_profile = dict(load_base_profile(base_profile_path or _DEFAULT_BASE_PROFILE_PATH))
        user_profiles = load_user_profiles(profile_dir)
    except (SlurmProfileStoreError, SlurmSchemaError) as exc:
        raise SlurmResolutionError(str(exc)) from exc

    if normalized_profile_name not in user_profiles:
        known = ", ".join(sorted(user_profiles)) or "(none)"
        raise SlurmResolutionError(
            f"Unknown Slurm profile {normalized_profile_name!r}. "
            f"Known profiles in {profile_dir}: {known}"
        )

    chain = _resolve_profile_chain(
        profile_name=normalized_profile_name,
        user_profiles=user_profiles,
    )

    merged: dict[str, object] = {}
    origins: dict[str, str] = {}
    _merge_layer(merged=merged, origins=origins, payload=base_profile, origin="base")
    for name in chain:
        _merge_layer(
            merged=merged,
            origins=origins,
            payload=user_profiles[name],
            origin=f"profile:{name}",
        )

    _validate_cli_overrides(cli_overrides)
    _merge_layer(
        merged=merged,
        origins=origins,
        payload=dict(cli_overrides),
        origin="cli",
    )

    try:
        settings = build_settings(
            merged=merged,
            source_name=f"resolved profile {normalized_profile_name!r}",
        )
    except SlurmSchemaError as exc:
        raise SlurmResolutionError(str(exc)) from exc

    relevant_keys = set(settings.to_dict())
    normalized_origins = {key: origins[key] for key in sorted(origins) if key in relevant_keys}
    return ResolvedSlurmSettings(
        profile_name=normalized_profile_name,
        settings=settings,
        origins=normalized_origins,
    )


def _merge_layer(
    *,
    merged: dict[str, object],
    origins: dict[str, str],
    payload: Mapping[str, object],
    origin: str,
) -> None:
    has_conda_env = "conda_env" in payload
    has_conda_prefix = "conda_prefix" in payload
    if has_conda_env and not has_conda_prefix:
        merged.pop("conda_prefix", None)
        origins.pop("conda_prefix", None)
    if has_conda_prefix and not has_conda_env:
        merged.pop("conda_env", None)
        origins.pop("conda_env", None)

    for key, value in payload.items():
        if key == "extends":
            continue
        merged[key] = value
        origins[key] = origin


def _resolve_profile_chain(
    *,
    profile_name: str,
    user_profiles: Mapping[str, Mapping[str, object]],
) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(name: str, depth: int) -> None:
        if depth > _MAX_EXTENDS_DEPTH:
            raise SlurmResolutionError(
                f"Profile inheritance depth exceeded {_MAX_EXTENDS_DEPTH} while resolving "
                f"{profile_name!r}."
            )
        if name in visiting:
            raise SlurmResolutionError(
                f"Detected Slurm profile inheritance cycle while resolving {profile_name!r}."
            )
        if name in ordered:
            return

        payload = user_profiles.get(name)
        if payload is None:
            raise SlurmResolutionError(f"Profile {name!r} does not exist.")

        visiting.add(name)
        extends = payload.get("extends")
        if isinstance(extends, str):
            parent = extends.strip()
            if parent and parent != _BASE_PROFILE_NAME:
                if parent not in user_profiles:
                    raise SlurmResolutionError(
                        f"Profile {name!r} extends unknown profile {parent!r}."
                    )
                visit(parent, depth + 1)
        visiting.remove(name)
        ordered.append(name)

    visit(profile_name, 1)
    return ordered


def _validate_cli_overrides(cli_overrides: Mapping[str, object]) -> None:
    allowed = allowed_profile_keys() - {"extends"}
    unknown = sorted(key for key in cli_overrides if key not in allowed)
    if unknown:
        joined = ", ".join(repr(key) for key in unknown)
        raise SlurmResolutionError(f"Unknown CLI override key(s): {joined}.")
