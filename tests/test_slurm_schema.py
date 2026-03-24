from pathlib import Path

from gensie.slurm.resolver import resolve_slurm_settings


def test_resolve_slurm_settings_uses_base_conda_defaults(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "default.toml").write_text(
        '\n'.join(
            [
                'partition = "gpu"',
                'time = "01:00:00"',
                'memory = "64G"',
                'conda_executable = "/opt/miniconda/bin/conda"',
                'conda_env = "gensie-slurm"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_slurm_settings(
        profile_name="default",
        profile_dir=profile_dir,
        cli_overrides={},
    )

    assert resolved.settings.conda_executable == "/opt/miniconda/bin/conda"
    assert resolved.settings.conda_env == "gensie-slurm"
    assert resolved.origins["conda_executable"] == "profile:default"
    assert resolved.origins["conda_env"] == "profile:default"
