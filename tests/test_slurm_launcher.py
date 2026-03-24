from pathlib import Path

from gensie.slurm.launcher import prepare_rendered_launch
from gensie.slurm.resolver import resolve_slurm_settings
from gensie.slurm.specs import load_eval_spec


def _write_profile(profile_dir: Path) -> None:
    profile_dir.mkdir()
    (profile_dir / "default.toml").write_text(
        '\n'.join(
            [
                'partition = "gpu"',
                'time = "02:00:00"',
                'memory = "80G"',
                'conda_executable = "/opt/miniconda/bin/conda"',
                'conda_env = "gensie-slurm"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_spec(root: Path, name: str, model_source: str) -> Path:
    data_dir = root / f"data-{name}"
    data_dir.mkdir()
    spec_path = root / f"{name}.yaml"
    spec_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                f"data_path: ./data-{name}",
                f"model_source: {model_source}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return spec_path


def test_prepare_rendered_launch_derives_base_ports(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    _write_profile(profile_dir)
    resolved = resolve_slurm_settings(
        profile_name="default",
        profile_dir=profile_dir,
        cli_overrides={},
    )
    spec = load_eval_spec(_write_spec(tmp_path, "baseline", "foo/bar"))

    launch = prepare_rendered_launch(
        run_id="slurm-test-run",
        run_dir=tmp_path / "run",
        resolved=resolved,
        specs=(spec,),
        normalized_manifest_path=None,
    )

    assert len(launch.tasks) == 1
    task = launch.tasks[0]
    assert task.vllm_port != 8000
    assert task.gensie_port != 8100
    assert task.vllm_port_origin == "derived:base"
    assert task.gensie_port_origin == "derived:base"
    assert task.resolved_spec_path.exists()
    assert task.report_path.parent.exists()
    launcher_text = launch.launcher_path.read_text(encoding="utf-8")
    assert "gensie serve --host 127.0.0.1 --port \"$GENSIE_PORT\" --no-reload" in launcher_text


def test_prepare_rendered_launch_preserves_explicit_ports(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    _write_profile(profile_dir)
    resolved = resolve_slurm_settings(
        profile_name="default",
        profile_dir=profile_dir,
        cli_overrides={"vllm_port": 8123, "gensie_port": 8124},
    )
    spec = load_eval_spec(_write_spec(tmp_path, "explicit", "foo/bar"))

    launch = prepare_rendered_launch(
        run_id="slurm-explicit-run",
        run_dir=tmp_path / "run",
        resolved=resolved,
        specs=(spec,),
        normalized_manifest_path=None,
    )

    task = launch.tasks[0]
    assert task.vllm_port == 8123
    assert task.gensie_port == 8124
    assert task.vllm_port_origin == "cli"
    assert task.gensie_port_origin == "cli"


def test_prepare_rendered_launch_renders_arrays(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    _write_profile(profile_dir)
    resolved = resolve_slurm_settings(
        profile_name="default",
        profile_dir=profile_dir,
        cli_overrides={},
    )
    spec_a = load_eval_spec(_write_spec(tmp_path, "a", "foo/a"))
    spec_b = load_eval_spec(_write_spec(tmp_path, "b", "foo/b"))

    launch = prepare_rendered_launch(
        run_id="slurm-array-run",
        run_dir=tmp_path / "run",
        resolved=resolved,
        specs=(spec_a, spec_b),
        normalized_manifest_path=tmp_path / "manifest.yaml",
    )

    assert launch.is_array is True
    assert launch.sbatch_output_path.endswith("task-%a.out")
    assert launch.sbatch_error_path.endswith("task-%a.err")
    launcher_text = launch.launcher_path.read_text(encoding="utf-8")
    assert 'TASK_SLOT="${SLURM_ARRAY_TASK_ID:-}"' in launcher_text
    assert 'case "$TASK_SLOT" in' in launcher_text
