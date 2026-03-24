import json
from pathlib import Path

from typer.testing import CliRunner

import gensie.cli
from gensie.slurm.run_artifacts import write_submission_manifest


runner = CliRunner()


def _write_profile(profile_dir: Path) -> None:
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


def _write_spec(root: Path, name: str) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    spec_path = root / f"{name}.yaml"
    spec_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "data_path: ./data",
                "model_source: meta-llama/Llama-3.1-8B-Instruct",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return spec_path


def test_serve_supports_no_reload(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(gensie.cli.uvicorn, "run", fake_run)

    result = runner.invoke(gensie.cli.app, ["serve", "--host", "127.0.0.1", "--port", "8123", "--no-reload"])

    assert result.exit_code == 0
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8123
    assert captured["kwargs"]["reload"] is False


def test_slurm_validate_accepts_profile_only(monkeypatch, tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    _write_profile(profile_dir)

    monkeypatch.setattr("gensie.slurm.cli.validate_conda_execution", lambda settings: None)
    monkeypatch.setattr("gensie.slurm.cli.validate_required_env", lambda settings: None)
    monkeypatch.setattr("gensie.slurm.cli.validate_optional_runtime_paths", lambda settings: None)

    result = runner.invoke(
        gensie.cli.app,
        ["slurm", "validate", "--profile-dir", str(profile_dir), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["selection_type"] == "profile_only"
    assert payload["task_count"] == 0
    assert payload["settings"]["conda_env"] == "gensie-slurm"


def test_slurm_eval_render_outputs_submission_manifest(tmp_path: Path):
    profile_dir = tmp_path / "profiles"
    _write_profile(profile_dir)
    spec_path = _write_spec(tmp_path, "baseline")
    run_root = tmp_path / "runs"

    result = runner.invoke(
        gensie.cli.app,
        [
            "slurm",
            "eval",
            "render",
            "--profile-dir",
            str(profile_dir),
            "--spec",
            str(spec_path),
            "--run-root",
            str(run_root),
            "--run-id",
            "slurm-render-test",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task_count"] == 1
    assert payload["task_specs"][0]["api_key"] == "<redacted>"
    assert Path(payload["launcher_path"]).exists()
    assert Path(payload["resolved_spec_paths"][0]).exists()


def test_slurm_logs_reads_array_globs(tmp_path: Path):
    run_root = tmp_path / "runs"
    run_id = "slurm-log-test"
    run_dir = run_root / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    write_submission_manifest(
        run_dir,
        {
            "stdout_path": "logs/task-%a.out",
            "stderr_path": "logs/task-%a.err",
            "job_id": "4242",
        },
    )
    (logs_dir / "task-0.out").write_text("first-task\n", encoding="utf-8")
    (logs_dir / "task-1.out").write_text("second-task\n", encoding="utf-8")

    result = runner.invoke(
        gensie.cli.app,
        [
            "slurm",
            "logs",
            "--run-id",
            run_id,
            "--run-root",
            str(run_root),
            "--stream",
            "out",
        ],
    )

    assert result.exit_code == 0
    assert "first-task" in result.stdout
    assert "second-task" in result.stdout
