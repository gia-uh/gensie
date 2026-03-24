from pathlib import Path

import pytest

from gensie.slurm.specs import EvalSpecError, load_eval_spec, load_manifest_specs


def test_load_eval_spec_applies_defaults(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spec_path = tmp_path / "baseline.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "name: baseline-run",
                "data_path: ./data",
                "model_source: meta-llama/Llama-3.1-8B-Instruct",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_eval_spec(spec_path)

    assert spec.name == "baseline-run"
    assert spec.data_path == data_dir.resolve()
    assert spec.pipeline == "baseline"
    assert spec.participant_path == "gensie.baseline.OfficialParticipant"
    assert spec.served_model_name == spec.model_source
    assert spec.limit is None


def test_load_manifest_specs_resolves_relative_paths(tmp_path: Path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    first = specs_dir / "a.yaml"
    second = specs_dir / "b.yaml"
    first.write_text("name: a\ndata_path: ../data\nmodel_source: foo\n", encoding="utf-8")
    second.write_text("name: b\ndata_path: ../data\nmodel_source: bar\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "evaluations:\n  - ./specs/a.yaml\n  - ./specs/b.yaml\n",
        encoding="utf-8",
    )

    specs = load_manifest_specs(manifest_path)

    assert specs == (first.resolve(), second.resolve())


def test_load_manifest_specs_rejects_wrong_shape(tmp_path: Path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("items: []\n", encoding="utf-8")

    with pytest.raises(EvalSpecError):
        load_manifest_specs(manifest_path)
