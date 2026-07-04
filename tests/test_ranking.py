"""Tests for the gap-closed-over-baseline ranking (the official primary leaderboard).

Spec: docs/description.md -> "Primary Leaderboard: Gap Closed over Baseline".
gap_closed = max(0, (F1_system - F1_baseline) / (1 - F1_baseline)), averaged over models.
"""

import json

import pytest

from gensie.eval import gap_closed
from gensie.ranking import compute_ranking, load_reports


# --- gap_closed pure function ---


@pytest.mark.parametrize(
    "f1_s, f1_b, expected",
    [
        (0.8, 0.6, 0.5),  # closed 20 of 40 points of error
        (0.9, 0.6, 0.75),  # closed 30 of 40
        (0.6, 0.6, 0.0),  # no improvement
        (0.5, 0.6, 0.0),  # worse than baseline -> clamped to 0
        (1.0, 0.6, 1.0),  # perfect -> closed the whole gap
        (1.0, 1.0, 1.0),  # baseline already perfect, system matches it
        (0.7, 1.0, 0.0),  # baseline perfect, system worse
    ],
)
def test_gap_closed(f1_s, f1_b, expected):
    assert gap_closed(f1_s, f1_b) == pytest.approx(expected)


# --- compute_ranking ---


def _report(team, pipeline, model, f1):
    return {
        "participant": {"team_name": team, "institution": "X"},
        "config": {"model": model, "pipeline": pipeline, "data_source": "/data/test"},
        "metrics": {"precision": f1, "recall": f1, "f1": f1},
        "tasks": [],
    }


def test_ranking_averages_gap_over_models():
    reports = [
        # model m1: baseline 0.6
        _report("GenSIE Baseline Team", "baseline", "m1", 0.6),
        _report("Team A", "p1", "m1", 0.8),  # gap 0.5
        _report("Team B", "p1", "m1", 0.7),  # gap 0.25
        _report("Team B", "p2", "m1", 0.9),  # gap 0.75  <- B's best on m1
        # model m2: baseline 0.5
        _report("GenSIE Baseline Team", "baseline", "m2", 0.5),
        _report("Team A", "p1", "m2", 0.75),  # gap 0.5
        _report("Team B", "p1", "m2", 0.6),  # gap 0.2  <- B's best on m2
        _report("Team B", "p2", "m2", 0.55),  # gap 0.1
    ]
    result = compute_ranking(reports, baseline_pipeline="baseline")

    # Per-model baselines recorded
    assert result["baselines"] == {"m1": 0.6, "m2": 0.5}

    lb = {row["team"]: row for row in result["leaderboard"]}
    # Team A: (0.5 + 0.5) / 2 = 0.5 ; Team B: (0.75 + 0.2) / 2 = 0.475
    assert lb["Team A"]["avg_gap_closed"] == pytest.approx(0.5)
    assert lb["Team B"]["avg_gap_closed"] == pytest.approx(0.475)
    # Team A is ranked above Team B despite B having the single best per-model result
    assert [row["team"] for row in result["leaderboard"]] == ["Team A", "Team B"]
    # Best pipeline per model is the one that achieved the max gap
    assert lb["Team B"]["per_model"]["m1"]["pipeline"] == "p2"
    assert lb["Team B"]["per_model"]["m2"]["pipeline"] == "p1"


def test_ranking_warns_when_model_has_no_baseline():
    reports = [
        _report("GenSIE Baseline Team", "baseline", "m1", 0.6),
        _report("Team A", "p1", "m1", 0.8),
        _report("Team A", "p1", "m2", 0.9),  # m2 has no baseline -> ignored + warning
    ]
    result = compute_ranking(reports, baseline_pipeline="baseline")
    assert result["baselines"] == {"m1": 0.6}
    assert any("m2" in w for w in result["warnings"])
    # Only m1 counts toward the average
    assert result["leaderboard"][0]["avg_gap_closed"] == pytest.approx(0.5)


def test_ranking_prefers_canonical_baseline_slug(tmp_path):
    """When multiple pipeline='baseline' reports exist for a model, the one whose
    filename slug matches `baseline_slug` is the official baseline; others are
    participant submissions. Regression test for the case where FranRodrigo named
    their pipeline 'baseline' and silently displaced the official baseline."""
    official = _report("GenSIE Baseline Team", "baseline", "m1", 0.6)
    franrodrigo = _report("FranRodrigo", "baseline", "m1", 0.7)
    submission = _report("Team B", "p1", "m1", 0.8)
    (tmp_path / "baseline--baseline.json").write_text(json.dumps(official))
    (tmp_path / "franrodrigo--baseline.json").write_text(json.dumps(franrodrigo))
    (tmp_path / "teamb--p1.json").write_text(json.dumps(submission))

    reports = load_reports(tmp_path)
    result = compute_ranking(reports, baseline_pipeline="baseline", baseline_slug="baseline")

    # Baseline is the official one (F1=0.6), NOT the highest-F1 collision (0.7).
    assert result["baselines"] == {"m1": 0.6}
    # FranRodrigo is treated as a participant submission.
    teams = {row["team"]: row for row in result["leaderboard"]}
    assert "FranRodrigo" in teams
    # FranRodrigo gap_closed = (0.7 - 0.6) / (1 - 0.6) = 0.25
    assert teams["FranRodrigo"]["avg_gap_closed"] == pytest.approx(0.25)
    # The collision is surfaced as a warning.
    assert any("franrodrigo" in w for w in result["warnings"])


def test_load_reports_reads_and_filters(tmp_path):
    good = _report("Team A", "p1", "m1", 0.8)
    (tmp_path / "good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps({"nope": 1}), encoding="utf-8")
    (tmp_path / "notjson.txt").write_text("ignored", encoding="utf-8")

    reports = load_reports(tmp_path)
    assert len(reports) == 1
    assert reports[0]["participant"]["team_name"] == "Team A"


def test_rank_command_runs_on_examples():
    """`gensie rank results/examples` exits 0 and prints the primary leaderboard."""
    from typer.testing import CliRunner

    from gensie.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["rank", "results/examples", "--plain"])
    assert result.exit_code == 0, result.output
    assert "Primary leaderboard" in result.output
    assert "Gamma Group" in result.output


def test_rank_command_errors_on_empty_dir(tmp_path):
    from typer.testing import CliRunner

    from gensie.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["rank", str(tmp_path)])
    assert result.exit_code == 1


def test_example_fixtures_rank_as_expected():
    """The committed example reports under results/examples/ exercise the command end-to-end."""
    from pathlib import Path

    examples_dir = Path("results/examples")
    reports = load_reports(examples_dir)
    assert reports, "results/examples/ should contain example reports"
    result = compute_ranking(reports, baseline_pipeline="baseline")
    # There should be at least one baseline and a non-trivial leaderboard.
    assert result["baselines"]
    assert len(result["leaderboard"]) >= 2
    # Leaderboard is sorted by avg_gap_closed descending.
    scores = [row["avg_gap_closed"] for row in result["leaderboard"]]
    assert scores == sorted(scores, reverse=True)
