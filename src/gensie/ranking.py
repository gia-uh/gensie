"""Gap-closed-over-baseline ranking — the official primary leaderboard.

Reads evaluation reports produced by ``gensie eval --output ...`` (JSON with
``participant`` / ``config`` / ``metrics`` keys), and for each model computes
how much of the baseline-to-perfect F1 gap each team closes, then averages over
models. See ``docs/description.md`` → "Primary Leaderboard: Gap Closed over
Baseline".

A report is treated as the official baseline for its model when its
``config.pipeline`` matches ``baseline_pipeline`` (default ``"baseline"``) AND,
when multiple such candidates exist, the report's filename slug matches
``baseline_slug`` (default ``"baseline"``). The filename slug is the chunk
before the first ``--`` or ``__`` in the report filename — e.g.
``baseline--baseline.json`` has slug ``"baseline"``. This disambiguates the
official baseline from a participant whose own pipeline happens to be named
``baseline``: their report still ranks as a regular submission instead of
silently displacing the canonical baseline.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from gensie.eval import gap_closed

REQUIRED_KEYS = ("participant", "config", "metrics")


def _slug_from_path(path: Path) -> str:
    """Return the filename slug — the chunk before the first ``--`` or ``__``."""
    stem = path.stem
    for sep in ("--", "__"):
        if sep in stem:
            return stem.split(sep, 1)[0]
    return stem


def load_reports(results_dir: Path) -> List[Dict[str, Any]]:
    """Load every ``*.json`` evaluation report under ``results_dir``.

    Files that are not valid JSON, or that lack the required top-level keys, are
    silently skipped (so a directory can mix reports with unrelated files).
    Each report dict is annotated with ``_source_slug`` (extracted from filename).
    """
    results_dir = Path(results_dir)
    reports: List[Dict[str, Any]] = []
    if not results_dir.is_dir():
        return reports
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or not all(k in data for k in REQUIRED_KEYS):
            continue
        data["_source_slug"] = _slug_from_path(path)
        reports.append(data)
    return reports


def _entry(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "team": report["participant"].get("team_name", "Unknown"),
        "model": report["config"].get("model", "Unknown"),
        "pipeline": report["config"].get("pipeline", "Unknown"),
        "f1": float(report["metrics"].get("f1", 0.0) or 0.0),
        "slug": report.get("_source_slug", ""),
    }


def compute_ranking(
    reports: List[Dict[str, Any]],
    baseline_pipeline: str = "baseline",
    baseline_slug: str = "baseline",
) -> Dict[str, Any]:
    """Build the primary (gap-closed) and secondary (raw F1) leaderboards.

    Returns a dict with:
      - ``baselines``: ``{model: baseline_f1}`` for models that have a baseline report.
      - ``per_model``: ``{model: [{team, pipeline, f1, gap_closed}, ...]}`` (non-baseline,
        sorted by gap_closed desc).
      - ``leaderboard``: ``[{team, avg_gap_closed, per_model: {model: {pipeline, f1, gap_closed}}}]``
        sorted by ``avg_gap_closed`` desc — using each team's best pipeline per model.
      - ``raw_f1_leaderboard``: ``[{team, avg_f1, per_model: {model: {pipeline, f1}}}]``
        sorted by ``avg_f1`` desc — secondary, for reference only.
      - ``warnings``: list of human-readable strings (e.g. a model with no baseline).
    """
    entries = [_entry(r) for r in reports]
    models = sorted({e["model"] for e in entries})
    warnings: List[str] = []

    # Baseline F1 per model. When multiple pipeline=baseline candidates exist for
    # a model, prefer the one whose filename slug matches baseline_slug (the
    # canonical official-baseline naming convention). The other candidates are
    # not treated as baselines and remain available as participant submissions.
    baselines: Dict[str, float] = {}
    baseline_keys: Dict[str, tuple] = {}  # model -> (pipeline, slug) of chosen baseline
    for model in models:
        candidates = [
            e
            for e in entries
            if e["model"] == model and e["pipeline"] == baseline_pipeline
        ]
        if not candidates:
            warnings.append(
                f"model '{model}' has no baseline report (pipeline='{baseline_pipeline}') "
                f"— it is excluded from the ranking"
            )
            continue

        canonical = [c for c in candidates if c["slug"] == baseline_slug]
        if canonical:
            chosen = max(canonical, key=lambda c: c["f1"])
            if len(canonical) > 1:
                warnings.append(
                    f"model '{model}' has {len(canonical)} canonical baseline reports "
                    f"(slug='{baseline_slug}') — using the highest F1"
                )
            non_canonical = sorted(
                {c["slug"] for c in candidates if c["slug"] != baseline_slug}
            )
            if non_canonical:
                warnings.append(
                    f"model '{model}': pipeline='{baseline_pipeline}' also used by "
                    f"submission(s) from {', '.join(non_canonical)} — treating those "
                    f"as participant entries, not baselines"
                )
        else:
            chosen = max(candidates, key=lambda c: c["f1"])
            if len(candidates) > 1:
                warnings.append(
                    f"model '{model}' has {len(candidates)} baseline reports and none "
                    f"match the canonical slug '{baseline_slug}' — using the highest F1"
                )

        baselines[model] = chosen["f1"]
        baseline_keys[model] = (chosen["pipeline"], chosen["slug"])

    ranked_models = [m for m in models if m in baselines]

    # Per-model, per-(team, pipeline) gap_closed for non-baseline entries.
    per_model: Dict[str, List[Dict[str, Any]]] = {}
    # team -> model -> best {pipeline, f1, gap_closed}
    team_best: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    # team -> model -> best {pipeline, f1} by raw f1
    team_best_f1: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for model in ranked_models:
        f1_b = baselines[model]
        base_pipe, base_slug = baseline_keys[model]
        rows: List[Dict[str, Any]] = []
        for e in entries:
            if e["model"] != model:
                continue
            # Skip only the chosen official baseline. A submission whose pipeline
            # happens to be named "baseline" but has a different filename slug is
            # still treated as a regular entry.
            if e["pipeline"] == base_pipe and e["slug"] == base_slug:
                continue
            gc = gap_closed(e["f1"], f1_b)
            rows.append(
                {
                    "team": e["team"],
                    "pipeline": e["pipeline"],
                    "f1": e["f1"],
                    "gap_closed": gc,
                }
            )

            cur = team_best[e["team"]].get(model)
            if cur is None or gc > cur["gap_closed"]:
                team_best[e["team"]][model] = {
                    "pipeline": e["pipeline"],
                    "f1": e["f1"],
                    "gap_closed": gc,
                }

            cur_f1 = team_best_f1[e["team"]].get(model)
            if cur_f1 is None or e["f1"] > cur_f1["f1"]:
                team_best_f1[e["team"]][model] = {
                    "pipeline": e["pipeline"],
                    "f1": e["f1"],
                }

        rows.sort(key=lambda r: r["gap_closed"], reverse=True)
        per_model[model] = rows

    # Primary leaderboard: average gap_closed over the ranked models for which the
    # team submitted a result. (Missing a model counts as 0 on that model.)
    leaderboard: List[Dict[str, Any]] = []
    for team, by_model in team_best.items():
        total = sum(by_model[m]["gap_closed"] for m in ranked_models if m in by_model)
        avg = total / len(ranked_models) if ranked_models else 0.0
        leaderboard.append(
            {"team": team, "avg_gap_closed": avg, "per_model": dict(by_model)}
        )
    leaderboard.sort(key=lambda r: r["avg_gap_closed"], reverse=True)

    # Secondary leaderboard: average raw F1 over the ranked models.
    raw_f1_leaderboard: List[Dict[str, Any]] = []
    for team, by_model in team_best_f1.items():
        total = sum(by_model[m]["f1"] for m in ranked_models if m in by_model)
        avg = total / len(ranked_models) if ranked_models else 0.0
        raw_f1_leaderboard.append(
            {"team": team, "avg_f1": avg, "per_model": dict(by_model)}
        )
    raw_f1_leaderboard.sort(key=lambda r: r["avg_f1"], reverse=True)

    return {
        "models": ranked_models,
        "baselines": baselines,
        "per_model": per_model,
        "leaderboard": leaderboard,
        "raw_f1_leaderboard": raw_f1_leaderboard,
        "warnings": warnings,
    }
