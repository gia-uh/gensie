"""Tests for the evaluator's per-instance timing summary.

Spec (docs/submission.md §4): the 60s timeout is a *target averaged over the
test set*, enforced softly — the evaluator records per-instance wall time and
reports the average plus which instances exceeded the budget, instead of
hard-stopping at 60s.
"""

import pytest

from gensie.eval import summarize_timing


def test_summarize_timing_basic():
    s = summarize_timing([10.0, 20.0, 90.0], budget_s=60.0)
    assert s["avg_elapsed_s"] == pytest.approx(40.0)
    assert s["max_elapsed_s"] == pytest.approx(90.0)
    assert s["over_budget_count"] == 1
    assert s["n"] == 3
    # Average (40s) is within the 60s budget even though one instance overran.
    assert s["avg_within_budget"] is True


def test_summarize_timing_average_over_budget():
    s = summarize_timing([70.0, 80.0, 90.0], budget_s=60.0)
    assert s["avg_elapsed_s"] == pytest.approx(80.0)
    assert s["over_budget_count"] == 3
    assert s["avg_within_budget"] is False


def test_summarize_timing_empty():
    s = summarize_timing([], budget_s=60.0)
    assert s["n"] == 0
    assert s["avg_elapsed_s"] == 0.0
    assert s["over_budget_count"] == 0
    assert s["avg_within_budget"] is True
