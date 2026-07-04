import pytest

from gensie.eval import summarize_token_usage


def _u(inp, out, calls=1):
    return {"input": inp, "output": out, "total": inp + out, "calls": calls}


def test_summary_basic():
    s = summarize_token_usage(
        [_u(10_000, 5_000), _u(20_000, 10_000), _u(50_000, 30_000)]
    )
    assert s["n"] == 3
    assert s["total_input"] == 80_000
    assert s["total_output"] == 45_000
    assert s["total_tokens"] == 125_000
    assert s["avg_total_per_instance"] == pytest.approx(125_000 / 3)
    assert s["max_total"] == 80_000
    assert s["over_target_count"] == 1  # only the 80K instance > 32K
    assert s["over_soft_count"] == 1  # 80K > 64K
    assert s["calls_total"] == 3
    # 125000/3 ≈ 41667 > 32000
    assert s["avg_within_target"] is False


def test_avg_over_target():
    s = summarize_token_usage([_u(20_000, 20_000), _u(20_000, 20_000)])  # 40K each
    assert s["avg_total_per_instance"] == 40_000
    assert s["over_target_count"] == 2
    assert s["over_soft_count"] == 0
    assert s["avg_within_target"] is False


def test_avg_within_target():
    s = summarize_token_usage([_u(10_000, 5_000), _u(8_000, 4_000)])  # 15K, 12K
    assert s["avg_within_target"] is True
    assert s["over_target_count"] == 0


def test_ignores_none_and_handles_empty():
    s = summarize_token_usage([_u(1000, 500), None, None])
    assert s["n"] == 1
    assert s["total_tokens"] == 1500
    e = summarize_token_usage([])
    assert e["n"] == 0 and e["total_tokens"] == 0 and e["avg_within_target"] is True
    e2 = summarize_token_usage([None, None])
    assert e2["n"] == 0
