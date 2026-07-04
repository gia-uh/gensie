import json

from gensie.usage import (
    UsageTracker,
    aggregate_rows,
    parse_usage_header,
    usage_disagrees,
    usage_rows,
)


def test_usage_rows_filters_by_key_and_skips_garbage(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "t1",
                        "api_key": "K1",
                        "model": "m",
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                    }
                ),
                "not json",
                json.dumps(
                    {
                        "ts": "t2",
                        "api_key": "K2",
                        "model": "m",
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                    }
                ),
                json.dumps(
                    {
                        "ts": "t3",
                        "api_key": "K1",
                        "model": "m",
                        "prompt_tokens": 200,
                        "completion_tokens": 80,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    rows = usage_rows(log, "K1")
    assert [r["prompt_tokens"] for r in rows] == [100, 200]
    assert len(usage_rows(log, None)) == 3
    assert usage_rows(tmp_path / "nope.jsonl", "K1") == []


def test_aggregate_rows():
    rows = [
        {"prompt_tokens": 100, "completion_tokens": 50},
        {"prompt_tokens": 200, "completion_tokens": 80},
    ]
    assert aggregate_rows(rows) == {
        "input": 300,
        "output": 130,
        "total": 430,
        "calls": 2,
    }
    assert aggregate_rows([]) == {"input": 0, "output": 0, "total": 0, "calls": 0}


def test_parse_usage_header():
    assert parse_usage_header(None) is None
    assert parse_usage_header("garbage") is None
    assert parse_usage_header(
        json.dumps(
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "calls": 1}
        )
    ) == {"input": 10, "output": 5, "total": 15, "calls": 1}
    assert (
        parse_usage_header(json.dumps({"input_tokens": 10, "output_tokens": 5}))[
            "total"
        ]
        == 15
    )


def test_usage_disagrees():
    assert usage_disagrees(10_000, 10_050) is False
    assert usage_disagrees(10_000, 20_000) is True
    assert usage_disagrees(500, 1_400) is False
    assert usage_disagrees(500, 1_600) is True


def test_usage_tracker_accumulates_attr_and_dict():
    t = UsageTracker()

    class U:
        prompt_tokens = 100
        completion_tokens = 40

    t.add(U())
    t.add({"prompt_tokens": 200, "completion_tokens": 60})
    t.add(None)
    assert t.snapshot() == {
        "input_tokens": 300,
        "output_tokens": 100,
        "total_tokens": 400,
        "calls": 3,
    }
    assert json.loads(t.header_value()) == t.snapshot()
    t.reset()
    assert t.snapshot() == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
