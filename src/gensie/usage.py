"""Token-usage helpers: read the inference server's usage log, parse the
``X-GenSIE-Token-Usage`` response header, and tally usage agent-side.

See ``plans/2026-05-12-token-budget-enforcement-spec.md`` in the gensie-internal
repo.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def usage_rows(log_path: Path, api_key: Optional[str]) -> List[Dict[str, Any]]:
    """Read the JSONL usage log and return rows whose ``api_key`` matches.

    ``api_key=None`` returns every well-formed row. Missing file or malformed
    lines are skipped silently — the log is append-only and best-effort.
    """
    path = Path(log_path)
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if api_key is not None and row.get("api_key") != api_key:
            continue
        rows.append(row)
    return rows


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Sum ``prompt_tokens`` / ``completion_tokens`` over usage-log rows."""
    inp = sum(int(r.get("prompt_tokens", 0) or 0) for r in rows)
    out = sum(int(r.get("completion_tokens", 0) or 0) for r in rows)
    return {"input": inp, "output": out, "total": inp + out, "calls": len(rows)}


def parse_usage_header(value: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse an ``X-GenSIE-Token-Usage`` header value into {input, output, total, calls}.

    Returns ``None`` if absent or unparseable. ``total`` defaults to input+output.
    """
    if not value:
        return None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    inp = int(data.get("input_tokens", 0) or 0)
    out = int(data.get("output_tokens", 0) or 0)
    total = data.get("total_tokens")
    total = int(total) if isinstance(total, (int, float)) else inp + out
    calls = int(data.get("calls", 0) or 0)
    return {"input": inp, "output": out, "total": total, "calls": calls}


def usage_disagrees(total_a: int, total_b: int) -> bool:
    """True if two token totals differ by more than max(10% of the larger, 1000)."""
    return abs(total_a - total_b) > max(0.1 * max(total_a, total_b), 1000)


class UsageTracker:
    """Accumulates OpenAI ``response.usage`` across calls within one task.

    Agents reset it at the start of ``run()`` and ``add(response.usage)`` after
    each completion call; ``header_value()`` is the JSON for the
    ``X-GenSIE-Token-Usage`` response header.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.input = 0
        self.output = 0
        self.calls = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        if usage is None:
            return
        get = (
            usage.get
            if isinstance(usage, dict)
            else (lambda k: getattr(usage, k, None))
        )
        self.input += int(get("prompt_tokens") or 0)
        self.output += int(get("completion_tokens") or 0)

    def snapshot(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input,
            "output_tokens": self.output,
            "total_tokens": self.input + self.output,
            "calls": self.calls,
        }

    def header_value(self) -> str:
        return json.dumps(self.snapshot())
