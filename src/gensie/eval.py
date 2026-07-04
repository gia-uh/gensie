import math
import re
from typing import Any, Dict, List, Optional


def flatten_json(
    data: Any, parent_key: str = "", sep: str = ".", expand_lists: bool = True
) -> Dict[str, Any]:
    """
    Official Transformation Phi(J): Flattens nested JSON into dot-notation.
    If expand_lists=False, it stops at list boundaries (preserving the list as a value).
    """
    items: Dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep, expand_lists=expand_lists))
    elif isinstance(data, list) and expand_lists:
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_json(v, new_key, sep=sep, expand_lists=expand_lists))
    else:
        items[parent_key] = data
    return items


def gap_closed(f1_system: float, f1_baseline: float) -> float:
    """Fraction of the baseline-to-perfect F1 gap that a system closes.

    gap_closed = max(0, (F1_system - F1_baseline) / (1 - F1_baseline))

    Clamped at 0 (a system worse than the baseline gets 0). If the baseline is
    already perfect, the gap is empty: a system that matches it scores 1.0,
    anything less scores 0.0.
    """
    if f1_baseline >= 1.0:
        return 1.0 if f1_system >= 1.0 else 0.0
    return max(0.0, (f1_system - f1_baseline) / (1.0 - f1_baseline))


def summarize_timing(
    elapsed_list: List[float], budget_s: float = 60.0
) -> Dict[str, Any]:
    """Summarize per-instance wall times against the soft per-instance budget.

    The submission spec treats the 60s timeout as a target *averaged over the
    test set*: a single instance may overrun if others compensate. This returns
    the average, the max, how many instances exceeded ``budget_s``, and whether
    the average stays within budget. The evaluator records these instead of
    hard-stopping a run at the budget.
    """
    n = len(elapsed_list)
    if n == 0:
        return {
            "n": 0,
            "avg_elapsed_s": 0.0,
            "max_elapsed_s": 0.0,
            "over_budget_count": 0,
            "budget_s": budget_s,
            "avg_within_budget": True,
        }
    avg = sum(elapsed_list) / n
    return {
        "n": n,
        "avg_elapsed_s": avg,
        "max_elapsed_s": max(elapsed_list),
        "over_budget_count": sum(1 for t in elapsed_list if t > budget_s),
        "budget_s": budget_s,
        "avg_within_budget": avg <= budget_s,
    }


def summarize_token_usage(
    per_instance: List[Optional[Dict[str, int]]],
    target: int = 32000,
    soft_multiple: float = 2.0,
) -> Dict[str, Any]:
    """Summarize per-instance token usage against the soft 32K-average budget.

    ``per_instance`` items are ``{input, output, total, calls}`` dicts (or ``None``
    for instances with no usage record). Mirrors :func:`summarize_timing`: the
    32K target is an average over the test set; an instance may reach
    ``target * soft_multiple`` (64K) if others compensate; nothing is hard-stopped.
    """
    recs = [r for r in per_instance if r is not None]
    n = len(recs)
    if n == 0:
        return {
            "n": 0,
            "total_input": 0,
            "total_output": 0,
            "total_tokens": 0,
            "avg_total_per_instance": 0.0,
            "max_total": 0,
            "over_target_count": 0,
            "over_soft_count": 0,
            "calls_total": 0,
            "target": target,
            "avg_within_target": True,
        }
    totals = [int(r["total"]) for r in recs]
    total_input = sum(int(r["input"]) for r in recs)
    total_output = sum(int(r["output"]) for r in recs)
    grand_total = sum(totals)
    avg = grand_total / n
    soft = target * soft_multiple
    return {
        "n": n,
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": grand_total,
        "avg_total_per_instance": avg,
        "max_total": max(totals),
        "over_target_count": sum(1 for t in totals if t > target),
        "over_soft_count": sum(1 for t in totals if t > soft),
        "calls_total": sum(int(r["calls"]) for r in recs),
        "target": target,
        "avg_within_target": avg <= target,
    }


def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))


def magnitude(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    m1 = magnitude(v1)
    m2 = magnitude(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot_product(v1, v2) / (m1 * m2)


class Evaluator:
    """
    Implements official GenSIE metrics: Flattened Schema Scoring.
    Note: Requires 'fastembed' (install via 'pip install gensie[eval]' or 'uv sync --group dev').
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding

            self.model = TextEmbedding(model_name=model_name)
        except ImportError:
            self.model = None
        self.alpha = 0.7  # Weight for semantic similarity

    def lexical_similarity(self, s1: str, s2: str) -> float:
        """Normalized token overlap score."""

        def tokenize(text):
            return set(re.findall(r"\w+", str(text).lower()))

        tokens1 = tokenize(s1)
        tokens2 = tokenize(s2)
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)

    def semantic_similarity(self, s1: str, s2: str) -> float:
        """Cosine similarity of embeddings using fastembed. Falls back to lexical if missing."""
        if self.model is None:
            return self.lexical_similarity(s1, s2)

        embeddings = list(self.model.embed([str(s1), str(s2)]))
        sim = cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
        return float(max(0.0, sim))

    def resolve_ref(self, schema: Dict[str, Any], ref_path: str) -> Dict[str, Any]:
        """Resolves $defs references in JSON Schema."""
        if not ref_path.startswith("#/"):
            return {}
        parts = ref_path.split("/")
        curr = schema
        for p in parts[1:]:
            curr = curr.get(p, {})
        return curr

    def _greedy_match(
        self,
        gold_list: List[Any],
        system_list: List[Any],
        item_schema: Dict[str, Any],
        root_schema: Dict[str, Any],
    ) -> float:
        """
        Calculates similarity sum between two lists using Greedy Bipartite Matching.
        """
        if not gold_list or not system_list:
            return 0.0

        # Precompute similarity matrix
        matrix = []
        for g_item in gold_list:
            row = []
            for s_item in system_list:
                # Recursive score for nested items
                sim = self.score_instance(
                    g_item,
                    s_item,
                    item_schema,
                    root_schema=root_schema,
                    _normalize=True,
                )
                row.append(sim)
            matrix.append(row)

        matches = 0.0
        used_g = set()
        used_s = set()

        # Find best matches greedily
        while len(used_g) < len(gold_list) and len(used_s) < len(system_list):
            best_sim = -1.0
            best_pair = (-1, -1)

            for i in range(len(gold_list)):
                if i in used_g:
                    continue
                for j in range(len(system_list)):
                    if j in used_s:
                        continue
                    if matrix[i][j] > best_sim:
                        best_sim = matrix[i][j]
                        best_pair = (i, j)

            if best_sim >= 0:
                matches += best_sim
                used_g.add(best_pair[0])
                used_s.add(best_pair[1])
            else:
                break

        return matches

    def compute_value_similarity(self, g_val: Any, s_val: Any, is_rigid: bool) -> float:
        """
        Scoring logic based on data type.
        """
        # Null / hallucination check (spec "Case C"): a null is correct only
        # against another null; null-vs-value scores 0 regardless of field type.
        if g_val is None or s_val is None:
            return 1.0 if (g_val is None and s_val is None) else 0.0

        if g_val == s_val:
            return 1.0

        if is_rigid:
            # Numbers, Dates, Booleans, Enums require exact match
            return 0.0

        # Free Text: Hybrid semantic/lexical
        sem = self.semantic_similarity(g_val, s_val)
        lex = self.lexical_similarity(g_val, s_val)
        return self.alpha * sem + (1 - self.alpha) * lex

    def get_field_type_info(
        self, schema: Dict[str, Any], key: str, root_schema: Dict[str, Any]
    ) -> bool:
        """
        Determines if a field is rigid (Enum, Bool, Number) or free-text (String).
        Performs recursive lookup on dot-notation keys.
        """
        if not schema:
            return True

        parts = [p for p in key.split(".") if p]
        curr = schema

        for p in parts:
            if curr.get("type") == "array":
                curr = curr.get("items", {})
            elif curr.get("type") == "object":
                curr = curr.get("properties", {}).get(p, {})

            # Resolve refs
            if "$ref" in curr:
                curr = self.resolve_ref(root_schema, curr["$ref"])

            if not curr:
                return True

        # Resolve final ref if any
        if "$ref" in curr:
            curr = self.resolve_ref(root_schema, curr["$ref"])

        # Deciding rigidity
        if "enum" in curr:
            return True
        if curr.get("type") in ["number", "integer", "boolean"]:
            return True
        if curr.get("type") == "string":
            # Dates are rigid (spec "Case A"): formatted date/time strings need
            # an exact match, not partial semantic credit.
            return curr.get("format") in ("date", "date-time", "time")

        return True

    def score_instance(
        self,
        gold: Any,
        system: Any,
        schema: Dict[str, Any],
        root_schema: Dict[str, Any] = None,
        _normalize: bool = False,
    ) -> float:
        """
        Calculates Total Match Score (TMS) for a single instance or object.
        Supports Jaccard Similarity for lists. Returns TPS unless _normalize=True.
        """
        if root_schema is None:
            root_schema = schema

        # Handle simple types (Base case)
        if not isinstance(gold, (dict, list)):
            is_rigid = self.get_field_type_info(schema, "", root_schema)
            return self.compute_value_similarity(gold, system, is_rigid)

        # Handle Objects
        if isinstance(gold, dict):
            if not isinstance(system, dict):
                return 0.0

            # Shallow flatten to get top-level keys
            g_flat = flatten_json(gold, expand_lists=False)
            s_flat = flatten_json(system, expand_lists=False)

            if not g_flat:
                return 1.0 if not s_flat else 0.0

            total_similarity = 0.0
            properties = schema.get("properties", {}) if schema else {}

            for k, g_val in g_flat.items():
                # Get schema for this specific field
                field_schema = properties.get(k, {})
                if "$ref" in field_schema:
                    field_schema = self.resolve_ref(root_schema, field_schema["$ref"])

                if k in s_flat:
                    total_similarity += self.score_instance(
                        g_val,
                        s_flat[k],
                        field_schema,
                        root_schema=root_schema,
                        _normalize=_normalize,
                    )

            if _normalize:
                return total_similarity / max(len(g_flat), len(s_flat))
            return total_similarity

        # Handle Lists (Bipartite Matching)
        if isinstance(gold, list):
            if not isinstance(system, list):
                return 0.0
            if not gold:
                return 1.0 if not system else 0.0

            item_schema = schema.get("items", {}) if schema else {}
            if "$ref" in item_schema:
                item_schema = self.resolve_ref(root_schema, item_schema["$ref"])

            matches = self._greedy_match(
                gold, system, item_schema, root_schema=root_schema
            )
            # IoU (Jaccard) for lists
            denom = len(gold) + len(system) - matches
            return matches / denom if denom > 0 else 0.0

        return 0.0

    def calculate_metrics(
        self, tps_list: List[float], gold_counts: List[int], system_counts: List[int]
    ) -> Dict[str, float]:
        """Calculates Micro-F1 across a set of instances."""
        total_tps = sum(tps_list)
        total_g = sum(gold_counts)
        total_s = sum(system_counts)

        precision = total_tps / total_s if total_s > 0 else 0.0
        recall = total_tps / total_g if total_g > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}
