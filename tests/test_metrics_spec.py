"""Tests that pin the evaluator to the documented spec (docs/description.md).

Covers the null-handling table, date rigidity, and corpus-level Micro-F1 with
mismatched key counts. These are the parts the spec is explicit about but that
were not previously exercised by tests.
"""

from gensie.eval import Evaluator


# --- Null handling (spec: "Case C: Null Values (Hallucination Penalty)") ---


def test_gold_value_system_null_scores_zero():
    """Gold has a value, system returns null on a present key -> 0.0 (hallucinated null)."""
    evaluator = Evaluator()
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    gold = {"summary": "El paciente presenta fiebre alta y tos persistente."}
    system = {"summary": None}
    assert evaluator.score_instance(gold, system, schema) == 0.0


def test_gold_null_system_value_scores_zero():
    """Gold is null, system returns a value -> 0.0 (hallucinated value)."""
    evaluator = Evaluator()
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    gold = {"summary": None}
    system = {"summary": "Algo inventado."}
    assert evaluator.score_instance(gold, system, schema) == 0.0


def test_both_null_scores_one():
    """Gold null, system null -> 1.0 (correct null)."""
    evaluator = Evaluator()
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    assert evaluator.score_instance({"summary": None}, {"summary": None}, schema) == 1.0


# --- Date rigidity (spec: rigid types are "Numbers, Dates, Booleans, and Enum Strings") ---


def test_date_field_is_rigid_near_miss_scores_zero():
    """A string field with format=date is scored by exact match, not semantic similarity."""
    evaluator = Evaluator()
    schema = {
        "type": "object",
        "properties": {"effective_date": {"type": "string", "format": "date"}},
    }
    gold = {"effective_date": "2026-05-22"}
    system = {"effective_date": "2026-05-21"}
    assert evaluator.score_instance(gold, system, schema) == 0.0


def test_date_field_exact_match_scores_one():
    evaluator = Evaluator()
    schema = {
        "type": "object",
        "properties": {"effective_date": {"type": "string", "format": "date-time"}},
    }
    gold = {"effective_date": "2026-05-22T10:00:00"}
    system = {"effective_date": "2026-05-22T10:00:00"}
    assert evaluator.score_instance(gold, system, schema) == 1.0


# --- Corpus-level Micro-F1 (spec: "Aggregated Metrics") ---


def test_micro_f1_with_mismatched_counts():
    """Precision = TPS/|S|, Recall = TPS/|G|, F1 harmonic mean — over the whole corpus."""
    evaluator = Evaluator()
    # Two instances: instance 1 fully correct (2 keys), instance 2 has a missing
    # gold key and an extra system key.
    tps_list = [2.0, 1.0]
    gold_counts = [2, 2]  # instance 2 had 2 gold keys, 1 matched
    system_counts = [2, 2]  # instance 2 had 2 system keys, 1 matched, 1 extra wrong
    metrics = evaluator.calculate_metrics(tps_list, gold_counts, system_counts)
    # total_tps = 3, |S| = 4, |G| = 4 -> P = R = 0.75 -> F1 = 0.75
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1"] == 0.75
