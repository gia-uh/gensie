import math
import re
from typing import Any, Dict, List


def flatten_json(data: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Official Transformation Phi(J): Flattens nested JSON into dot-notation.
    """
    items: Dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_json(v, new_key, sep=sep))
    else:
        items[parent_key] = data
    return items


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
        """Cosine similarity of embeddings using fastembed."""
        if self.model is None:
            raise ImportError(
                "Evaluation requires 'fastembed'. Install it with 'pip install fastembed'."
            )

        embeddings = list(self.model.embed([str(s1), str(s2)]))
        sim = cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
        return float(max(0.0, sim))

    def compute_value_similarity(self, g_val: Any, s_val: Any, is_rigid: bool) -> float:
        """
        Scoring logic based on data type.
        """
        if g_val == s_val:
            return 1.0

        if is_rigid:
            # Numbers, Dates, Booleans, Enums require exact match
            return 0.0

        # Free Text: Hybrid semantic/lexical
        sem = self.semantic_similarity(g_val, s_val)
        lex = self.lexical_similarity(g_val, s_val)
        return self.alpha * sem + (1 - self.alpha) * lex

    def get_field_type_info(self, schema: Dict[str, Any], key: str) -> bool:
        """
        Determines if a flattened key corresponds to a rigid or free-text field.
        """
        if isinstance(key, str) and (
            "description" in key.lower() or "summary" in key.lower()
        ):
            return False
        return True

    def score_instance(
        self, gold: Dict[str, Any], system: Dict[str, Any], schema: Dict[str, Any]
    ) -> float:
        """Calculates True Positive Score (TPS) for a single instance."""
        g_flat = flatten_json(gold)
        s_flat = flatten_json(system)

        tps = 0.0

        for k in g_flat:
            if k in s_flat:
                # Determine if rigid based on key name/schema
                is_rigid = self.get_field_type_info(schema, k)
                tps += self.compute_value_similarity(g_flat[k], s_flat[k], is_rigid)

        return tps

    def calculate_metrics(
        self, tps_list: List[float], gold_counts: List[int], system_counts: List[int]
    ) -> Dict[str, float]:
        """Calculates Micro-F1 across a set of instances."""
        total_tps = sum(tps_list)
        total_g = sum(gold_counts)
        total_s = sum(system_counts)

        precision = total_tps / total_s if total_s > 0 else 0.0
        recall = total_tps / total_g if total_g > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}
