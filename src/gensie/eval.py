import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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



@dataclass
class GreedyMatchResult:
    """Result of greedy bipartite matching between two lists."""

    pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    unmatched_gold: List[int] = field(default_factory=list)
    unmatched_system: List[int] = field(default_factory=list)
    total: float = 0.0
    

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

    def _greedy_match_pairs(
        self,
        gold_list: List[Any],
        system_list: List[Any],
        item_schema: Dict[str, Any],
        root_schema: Dict[str, Any],
        input_text: Optional[str] = None,
    ) -> GreedyMatchResult:
        """
        Greedy Bipartite Matching returning matched pairs and unmatched indices.
        """
        result = GreedyMatchResult()

        if not gold_list or not system_list:
            result.unmatched_gold = list(range(len(gold_list)))
            result.unmatched_system = list(range(len(system_list)))
            return result

        # Precompute similarity matrix
        matrix = []
        for g_item in gold_list:
            row = []
            for s_item in system_list:
                sim = self.score_instance(
                    g_item,
                    s_item,
                    item_schema,
                    root_schema=root_schema,
                    input_text=input_text,
                )
                row.append(sim)
            matrix.append(row)

        used_g: set = set()
        used_s: set = set()

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
                result.pairs.append((best_pair[0], best_pair[1], best_sim))
                result.total += best_sim
                used_g.add(best_pair[0])
                used_s.add(best_pair[1])
            else:
                break

        result.unmatched_gold = [i for i in range(len(gold_list)) if i not in used_g]
        result.unmatched_system = [
            j for j in range(len(system_list)) if j not in used_s
        ]
        return result

    def _greedy_match(
        self,
        gold_list: List[Any],
        system_list: List[Any],
        item_schema: Dict[str, Any],
        root_schema: Dict[str, Any],
        input_text: Optional[str] = None,
    ) -> float:
        """
        Calculates similarity sum between two lists using Greedy Bipartite Matching.
        """
        return self._greedy_match_pairs(
            gold_list,
            system_list,
            item_schema,
            root_schema,
            input_text=input_text,
        ).total


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
            return False

        return True

    def score_instance(
        self,
        gold: Any,
        system: Any,
        schema: Dict[str, Any],
        root_schema: Dict[str, Any] = None,
    ) -> float:
        """
        Calculates Total Match Score (TMS) for a single instance or object.
        Supports Jaccard Similarity for lists. Returns normalized 0-1 score.
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
                    s_val = s_flat[k]
                    total_similarity += self.score_instance(
                        g_val, s_val, field_schema, root_schema=root_schema
                    )

            # Normalize by max possible keys (Precision/Recall blend at instance level)
            return total_similarity / max(len(g_flat), len(s_flat))

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
