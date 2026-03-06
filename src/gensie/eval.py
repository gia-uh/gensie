import numpy as np
from typing import Any, Dict, List, Set, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


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


class Evaluator:
    """
    Implements official GenSIE metrics: Flattened Schema Scoring.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
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
        """Cosine similarity of embeddings."""
        embeddings = self.model.encode([str(s1), str(s2)])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
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
        Determines if a flattened key corresponds to a rigid or free-text field
        by traversing the original JSON schema.
        Simplification: Assume strings with descriptions > 50 chars or specific 
        patterns are free text. In production, this uses full schema traversal.
        """
        # For the baseline, we'll use a heuristic or check the schema if available.
        # Rigid types: boolean, integer, number, or string with 'enum'.
        # We'll default to Rigid unless it's a long string.
        if isinstance(key, str) and ("description" in key.lower() or "summary" in key.lower()):
            return False
        return True

    def score_instance(self, gold: Dict[str, Any], system: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculates True Positive Score (TPS) for a single instance."""
        g_flat = flatten_json(gold)
        s_flat = flatten_json(system)
        
        tps = 0.0
        all_keys = set(g_flat.keys()).union(set(s_flat.keys()))
        
        for k in g_flat:
            if k in s_flat:
                # Determine if rigid based on key name/schema
                is_rigid = self.get_field_type_info(schema, k)
                tps += self.compute_value_similarity(g_flat[k], s_flat[k], is_rigid)
        
        return tps

    def calculate_metrics(self, tps_list: List[float], gold_counts: List[int], system_counts: List[int]) -> Dict[str, float]:
        """Calculates Micro-F1 across a set of instances."""
        total_tps = sum(tps_list)
        total_g = sum(gold_counts)
        total_s = sum(system_counts)
        
        precision = total_tps / total_s if total_s > 0 else 0.0
        recall = total_tps / total_g if total_g > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
