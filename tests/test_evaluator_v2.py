import pytest
from gensie.eval import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator()


@pytest.fixture
def entities_schema():
    return {
        "$defs": {
            "Entity": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"enum": ["PERSON", "ORG", "LOC"]},
                },
            }
        },
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"$ref": "#/$defs/Entity"}}
        },
    }


def test_list_order_independence(evaluator, entities_schema):
    gold = {
        "entities": [
            {"text": "Apple", "label": "ORG"},
            {"text": "Steve Jobs", "label": "PERSON"},
        ]
    }
    # Swapped order
    system = {
        "entities": [
            {"text": "Steve Jobs", "label": "PERSON"},
            {"text": "Apple", "label": "ORG"},
        ]
    }

    score = evaluator.score_instance(gold, system, entities_schema)
    # 2 matches / (2 + 2 - 2) = 1.0
    assert score == 1.0


def test_partial_list_matching(evaluator, entities_schema):
    gold = {
        "entities": [
            {"text": "Apple", "label": "ORG"},
            {"text": "Steve Jobs", "label": "PERSON"},
        ]
    }
    # One correct, one missing, one extra (wrong)
    system = {
        "entities": [
            {"text": "Apple", "label": "ORG"},
            {"text": "Microsoft", "label": "ORG"},
        ]
    }

    # Apple matches perfectly (1.0)
    # Jaccard: 1.0 / (2 + 2 - 1.0) = 1.0 / 3.0 = 0.333 (base)
    # Higher because of cross-item semantic noise
    score = evaluator.score_instance(gold, system, entities_schema)
    assert 0.32 < score < 0.5


def test_rigid_enum_matching(evaluator, entities_schema):
    gold = {"text": "Apple", "label": "ORG"}
    system = {"text": "Apple", "label": "PERSON"}

    # Label is enum, should be rigid (score 0.0)
    # Text is string, should be semantic (score 1.0)
    # Total score = (0.0 + 1.0) / 2 = 0.5

    schema = entities_schema["$defs"]["Entity"]
    score = evaluator.score_instance(gold, system, schema, root_schema=entities_schema)
    assert score == 0.5


def test_free_text_semantic_scoring(evaluator):
    schema = {"type": "object", "properties": {"description": {"type": "string"}}}
    gold = {"description": "A fast red car."}
    system = {"description": "A quick crimson vehicle."}

    score = evaluator.score_instance(gold, system, schema)
    # Should be > 0.6 due to semantic similarity even if exact match fails
    assert score > 0.6
    assert score < 1.0


def test_empty_lists(evaluator, entities_schema):
    gold = {"entities": []}
    system = {"entities": []}
    score = evaluator.score_instance(gold, system, entities_schema)
    assert score == 1.0

    system_extra = {"entities": [{"text": "A", "label": "ORG"}]}
    score_extra = evaluator.score_instance(gold, system_extra, entities_schema)
    assert score_extra == 0.0


def test_nested_complex_matching(evaluator):
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    gold = {"tags": ["AI", "Tech"]}
    system = {"tags": ["Technology", "Artificial Intelligence"]}

    # Strings in list should use semantic matching
    score = evaluator.score_instance(gold, system, schema)
    assert score > 0.4  # Better than 0.0
