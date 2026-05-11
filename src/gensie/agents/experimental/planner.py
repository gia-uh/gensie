
from gensie.agents.experimental.strategies.Categorial import Categorial
from gensie.agents.experimental.strategies.Complex import Complex
from gensie.agents.experimental.strategies.Direct import Direct
from gensie.agents.experimental.strategies.FixedEntities import FixedEntities
from gensie.agents.experimental.strategies.SoftEntities import SoftEntities

from .categorizer import classify_types, get_types

strategies ={
    "direct": Direct,
    "categorical": Categorial,
    "fixed_entities": FixedEntities,
    "soft_entities": SoftEntities,
    "complex": Complex,
}

def generate_plan(task):
    schema = task.target_schema
    # print("Required fields:", schema.get("required", []))
    fields_by_cat = {}
    for field in schema.get("properties", []):
        types = get_types(field, schema)
        cat = classify_types(types)
        fields_by_cat[cat] = fields_by_cat.get(cat, []) + [field]

    plan =[]

    for cat, fields in fields_by_cat.items():
        # print(f"Category: {cat}, Fields: {fields}")
        fields_are_required = any(f in schema.get("required", []) for f in fields)
        strategy = strategies.get(cat, Direct)(llm=None)
        strategy.estimate(task, fields)

        plan.append({
            "category": cat,
            "fields": fields,
            "priority": 1 if fields_are_required else 2,
            "estimated_time": strategy.estimated_time,
            "strategy": strategy
        })

    plan.sort(key=lambda x: x["priority"])

    return plan