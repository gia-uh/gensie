import copy
from typing import Any, Dict


def _normalize_schema_for_strict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI strict mode requires every key in 'properties' to appear in 'required'.
    This function makes a deep copy of the schema and adds missing property keys
    to 'required' for every object node, including those inside '$defs'.
    """
    schema = copy.deepcopy(schema)

    def _fix_node(node: Any) -> None:
        if not isinstance(node, dict):
            return
        # $ref cannot have sibling keywords in strict mode — keep only $ref
        if "$ref" in node and len(node) > 1:
            ref_value = node["$ref"]
            node.clear()
            node["$ref"] = ref_value
            return  # nothing else to process on this node
        if node.get("type") == "object" and "properties" in node:
            all_keys = list(node["properties"].keys())
            node["required"] = all_keys
            node.setdefault("additionalProperties", False)
        # Recurse into all nested values
        for value in node.values():
            if isinstance(value, dict):
                _fix_node(value)
            elif isinstance(value, list):
                for item in value:
                    _fix_node(item)

    _fix_node(schema)
    # Also fix definitions
    for def_node in schema.get("$defs", {}).values():
        _fix_node(def_node)
    return schema
