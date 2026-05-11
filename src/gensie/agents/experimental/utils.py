from __future__ import annotations

import copy
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import yaml

if TYPE_CHECKING:
    from gensie.task import Task


def get_prompts():
    "read yaml file in same folder"
    with open(os.path.join(os.path.dirname(__file__),"prompts.yaml"), "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_labels_for_fields(schema: Dict[str, Any], fields: List[str]) -> Dict[str, List[str]]:
    """Return enum values keyed by field name for the given fields in the schema."""
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})
    result: Dict[str, List[str]] = {}

    for field in fields:
        body = props.get(field, {})
        enum_refs: set = set()

        ref = body.get("$ref", "")
        if ref:
            enum_refs.add(ref.split("/")[-1])
        for variant in body.get("anyOf", []):
            if isinstance(variant, dict) and "$ref" in variant:
                enum_refs.add(variant["$ref"].split("/")[-1])
        if body.get("type") == "array":
            items = body.get("items", {})
            ref = items.get("$ref", "")
            if ref:
                ref_name = ref.split("/")[-1]
                enum_refs.add(ref_name)
                # Descend into object defs (e.g. Entity) to find nested enum refs (e.g. EntityType)
                ref_body = defs.get(ref_name, {})
                label_prop = ref_body.get("properties", {}).get("label", {})
                if "$ref" in label_prop:
                    enum_refs.add(label_prop["$ref"].split("/")[-1])
            for variant in items.get("anyOf", []):
                if isinstance(variant, dict) and "$ref" in variant:
                    ref_name = variant["$ref"].split("/")[-1]
                    enum_refs.add(ref_name)
                    ref_body = defs.get(ref_name, {})
                    label_prop = ref_body.get("properties", {}).get("label", {})
                    if "$ref" in label_prop:
                        enum_refs.add(label_prop["$ref"].split("/")[-1])

        labels = [
            v
            for def_name, def_body in defs.items()
            if def_name in enum_refs and "enum" in def_body
            for v in def_body["enum"]
        ]
        if labels:
            result[field] = labels

    return result


def check_in_text(value: str|dict|list|float, input_text: str) -> list[str]:
    """
    Utility method to check if a given value appears in the input text.
    This can be used to provide feedback in the Grad evaluation detail table.
    """
    values_non_in_text = []
    if isinstance(value, str):
        if str(value) not in input_text:
            values_non_in_text.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            values_in_dict = check_in_text(v, input_text)
            if  values_in_dict:
                values_non_in_text.extend(values_in_dict)
    elif isinstance(value, list):
        for v in value:
            if str(v) not in input_text:
                values_in_lsit = check_in_text(v, input_text)
                if values_in_lsit:
                    values_non_in_text.extend(values_in_lsit)
    elif isinstance(value, float):
        if str(value) not in input_text:
            values_non_in_text.append(value)
    else:
        if str(value) not in input_text:
            values_non_in_text.append(value)

    return values_non_in_text



def _sort_entities_by_position( result: Dict[str, Any], input_text: str) -> Dict[str, Any]:
    """Sort entities list by order of first appearance in input_text."""
    entities = result.get("entities")
    if not isinstance(entities, list):
        return result
    def _pos(entity: Dict) -> int:
        text = entity.get("text", "")
        idx = input_text.find(text)
        return idx if idx >= 0 else len(input_text)
    result["entities"] = sorted(entities, key=_pos)
    return result


def new_entities(prev_result, result):
    if "entities" in prev_result and "entities" in result:
        prev_entities = [(e.get("text"), e.get("label")) for e in prev_result["entities"]]
        new_entities = [(e.get("text"), e.get("label")) for e in result["entities"]]

        return set([t for t,l in new_entities]) - set([t for t,l in prev_entities])
    return set()



def get_all_labels(task: Task) -> str:
    """Return enum values only from $defs referenced by a field named 'label' anywhere in the schema.
    This avoids sending severity/frequency enums to the entity tagger.
    """
    schema = task.target_schema
    defs = schema.get("$defs", {})

    # Walk schema to find all $ref values used by any property named 'label'
    label_refs: set = set()
    def _find_label_refs(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties", {})
            for prop_name, prop_body in props.items():
                if prop_name == "label" and "$ref" in prop_body:
                    label_refs.add(prop_body["$ref"].split("/")[-1])
            for v in node.values():
                _find_label_refs(v)
        elif isinstance(node, list):
            for item in node:
                _find_label_refs(item)
    _find_label_refs(schema)

    return ", ".join(
        v
        for def_name, body in defs.items()
        if def_name in label_refs and "enum" in body
        for v in body["enum"]
    )


def get_param_names(schema: Dict[str, Any]) -> List[str]:
    return list(schema.get("properties", {}).keys())


def get_complexity(task: Task) -> int:
    """Ejemplo de función para determinar la complejidad de una task."""
    description = task.target_schema.get("description", "")
    complexity = description.split("Complexity:")[-1].strip().split()[0].strip("L")
    return int(complexity) if complexity.isdigit() else 0

def _get_field_descriptions(task: Task) -> str:
    """Build a human-readable summary of each schema field's description.
    Resolves $ref to include enum valid values, and descends into array item schemas.
    """
    defs = task.target_schema.get("$defs", {})
    lines = []

    def _enum_hint(body: Dict[str, Any]) -> str:
        ref = body.get("$ref", "")
        if ref:
            def_name = ref.split("/")[-1]
            enum_vals = defs.get(def_name, {}).get("enum", [])
            if enum_vals:
                return f"Valid values: {', '.join(str(v) for v in enum_vals)}"
        return ""

    def _field_line(name: str, body: Dict[str, Any]) -> str | None:
        parts = []
        desc = body.get("description", "")
        if desc:
            parts.append(desc)
        hint = _enum_hint(body)
        if hint:
            parts.append(hint)
        return f"- {name}: {'. '.join(parts)}" if parts else None

    for field, body in task.target_schema.get("properties", {}).items():
        line = _field_line(field, body)
        if line:
            lines.append(line)
        # Descend into array items that reference a $def with its own properties
        items = body.get("items", {})
        if isinstance(items, dict) and "$ref" in items:
            def_name = items["$ref"].split("/")[-1]
            def_body = defs.get(def_name, {})
            for sub_field, sub_body in def_body.get("properties", {}).items():
                sub_line = _field_line(f"  {field}[].{sub_field}", sub_body)
                if sub_line:
                    lines.append(sub_line)

    return "\n".join(lines)



def _strip_tag_markup(text: str) -> str:
    """Remove inline tagger annotations, keeping only the mention text."""
    # [mention]text[/mention](TYPE) → text
    text = re.sub(r'\[mention\](.*?)\[/mention\]\([^)]+\)', r'\1', text)
    # [text](TYPE) → text  (TYPE is uppercase, avoids collisions with URLs)
    text = re.sub(r'\[([^\]]+)\]\([A-Z_]+\)', r'\1', text)
    return text


def _decompose_schema(schema: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Split a (strict-mode) schema into one minimal sub-schema per top-level field.

    Each sub-schema wraps exactly one property and inherits the parent's $defs
    so that $ref references (e.g. shared enum types) still resolve correctly.

    Returns a list of (field_name, sub_schema) tuples in property-definition
    order.  Used to enable per-field extraction strategies.

    Example
    -------
    Schema with fields {name, age, tags} →
        [("name", {…only name…}),
         ("age",  {…only age…}),
         ("tags", {…only tags…})]
    """
    properties = schema.get("properties", {})
    defs = schema.get("$defs", {})
    result = []
    for field_name, field_def in properties.items():
        sub: Dict[str, Any] = {
            # Title helps the LLM understand what it is extracting
            "title": f"{schema.get('title', 'Extraction')}_{field_name}",
            "type": "object",
            "properties": {field_name: copy.deepcopy(field_def)},
            # OpenAI strict mode requires all properties to be in required
            "required": [field_name],
            # Prevent the model from adding extra keys to the response object
            "additionalProperties": False,
        }
        if defs:
            # Preserve shared type definitions so $ref nodes resolve correctly
            sub["$defs"] = copy.deepcopy(defs)
        result.append((field_name, sub))
    return result


def _is_array_field(body: Dict[str, Any]) -> bool:
    """Return True if a field body resolves to an array type (including nullable arrays)."""
    if body.get("type") == "array":
        return True
    for variant in body.get("anyOf", []):
        if isinstance(variant, dict) and variant.get("type") == "array":
            return True
    return False




def _build_combined_schema(
    fields: List[Tuple[str, Dict[str, Any]]],
    base_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a subset of single-field sub-schemas (from _decompose_schema) back into one schema."""
    combined: Dict[str, Any] = {
        "title": base_schema.get("title", "Extraction"),
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    defs = base_schema.get("$defs", {})
    if defs:
        combined["$defs"] = copy.deepcopy(defs)
    for field_name, sub_schema in fields:
        combined["properties"][field_name] = copy.deepcopy(sub_schema["properties"][field_name])
        combined["required"].append(field_name)
    return combined




def _is_ner_schema( schema: Dict[str, Any]) -> bool:
    """Return True if the schema is a flat NER schema: top-level output is a list of {text, label} entities.

    This pattern causes problems with the tag→extract pipeline because the tagger
    invents sub-types not in the enum, and the extractor then copies the type tag
    into the 'text' field instead of the verbatim span.
    """
    props = schema.get("properties", {})
    if len(props) != 1:
        return False
    field_body = next(iter(props.values()))
    if field_body.get("type") != "array":
        return False
    items = field_body.get("items", {})
    ref = items.get("$ref", "")
    if not ref:
        return False
    def_name = ref.split("/")[-1]
    def_body = schema.get("$defs", {}).get(def_name, {})
    sub_props = set(def_body.get("properties", {}).keys())
    return {"text", "label"} <= sub_props