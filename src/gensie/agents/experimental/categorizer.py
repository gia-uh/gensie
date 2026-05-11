from typing import List


def _resolve_type(path, schema):
    steps = path.split("/")[1:]
    out = schema
    for step in steps:
        out = out.get(step)
    return out


def get_types(field, schema) -> set:
    types = set()
    type_field = schema["properties"][field].get("type", "unknown")
    if type_field == "unknown":
        if "anyOf" in schema["properties"][field]:
            types.add("anyOf")
            for type_anyofs in schema["properties"][field]["anyOf"]:
                type_field = type_anyofs.get("type", "unknown")
                types.add(type_field)
        elif "$ref" in schema["properties"][field]:
            types.add("$ref")
            path_to_type = schema["properties"][field]["$ref"]
            schema_type = _resolve_type(path_to_type, schema)

            if "type" in schema_type:
                types.add(schema_type["type"])
            else:
                types.add(type_field)
        else:
            types.add(type_field)

    elif type_field == "array":
        types.add("array")
        items = schema["properties"][field].get("items", {})
        type_items = items.get("type", "unknown")
        if type_items == "unknown":
            if "anyOf" in items:
                types.add("array.anyOf")
                for type_anyofs in items["anyOf"]:
                    type_field = type_anyofs.get("type", "unknown")
                    if type_field == "unknown":
                        types.add(type_field)
                    else:
                        types.add(f"array.{type_field}")
            elif "$ref" in items:
                types.add("array.$ref")
                path_to_type = items["$ref"]
                schema_type = _resolve_type(path_to_type, schema)

                if "type" in schema_type:
                    types.add(f"array.{schema_type['type']}")
                else:
                    types.add("array.unknown")
            else:
                types.add("array.unknown")
        else:
            types.add(f"array.{type_items}")
    else:
        types.add(type_field)

    return types

# categories: direct, categorical, fixed_entities, soft_entities, complex, unknown
def classify_types(types: set) -> str:
    if "array" in types:
        if "array.object" in types:
            return "complex"
        if "array.$ref" in types:
            return "fixed_entities"
        if "array.string" in types:
            return "soft_entities"

        return "unknown"

    if "string" in types or "number" in types or "integer" in types or "boolean" in types or "null" in types:
        if "$ref" in types:
            return "categorical"
        return "direct"
    else:
        return "complex"

def _body_to_ts(body: dict, defs: dict, indent: int = 2) -> str:
    if "$ref" in body:
        def_name = body["$ref"].split("/")[-1]
        return _def_to_ts(defs.get(def_name, {}), defs, indent)

    if "anyOf" in body:
        return " | ".join(_body_to_ts(v, defs, indent) for v in body["anyOf"])

    t = body.get("type", "unknown")

    if t == "array":
        items = body.get("items", {})
        return f"Array<{_body_to_ts(items, defs, indent)}>"

    if t == "object":
        props = body.get("properties", {})
        if not props:
            return "object"
        pad = " " * indent
        sub_lines = []
        for k, v in props.items():
            ts_type = _body_to_ts(v, defs, indent + 2)
            desc = v.get("description", "")
            comment = f"  // {desc}" if desc else ""
            sub_lines.append(f"{pad}  {k}: {ts_type},{comment}")
        return "{\n" + "\n".join(sub_lines) + "\n" + pad + "}"

    if t == "null":
        return "null"

    return t


def _def_to_ts(def_body: dict, defs: dict, indent: int = 2) -> str:
    if "enum" in def_body:
        return " | ".join(f'"{v}"' for v in def_body["enum"])
    return _body_to_ts(def_body, defs, indent)

def get_fields_descriptions(schema, fields: list[str]) -> dict:
    props = schema.get("properties", {})
    return {field: props[field].get("description", "") for field in props if field in fields}

def _body_to_plain(body: dict, defs: dict, indent: int = 0) -> str:
    """Recursively describe a schema body as plain text."""
    if "$ref" in body:
        def_name = body["$ref"].split("/")[-1]
        return _body_to_plain(defs.get(def_name, {}), defs, indent)

    if "anyOf" in body:
        parts = []
        for v in body["anyOf"]:
            if v.get("type") == "null":
                parts.append("null")
            else:
                parts.append(_body_to_plain(v, defs, indent))
        return " or ".join(parts)

    if "enum" in body:
        return "one of: " + ", ".join(f'"{v}"' for v in body["enum"])

    t = body.get("type", "unknown")

    if t == "array":
        items = body.get("items", {})
        inner = _body_to_plain(items, defs, indent + 2)
        if "\n" in inner:
            pad = " " * indent
            return f"list of objects:\n{inner}"
        return f"list of {inner}"

    if t == "object":
        props = body.get("properties", {})
        if not props:
            return "object"
        pad = " " * (indent + 2)
        lines = []
        for k, v in props.items():
            desc = v.get("description", "")
            desc_str = f" ({desc})" if desc else ""
            val_str = _body_to_plain(v, defs, indent + 2)
            lines.append(f"{pad}- {k}{desc_str}: {val_str}")
        return "\n".join(lines)

    if t == "null":
        return "null"

    return t


def get_schema_to_field_plain(fields: list[str], schema) -> str:
    """Return a plain-text description of the given fields, without TypeScript syntax."""
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})
    lines = []

    schema_desc = schema.get("description", "").split("Complexity")[0].strip()
    if schema_desc:
        lines.append(schema_desc)
        lines.append("")

    for field in fields:
        if field not in props:
            print(f"Field {field} not found in schema")
            continue
        body = props[field]
        desc = body.get("description", "")
        desc_str = f": {desc}" if desc else ""
        val_str = _body_to_plain(body, defs, indent=2)
        if "\n" in val_str:
            lines.append(f"- {field}{desc_str}")
            lines.append(val_str)
        else:
            lines.append(f"- {field}{desc_str} → {val_str}")

    return "\n".join(lines)

def fit_schema_to_fields(fields: List[str], schema) -> str:
    """Return a TypeScript-like type string for the given fields, for use in LLM prompts."""
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})
    lines = []
    for field in fields:
        if field not in props:
            print(f"Field {field} not found in schema")
            continue
        body = props[field]
        ts_type = _body_to_ts(body, defs, indent=2)
        desc = body.get("description", "")
        # desc = desc.split("Complexity")[0].strip()  # Remove complexity info from description
        comment = f"  // {desc}" if desc else ""
        lines.append(f"  {field}: {ts_type},{comment}")

    schema_desc = schema.get("description", "")
    schema_desc = schema_desc.split("Complexity")[0].strip()  # Remove complexity info from description
    header = f"// {schema_desc}\n" if schema_desc else ""
    return header + "{\n" + "\n".join(lines) + "\n}"





