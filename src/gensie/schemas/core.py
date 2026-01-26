import json
from typing import Any, Type
from pydantic import BaseModel, ConfigDict

class GenSIESchema(BaseModel):
    """
    The base class for all GenSIE extraction schemas.

    This class enforces strict configuration (to ensure clean JSON schemas)
    and provides the standard 'flattening' logic used for scoring.
    """

    # Strict config prevents extra fields and ensures Enums are serialized correctly
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        populate_by_name=True,
        validate_assignment=True
    )

    @classmethod
    def get_json_schema(cls) -> dict[str, Any]:
        """
        Returns the clean JSON Schema for the task input.
        We strip out Pydantic-specific titles/versions to keep the prompt clean.
        """
        schema = cls.model_json_schema()
        # Cleanup: Remove 'title' from definitions to save tokens and reduce bias
        # (The LLM should focus on field names/descriptions, not the class name)
        if "title" in schema:
            del schema["title"]
        return schema

    def flatten(self) -> dict[str, Any]:
        """
        Converts the instance data into a flat dot-notation dictionary.
        This is the Official Transformation \Phi(J) for the metric.

        Example:
            {"event": {"location": "Madrid", "tags": ["A", "B"]}}
            ->
            {
                "event.location": "Madrid",
                "event.tags.0": "A",
                "event.tags.1": "B"
            }
        """
        return self._flatten_dict(self.model_dump(mode='json'))

    @staticmethod
    def _flatten_dict(data: dict | list | Any, parent_key: str = '') -> dict[str, Any]:
        items: dict[str, Any] = {}

        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.update(GenSIESchema._flatten_dict(v, new_key))
                else:
                    items[new_key] = v

        elif isinstance(data, list):
            for i, v in enumerate(data):
                # Lists are flattened by index to preserve order in scoring
                new_key = f"{parent_key}.{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.update(GenSIESchema._flatten_dict(v, new_key))
                else:
                    items[new_key] = v

        else:
            # Primitive values (should be caught by the parent recursion, but safety check)
            items[parent_key] = data

        return items

    @classmethod
    def load(cls, data: dict) -> "GenSIESchema":
        """Safe loader that validates the input JSON against the schema."""
        return cls.model_validate(data)
