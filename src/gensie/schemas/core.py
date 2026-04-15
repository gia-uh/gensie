from typing import Any, TypedDict, Literal
from pydantic import BaseModel, ConfigDict


class ComplexityDimensions(TypedDict):
    depth: Literal[1, 2, 3, 4]
    distance: Literal[1, 2, 3, 4]
    dispersion: Literal[1, 2, 3, 4]
    rigidity: Literal[1, 2, 3, 4]
    grounding: Literal[1, 2, 3]


class ComplexityMetadata(BaseModel):
    overall_level: Literal["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"]
    dimensions: ComplexityDimensions


def complexity(overall_level: str, dimensions: dict[str, int]):
    """
    Decorator to annotate a GenSIESchema with its complexity profile.
    """

    def decorator(cls):
        metadata = ComplexityMetadata(
            overall_level=overall_level,
            dimensions=dimensions,  # type: ignore
        )
        setattr(cls, "__complexity__", metadata)
        return cls

    return decorator


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
        validate_assignment=True,
    )

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """
        Returns the clean JSON Schema for the task input.
        """
        return cls.model_json_schema()

    def flatten(self) -> dict[str, Any]:
        """
        Converts the instance data into a flat dot-notation dictionary.
        This is the Official Transformation Phi(J) for the metric.

        Example:
            {"event": {"location": "Madrid", "tags": ["A", "B"]}}
            ->
            {
                "event.location": "Madrid",
                "event.tags.0": "A",
                "event.tags.1": "B"
            }
        """
        return self._flatten_dict(self.model_dump(mode="json"))

    @staticmethod
    def _flatten_dict(data: dict | list | Any, parent_key: str = "") -> dict[str, Any]:
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
