import json
from pathlib import Path
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

from gensie.schemas.core import GenSIESchema


class Task(BaseModel):
    """
    Represents a single instance in the GenSIE dataset.
    This object is what participants will receive (minus the 'output' in the test set).
    """

    id: str = Field(..., description="Unique identifier (e.g., 'legal_001')")
    input_text: str = Field(..., description="The raw context text")
    instruction: str = Field(..., description="The natural language task description")
    target_schema: Dict[str, Any] = Field(
        ..., description="The full JSON Schema definition"
    )
    output: Optional[Dict[str, Any]] = Field(
        None, description="The extracted JSON (Silver/Gold). None for Test set."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Provenance info (source url, domain, split)"
    )

    @classmethod
    def create(
        cls,
        text: str,
        schema_class: Type[GenSIESchema],
        output: Optional[Dict[str, Any]] = None,
        task_id: str = "",
        instruction: str = "Extract the structured data from the text following the provided schema.",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Task":
        """
        Factory method to build a Task directly from a Pydantic Schema class.
        This automatically serializes the Pydantic class into JSON Schema.
        """
        return cls(
            id=task_id,
            input_text=text,
            instruction=instruction,
            target_schema=schema_class.get_schema(),  # Serializes the class to dict
            output=output,
            metadata=metadata or {},
        )

    def save(self, directory: Path) -> Path:
        """Saves the task to a JSON file named {id}.json in the given directory."""
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"{self.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2, exclude_none=True))
        return file_path

    @classmethod
    def load(cls, path: Path) -> "Task":
        """Loads a Task from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def get_input_prompt(self) -> str:
        """
        Helper for participants: Generates a standard formatted prompt
        combining instruction, schema, and text.
        """
        return (
            f"{self.instruction}\n\n"
            f"SCHEMA:\n{json.dumps(self.target_schema, indent=2)}\n\n"
            f"TEXT:\n{self.input_text}"
        )
