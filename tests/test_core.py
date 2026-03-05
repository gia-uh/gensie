from gensie.task import Task
from gensie.schemas.core import GenSIESchema
from pydantic import Field


class SimpleSchema(GenSIESchema):
    name: str = Field(..., description="A simple name")


def test_task_creation():
    """Verify that a Task can be created from a schema class."""
    task = Task.create(
        text="My name is Gemini.", schema_class=SimpleSchema, task_id="test_001"
    )
    assert task.id == "test_001"
    assert "name" in task.target_schema["properties"]


def test_schema_flattening():
    """Verify the flattening logic for scoring."""
    obj = SimpleSchema(name="Gemini")
    flattened = obj.flatten()
    assert flattened == {"name": "Gemini"}
