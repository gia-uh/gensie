from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from gensie.task import Task


class PipelineInfo(BaseModel):
    name: str = Field(..., description="Unique name of the pipeline")
    description: str = Field(..., description="Brief technical description of the pipeline")


class ParticipantInfo(BaseModel):
    team_name: str
    institution: str
    pipelines: List[PipelineInfo]


class GenSIEAgent(ABC):
    """
    Abstract base class for all GenSIE extraction agents.
    """

    @abstractmethod
    def run(self, task: Task, model: str) -> Dict[str, Any]:
        """
        Processes a single Task and returns the extracted JSON data.
        The returned dictionary must adhere to task.target_schema.
        """
        pass


class Participant(ABC):
    """
    Base class for a competition entry.
    Participants must implement this to provide their info and their agent pipelines.
    """

    @abstractmethod
    def get_info(self) -> ParticipantInfo:
        """Returns metadata about the team and available pipelines."""
        pass

    @abstractmethod
    def get_agent(self, pipeline_name: str) -> GenSIEAgent:
        """Returns the agent instance corresponding to the pipeline name."""
        pass
