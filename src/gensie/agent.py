from abc import ABC, abstractmethod
from typing import Any, Dict
from gensie.task import Task


class GenSIEAgent(ABC):
    """
    Abstract base class for all GenSIE extraction agents.
    Participants must inherit from this class and implement the 'run' method.
    """

    @abstractmethod
    def run(self, task: Task) -> Dict[str, Any]:
        """
        Processes a single Task and returns the extracted JSON data.
        The returned dictionary must adhere to task.target_schema.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the participant and the agent.
        Expected keys: 'team_name', 'institution', 'model_name', 'description'.
        """
        pass
