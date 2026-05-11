
from abc import abstractmethod
from typing import Any, Dict
import tiktoken
from ..grad_llm import GradLLM
from ..utils import get_prompts
from abc import ABC

ESTIMATOR = tiktoken.get_encoding("cl100k_base")  # compatible con la mayoría de modelos



class StrategyV2(ABC):
    def __init__(self, llm: GradLLM):
        self.llm = llm
        self.prompts = get_prompts()
        self.encoder=tiktoken.get_encoding("cl100k_base")  # compatible con la mayoría de modelos
        self.estimated_time = 0
        self.token_per_second ={
            "standard": 100,
            "small": 200,
            "medium": 300,
            "big": 400,
        }
        self.methods={}

    def use_model(self, llm: GradLLM):
        self.llm = llm

    @abstractmethod
    def estimate(self,task,fields):
        pass

    @abstractmethod
    def execute(self, in_time=0)-> Dict[str, Any]:
        pass