import os
import time
from typing import Any, Dict
from openai import OpenAI
from gensie.agent import GenSIEAgent
from gensie.task import Task

from gensie.agents.experimental.planner import generate_plan
from gensie.agents.experimental.grad_llm import GradLLM




class ExperimentalAgent(GenSIEAgent):
    # Hard wall-clock limit per task instance (seconds), matching competition rules
    MAX_TIME = 60

    def __init__(self):
        # OpenAI-compatible client — works with local LMStudio, vLLM, or OpenAI
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        # Set at the start of each run() call to enforce MAX_TIME across all phases
        self.initial_time: float = 0.0
        self.current_task_id: str = ""  # Set at the start of run() for log tracing
        self.model: str = ""  # Set at the start of run() for log tracing
        self.llm = GradLLM(self.client, None)


    def run(self, task: Task, model: str) -> Dict[str, Any]:
        self.initial_time = time.time()
        self.current_task_id = task.id
        self.model = model

        self.llm.model = model

        # 1. generate plan
        plan = generate_plan(task)
        print("Time generating plan:", time.time() - self.initial_time)

        print("Processing plan for: ", ",".join([f"{step.get('category')}({step.get('fields', [])})" for step in plan]))
        # TEMP: test direct strategy on all tasks
        # 2. execute strategies
        results = []

        for step in plan:
            print("PROCESSING", step.get("category"), step.get("fields", []))
            init = time.time()
            strategy = step.get("strategy", None)
            strategy.use_model(self.llm)
            result = strategy.execute(in_time=self.MAX_TIME - (time.time() - self.initial_time))
            results.append(result)
            print(f"Step {step.get('category')}, Execution Time:", time.time() - init)

        # 3. join results
        init = time.time()
        final_result = {}
        for res in results:
            final_result.update(res)

        print("Time joining results:", time.time() - init)
        return final_result
