from fastapi import FastAPI, HTTPException
from gensie.task import Task
from gensie.agent import GenSIEAgent
from typing import Any, Dict
import importlib
import os

app = FastAPI(title="GenSIE Agent Server")

# Global agent instance
agent: GenSIEAgent = None

def get_agent() -> GenSIEAgent:
    global agent
    if agent is None:
        # Load agent from environment or default to BasicAgent
        agent_path = os.getenv("AGENT_PATH", "gensie.baseline.BasicAgent")
        try:
            module_name, class_name = agent_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            agent = agent_class()
        except Exception as e:
            raise RuntimeError(f"Failed to load agent {agent_path}: {e}")
    return agent

@app.get("/info")
async def info():
    """Returns metadata about the participant's agent."""
    return get_agent().get_info()

@app.post("/run")
async def run_task(task: Task) -> Dict[str, Any]:
    """Executes the extraction task and returns the JSON result."""
    try:
        result = get_agent().run(task)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
