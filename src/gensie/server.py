from fastapi import FastAPI, HTTPException, Query
from gensie.task import Task
from gensie.agent import Participant
from typing import Any, Dict, Optional
import importlib
import os

app = FastAPI(title="GenSIE Agent Server")

# Global participant instance
participant: Participant = None

def get_participant() -> Participant:
    global participant
    if participant is None:
        # Load participant from environment or default to OfficialParticipant
        participant_path = os.getenv("PARTICIPANT_PATH", "gensie.baseline.OfficialParticipant")
        try:
            module_name, class_name = participant_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            participant_class = getattr(module, class_name)
            participant = participant_class()
        except Exception as e:
            raise RuntimeError(f"Failed to load participant {participant_path}: {e}")
    return participant

@app.get("/info")
async def info():
    """Returns metadata about the participant and available pipelines."""
    return get_participant().get_info()

@app.post("/run")
async def run_task(
    task: Task, 
    pipeline: str = Query("baseline", description="Name of the pipeline to execute"),
    model: Optional[str] = Query(None, description="Optional model override for the agent")
) -> Dict[str, Any]:
    """Executes the extraction task using the specified pipeline."""
    try:
        p = get_participant()
        agent = p.get_agent(pipeline)
        
        # If a model override is provided, we try to apply it
        # This assumes agents can handle dynamic model switching or are initialized per request
        # For the baseline BasicAgent, we'll re-initialize it if model changes, 
        # but a better way is to pass the model to the run method.
        # However, to keep GenSIEAgent.run signature simple, we check if the agent has a 'model' attr.
        if model and hasattr(agent, 'model'):
            # Temporary override logic: re-initialize if it's a BasicAgent or similar
            # A more robust participant implementation would handle this.
            from gensie.baseline import BasicAgent
            if isinstance(agent, BasicAgent):
                agent = BasicAgent(model=model)

        result = agent.run(task)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
