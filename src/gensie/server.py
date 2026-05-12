from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from gensie.task import Task
from gensie.agent import Participant
from typing import Any
import importlib
import os

from logging import getLogger

logger = getLogger("gensie")

app = FastAPI(title="GenSIE Agent Server")

# Global participant instance
participant: Participant = None


def get_participant() -> Participant:
    global participant
    if participant is None:
        # Load participant from environment or default to OfficialParticipant
        participant_path = os.getenv(
            "PARTICIPANT_PATH", "gensie.baseline.OfficialParticipant"
        )
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
    model: str = Query(..., description="The exact model name to use for inference"),
) -> Any:
    """Executes the extraction task using the specified pipeline and model.

    The response body is the extracted JSON object. If the agent exposes a
    ``usage`` tracker (see ``gensie.usage.UsageTracker``), its tally is reported
    in the ``X-GenSIE-Token-Usage`` response header.
    """
    try:
        p = get_participant()
        agent = p.get_agent(pipeline)

        result = agent.run(task, model=model)
        headers = {}
        tracker = getattr(agent, "usage", None)
        if tracker is not None and hasattr(tracker, "header_value"):
            headers["X-GenSIE-Token-Usage"] = tracker.header_value()
        return JSONResponse(content=result, headers=headers)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
