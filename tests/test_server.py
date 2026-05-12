"""The /run endpoint reports the agent's token tally in X-GenSIE-Token-Usage."""

import gensie.server as server
from fastapi.testclient import TestClient
from gensie.agent import GenSIEAgent, Participant, ParticipantInfo, PipelineInfo
from gensie.usage import UsageTracker


class _FakeAgent(GenSIEAgent):
    def __init__(self):
        self.usage = UsageTracker()

    def run(self, task, model):
        self.usage.reset()
        self.usage.add({"prompt_tokens": 1234, "completion_tokens": 567})
        return {"answer": "ok"}


class _FakeParticipant(Participant):
    def __init__(self):
        self._agent = _FakeAgent()

    def get_info(self):
        return ParticipantInfo(
            team_name="Fake",
            institution="Fake",
            pipelines=[PipelineInfo(name="baseline", description="x")],
        )

    def get_agent(self, pipeline_name):
        return self._agent


_TASK = {
    "id": "t1",
    "input_text": "x",
    "instruction": "y",
    "target_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    "output": {"answer": "ok"},
}


def test_run_sets_token_usage_header(monkeypatch):
    monkeypatch.setattr(server, "participant", _FakeParticipant())
    client = TestClient(server.app)
    resp = client.post("/run", params={"model": "demo"}, json=_TASK)
    assert resp.status_code == 200
    assert resp.json() == {"answer": "ok"}
    import json as _json

    usage = _json.loads(resp.headers["x-gensie-token-usage"])
    assert usage == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "total_tokens": 1801,
        "calls": 1,
    }
