from .Strategy import StrategyV2
from ..categorizer import fit_schema_to_fields
from ..corector import correct_in_text

from typing import Dict, Any


class Direct(StrategyV2):

    def estimate(self, task, fields):
        self.task = task
        ts_schema = fit_schema_to_fields(fields, task.target_schema)

        self.direct_user = "Instrucción:{task.instruction}\n\nTexto de entrada:{task.input_text}\n\nReturn following schema:\n{ts_schema}".format(
            task=task,
            ts_schema=ts_schema
        )
        self.direct_system = self.prompts.get("direct_system", "")
        tokens = len(self.encoder.encode(self.direct_user + self.direct_system))
        self.estimated_time = tokens / self.token_per_second["standard"]
        self.tokens = tokens

    def execute(self, in_time=0)-> Dict[str, Any]:

        result = self.llm.call_llm(
            user_prompt=self.direct_user,
            system_prompt=self.direct_system,
            seed=43,
            use_schema=True,
            temperature=0.3,
        )

        if not result:
            print("Usando llamada directa")
            result = self.llm.direct_call(self.task, seed=43, temperature=0.3)

        result = correct_in_text(result, self.task.input_text, schema=self.task.target_schema)
        return result