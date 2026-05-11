from .Strategy import StrategyV2
from ..categorizer import fit_schema_to_fields, get_schema_to_field_plain
from ..corector import correct_in_text

from typing import Dict, Any

class Complex(StrategyV2):

    def estimate(self, task, fields):
        self.task = task
        self.plain_schema = get_schema_to_field_plain(fields, task.target_schema)
        self.ts_schema = fit_schema_to_fields(fields, task.target_schema)
        hints_estimation = 500
        estimated_user_prompt = "{instruction}\n\nText:{input_text}:\n\n\nReturn following schema:\n{ts_schema}".format(
            instruction=self.task.instruction,
            input_text=self.task.input_text,
            ts_schema=self.ts_schema
        )

        complex_system = self.prompts.get("complex_system", "You are a precise data extraction agent.")

        tokens = len(self.encoder.encode(estimated_user_prompt + complex_system)) + hints_estimation
        self.estimated_time = tokens / self.token_per_second["standard"]
        self.tokens = tokens


    def execute(self, in_time=0)-> Dict[str, Any]:

        # 1. tag text
        hints = self.llm.get_candidates(self.task, self.plain_schema, self.task.input_text)

        #2. direct call on tagged text
        result={}
        user_prompt = "{instruction}\n\nText:{input_text}:\nHints:{hints}\n\nReturn following schema:\n{ts_schema}".format(
            instruction=self.task.instruction,
            input_text=self.task.input_text,
            hints=hints,
            ts_schema=self.ts_schema
        )

        result = self.llm.direct_call(self.task, use_schema=True, seed=43, temperature=0.5, user_prompt=user_prompt, system_name="complex_system")
        result = correct_in_text(result, self.task.input_text, schema=self.task.target_schema)
        return result