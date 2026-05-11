from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict

import yaml

if TYPE_CHECKING:
    from gensie.task import Task
from gensie.agents.utils import _normalize_schema_for_strict

from .utils import _get_field_descriptions


def get_prompts():
    "read yaml file in same folder"
    with open(os.path.join(os.path.dirname(__file__),"prompts.yaml"), "r") as f:
        data = yaml.safe_load(f)
    return data or {}


class GradLLM:
    client: Any
    model: str
    prompts: Dict[str, str]

    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.prompts = get_prompts()



    def direct_call(self, task: Task, seed=42, temperature=0.7, use_schema=False, use_strict_schema=False, system_name = "direct_system", user_prompt = None) -> Dict[str, Any]:
        """ Simply call with the prompt in the task and its descriptions"""

        direct_system = self.prompts.get(system_name, "You are a powerful QA agent.")

        if user_prompt is None:
            field_descriptions = _get_field_descriptions(task)
            user_prompt = task.get_input_prompt()
            if field_descriptions:
                user_prompt = f"Field descriptions:\n{field_descriptions}\n\n{user_prompt}"

        strict_schema = _normalize_schema_for_strict(task.target_schema) if use_strict_schema else None

        result = self.call_llm(
            user_prompt=user_prompt,
            system_prompt=direct_system,
            use_schema=use_schema,
            force_schema=strict_schema,
            seed=seed,
            temperature=temperature,
        )
        return result

    def get_hints(self, task, fields, labels, properties, prev_hints):

        task_description = task.target_schema.get("description", "").split("Complexity")[0].strip()
        hints_system = self.prompts.get("hints_with_text_system", "").format(
            descripcion=task_description,
            instruction=task.instruction,
        )
        hints_user = self.prompts.get("hints_with_text_user", "").format(
            hints=prev_hints,
            fields=fields,
            labels=labels,
            properties=properties,
            input_text = task.input_text,
        )

        hints = self.call_llm(
            user_prompt=hints_user,
            system_prompt=hints_system,
            temperature=0.2,
        )
        print("Hint corregido",hints)
        return hints

    def get_hints_blind(self, task, fields, labels, properties):

        task_description = task.target_schema.get("description", "").split("Complexity")[0].strip()
        hints_system = self.prompts.get("hints_system", "").format(
            descripcion=task_description,
            instruction=task.instruction,
        )
        hints_user = self.prompts.get("hints_user", "").format(
            fields=fields,
            labels=labels,
            properties=properties,
            # input_text = task.instruction,
        )

        hints = self.call_llm(
            user_prompt=hints_user,
            system_prompt=hints_system,
            temperature=0.9,
        )
        print("Hint ciego",hints)
        return hints

    def get_candidates(self, task, plain_schema, input_text):

        task_description = task.target_schema.get("description", "").split("Complexity")[0].strip()
        hints_system = self.prompts.get("candidates_system", "").format(
            descripcion=task_description,
            instruction=task.instruction,
        )
        hints_user = self.prompts.get("candidates_user", "").format(
            plain_schema=plain_schema,
            input_text = input_text,
        )

        candidates = self.call_llm(
            user_prompt=hints_user,
            system_prompt=hints_system,
            temperature=0.9,
        )
        print("Candidates",candidates)
        return candidates

    def tag_text(self, task: Task, labels, properties, input_text) -> str:
        tag_system = self.prompts.get("tag_system", "")

        print("LABELS",labels)
        if not labels:
            labels = "ANSWER"

        tag_user = self.prompts.get("tag_user", "").format(
            input_text=input_text,
            labels=labels,
            task_instruction = task.instruction,
            properties = properties
        )
        max_tokens = max(250, len(task.input_text.split()) * 5) +2000
        num_ctx = max(250,len(task.input_text.split()) * 2) + 1000

        print("Max tokens for tagging:", max_tokens)
        tagged_text = self.call_llm(
            user_prompt=tag_user,
            system_prompt=tag_system,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
        )
        if not isinstance(tagged_text, str):
            print("Warning: tag_text did not return a string, falling back to original text")
            return task.input_text
        return tagged_text






    def call_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        use_schema: bool=False,
        force_schema: Any = None,
        seed=42,
        temperature=0.7,
        max_tokens=4000,
        num_ctx=4096,
    ) -> Dict[str, Any]:
        """
        Example LLM call method that demonstrates how to use self.client to make
        a call to the LLM and handle the response. This is a placeholder and
        should be replaced with actual logic to construct prompts, call the LLM,
        and parse responses according to the task requirements.
        """

        # # Make the API call to the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            # extra_body={
            #     "num_ctx": num_ctx
            # },
            response_format=None if not use_schema else {"type": "json_object"} if not force_schema else force_schema
        )
        print("Token usage:", response.usage)
        # Placeholder response parsing — replace with actual parsing logic
        try:
            content = response.choices[0].message.content
            # Ollama sometimes wraps JSON in markdown fences — strip them
            if isinstance(content, str):
                stripped = re.sub(r'^```(?:json)?\s*', '', content.strip())
                stripped = re.sub(r'\s*```$', '', stripped)
                content = stripped
            result = json.loads(content)
            return result
        except (KeyError, json.JSONDecodeError):
            # Handle parsing errors gracefully
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if isinstance(content, str):
                    if use_schema is False:
                        return content
                    # Try to extract a JSON object/array embedded in free text
                    for pattern in (r'\{.*\}', r'\[.*\]'):
                        m = re.search(pattern, content, re.DOTALL)
                        if m:
                            try:
                                return json.loads(m.group())
                            except json.JSONDecodeError:
                                pass
                    print(f"Warning: call_llm could not parse JSON. Content preview: {content[:200]}")
            return {}

