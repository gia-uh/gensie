"""
gradiant_agents.py — Experimental extraction pipeline for GenSIE 2026.
──────────────────────────────────────────
Smaller models tend to be less reliable on:
  - Verbatim copying (they paraphrase more aggressively)
  - Enum case sensitivity (they often return lowercase)
  - Long-context array extraction (they over-extract or miss items)

The current implementation already mitigates these by:
  - Returning the grounded span directly for verbatim fields (Path 1)
  - Applying free enum normalization after every call (Path 2)
  - Using a conservative anti-hallucination system prompt for arrays (Path 2)

For very small models (< 3 B) consider enabling _self_consistency for
boolean fields and _null_review for fields that are frequently missed.

Architecture overview
─────────────────────
The agent routes each task through one of two paths depending on the schema:

  1. Single-field verbatim path  (L1 extraction schemas)
     ┌─────────────┐   Step 1: find the exact sentence span in the text
     │ Quote-then- │   Step 2: skipped — the span IS the answer
     │   extract   │
     └─────────────┘
     Used when the schema has exactly one string field whose description
     contains "verbatim" or "fragment".  The evaluator treats these fields
     as rigid (exact match), so returning the full sentence instead of just
     the entity name is critical.

  2. Full-schema path  (all other schemas)
     ┌──────────────────────────────────────────┐
     │  Single LLM call with the full schema    │
     │  + conservative anti-hallucination       │
     │    system prompt                         │
     │  + enum normalization post-processing    │
     └──────────────────────────────────────────┘
     Decomposing complex schemas was found to cause over-extraction in
     array fields (model gets "tunnel vision") and context loss for string
     fields that depend on the full instruction.  A single call with the
     full schema avoids both problems.

Additional methods that are implemented but not currently invoked
(available for future experiments):
  - _self_consistency   : majority-vote over N runs for boolean/enum fields
  - _null_review        : retry null fields with a "search carefully" nudge
  - _verification_pass  : holistic review of the full draft against the text
"""

import copy
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from gensie.agents.utils import _normalize_schema_for_strict
from openai import OpenAI

from gensie.agent import GenSIEAgent
from gensie.task import Task

# ── Module-level schema helpers ────────────────────────────────────────────────
#
# These pure functions operate on JSON Schema dicts and carry no state.
# They are defined at module level so they can be unit-tested independently
# from the agent class.


SYSTEM_PROMPT_EXTRACTION = """
Eres un asistente especializado en extracción precisa de información a partir de texto.

Tu tarea consiste en extraer información del texto proporcionado y devolverla exclusivamente en el formato y esquema especificados en la instrucción. Debes basar todas tus respuestas únicamente en el contenido explícito del texto original.

Reglas generales:
- Tu salida tiene debe estar integramente en español.
- Extrae únicamente información que aparezca de forma clara y directa en el texto.
- No infieras, no generalices y no completes información que no esté explícitamente indicada.
- No reformules ni normalices valores: utiliza, siempre que sea posible, la misma redacción que aparece en el texto.
- No uses abreviaturas si en el texto no aparece abreviado (es decir, copia el campo tal cual aparece en el texto).
- En la mayoría de casos el texto de las entidades que debes extraer habrá sido subrayado en el análisis.
- A la hora de extraer campos al pie de la letra estos deben aparecer tal cual en el texto original (no en el análisis hecho posterior). En el análisis realizado se incluyen las partes relevantes en negrita del texto roginal. Coge una parte de la frase (la más relevante para lo que piden) pero que aparezca así tal cual en el texto original, nunca reformules la frase bajo ningún concepto (salvo que se te pida expresamente).
- Cuando existan varias opciones para resolver la instrucción general del usuario escoge la principal.
- No acortes o abrevies el nombre de los artículos.

Para campos de tipo array:
- Incluye solo los elementos que estén explícita y directamente mencionados en el texto.
- No añadas elementos implícitos, derivados o no listados.
- No crees arrays con valores estimados o inferidos.

Cuando la información solicitada no esté presente en el texto:
- Sigue estrictamente las instrucciones definidas para ese tipo de entidad o campo (por ejemplo, dejar el campo vacío, usar null o el valor indicado).
- No intentes “rellenar” el campo con suposiciones.

No añadas campos nuevos.
No elimines campos del esquema.
No modifiques la estructura solicitada.
Tu salida debe estar integramente en español

Tu salida debe ser únicamente la extracción resultante conforme al esquema indicado.
"""

SYSTEM_PROMPT_NORMALIZATION = """

Eres un asistente experto en simplificación de tareas de  estructuración de texto para tareas de extracción de información.

Tu objetivo es etiquetar el texto de entrada de forma que facilite la extracción automática de campos específicos en iteraciones posteriores. No debes extraer la información tú mismo, sino presentar el contenido de manera más clara, ordenada y coherente para que esa extracción sea sencilla y fiable en el futuro. No debes modificar el texto de entrada, tu objetivo es poner en negrita con ** y ** las partes del texto original donde están las entidades que se piden. Limítalo al texto necesario. Además deberás añadir una explicación de los tipos de entidades y listar las extracciones realizadas tal y como aparecen en el texto. Ten en cuenta para este resaltado en los tipos de entidades que te piden extraer y márca ese texto con **

Recibirás como entrada:
- Un texto original sin estructurar que contiene la información relevante.
- Una descripción de los tipos de entidades o campos que se necesitarán extraer posteriormente. Descríbelos de forma que sean fáciles de entender por un agente para evitar ambigüedades y facilitar su proceso de extracción

Tu tarea consiste en:
- Poner en negrita las partes donde están las entidades
- Reordenar la información del texto original en la parte del análisis para facilitar la extracción.
- Agrupar contenidos relacionados por el tipo de entidad
- Explicar los tipos de entidad para que sean más fáciles de extraer
- Clarificar ambigüedades cuando sea posible sin añadir información nueva.
- Mantener todo el contenido original, sin eliminar ni inventar datos.
- Pon especial énfasis en el tipo de entidades para facilitar el trabajo de extracción
- No cambies el formato de las entidades en tu análisis (p.e. fechas salvo que te lo indiquen, o añadas abreviaturas).
- Salvo que se te pida generar una pregunta las respuestas a los campos deben obtenerse **tal cual** aparecen en el texto original
- Cuando existan varias opciones para resolver la instrucción del usuario escoge la principal.

Además, debes explicar brevemente la lógica de la estructura que has aplicado, de forma que una persona encargada de revisar o verificar tu trabajo pueda entender fácilmente las decisiones tomadas.

Tu salida debe:
- Estar íntegramente en español.
- Contener únicamente el texto etiquetado y la explicación y análisis aplicada.
- Tienes 1300 tokens para tu explicación (debes ser escueto en tus análisis)

Tu salida consta de dos partes:
- Primera parte: devuelves el texto original con los textos en negrita
- Segunda parte: devuelves el análisis de tu etiquetado


"""

SYSTEM_PROMPT_NORMALIZATION = """

Eres un asistente experto en preparar textos para tareas de extracción de información.

Tu objetivo NO es extraer información, sino PREPARAR el texto para que otros sistemas puedan extraerla fácilmente más adelante.

### Qué debes hacer con el texto de entrada

1. NO modifiques el contenido original.
2. Identifica en el texto las entidades que se te indiquen.
3. Marca SOLO esas entidades usando negrita: **texto**.
4. Limita el marcado únicamente al texto necesario.

### Entrada que recibirás

- Un texto original sin estructurar.
- Una lista o descripción de los tipos de entidades que deberán extraerse después.

### Tu tarea consiste en

- Resaltar en negrita (** **) las entidades dentro del texto original.
- Agrupar y reorganizar la información SOLO en la parte de análisis.
- Mantener todo el contenido original (no eliminar ni inventar datos).
- No cambiar el formato original de las entidades (fechas, nombres, números, etc.).
- Resolver ambigüedades solo si es posible usando el propio texto.
- Si hay varias interpretaciones, escoge la principal.
- Las respuestas deben aparecer exactamente como están en el texto original.

### Explicaciones adicionales

Debes:
- Explicar claramente cada tipo de entidad.
- Listar las entidades detectadas tal como aparecen en el texto.
- Explicar brevemente la lógica usada para estructurar y marcar el contenido.

### Formato de salida (OBLIGATORIO)

Tu salida tendrá SOLO dos partes, ambas en español:

1. **Texto original**, con las entidades marcadas en negrita.
2. **Análisis del etiquetado**, que incluye:
   - Explicación de los tipos de entidad.
   - Lista de entidades encontradas.
   - Breve explicación de la estructura aplicada.

### Restricciones finales

- No añadas información nueva.
- No cambies el contenido del texto original.
- No extraigas valores como respuesta directa.
- Usa un lenguaje claro y simple.
- Máximo 1300 tokens en la explicación.


"""

SYSTEM_PROMPT_NORMALIZATION = """

Eres un asistente de marcado de texto.

⚠️ REGLA CRÍTICA
Debes COPIAR el texto original EXACTAMENTE, carácter por carácter.
Está PROHIBIDO:
- reescribir
- resumir
- parafrasear
- corregir
- reordenar
- cambiar signos de puntuación
- cambiar saltos de línea

La ÚNICA modificación permitida es:
añadir ** ** alrededor de fragmentos del texto original.

Si cambias cualquier otro carácter, la tarea se considera incorrecta.

---

### Entrada
Recibirás:
1. Un texto original.
2. Una lista de tipos de entidades a marcar.

---

### Tarea

1. Devuelve el texto original EXACTO.
2. Marca SOLO las entidades indicadas usando **texto**.
3. No marques nada más.
4. No añadas texto nuevo.

---

### Salida (OBLIGATORIA)

Tu salida tiene DOS BLOQUES claramente separados:


[BLOQUE 1 – TEXTO MARCADO]
Incluye únicamente el texto original con ** ** añadidos donde haya entidades o salidas esperadas.

[BLOQUE 2 – ANÁLISIS]
Aquí sí puedes:
- explicar los tipos de entidades
- listar las entidades detectadas
- explicar brevemente la lógica

⚠️ En el BLOQUE 2 NO repitas ni modifiques el texto original.

---

### Restricciones finales

- El BLOQUE 1 debe ser una copia exacta del texto original con ** ** añadidos donde haya entidades o salidas esperadas.
- El BLOQUE 1 no puede contener análisis ni comentarios.
- Todo el texto debe estar en español.

"""

SYSTEM_PROMPT_NORMALIZATION_ANALYSIS = """


Eres un asistente de ANÁLISIS para tareas de marcado de texto.

⚠️ PROHIBIDO
- Copiar el texto original
- Reescribir el texto
- Parafrasear el texto
- Devolver el texto original total o parcialmente

Tu tarea NO es modificar ni devolver el texto.

---

### Entrada

Recibirás:
1. Un texto original sin estructurar.
2. Una descripción de los tipos de entidades que deben marcarse.

---

### Tu tarea

Debes analizar el texto y producir una GUÍA DE MARCADO que incluya:

1. Lista de tipos de entidades, explicados de forma clara y simple.
2. Para cada tipo de entidad:
   - Descripción breve.
   - Ejemplos literales EXACTOS tomados del texto (copiados tal cual).
3. Lista final de todas las entidades detectadas,
   agrupadas por tipo.

⚠️ IMPORTANTE
- Las entidades deben aparecer exactamente como están en el texto.
- No corrijas ni reformules.
- No inventes entidades.
- Si hay varias opciones, escoge la principal.

---

### Formato de salida (OBLIGATORIO)

Devuelve SOLO estas secciones:

TIPOS DE ENTIDAD
- …

ENTIDADES DETECTADAS
- Tipo A:
  - "texto exacto"
- Tipo B:
  - "texto exacto"

LOGICA DEL MARCADO
- Breve explicación de cómo se decidirá qué marcar.

---

### Restricciones finales

- No incluyas el texto original completo.
- No marques en negrita.
- No añadas ** **.
- Español neutro y claro.


"""

SYSTEM_PROMPT_NORMALIZATION_TEXTTAG = """

Eres un asistente de MARCADO DE TEXTO.

⚠️ REGLA CRÍTICA
Debes COPIAR el texto original EXACTAMENTE, carácter por carácter.

Está ABSOLUTAMENTE PROHIBIDO:
- reescribir
- resumir
- corregir
- cambiar palabras
- cambiar puntuación
- cambiar mayúsculas/minúsculas
- cambiar saltos de línea

La ÚNICA modificación permitida es:
añadir ** **  alrededor de los fragmentos requeridos dejando el resto del texto igual

Si modificas cualquier otro carácter, la respuesta es incorrecta.

---

### Entrada

Recibirás:
1. El texto original.
2. Una lista de elementos EXACTOS que deben marcarse.

---

### Tu tarea

1. Copia el texto original sin cambios.
2. Rodea con ** ** únicamente los fragmentos indicados en el análisis.
3. No marques nada más.
4. Si un elemento aparece varias veces, márcala en todas sus apariciones.

---

### Salida (OBLIGATORIA)

[BLOQUE UNICO – TEXTO MARCADO]

- Contiene SOLO el texto original con ** ** añadidos.
- No añadas análisis.
- No añadas listas.
- No añadas comentarios.
- No repitas las entidades fuera del texto.

---

### Restricciones finales

- Español.
- Ninguna explicación adicional.
- Ningún texto fuera del bloque.


"""

def _decompose_schema(schema: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Split a (strict-mode) schema into one minimal sub-schema per top-level field.

    Each sub-schema wraps exactly one property and inherits the parent's $defs
    so that $ref references (e.g. shared enum types) still resolve correctly.

    Returns a list of (field_name, sub_schema) tuples in property-definition
    order.  Used to enable per-field extraction strategies.

    Example
    -------
    Schema with fields {name, age, tags} →
        [("name", {…only name…}),
         ("age",  {…only age…}),
         ("tags", {…only tags…})]
    """
    properties = schema.get("properties", {})
    defs = schema.get("$defs", {})
    result = []
    for field_name, field_def in properties.items():
        sub: Dict[str, Any] = {
            # Title helps the LLM understand what it is extracting
            "title": f"{schema.get('title', 'Extraction')}_{field_name}",
            "type": "object",
            "properties": {field_name: copy.deepcopy(field_def)},
            # OpenAI strict mode requires all properties to be in required
            "required": [field_name],
            # Prevent the model from adding extra keys to the response object
            "additionalProperties": False,
        }
        if defs:
            # Preserve shared type definitions so $ref nodes resolve correctly
            sub["$defs"] = copy.deepcopy(defs)
        result.append((field_name, sub))
    return result


def _resolve_field_info(schema: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    """
    Return a normalised dict with type, description, enum values and rigidity
    flag for a single field, resolving $ref pointers when necessary.

    JSON Schema allows a field to be defined either inline or via a $ref to
    a shared definition in $defs.  This helper handles both cases so callers
    always get a consistent result regardless of how the schema was written.

    Also handles nullable fields expressed as anyOf: [{type: X}, {type: null}],
    which is the OpenAI-recommended pattern for optional fields in strict mode.

    Returns
    -------
    {
        "type":        str   — JSON Schema primitive type of the field
        "description": str   — human-readable description from the schema
        "enum_values": list | None — allowed enum values, or None if not an enum
        "is_rigid":    bool  — True for types that require exact match in scoring
                               (numbers, booleans, enums); False for free text
    }
    """
    prop = schema.get("properties", {}).get(field_name, {})

    # Resolve $ref: replace the reference with the actual definition
    if "$ref" in prop:
        ref_name = prop["$ref"].split("/")[
            -1
        ]  # e.g. "#/$defs/EntityType" → "EntityType"
        prop = schema.get("$defs", {}).get(ref_name, prop)

    field_type = prop.get("type", "string")
    enum_values: Optional[List[Any]] = prop.get("enum")
    description: str = prop.get("description", "")

    # Handle anyOf nullable pattern: pick the non-null branch
    for item in prop.get("anyOf", []):
        if item.get("type") != "null":
            field_type = item.get("type", field_type)
            enum_values = item.get("enum", enum_values)

    # Rigid fields require exact match in the evaluator (Case A in the spec)
    is_rigid = field_type in ("integer", "number", "boolean") or enum_values is not None

    return {
        "type": field_type,
        "description": description,
        "enum_values": enum_values,
        "is_rigid": is_rigid,
    }


def _normalize_enum(value: Any, enum_values: List[Any]) -> Any:
    """
    Fix enum case mismatches without an LLM call.

    The evaluator uses exact string comparison for enum fields (Case A / rigid
    types).  A model that returns "positive" instead of "POSITIVE" scores 0.
    This function applies a case-insensitive lookup against the allowed values
    as a free post-processing step — no extra token cost.

    If the value already matches an enum member, or is None, it is returned
    unchanged.  If no case-insensitive match exists, the original value is
    returned (the model may have hallucinated an invalid label).
    """
    if value is None or value in enum_values:
        return value
    lower_map = {str(v).lower(): v for v in enum_values}
    return lower_map.get(str(value).lower(), value)


def _sort_arrays_by_enum(result: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Sort array items by their enum-typed field, using the enum's definition order.
    Mutates *result* in place.

    Motivation
    ----------
    Gold annotations group entities by label following the enum order defined
    in the schema (e.g. PERSON → ORGANIZATION → LOCATION → DATE → EVENT →
    MISCELLANEOUS).  The LLM extracts entities in reading order (chronological),
    so the flat keys produced by the evaluator's flatten_json mismatch those of
    the gold even when all values are correct.  Sorting fixes the mismatch for
    the keys that happen to align positionally.

    Note: this is a partial mitigation.  The root issue is that the evaluator
    uses index-based array alignment (entities.0, entities.1, …) instead of
    best-match alignment.  Sorting by enum order makes the system output follow
    the same convention as the gold, maximising positional overlaps.

    Algorithm
    ---------
    For each array field in the schema:
      1. Resolve the items definition (following $ref if needed).
      2. Find the first property of the item object that is an enum type
         (again following $ref for shared enum definitions like EntityType).
      3. Build an order map {enum_value: index} from the enum definition.
      4. Sort the array using that map as a key.  Items whose enum value is
         not in the map (unexpected values) are placed at the end.
    """
    defs = schema.get("$defs", {})

    def _resolve(node: Dict[str, Any]) -> Dict[str, Any]:
        """Follow a single $ref level, returning the referenced definition."""
        if "$ref" in node:
            ref_name = node["$ref"].split("/")[-1]
            return defs.get(ref_name, node)
        return node

    for field_name, field_def in schema.get("properties", {}).items():
        if field_def.get("type") != "array":
            continue

        # Resolve items definition (may be a $ref to a shared object type)
        items_def = _resolve(field_def.get("items", {}))
        item_props = items_def.get("properties", {})

        # Find the first property that carries an enum constraint
        enum_field: Optional[str] = None
        enum_order: Optional[Dict[Any, int]] = None
        for prop_name, prop_def in item_props.items():
            resolved = _resolve(prop_def)
            if "enum" in resolved:
                enum_field = prop_name
                enum_order = {v: i for i, v in enumerate(resolved["enum"])}
                break

        if not enum_field or field_name not in result:
            continue

        arr = result[field_name]
        if isinstance(arr, list):
            result[field_name] = sorted(
                arr,
                key=lambda item: (
                    enum_order.get(item.get(enum_field, ""), len(enum_order))
                    if isinstance(item, dict)
                    else 0
                ),
            )


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models (DeepSeek-R1, etc.)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _build_question(field_name: str, description: str, instruction: str) -> str:
    """
    Compose a focused natural-language question for a single schema field.

    Combines the task instruction with the field's schema description to create
    a question that is more informative than the raw instruction alone.  This
    helps smaller models understand exactly what to extract.

    If the field has no description (e.g. a bare {"type": "string"}), falls
    back to a generic phrasing that at least names the field.
    """
    if description:
        return f"{instruction}\n\nField '{field_name}': {description}"
    return f"{instruction} — Extract only the field: '{field_name}'"


_VERBATIM_SYSTEM_PROMPT: str = (
    "You are a precise text extraction agent specialised in locating and copying "
    "verbatim fragments from source documents. "
    "Your only job is to find the exact sentence or passage in the text that "
    "answers the question, also could be a part of the sentence or passage, and return it word-for-word without any modification, "
    "summarisation, or reformulation."
    "Do not rephrase, translate, or shorten the extracted text. "
    "Do not add information that is not explicitly present in the source. "
    "If the requested information is not found in the text, return null."
)

_GROUNDING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {
            "type": "boolean",
            # True  → the field's information was located in the text
            # False → the information is absent; span will be an empty string
            "description": "True if the information for the field is present in the text",
        },
        "span": {
            "type": "string",
            # Verbatim sentence(s) copied from the source text.
            # Empty string when found=False.
            "description": "The verbatim sentence(s) from the text, or empty string if not found",
        },
    },
    "required": ["found", "span"],
    "additionalProperties": False,
}


# ── Agent class ────────────────────────────────────────────────────────────────


class StableAgent(GenSIEAgent):
    """
    Stable extraction agent for the GenSIE 2026 competition.

    Inherits from GenSIEAgent and must implement run(task, model) → dict.

    Key design decisions
    --------------------
    * Model-agnostic: the model name is passed in at runtime by the evaluator
      and forwarded unchanged to the endpoint.  Any model served via an
      OpenAI-compatible API (OpenAI, vLLM, LM Studio, …) works as long as
      it supports structured outputs (response_format = json_schema).

    * Uses OpenAI's structured-output API (response_format = json_schema) so
      the model is constrained to produce JSON that matches the target schema.
      This eliminates JSON parsing errors at the cost of requiring strict-mode
      compatible schemas (handled by _normalize_schema_for_strict).

    * Routes tasks to specialised strategies based on schema shape rather than
      applying one algorithm to everything.  This avoids the "one size fits all"
      trap that hurts performance on simpler or more complex schemas alike.

    * All LLM calls respect the 60-second per-instance wall-clock budget defined
      by the competition rules (MAX_TIME).

    Unused methods (_self_consistency, _null_review, _verification_pass) were
    built and tested during development.
    """

    # Hard wall-clock limit per task instance (seconds), matching competition rules
    MAX_TIME = 60

    def __init__(self):
        # OpenAI-compatible client — works with local LMStudio, vLLM, or OpenAI
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
        )
        # Set at the start of each run() call to enforce MAX_TIME across all phases
        self.initial_time: float = 0.0
        self.current_task_id: str = ""  # Set at the start of run() for log tracing

    # ── Internal utilities ──────────────────────────────────────────────────

    def _call_llm(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        system: str = "You are a precise data extraction agent.",
    ) -> Dict[str, Any]:
        """
        Single LLM call with structured-output enforcement.

        Sends a (system, user) message pair and enforces the response schema
        via OpenAI's json_schema response_format with strict=True.  This means
        the model MUST return JSON that matches the schema — it cannot add extra
        keys or omit required ones.

        Parameters
        ----------
        model  : exact model name string forwarded from the evaluator
        prompt : the user-turn message (task instruction + schema + text)
        schema : a strict-mode compatible JSON Schema dict for the response
        system : optional system-turn override (default is a generic agent persona)

        Returns
        -------
        Parsed JSON dict, or {"error": "…"} if parsing fails.
        """
        # Disable chain-of-thought thinking for models that support it (e.g. Qwen3).
        # Ignored by models that don't recognise the token, so safe for all models.
        system = system + "\n/nothink"
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema, "strict": True},
            },
            temperature= 0.1,
            reasoning_effort= None,
            max_tokens=2000

        )
        try:
            result = json.loads(response.choices[0].message.content or "")
        except (json.JSONDecodeError, AttributeError, IndexError) as exc:
            result = {"error": str(exc)}

        # Append one JSON record per call to the log file so every LLM
        # interaction can be reviewed offline.  The log path can be overridden
        # via the LLM_LOG_FILE env var; set it to "" to disable logging.
        log_path = os.getenv("LLM_LOG_FILE", "llm_calls.jsonl")
        if log_path:
            record = {
                "task_id": self.current_task_id,
                "ts": time.time(),
                "model": model,
                "system": system,
                "prompt": prompt,
                "schema": schema,
                "response": result,
            }
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(record, ensure_ascii=False, indent=2))
                lf.write("\n" + "─" * 80 + "\n")

        return result

    def _time_left(self, reserve: float = 0.0) -> float:
        """
        Seconds remaining in the per-instance budget minus an optional reserve.

        Use this before starting any LLM call to check whether there is still
        enough budget to complete it.  The reserve parameter lets callers
        guarantee a minimum buffer for downstream steps.

        Example: self._time_left(reserve=10) > 0  →  at least 10 s still free
        """
        return self.MAX_TIME - (time.time() - self.initial_time) - reserve

    # ── Extraction strategies ───────────────────────────────────────────────
    #
    # Each strategy below is a self-contained extraction approach designed for
    # a specific type of field or schema.  They all return the extracted VALUE
    # for the target field (not a full dict), or None if nothing was found.

    def _quote_then_extract(
        self,
        model: str,
        field_name: str,
        sub_schema: Dict[str, Any],
        text: str,
        question: str,
        verbatim: bool = False,
    ) -> Any:
        """
        Two-step grounded extraction that anchors the answer to a source span.

        Motivation
        ----------
        A vanilla LLM call for "who developed 7-Zip?" returns "Igor Pavlov".
        The gold annotation expects the full verbatim sentence: "7-Zip es un
        archivador de ficheros libre desarrollado por Igor Pavlov."  Because the
        evaluator treats the `answer` field as rigid (exact match), the short
        answer scores 0.  Returning the full grounded span fixes this.

        Step 1 — Grounding
            Ask the model to locate and copy the exact sentence(s) from the
            text that contain the relevant information.  Returns a boolean
            `found` flag and a `span` string.  Uses _GROUNDING_SCHEMA which is
            already strict-mode compatible.

        Step 2 — Extraction  (skipped when verbatim=True)
            Extract the structured field value from the grounded span.  Using
            the span instead of the full text reduces noise and focuses the
            model on the relevant evidence.

        Parameters
        ----------
        verbatim : if True, the span from Step 1 is returned directly as the
                   field value, skipping Step 2.  Use this when the field
                   description says "verbatim" or "fragment" — in those cases
                   the span IS the correct answer and Step 2 would only
                   paraphrase it (e.g. turning a full sentence into just a name).
        """
        # ── Step 1: locate the evidence span ──────────────────────────────
        grounding_prompt = (
            f"TEXT:\n{text}\n\n"
            f"TASK: {question}\n\n"
            f"Copy the exact sentence(s) from the TEXT that contain the information "
            f"for the field '{field_name}'. "
            f"Set 'found' to false and 'span' to an empty string if the information "
            f"is not present in the text."
        )
        grounding = self._call_llm(
            model,
            grounding_prompt,
            _GROUNDING_SCHEMA,
            system=_VERBATIM_SYSTEM_PROMPT,
        )

        if not grounding.get("found"):
            return None  # Information genuinely absent → caller should return null
        span = grounding.get("span", "").strip()
        if not span:
            return None  # Model set found=True but gave no span — treat as absent

        # ── Verbatim shortcut: span = answer, no Step 2 needed ────────────
        if verbatim:
            # Strip a trailing comma that some models append to copied sentences
            return span.rstrip(",").strip()

        # ── Step 2: extract structured value from the grounded span ───────
        extract_prompt = (
            f"Q: {question}\n"
            f"A: [Respond with a JSON object matching the schema below]\n\n"
            f"SCHEMA:\n{json.dumps(sub_schema, indent=2)}\n\n"
            f"TEXT:\n{span}\n\n"
            f"Important: do not paraphrase. Use the exact wording from the text."
        )
        return self._call_llm(
            model,
            extract_prompt,
            sub_schema,
            system=_VERBATIM_SYSTEM_PROMPT,
        ).get(field_name)

    def _self_consistency(
        self,
        model: str,
        field_name: str,
        sub_schema: Dict[str, Any],
        text: str,
        question: str,
        runs: int = 3,
    ) -> Any:
        """
        Majority-vote extraction for boolean and small-enum fields.

        Motivation
        ----------
        Boolean and enum fields require exact match in the evaluator (Case A /
        rigid types).  A single LLM call can be noisy, especially for smaller
        models making inference-heavy decisions (e.g. "is X legally permitted?").
        Running several independent calls and taking the most frequent answer
        reduces variance at the cost of extra token usage.

        Implementation
        --------------
        Values are serialised to JSON strings for hashing (handles dicts and
        lists as well as primitives).  The Counter picks the winner; ties are
        broken by Counter.most_common(1) which returns the first encountered.

        Note: currently not invoked in run() but kept for future experiments.
        """
        values = []
        for _ in range(runs):
            # Stop early if we are running low on time budget
            if self._time_left(reserve=8) <= 0:
                break
            prompt = (
                f"Q: {question}\n"
                f"A: [Respond with a JSON object matching the schema below]\n\n"
                f"SCHEMA:\n{json.dumps(sub_schema, indent=2)}\n\n"
                f"TEXT:\n{text}"
            )
            result = self._call_llm(model, prompt, sub_schema)
            val = result.get(field_name)
            if val is not None:
                values.append(val)
        if not values:
            return None
        # Serialise to string for hashing, then deserialise the winner back
        winner_key = Counter(json.dumps(v, sort_keys=True) for v in values).most_common(
            1
        )[0][0]
        return json.loads(winner_key)

    def _standard_extract(
        self,
        model: str,
        field_name: str,
        sub_schema: Dict[str, Any],
        text: str,
        question: str,
    ) -> Any:
        """
        Single focused LLM call for a specific field.

        Used for array fields, numbers, and complex nested objects where
        multi-step strategies (quote-then-extract, self-consistency) do not
        apply.  The sub_schema restricts the response to just the target field,
        giving the model a small, well-scoped task.

        Note: currently only used as a fallback in the verbatim path (when
        grounding returns no span).  The full-schema path is preferred for
        multi-field schemas to preserve cross-field context.
        """
        prompt = (
            f"Q: {question}\n"
            f"A: [Respond with a JSON object matching the schema below]\n\n"
            f"SCHEMA:\n{json.dumps(sub_schema, indent=2)}\n\n"
            f"TEXT:\n{text}"
        )
        return self._call_llm(model, prompt, sub_schema).get(field_name)

    # ── Multi-pass helpers (not currently active) ───────────────────────────

    def _null_review(
        self,
        model: str,
        merged: Dict[str, Any],
        sub_schemas: List[Tuple[str, Dict[str, Any]]],
        original_schema: Dict[str, Any],
        text: str,
        instruction: str,
    ) -> None:
        """
        Retry every still-null field with an explicit 'search carefully' nudge.

        After the main extraction pass, some fields may be None because the
        model did not find the information on the first attempt.  A focused
        retry with an explicit instruction to read the full text carefully can
        recover some of these cases without adding hallucination risk (the
        nudge explicitly says to return null if truly absent).

        Mutates *merged* in place so the caller sees the updated values.

        Note: currently not invoked in run(). FOR FUTURE EXPERIMENTS.
        """
        for field_name, sub_schema in sub_schemas:
            if self._time_left(reserve=6) <= 0:
                break
            if merged.get(field_name) is not None:
                continue  # Already extracted, skip
            field_info = _resolve_field_info(original_schema, field_name)
            question = _build_question(
                field_name, field_info["description"], instruction
            )
            prompt = (
                f"Q: {question}\n"
                f"A: [Respond with a JSON object matching the schema below]\n\n"
                f"SCHEMA:\n{json.dumps(sub_schema, indent=2)}\n\n"
                f"TEXT:\n{text}\n\n"
                f"Note: Read the entire text carefully. "
                f"Only leave '{field_name}' as null if the information is truly absent."
            )

            result = self._call_llm(model, prompt, sub_schema)
            val = result.get(field_name)
            if val is not None:
                # Apply enum normalization in case the retry returned wrong case
                if field_info["enum_values"]:
                    val = _normalize_enum(val, field_info["enum_values"])
                merged[field_name] = val

    def _verification_pass(
        self,
        model: str,
        draft: Dict[str, Any],
        strict_schema: Dict[str, Any],
        text: str,
        instruction: str,
    ) -> Dict[str, Any]:
        """
        Holistic correctness check on the full draft extraction.

        Shows the model its own previous output alongside the source text and
        asks it to correct any wrong or ungrounded values.  This catches errors
        that per-field extraction might miss (e.g. a field extracted from the
        wrong sentence, or a value that seemed plausible but contradicts another
        field).

        Returns the corrected dict, or the original draft if the LLM call fails.

        Note: currently not invoked in run(). FOR FUTURE EXPERIMENTS.
        """

        system_prompt = (
            """

Eres un asistente especializado en validación y corrección de extracciones de información basadas en texto.

Tu tarea consiste en revisar una extracción previa en formato JSON y verificar, campo por campo, si los valores extraídos están correcta y explícitamente fundamentados en el texto original.

Regla general:
- Todo campo que NO sea un campo libre debe aparecer en el texto original de forma explícita y con la misma redacción o significado inequívoco.
- Si un valor no aparece en el texto, es incorrecto, está incompleto o mal formulado, debes corregirlo para que se ajuste exactamente a la información presente en el texto.
- No debes inferir ni completar información que no esté claramente justificada por el texto.
- Revisa los tipos de las entidades, modifica aquellas donde la asignación no se hizo correctamente
- Cuando hablamos de que debe aparecer literalmente en el texto es que no puede haber ninguna otra palabra entre medias. Si esto ocurre incluye solo las partes del texto consecutivo que están relacionados con la entidad que se quiere extraer.

Excepción:
- Los campos definidos como “campos libres” pueden no aparecer literalmente en el texto (por ejemplo, preguntas generadas a partir del contenido). En estos casos, solo debes comprobar que el valor sea coherente con el texto, no que aparezca explícitamente.

Tu salida debe ser exclusivamente el objeto JSON corregido, manteniendo el mismo esquema que la extracción original.

            """

        )


        prompt = (
            f"Instrucción original: {instruction}\n\n"
            f"Schema con las entidades que se solicitan extraer:\n{json.dumps(strict_schema, indent=2)}\n\n"
            f"Texto:\n{text}\n\n"
            f"Extracción del sistema en el paso anterior:\n{json.dumps(draft, indent=2)}\n\n"
            f"Revisa cada campo extraido por el sistema . "
            f"Fix any value that is incorrect or not grounded in the text. "
            f"Devuelve el objeto JSON original "
        )


        prompt = (
           f"""

Instrucción original:
{instruction}

Esquema estricto con las entidades que debían extraerse:
{json.dumps(strict_schema, indent=2)}

Texto original del que debía realizarse la extracción:
{text}

Resultado de la extracción realizada en el paso anterior:
{json.dumps(draft, indent=2)}

Tarea:
Revisa cuidadosamente cada campo del JSON extraído y verifica si su valor está correctamente fundamentado en el texto original.

- Si un valor es correcto y aparece claramente en el texto, mantenlo tal como está.
- Si un valor es incorrecto, impreciso o no está respaldado por el texto, corrígelo para que coincida con la información real del texto.
- Si el campo es un campo libre, comprueba únicamente su coherencia con el contenido del texto.

Devuelve exclusivamente el objeto JSON final corregido, respetando exactamente la misma estructura.

            """
        )

        result = self._call_llm(model, prompt, strict_schema, system_prompt)

        if result.get("error"):
            print ("Validation Error")

        else:
            print ("Everything ok")

        return result if not result.get("error") else draft

    # ── Entry point ─────────────────────────────────────────────────────────

    def _normalize_text(self, task: Task, full_schema: Dict[str, Any], model: str):
        """ Re-structures the input text to simplify the extraction of content """

        # 1st step

        input_text = task.input_text

        norm_prompt_user = f"""
        Texto original:
        {input_text}

        Estas son las entidades a extraer:
        {full_schema}

        Sé conciso. Limita tu análisis a un máximo de 1000 palabras.
        """

        norm_prompt_system = SYSTEM_PROMPT_NORMALIZATION_ANALYSIS + "\n/nothink"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": norm_prompt_system},
                {"role": "user", "content": norm_prompt_user},
            ],
            temperature=0.1,
            max_tokens=1300,
            reasoning_effort = None
        )

        response_text_analysis = _strip_thinking(response.choices[0].message.content or "")

        # 2nd step

        input_text = task.input_text + f"\n Análsis del texto para ayudarte en la extracción: {response_text_analysis}"

        norm_prompt_user = f"""
        Texto original:
        {input_text}

        Entidades a extraer:
        {full_schema}

        Etiqueta el texto original con lo que se pide en el esquema usando el análisis para ayudarte.
        """

        norm_prompt_system = SYSTEM_PROMPT_NORMALIZATION_TEXTTAG + "\n/nothink"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": norm_prompt_system},
                {"role": "user", "content": norm_prompt_user},
            ],
            temperature=0.1,
            max_tokens=1300,
            reasoning_effort = None
        )

        response_text_sub = _strip_thinking(response.choices[0].message.content or "")

        # Monkey patch input_text
        task.input_text = f"Texto original del que se deben extraer las entidades: {task.input_text} \n Análsis del texto para ayudarte en la extracción: \n Texto etiquetado {response_text_sub} \n Analisis {response_text_analysis}"

        print ("============ Task input Text  ==============")
        print (task.input_text)


    def run(self, task: Task, model: str) -> Dict[str, Any]:
        """
        Main extraction method called by the server for every task instance.

        Routing logic
        -------------
        The method inspects the target schema and routes to the most effective
        strategy.  This avoids the "one algorithm for everything" trap:

        Path 1 — Single-field verbatim  (L1 extraction schemas)
        ─────────────────────────────────────────────────────────
        Condition: the schema has exactly one top-level field, its type is
        string, it has no enum constraint, and its description contains the
        word "verbatim" or "fragment".

        Strategy: quote-then-extract with verbatim=True.
        The grounded span is returned directly as the field value.

        Why: these schemas expect the full source sentence, not a paraphrased
        entity.  The evaluator treats the field as rigid (exact string match)
        because the key name doesn't match "description" or "summary".
        Returning only the entity name (e.g. "Igor Pavlov") instead of the
        full sentence scores 0.

        Fallback: if the grounding step finds no span, a standard extraction
        call is made so the task is never left completely empty.

        Path 2 — Full-schema baseline+  (all other schemas)
        ─────────────────────────────────────────────────────
        Condition: any schema that does not match Path 1.

        Strategy: single LLM call with the complete schema and a conservative
        system prompt, followed by free enum normalization on the result.

        Why not always decompose?
          - Array fields over-extract when extracted in isolation: without the
            context of other fields, the model tries to fill the array as
            completely as possible, extracting more items than the gold.
          - Non-verbatim string fields (e.g. the `question` field in legal
            schemas) lose important context from the task instruction when
            extracted from a grounded span rather than the full text.
          Both effects were confirmed empirically: decomposition reduced F1 on
          software, media, and legal tasks while only helping on L1 schemas.

        Enum normalization (post-processing, no LLM cost)
          After the LLM call, each field that has an enum constraint is checked
          for case mismatches (e.g. "positive" → "POSITIVE").  This is a free
          fix that catches a common failure mode without any additional tokens.
        """
        self.initial_time = time.time()
        self.current_task_id = task.id

        # Prepare the strict-mode schema (adds missing `required` lists and
        # additionalProperties: false throughout the schema tree)
        strict_schema = _normalize_schema_for_strict(task.target_schema)

        # Build the per-field sub-schema list (used in both paths)
        sub_schemas = _decompose_schema(strict_schema)

        if self._time_left(reserve=30) > 0:
            self._normalize_text(task, strict_schema, model)

        # ── Path 1: single-field verbatim extraction ───────────────────────
        if len(sub_schemas) == 1:
            print(
                f"Task {task.id}: single-field schema detected, applying verbatim path"
            )
            field_name, sub_schema = sub_schemas[0]
            field_info = _resolve_field_info(task.target_schema, field_name)
            desc_lower = field_info["description"].lower()

            # Detect verbatim-string fields by their description keywords
            is_verbatim = (
                field_info["type"] == "string"
                and not field_info["enum_values"]
                and ("verbatim" in desc_lower or "fragment" in desc_lower)
            )

            if is_verbatim:
                question = _build_question(
                    field_name, field_info["description"], task.instruction
                )
                # Step 1: find the evidence span; Step 2: skipped (span = answer)
                val = self._quote_then_extract(
                    model,
                    field_name,
                    sub_schema,
                    task.input_text,
                    question,
                    verbatim=True,
                )
                if val is None:
                    # Grounding returned nothing — fall back to a direct call
                    # so the task is not left completely unanswered
                    val = self._standard_extract(
                        model, field_name, sub_schema, task.input_text, question
                    )
                result = {field_name: val}
                _sort_arrays_by_enum(result, task.target_schema)
                return result

        # ── Path 2: full-schema extraction with anti-hallucination prompt ──
        prompt = task.get_input_prompt()  # instruction + schema JSON + source text
        result = self._call_llm(
            model,
            prompt,
            strict_schema,
            system=(
                SYSTEM_PROMPT_EXTRACTION
            ),
        )

        if self._time_left(reserve=17) > 0:
            result = self._verification_pass(model, result, strict_schema,
                                             task.input_text, task.instruction)

        # Post-process: fix enum case mismatches without an extra LLM call
        # (e.g. model returns "positive" but schema requires "POSITIVE")
        for field_name, _ in sub_schemas:
            field_info = _resolve_field_info(task.target_schema, field_name)
            if field_info["enum_values"] and field_name in result:
                result[field_name] = _normalize_enum(
                    result[field_name], field_info["enum_values"]
                )

        # Post-process: sort array items by their enum field order so that
        # positional keys (entities.0, entities.1, …) align with the gold
        # annotations, which follow the enum definition order by convention.
        _sort_arrays_by_enum(result, task.target_schema)

        return result
