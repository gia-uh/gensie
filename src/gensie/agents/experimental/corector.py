import re
from pathlib import Path
from typing import Any

import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable
# ── Intentar usar rapidfuzz si está disponible ────────────────────────────────
try:
    from rapidfuzz import fuzz as _rfuzz

    def _fuzzy_ratio(a: str, b: str) -> float:
        return _rfuzz.token_sort_ratio(a, b) / 100.0

    _BACKEND = "rapidfuzz"
except ImportError:
    def _fuzzy_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b, autojunk=False).ratio()

    _BACKEND = "difflib"

CURRENT_PATH = Path(__file__).parent


def get_all_enum_values(schema: dict | None) -> set[str]:
    """Collect ALL enum values from every $defs entry in the schema.

    Unlike get_all_labels(), this covers enums on any field (category, etiology_type, etc.),
    not only fields named 'label'.
    """
    if not schema:
        return set()
    return {
        v
        for body in schema.get("$defs", {}).values()
        if "enum" in body
        for v in body["enum"]
    }


def _strip_tag_markup(text: str) -> str:
    """Remove inline tagger annotations, keeping only the mention text."""
    # [mention]text[/mention](TYPE) → text
    text = re.sub(r'\[mention\](.*?)\[/mention\]\([^)]+\)', r'\1', text)
    # [text](TYPE) → text  (TYPE is uppercase, avoids collisions with URLs)
    text = re.sub(r'\[([^\]]+)\]\([A-Z_]+\)', r'\1', text)
    return text


_TRAILING_PUNCT = frozenset('.,;:!?')


def _strip_span_punctuation(span: str, original: str) -> str:
    """Remove leading/trailing punctuation from a found span that the original didn't have.

    find_best_span may return a span that includes adjacent punctuation from the
    source text (e.g. 'estirar demasiado su tramo final.' vs original without '.').
    """
    if original and original[-1] not in _TRAILING_PUNCT:
        span = span.rstrip()
        while span and span[-1] in _TRAILING_PUNCT:
            span = span[:-1].rstrip()
    if original and original[0] not in _TRAILING_PUNCT:
        span = span.lstrip()
        while span and span[0] in _TRAILING_PUNCT:
            span = span[1:].lstrip()
    return span


def _resolve_schema(schema: dict, root: dict) -> dict:
    """Follow a $ref to its definition in root['$defs']."""
    if schema and "$ref" in schema:
        def_name = schema["$ref"].split("/")[-1]
        return root.get("$defs", {}).get(def_name, {})
    return schema or {}


def _expected_types(schema: dict) -> set[str]:
    """Collect all expected JSON primitive types from a (possibly anyOf) schema node."""
    types: set[str] = set()
    t = schema.get("type")
    if t:
        types.update(t if isinstance(t, list) else [t])
    for sub in schema.get("anyOf", []) + schema.get("oneOf", []):
        types.update(_expected_types(sub))
    return types


def _coerce_to_schema_type(value: Any, schema: dict) -> Any:
    """If the schema expects a scalar but value is a malformed dict, extract the scalar.

    Example: {"value": 1, "type": "integer"} → 1  when schema type is integer.
    """
    if not isinstance(value, dict):
        return value
    expected = _expected_types(schema)
    if {"integer", "number"} & expected:
        for key in ("value", "count", "number"):
            if key in value and isinstance(value[key], (int, float)) and not isinstance(value[key], bool):
                return value[key]
    if "string" in expected:
        for key in ("value", "text", "name"):
            if key in value and isinstance(value[key], str):
                return value[key]
    if "boolean" in expected:
        if "value" in value and isinstance(value["value"], bool):
            return value["value"]
    return value






# ─────────────────────────────────────────────────────────────────────────────
# Dataclass de resultado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Match:
    text: str          # fragmento encontrado en el texto original
    score: float       # similitud [0-1]; 1 = idéntico
    start: int         # posición de carácter inicial en el texto
    end: int           # posición de carácter final (exclusive)
    word_start: int    # índice de palabra inicial
    word_end: int      # índice de palabra final (exclusive)

    def __str__(self) -> str:
        return (
            f"[{self.score:.2%}]  '{self.text}'  "
            f"(chars {self.start}:{self.end})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Minúsculas + quita acentos + colapsa espacios."""

    nfkd = unicodedata.normalize("NFKD", text.lower() if text else "")
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip()


def _char_ngram_cosine(a: str, b: str, n: int = 3) -> float:
    """Similitud coseno de n-gramas de caracteres entre a y b."""
    def ngrams(s: str) -> Counter:
        s = f"{'_' * (n - 1)}{s}{'_' * (n - 1)}"
        return Counter(s[i : i + n] for i in range(len(s) - n + 1))

    ca, cb = ngrams(_normalize(a)), ngrams(_normalize(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[k] * cb[k] for k in ca if k in cb)
    norm = (sum(v * v for v in ca.values()) * sum(v * v for v in cb.values())) ** 0.5
    return dot / norm if norm else 0.0


def _combined_score(query: str, candidate: str, ngram_n: int = 3) -> float:
    """
    Mezcla ratio difuso (thefuzz/difflib) y coseno de n-gramas.
    Ambos trabajan sobre texto normalizado.
    El ratio difuso captura ediciones; el coseno captura solapamiento léxico.
    """
    q = _normalize(query)
    c = _normalize(candidate)
    fuzzy = _fuzzy_ratio(q, c)
    cosine = _char_ngram_cosine(q, c, n=ngram_n)
    # Media ponderada: fuzzy pesa más porque maneja reordenamientos
    return 0.6 * fuzzy + 0.4 * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizador que preserva posiciones
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """
    Devuelve lista de (palabra, inicio_char, fin_char).
    Divide por espacios/puntuación pero conserva las posiciones originales.
    """
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def find_top_spans(
    query: str,
    text: str,
    top_k: int = 5,
    window_slack: int = 2,
    ngram_n: int = 3,
    scorer: Callable[[str, str], float] | None = None,
    min_score: float = 0.0,
) -> list[Match]:
    """
    Encuentra los `top_k` fragmentos de `text` más similares a `query`.

    Parámetros
    ----------
    query        : frase o entidad a buscar (idealmente ≤ 6 palabras)
    text         : texto completo donde buscar
    top_k        : cuántos resultados devolver
    window_slack : cuántas palabras más/menos del tamaño de la query probar
                   (útil cuando la entidad extraída tiene más/menos tokens)
    ngram_n      : tamaño de n-grama de caracteres para el scorer coseno
    scorer       : función de puntuación alternativa f(query, candidate) → [0,1]
    min_score    : descartar matches por debajo de este umbral
    """
    score_fn = scorer or (lambda q, c: _combined_score(q, c, ngram_n))

    tokens = _tokenize(text)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return []

    query= str(query) if query is not None else ""
    query_words = query.split() if query != "" else []
    q_len = len(query_words)

    # Rangos de ventana a probar
    min_w = max(1, q_len - window_slack)
    max_w = min(n_tokens, q_len + window_slack)

    results: list[Match] = []

    for w in range(min_w, max_w + 1):
        for i in range(n_tokens - w + 1):
            span_tokens = tokens[i : i + w]
            span_text = text[span_tokens[0][1] : span_tokens[-1][2]]
            score = score_fn(query, span_text)
            if score >= min_score:
                results.append(
                    Match(
                        text=span_text,
                        score=score,
                        start=span_tokens[0][1],
                        end=span_tokens[-1][2],
                        word_start=i,
                        word_end=i + w,
                    )
                )

    # Ordenar y eliminar solapamientos (NMS simple por posición)
    results.sort(key=lambda m: m.score, reverse=True)
    return _deduplicate(results, top_k)


def find_best_span(
    query: str,
    text: str,
    window_slack: int = 2,
    ngram_n: int = 3,
    scorer: Callable[[str, str], float] | None = None,
) -> Match | None:
    """Atajo: devuelve solo el mejor Match (o None si el texto está vacío)."""
    results = find_top_spans(
        query, text, top_k=1,
        window_slack=window_slack,
        ngram_n=ngram_n,
        scorer=scorer,
    )
    return results[0] if results else None


# ─────────────────────────────────────────────────────────────────────────────
# NMS: elimina candidatos que solapan con un resultado ya seleccionado
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(matches: list[Match], top_k: int) -> list[Match]:
    selected: list[Match] = []
    taken: set[tuple[int, int]] = set()

    for m in matches:
        overlaps = any(
            not (m.word_end <= ws or m.word_start >= we)
            for ws, we in taken
        )
        if not overlaps:
            selected.append(m)
            taken.add((m.word_start, m.word_end))
            if len(selected) == top_k:
                break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Backend activo: {_BACKEND}\n")

    TEXT = """
    La película fue distribuida por Warner en exclusiva para cines europeos.
    Es considerada una Triste imitador de marca blanca de las grandes producciones.
    El director trabaja habitualmente con Sony Pictures y Paramount Studios.
    Algunos críticos la califican de obra maestra, aunque otros discrepan.
    La banda sonora corrió a cargo de Universal Music Group.
    """

    casos = [
        ("Warner Bros.",                    "entidad recortada"),
        ("imperfecto copia de marca blanca","paráfrasis aproximada"),
        ("Sony Pictures Entertainment",     "nombre más largo"),
        ("obra maestra del cine",           "frase con palabras extra"),
    ]

    sep = "─" * 60
    for query, descripcion in casos:
        print(sep)
        print(f"  Query      : «{query}»  ({descripcion})")
        top = find_top_spans(query, TEXT, top_k=3, min_score=0.3)
        if top:
            for m in top:
                print(f"  {m}")
        else:
            print("  (sin resultados por encima del umbral)")
    print(sep)


def correct_in_text(
    value: Any,
    input_text: str,
    schema: dict | None = None,
    _root: dict | None = None,
) -> Any:
    """Recursively normalize a model output value against the input text and JSON Schema.

    Applies three corrections in order:
    1. Type coercion: malformed dicts and numeric strings are cast to the schema type.
    2. Enum normalization: values that match an enum case-insensitively are canonicalized.
    3. Span correction: short string values not present in the text are fuzzy-matched.
    """
    root = _root if _root is not None else schema or {}
    labels = get_all_enum_values(schema)
    resolved = _resolve_schema(schema, root) if schema else {}

    value = _coerce_to_schema_type(value, resolved)

    def _cast(v: str) -> str | int | float:
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return v

    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        value = _strip_tag_markup(value)
        value_cf = value.casefold()

        if resolved.get("enum"):
            for enum_val in resolved["enum"]:
                if enum_val.casefold() == value_cf:
                    return enum_val
            return value

        if value in labels:
            return value
        for lbl in labels:
            if lbl.casefold() == value_cf:
                return lbl

        expected = _expected_types(resolved)
        if expected and "string" not in expected:
            return _cast(value) if {"integer", "number"} & expected else value

        if len(value.split()) >= 5:
            return value

        is_numeric = bool(expected and {"integer", "number"} & expected)

        if value in input_text or value_cf in input_text.casefold():
            return _cast(value) if is_numeric else value

        match = find_best_span(value, input_text)
        if match and match.score >= 0.90:
            span = _strip_span_punctuation(match.text, value).strip('\u200b\u200c\u200d\ufeff\u00a0')
            if (
                span
                and span != value
                and not ("\n" in span and "\n" not in value)
                and not (value.startswith(span) and len(span) < len(value))
                and not (value in span and len(span) > len(value))
            ):
                return _cast(span)

        return _cast(value)

    if isinstance(value, dict):
        properties = resolved.get("properties", {})
        return {
            key: correct_in_text(v, input_text, properties.get(key), root)
            for key, v in value.items()
        }

    if isinstance(value, list):
        return [
            correct_in_text(v, input_text, resolved.get("items") or None, root)
            for v in value
        ]

    return value





