# Task Description

The objective of GenSIE is to extract structured knowledge from a text fragment according to a specific, arbitrary schema provided in the input.

> [Read the full task description](./gensie.pdf), including score metrics and detailed constraints.

## The Challenge

For each instance, the system receives:

1.  **Context:** A text fragment in Spanish (Wikipedia, news, scientific abstracts).
2.  **Instruction:** A natural language description of the goal.
3.  **Target Schema:** A JSON Schema definition (OpenAPI 3.0 subset).

The system must generate a **valid JSON object** that:

1.  Strictly adheres to the `Target Schema`.
2.  Contains information faithfully extracted from the `Context`.
3.  **Returns `null`** for information not present (Hallucination Check).

## Grounding & Hallucination Traps

A critical aspect of GenSIE is evaluating the model's ability to remain strictly grounded in the source text. To test this, we design specific schema fields that ask for information **not present** in the input context—even if that information is widely known (e.g., asking for the specific date of a famous historical event when the text only mentions the year). In such cases, the system must explicitly return a `null` value. The frequency of these "null" targets will be kept intentionally low to prevent systems from artificially inflating their scores by defaulting to empty outputs, but their presence is vital for penalizing parametric hallucinations. This design enforces a strict "retrieval-only" behavior essential for reliable, trustworthy downstream applications.

## Example Instance

!!! example "Input Context"
    "El ensayo clínico aleatorizado de Fase 3 de la vacuna mRNA-1273 (Moderna) evaluó a 30,420 participantes. Los resultados primarios mostraron una eficacia del 94.1%..."

!!! example "Input Instruction"
    "Extrae el nombre de la medicación y el resultado del ensayo, siendo positivo si es mayor de 90% de efectividad, negativo si es menor del 70%, inconcluso en cualquier otro caso."

!!! abstract "Input Target Schema"
    ```json
    {
      "type": "object",
      "properties": {
        "medication_name": { "type": "string" },
        "clinical_outcome": {
          "type": "string",
          "enum": ["POSITIVE", "NEGATIVE", "INCONCLUSIVE"],
          "description": "Infer the semantic success of the trial"
        }
      }
    }
    ```

!!! success "Expected Output"
    ```json
    {
      "medication_name": "mRNA-1273 (Moderna)",
      "clinical_outcome": "POSITIVE"
    }
    ```

Note that `clinical_outcome` requires **semantic reasoning** mapping "94.1%" to the enum `POSITIVE`, as explained in the instructions. These semantic reasoning subproblems can range in complexity.

## The Zero-Shot Challenge

A crucial aspect of the GenSIE challenge is that it is a **zero-shot schema** task.

* **Development Phase:** Participants are given the dev set (150 silver instances) with a set of schemas (e.g., `Event`, `Biography`, `Recipe`).
* **Test Phase:** The private evaluation set has **100 brand-new instances**, split roughly evenly:
    * **~50%** use schemas that *do* appear in the dev set, but extracted from **new source documents** you have not seen.
    * **~50%** use **entirely new schemas** (e.g., `LegalContract`, `MedicalProcedure`, `ProductSpec`) that do not appear anywhere in the dev set.

This forces participants to build systems that can generalize to *any* structure, rather than overfitting to specific entity types. You must leverage the schema's field names, types, and descriptions dynamically at inference time.

## Evaluation Metrics

Evaluating generative structured output requires a rigorous approach that assesses both **structural validity** and **semantic accuracy**. We employ a custom metric called **Flattened Schema Scoring**.

### Flattened Schema Scoring

Since JSON objects can be deeply nested, we first "flatten" both the Gold Standard and the System Output into a set of key-value pairs using dot notation.

Let $J$ be a JSON object. The flattening function $\Phi(J)$ transforms it:

$$\Phi(J) = \{ (k_1, v_1), (k_2, v_2), \dots, (k_n, v_n) \}$$

!!! example "Flattening Example"
    **Nested JSON:**
    ```json
    {
      "event": {
        "city": "Madrid",
        "details": {
          "attendees": 500
        }
      }
    }
    ```

    **Flattened Representation:**
    ```python
    {
      "event.city": "Madrid",
      "event.details.attendees": 500
    }
    ```

### Value Scoring Logic

Once aligned by keys, we compare the values. The scoring method depends strictly on the **data type** defined in the schema.

#### Case A: Rigid Types

For fields where precision is binary (Numbers, Dates, Booleans, and Enum Strings), we require an **Exact Match**.

$$ Sim(g_k, s_k) = \mathbb{I}(g_k = s_k) $$

* **Example:** If the Gold schema expects `"clinical_outcome": "POSITIVE"` and the system outputs `"positive"` (lowercase) or `"SUCCESS"`, the score is $0$.

#### Case B: Free Text

For fields describing content (Descriptions, Summaries), we use a hybrid metric combining semantic and lexical similarity.

$$ Sim(g_k, s_k) = \alpha \cdot \text{CosSim}(\mathbf{e}_{g_k}, \mathbf{e}_{s_k}) + (1 - \alpha) \cdot \text{Lexical}(g_k, s_k) $$

* **Semantic:** Cosine similarity of embeddings using a multilingual sentence transformer (e.g., `paraphrase-multilingual-mpnet-base-v2`).
* **Lexical:** A normalized token-overlap score.
* **$\alpha$:** Weighting parameter (default $\approx 0.7$).

#### Case C: Null Values (Hallucination Penalty)

The evaluator penalizes hallucinations where the system fabricates information not present in the source text:

| Gold | System | Score |
|:-----|:-------|:------|
| `"field": "value"` | `"field": null` | 0.0 ❌ (hallucinated null) |
| `"field": null` | `"field": "value"` | 0.0 ❌ (hallucinated value) |
| `"field": null` | `"field": null` | 1.0 ✅ (correct null) |
| missing key | `"field": "value"` | Ignored (precision penalty via denominator) |

- If gold has a value but system returns `null`: score 0.0
- If gold has `null` but system returns a value: score 0.0
- System keys not in gold do not contribute to similarity but increase the system size, reducing precision

#### Lists: Order-Independent Matching

Since JSON lists have no inherent order, we use a **greedy bipartite matching** heuristic to maximize the alignment score:

1. Construct a similarity matrix between all gold items and all system items
2. Iteratively pick the highest-similarity pair, remove both from consideration, repeat
3. Score = sum of matched similarities
4. Normalize by Jaccard index: `|gold| + |system| - |matches|`

This allows flexibility in list ordering while penalizing missing or extra items.

### Per-Model Score: Micro-F1

For a given evaluation model, we calculate the total **True Positive Score (TPS)** by summing the similarity scores of all shared keys, then aggregate into a **Micro-F1** over the whole test set:

$$ TPS = \sum_{k \in K} Sim(G[k], S[k]) $$

$$ Precision = \frac{TPS}{|S|} \quad, \quad Recall = \frac{TPS}{|G|} \quad, \quad F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall} $$

### Multi-Model Evaluation

Every submitted pipeline is run against **several models** — some published/recommended, some held out and not disclosed before the results (see the [Submission Guidelines](./submission.md#3-infrastructure--connectivity)). We compute the Micro-F1 above **independently for each model**, and we run the official zero-shot baseline against each model as well.

### Primary Leaderboard: Gap Closed over Baseline

The team ranking is **not** the raw F1. For each evaluation model, let $F1_b$ be the official baseline's Micro-F1 and $F1_s$ be your system's Micro-F1 on that model. The **gap closed** is the fraction of the baseline-to-perfect error that your system eliminates:

$$ \text{GapClosed} = \max\!\left(0,\ \frac{F1_s - F1_b}{1 - F1_b}\right) $$

!!! example "Why gap-closed, not raw F1"
    Baseline scores 60 F1 → it has 40 points of error left. System **A** scores 80 → 20 points of error left → it closed **50%** of the gap. System **B** scores 90 → 10 points of error left → it closed **75%** of the gap. **B ranks above A** — it made the bigger jump over the baseline.

Your **primary score is the average of `GapClosed` over all evaluation models**, and the team ranking is by that average.

**Rationale.** Improving from 90→95 F1 is far harder than 70→75 — returns diminish near the optimum, so a fixed F1 delta does not represent a fixed amount of progress. And raw macro/micro-F1 is hard to compare across models (a gain on one model and a loss on another do not cancel cleanly when the baselines differ). The gap-closed fraction is normalized per model, so it is **comparable and averageable across models**, and it directly rewards what GenSIE is about: closing the *innovation gap* — pulling small models as close as possible to frontier-model performance.

### Secondary Leaderboards

* **Raw Micro-F1:** the plain average of Micro-F1 across all evaluation models. Reported for reference, but not the basis of the team ranking.
* **Efficiency:** a **Performance-to-Cost Ratio** to reward sustainable AI:

    $$ \text{Ratio} = \frac{\text{Average F1}}{\text{Total Token Consumption}} $$

    * **Total Token Consumption:** the sum of all input and output tokens generated across every model call required to solve an instance (including reasoning steps, retries and self-corrections), as recorded by the inference server's usage log — the same numbers `gensie eval` reports in its `token_usage` block.

## Rules & Submission

### Submission Format

Participants must submit a **private GitHub repository** containing:

1.  **Source Code:** MIT or Apache 2.0 licensed.
2.  **Dockerfile:** A self-sufficient container for inference.
3.  **README:** Instructions for running the pipeline.

!!! warning "Network Restrictions"
    The evaluation environment is **isolated** (no internet). Your container must include all necessary code and dependencies. The only allowed network call is to the Organizer's Hosted Inference API (OpenAI-compatible).

### The "No-Training" Policy

You **cannot** fine-tune the Language Model weights.

* Allowed: Prompt Engineering, RAG, Grammar-Constrained Decoding, Synthetic Data generation for few-shotting.
* Prohibited: LoRA, QLoRA, Full Fine-tuning.

### Inference Environment

* **Hardware:** NVIDIA A100 GPUs.
* **Model Access:** Your code connects to a local endpoint (e.g., `http://inference-server:8000/v1`). Several models are served — some published/recommended (e.g. Llama 3, Salamandra, Qwen) and some **held out and not disclosed before the results**. All are small (<~14B) open-source models from different families. The model name is passed to your `/run` endpoint at evaluation time; your system must respect it and must not assume a specific model.
* **Resource budgets:** Token and time budgets are soft averages over the test set, not hard per-instance caps — see [Submission Guidelines §4](./submission.md#4-resource-quotas--qualification).
