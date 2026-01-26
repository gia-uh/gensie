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

## The Zero-Shot Constraint

This is a **Zero-Shot Schema** task.

* **Development Phase:** You will see schemas like `Biography`, `Recipe`, and others.
* **Test Phase:** You will encounter **entirely new schemas** (e.g., `LegalContract`, `ProductSpec`, and others).

Your system must generalize to the *structure*, not memorize the entity types.

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

### Aggregated Metrics

We calculate the total **True Positive Score (TPS)** by summing the similarity scores of all shared keys.

$$ TPS = \sum_{k \in K} Sim(G[k], S[k]) $$

Final rankings are based on **Micro-F1**:

$$ Precision = \frac{TPS}{|S|} \quad, \quad Recall = \frac{TPS}{|G|} $$

$$ F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall} $$

### Secondary Leaderboard: Efficiency

In addition to performance, we maintain an **Efficiency Leaderboard** to reward sustainable AI.

We calculate the **Performance-to-Cost Ratio**:

$$ \text{Ratio} = \frac{\text{Average F1}}{\text{Total Token Consumption}} $$

* **Total Token Consumption:** The sum of all input and output tokens generated across all API calls required to solve a single instance (including reasoning steps, retries, and self-corrections).

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
* **Model Access:** Your code connects to a local endpoint (e.g., `http://inference-server:8000/v1`) serving the official models (Llama 3, Salamandra, Qwen).
