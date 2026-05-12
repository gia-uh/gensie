# Submission Guidelines

To participate in GenSIE 2026, you must submit a **private GitHub repository** containing your full system and its environment configuration.

## 1. Submission Process

To participate in GenSIE 2026, you must:

1.  Maintain a **private GitHub repository** containing your full system and its environment configuration.
2.  Open a **New Issue** in the official competition repository using the **[Competition Submission]** template.
3.  Fill in your team name, participant details, and your repository URL.
4.  Grant read access to the designated organizer account (`apiad`).

## 2. Repository & License Requirements

*   **License:** The repository must be open-sourced under a permissive FOSS license (e.g., **MIT, Apache 2.0**). Copyleft licenses (e.g., GPL) are **not** accepted to facilitate broader adoption.
*   **Privacy & Release Cycle:** Submitted repositories must remain private during the competition phase. Teams will grant read access to a designated member of the organizing committee for evaluation. All repositories must be made publicly available immediately after the release of the official results.
*   **Documentation (README):** The repository must include a comprehensive `README.md` detailing the technical setup, specific runtime requirements, and instructions on how to run each specific pipeline (if multiple are submitted).

## 2. Core Requirement: Dockerization & Hardware

The primary requirement for submission is a **`Dockerfile`** located in the root of your repository. The resulting container must be self-sufficient, containing all necessary code, dependencies, and auxiliary data (like local RAG databases) required for inference.

### Hardware Constraints

Organizers will evaluate your system in a strictly controlled environment:

*   **Agent Side:** Your code will run on a **modest CPU-only environment**.
*   **NO GPUs:** Submissions must not depend on GPU-enabled hardware for the agent's logic.
*   **Lightweight Dependencies:** Avoid heavy machine learning frameworks like TensorFlow or PyTorch for local inference. Simple libraries like `scikit-learn` for intent parsing or `fastembed` for RAG are permitted, provided they run efficiently on a standard CPU.
*   **Local Knowledge Bases:** You may include local knowledge bases (e.g., SQLite, flat files) for RAG or other agentic strategies, provided they fit within the repository and container.

## 3. Infrastructure & Connectivity

The evaluation environment is **completely isolated from the internet**.

*   **Inference Server:** Organizers will host the inference engine (OpenAI-compatible server) on high-end hardware (8x NVIDIA A100 GPUs).
*   **Single Endpoint:** Your agent is **forbidden** from making any network calls other than to the provided inference server.
*   **No streaming:** completion requests must use `stream: false`. Streaming gives an extraction agent nothing here, and disallowing it lets the evaluator read exact token usage from every response.
*   **Model Selection:** The exact model to use will be decided by the evaluator at runtime. Your system must be model-agnostic and respect the `model` parameter passed during the evaluation.
*   **Multiple Evaluation Models:** Your system will be run against **several models**, not one. Some are the **published / recommended** models listed in the starter kit; others are **held-out models that we will not disclose** before the results. All of them are small open-source models (under ~14B) drawn from different model families — the kind of model a participant could realistically run on their own hardware. **Do not tune your pipeline to any single model**; the held-out models exist specifically to reward systems that generalize. Your final ranking aggregates performance across all evaluation models (see [Task Description → Evaluation Metrics](./description.md#evaluation-metrics)).

## 4. Resource Quotas & Qualification

We enforce token and time budgets to keep the playing field level and the evaluation tractable — **not** to penalize good systems. The budgets are **soft averages over the whole test set**, not hard per-instance caps:

*   **Token Budget:** A target of **32K total tokens** (input + output, including reasoning, retries and self-corrections) **averaged over the 100 test instances**. An individual instance may go up to **2× that budget (64K tokens)** as long as the rest of your run compensates so your average stays at or below 32K. Aim for 32K per instance; the slack exists because a held-out model may emit somewhat longer outputs than you expect, and you cannot tune against it.
*   **Timeout:** A target of **60 seconds of wall-clock time per instance, averaged over the 100 test instances** — i.e. roughly 100 minutes total. An individual instance may run longer if others run faster and your average stays at or below 60 s. Only **user-code time** counts toward this budget — time spent waiting on the inference server for token generation is **not** counted, so slow inference on our side never penalizes you.
*   **No hard stops:** We do not abort a run when it momentarily exceeds these budgets. We let the system finish and review the totals afterwards, case by case. A system that randomly overshoots ~5–10% on a handful of instances while producing strong results **will not be penalized** if the overshoot is statistical, has no material consequence, and our hardware can still run it. Systematic, sustained overshoot **will** be penalized — and only on the instances where it occurred.
*   **How tokens are counted:** the official inference server logs the prompt + completion tokens of every call your agent makes (completion tokens include any reasoning/thinking tokens), and the evaluator attributes them to the instance they occurred in — that is the authoritative number. As a transparency aid, the reference agent also reports its own tally on each `/run` response via the `X-GenSIE-Token-Usage` header (see §6); if your self-report and the server's count disagree materially, the instance is flagged for review, never auto-penalized. The server may additionally apply a high cumulative-token circuit breaker per instance purely as a cost backstop; legitimate systems never reach it.
*   **Qualification Phase:** All submissions must pass a sanity check on a subset of the Dev Set before entering final evaluation. The container must execute without hanging/crashing and must demonstrate performance superior to the provided zero-shot baseline.

## 5. Configuration & Environment Variables

Your container will be configured with the following environment variables to point your agent to our inference engine:

| Variable | Description |
| :--- | :--- |
| `OPENAI_BASE_URL` | The URL of the official GenSIE inference server. |
| `OPENAI_API_KEY` | The API key required to authenticate with the server. |

## 6. API Specification

Your containerized service must expose a web server (on port `8000`) with the following two endpoints.

### `GET /info`
Returns metadata about your team and your available extraction pipelines.

**Expected Response (JSON):**
```json
{
  "team_name": "DeepExtractors",
  "institution": "University of XYZ",
  "pipelines": [
    {
      "name": "baseline",
      "description": "Standard prompt-based extraction."
    },
    {
      "name": "rag-v1",
      "description": "RAG pipeline using a local SQLite index of domain manuals."
    }
  ]
}
```

### `POST /run`
Executes an extraction task.

**Query Parameters:**

- `pipeline` (string, required): The name of the pipeline to use.
- `model` (string, required): **MANDATORY.** You must use this specific model name when calling the inference server. Failing to use this model will result in an evaluation error.

**Request Body (JSON):**
The request body will be a `Task` object.

**Expected Response (JSON):**
A valid JSON object that strictly adheres to the provided `target_schema`.

**Optional response header — `X-GenSIE-Token-Usage`:**
A JSON object `{"input_tokens": int, "output_tokens": int, "total_tokens": int, "calls": int}` summing every model call your agent made for this task. Not required, but recommended for transparency; the starter kit's `gensie.usage.UsageTracker` populates it for you (`self.usage = UsageTracker()`; `self.usage.reset()` at the top of `run()`; `self.usage.add(response.usage)` after each call).

---

## 7. Multi-Pipeline Limit

To stimulate innovation, each team is permitted to submit **up to three distinct pipelines** within the same repository/container. We will evaluate each pipeline separately, against every evaluation model. The final Team Ranking will be based on the best result obtained by any of their submitted pipelines under the primary metric (see [Task Description → Evaluation Metrics](./description.md#evaluation-metrics)).

## 8. Reference Implementation

If you are developing in Python, the [**Starter Kit**](./starter-kit.md) provides a fully compliant reference implementation that respects these constraints, including robust connection logic (retries, backoff) for the inference server.
