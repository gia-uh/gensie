# Submission Guidelines

To participate in GenSIE 2026, you must submit a **private GitHub repository** containing your full system and its environment configuration.

## 1. Repository & License Requirements

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
*   **Model Selection:** The exact model to use will be decided by the evaluator at runtime. Your system must be model-agnostic and respect the `model` parameter passed during the evaluation.

## 4. Resource Quotas & Qualification

To ensure fair access and stability:
*   **Token Quota:** A strict maximum budget of **32K total tokens** (input + output) across all inference calls for each specific input example.
*   **Timeout:** A strict wall-clock timeout of **60 seconds** per test instance.
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

---

## 7. Multi-Pipeline Limit

To stimulate innovation, each team is permitted to submit **up to three distinct pipelines** within the same repository/container. We will evaluate each pipeline separately. The final Team Ranking will be based on the highest score obtained by any of their submitted pipelines.

## 8. Reference Implementation

If you are developing in Python, the [**Starter Kit**](./starter-kit.md) provides a fully compliant reference implementation that respects these constraints, including robust connection logic (retries, backoff) for the inference server.
