# GenSIE Starter Kit

The **GenSIE Starter Kit** is designed to get you up and running with the task in minutes. It includes a baseline agent, a local evaluation server, and an initial set of examples to guide your development.

## 1. The Starter Data

Inside the repository at `data/starter/`, you will find **40 JSON instances** covering multiple domains (Legal, Medical, Cultural, and Technical).

!!! warning "Data Curation"
    These examples have been generated using Gemini 3 Pro and **manually curated** for grounding and correctness. However, they have not yet passed the official "Gold Curation" protocol. Expect this set to be updated when the full development set is released. The **format**, however, is final and will not change.

## 1.5 Development Data (v0.1 - Silver)

**150 instances** are available in `data/dev/dev.jsonl` covering 8 domains:

| Domain | Subdomain | Tasks |
|:-------|:---------| ------:|
| cultural | literature, monuments, movie_reviews | 29 |
| environmental | ecology | 10 |
| general | disasters | 10 |
| legal | contracts, judicial, legislation | 22 |
| lifestyle | recipes | 10 |
| medical | diseases, drug_safety, health_news | 31 |
| stem | astronomy_detailed | 20 |
| technical | software | 18 |

### Data Collection
- Sources: Wikipedia, Wikinoticias, BOE, CIMA (drug safety), Espinof (movie reviews)
- Web crawling + synthetic generation pipelines

### Generation & Quality Assurance
- Generated with **Claude Opus 4.6** via specialized subagents
- Passed through **1-2 audit phases** with silver-auditor subagent
- Manually spot-checked for broad inconsistencies

### ⚠️ Known Limitations
This is **NOT gold-standard data** - may contain errors:
- Some grounding may be incomplete
- Schema alignment edge cases not fully resolved
- Use for **development** only

### Reporting Errors
Found an issue? Please submit at: https://github.com/gia-uh/gensie/issues

### Status
- **Silver** - Will be replaced with curated gold data when available

## 2. Reference Implementation & Baselines

The kit provides a high-quality Python baseline in `src/gensie/baseline.py` that is fully compliant with the competition's hardware and connectivity constraints.

*   **`BasicAgent`**: A reference agent that uses the OpenAI client. It is **model-agnostic**, meaning it respects the `model` parameter passed by the evaluator instead of hardcoding a specific model.
*   **`OfficialParticipant`**: The entry point for your submission. It acts as a factory that can host up to **three distinct pipelines**.

### Evaluation Models & Public Baselines

Your system will be evaluated against **several models**, not one. Some are the published/recommended models below; others are **held out and not disclosed before the results** — all of them are small (<~14B) open-source models from different families, the kind you could run on your own hardware. For each model the organizers also run an official zero-shot baseline (standard prompting with grammar-constrained decoding), and your ranking is based on how much of the baseline-to-perfect gap you close, averaged over all models — see the [Task Description → Evaluation Metrics](./description.md#evaluation-metrics). **Do not tune to any single model.**

Published/recommended models, spanning different compute tiers:

*   **Tiny:** Llama 3.2 3B Instruct
*   **Small:** Salamandra 7b Instruct (native Spanish language model)
*   **Medium:** Qwen 3 14b

### Initial Benchmarks

To provide a reference point for your experiments, we have performed internal testing on the **40 starter instances** using the provided zero-shot baseline:

| Model                    | Micro-F1 |
| :----------------------- | -------: |
| `llama-3.2-3b-instruct`  | `0.2950` |
| `llama-3.1-8b-instruct`  | `0.3109` |
| `gemini-3-flash-preview` | `0.4334` |
| `llama-4-maverick`       | `0.4549` |

While your mileage may vary depending on prompting and system configuration, these are reasonable "lower baselines" to aim for. We will report official **Inter-Annotator Agreement (IAA)** and human performance metrics upon the release of the final test set.

## 3. Environment Setup

We recommend using **`uv`** for dependency management. The environment is optimized for **CPU-only execution**, avoiding heavy ML frameworks.

1.  **Install dependencies:**

    ```bash
    uv sync --group dev
    ```
2.  **Configure Environment:**

    The evaluator will provide `OPENAI_API_KEY` and `OPENAI_BASE_URL` at runtime. For local development, create a `.env` file in the root of the `gensie` folder:

    ```bash
    OPENAI_API_KEY="your-api-key"
    OPENAI_BASE_URL="http://localhost:1234/v1" # Point to your local LMStudio or OpenAI
    ```

## 4. Local Development Workflow

### Serving your Agent

To make your agent available for evaluation, start the FastAPI server:

```bash
gensie serve --port 8000
```
This launches your agent at `http://localhost:8000`.

### Running Local Benchmarks

You can test your performance against the starter data. The `eval` command will automatically pass the `--model` flag to your agent, simulating the evaluator's behavior. You can also generate a JSON report for later analysis:

```bash
gensie eval
    --data data/starter/
    --url http://localhost:8000
    --pipeline baseline
    --model gpt-4o-mini
    --output report.json
```

!!! note "Error Penalization"
    If your agent fails to process a task (e.g., returns a 500 error or crashes), the evaluation engine will **penalize** the result with a 0 score for that instance. The task remains in the denominator for Recall calculation, effectively lowering your overall F1 score.

!!! note "First Run & Embeddings"
    On its first run, `gensie eval` will automatically download the lightweight `BAAI/bge-small-en-v1.5` embedding model via `fastembed`. This is used for fast semantic similarity calculations on your CPU. The official leaderboard evaluation will use a more robust, heavier model (e.g., `paraphrase-multilingual-mpnet-base-v2`).

!!! note "Timing in the report"
    `gensie eval` records the wall-clock time per instance and adds a `timing` block to the report (`avg_elapsed_s`, `max_elapsed_s`, `over_budget_count`, `avg_within_budget`). It does **not** hard-stop a run at the 60s target — the budget is a soft average over the test set (see [Submission Guidelines §4](./submission.md#4-resource-quotas--qualification)). Tune `--time-budget-s` / `--request-timeout-s` if needed.

!!! note "Token usage in the report"
    `gensie eval --usage-log <path>` attributes per-instance token usage from the inference server's JSONL usage log; without it, the evaluator falls back to the `X-GenSIE-Token-Usage` response header your agent emits. Either way the report gains a `token_usage` block (`avg_total_per_instance`, `max_total`, `over_target_count`, `over_soft_count`, `avg_within_target`, `source`). The 32K target is a soft average over the test set — same spirit as the time budget. The reference agent reports the header automatically via `gensie.usage.UsageTracker`; reuse it in your own agent: `self.usage = UsageTracker()` in `__init__`, `self.usage.reset()` at the top of `run()`, `self.usage.add(response.usage)` after each model call. (Completion requests must be non-streaming — `stream: false`.)

### Ranking against a baseline

Once you have one or more reports (and a baseline report with `--pipeline baseline`), `gensie rank` reproduces the official primary leaderboard — the fraction of the baseline→perfect F1 gap closed, averaged over models:

```bash
gensie rank results/            # rich tables
gensie rank results/ --plain    # Markdown
gensie rank results/examples    # a runnable example with synthetic data
```

See [Task Description → Evaluation Metrics](./description.md#evaluation-metrics) for what the numbers mean.

## 5. How to Hack

To start building your own solution:

1.  **Configure Inference:** Before launching your container, ensure your `.env` file points to a valid backend. You can host a local model using [**LMStudio**](https://lmstudio.ai/) or connect to an external provider by setting `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
2.  **Launch:** Start the environment:
    ```bash
    docker compose up --build
    ```
3.  **Implement:** Create a new agent class in `src/gensie/` that inherits from `GenSIEAgent`.
4.  **Register:** Add your agent to the `OfficialParticipant` class inside `baseline.py`.
5.  **Benchmark:** Run `gensie eval` to see your progress.

!!! tip "Fast Iteration with Docker"
    The provided `Dockerfile` installs the package in **editable mode**. You can use **Docker Compose** to build and run your agent with local volume mounting and environment variables pre-configured:

    ```bash
    docker compose up --build
    ```

    This will:

    1. Build the agent image.
    2. Mount your local code so changes are reflected instantly (via FastAPI reload).
    3. Load your API keys from the `.env` file.
