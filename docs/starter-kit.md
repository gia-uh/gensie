# GenSIE Starter Kit

The **GenSIE Starter Kit** is designed to get you up and running with the task in minutes. It includes a baseline agent, a local evaluation server, and an initial set of examples to guide your development.

## 1. The Starter Data

Inside the repository at `data/starter/`, you will find **40 JSON instances** covering multiple domains (Legal, Medical, Cultural, and Technical).

!!! warning "Data Curation"
    These examples have been generated using Gemini 3 Pro and **manually curated** for grounding and correctness. However, they have not yet passed the official "Gold Curation" protocol. Expect this set to be updated when the full development set is released. The **format**, however, is final and will not change.

## 2. Reference Implementation & Baselines

The kit provides a high-quality Python baseline in `src/gensie/baseline.py` that is fully compliant with the competition's hardware and connectivity constraints.

*   **`BasicAgent`**: A reference agent that uses the OpenAI client. It is **model-agnostic**, meaning it respects the `model` parameter passed by the evaluator instead of hardcoding a specific model.
*   **`OfficialParticipant`**: The entry point for your submission. It acts as a factory that can host up to **three distinct pipelines**.

### Public Baselines

During the evaluation period, the organizers will provide three open-source baselines executed via standard zero-shot prompting with grammar-constrained decoding. These target different compute tiers:

*   **Tiny:** Llama 3.2 3B Instruct
*   **Small:** Salamandra 7b Instruct (native Spanish language model)
*   **Medium:** Qwen 3 14b

### Initial Benchmarks

To provide a reference point for your experiments, we have performed internal testing on the **40 starter instances**:

*   **Model:** `Meta-Llama-3.1-8B-Instruct`
*   **Micro-F1:** `0.4126`

While your mileage may vary depending on prompting and system configuration, this is a reasonable "lower baseline" to aim for. We will report official **Inter-Annotator Agreement (IAA)** and human performance metrics upon the release of the final test set.

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
You can test your performance against the starter data. The `eval` command will automatically pass the `--model` flag to your agent, simulating the evaluator's behavior:
```bash
gensie eval --data data/starter/ --url http://localhost:8000 --pipeline baseline --model gpt-4o-mini
```

!!! note "First Run & Embeddings"
    On its first run, `gensie eval` will automatically download the lightweight `BAAI/bge-small-en-v1.5` embedding model via `fastembed`. This is used for fast semantic similarity calculations on your CPU. The official leaderboard evaluation will use a more robust, heavier model (e.g., `paraphrase-multilingual-mpnet-base-v2`).

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
