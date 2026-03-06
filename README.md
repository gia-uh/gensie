# GenSIE 2026 Public Starter Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](./Dockerfile)

**GenSIE (General-purpose Schema-guided Information Extraction)** is a shared task at [IberLEF 2026](https://sites.google.com/view/iberlef-2026). This repository provides the official starter kit for participants.

## 🚀 Quick Start

### 1. Installation
We recommend using [**uv**](https://github.com/astral-sh/uv) for fast dependency management:

```bash
git clone <repository-url>
cd gensie
uv sync --group dev
```

### 2. Configuration
Create a `.env` file to configure your inference backend:

```bash
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="http://localhost:1234/v1" # Optional: for local LLMs
```

### 3. Serving your Agent
Start the FastAPI server:
```bash
uv run gensie serve --port 8000
```

### 4. Running Benchmarks
Evaluate your agent against the 40 starter instances:
```bash
uv run gensie eval --data data/starter/ --url http://localhost:8000 --pipeline baseline --model gpt-4o-mini
```

## 🛠️ How to Participate

1.  **Inherit from `GenSIEAgent`**: Implement your extraction logic in `src/gensie/`.
2.  **Register your Pipelines**: Configure up to 3 pipelines in `OfficialParticipant` (see `src/gensie/baseline.py`).
3.  **Dockerize**: Use the provided `Dockerfile` and `docker-compose.yml` for testing and final submission.

```bash
docker compose up --build
```

## 📊 Dataset & Metrics

The kit includes **40 silver-generated instances** for initial testing. Official metrics use **Flattened Schema Scoring** (Micro-F1), which combines exact matches for rigid fields and semantic similarity for free-text fields.

## 📜 Documentation

For more details, see our guides:
*   🚀 [**Starter Kit Guide**](./docs/starter-kit.md)
*   📂 [**Submission Guidelines**](./docs/submission.md)
*   📊 [**Task Description**](./docs/description.md)

## ⚖️ License

This starter kit is licensed under the **MIT License**.
