# GenSIE 2026 - Organizing Repository

**[GenSIE](https://uhgia.org/gensie) (General-purpose Schema-guided Information Extraction)** is a shared task for [IberLEF 2026](https://sites.google.com/view/iberlef-2026).

This **monorepo** contains the entire lifecycle of the challenge:

1.  **Public Website:** The documentation hosted on GitHub Pages.
2.  **Starter Kit:** The Docker templates and baselines released to participants.
3.  **Internal Tools:** Scripts for generating the Golden Dataset and running evaluations.


## 🚀 Quick Start

This project uses [`uv`](https://github.com/astral-sh/uv) for fast dependency management.

### 1. Setup

```bash
# Install dependencies (including dev & docs groups)
uv sync

```

### 2. Website Development

The website is built with **MkDocs Material**.

```bash
# Run the local dev server ([http://127.0.0.1:8000](http://127.0.0.1:8000))
uv run mkdocs serve

# Deploy to GitHub Pages (Manual Trigger)
uv run mkdocs gh-deploy --force

```

### 3. Data Generation (Internal)

**TODO;**

## 📜 License

This repository is licensed MIT.
