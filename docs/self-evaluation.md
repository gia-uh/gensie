# Self-Evaluation on the Test Set

The **GenSIE 2026 test set is now public.** This page tells you how to run the
official evaluation harness on it yourself, so you can measure your system,
do your own analysis, and write your working note without waiting on us.

!!! info "Official numbers still come from the organizers"
    We are finishing the official evaluation of every team and will publish the
    **official results** ourselves. The evaluation script is deterministic, so
    your self-reported numbers should match ours closely; small differences can
    appear because the language models are stochastic. If your numbers and ours
    diverge materially, we will reconcile them with you before the results are
    final. **Run this to write your paper on time — not to replace the official
    ranking.**

---

## 1. Get the test set

The test set lives in the public repository under [`data/test/`](https://github.com/gia-uh/gensie/tree/main/data/test).
It contains **145 instances** across the eight task domains (cultural,
environmental, general, legal, medical, research, stem, technical). Each
instance is a single JSON file with the same shape as the dev set
(`input_text`, `instruction`, `target_schema`, `output`, `metadata`); the
`output` field is the gold annotation, so you can score locally.

```bash
git clone https://github.com/gia-uh/gensie
ls gensie/data/test/*.json | wc -l   # 145
```

---

## 2. Models to evaluate against

Report your system on **two models**:

| Model | Hugging Face id | GGUF mirror (for local serving) |
|---|---|---|
| Gemma 4 E4B | `google/gemma-4-E4B-it` | `bartowski/google_gemma-4-E4B-it-GGUF` |
| Qwen 3 14B | `qwen/qwen3-14b` | `lmstudio-community/Qwen3-14B-GGUF` |

These are the two models our **official results** will report. (Salamandra 7B
was part of an earlier plan but is being dropped for technical reasons.)
Because the test set is now public, there are **no held-out models** for these
final numbers.

!!! note "Reasoning mode"
    Both models have a reasoning ("thinking") toggle. Our official run evaluates
    **both modes** and keeps the better score per pipeline. You may report
    whichever mode you used — just **state which mode** in your paper.

You serve the models yourself, on any OpenAI-compatible server (LM Studio,
Ollama, vLLM, …). Point your agent at it exactly as you did during the
competition (`OPENAI_BASE_URL`, `OPENAI_API_KEY`). The `--model` value you pass
to `gensie eval` must be the **served model id** your inference server reports.

---

## 3. Run the evaluation

Boot your agent server the same way you submitted it (your container, or
`gensie serve`), then run `gensie eval` once per pipeline **and once for the
official `baseline`**, writing every report into one directory:

```bash
# your pipeline(s)
gensie eval \
  --data gensie/data/test \
  --url http://127.0.0.1:8000 \
  --pipeline <your-pipeline> \
  --model google/gemma-4-E4B-it \
  --output results/myteam--<your-pipeline>--gemma4b.json

# the official zero-shot baseline (ships in the public `gensie` package)
gensie eval \
  --data gensie/data/test \
  --url http://127.0.0.1:8000 \
  --pipeline baseline \
  --model google/gemma-4-E4B-it \
  --output results/baseline--gemma4b.json
```

Repeat both for `qwen/qwen3-14b`. Each report's `metrics` block holds your
**Micro-F1** (the per-model score; computed with Flattened Schema Scoring as
described in [Task Description → Evaluation Metrics](./description.md#evaluation-metrics)).
The report also carries `timing` and `token_usage` blocks so you can check the
soft 60 s / 32 K budgets — recorded, never hard-stopped.

---

## 4. Compute the primary metric (gap closed over baseline)

The primary leaderboard ranks by the **fraction of the baseline→perfect F1 gap
your system closes**, averaged over the models:

```
gap_closed = max(0, (F1_system − F1_baseline) / (1 − F1_baseline))
```

With your pipeline reports and the `baseline` reports in the same directory,
`gensie rank` does this for you:

```bash
gensie rank results/ --plain
```

It prints your per-model gap-closed, the average across models, and a secondary
raw-F1 table — the exact computation we use for the official ranking.

---

## 5. What to report in your paper

For each of your pipelines, report **Micro-F1 and gap-closed on both models**
(stating the reasoning mode used). You are free to add any further analysis the
test set enables — per-domain breakdowns, error analysis, complexity-level
trends (`metadata.complexity`), token/latency profiles, etc.

If you want us to cross-check your numbers against the official run, paste them
as a comment on your team's `[SUBMISSION]` issue. We will flag any material
disagreement and reconcile it before the official results are published.

---

## 6. Questions

Reply on your `[SUBMISSION]` issue in <https://github.com/gia-uh/gensie>, or
email Alejandro Piad Morffis (`apiad@matcom.uh.cu`). English or Spanish.

— The GenSIE 2026 Organizing Committee
(GIA-UH, University of Havana · GPLSI, University of Alicante)
