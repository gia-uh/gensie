# Example evaluation reports

Synthetic `gensie eval` reports (made-up teams, made-up models, made-up scores)
used to exercise the `gensie rank` command end-to-end and as a fixture for
`tests/test_ranking.py`. Not real results.

Two models (`demo-3b-instruct`, `demo-7b-instruct`), an official baseline, and
three teams (one with two pipelines). Try it:

```bash
gensie rank results/examples
gensie rank results/examples --plain
```

The ranking is by the fraction of the baseline→perfect F1 gap each team closes,
averaged over the models — see `docs/description.md` → "Primary Leaderboard:
Gap Closed over Baseline".
