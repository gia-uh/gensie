# Results

Final results of the **GenSIE @ IberLEF 2026** shared task.

Primary metric: **gap closed over baseline**, `gap = max(0, (F1 − F1_base) / (1 − F1_base))`, averaged over the two evaluation models (**Gemma 4 E4B**, **Qwen3-14B**). Each team's best-reported valid pipeline is used (any thinking config), scored with micro-F1 on the **125-instance** test set (the full 145 instances minus a 20-instance drop-set — see [Methodology](#methodology)).

Baselines (125-instance, official participant): gemma4-e4b = **0.7895**, qwen3-14b = **0.7805**.

## Leaderboard

| # | Team | Avg gap closed | gemma F1 | qwen F1 |
|--:|:--|--:|--:|--:|
| 1 | DRILLER | **0.2158** | 0.8285 | 0.8346 |
| 2 | Krishan | **0.1202** | 0.8142 | 0.8076 |
| 3 | CodeStrange | **0.0995** | 0.8090 | 0.8039 |
| 4 | FranRodrigo | **0.0629** | 0.7812 | 0.8081 |
| 5 | SEsml | **0.0481** | 0.7746 | 0.8017 |
| 6 | VerbaNex | **0.0473** | n/a | 0.8013 |
| 7 | UC3M | **0.0212** | 0.7984 | 0.3629 |
| 8 | GRADIANT | **0.0000** | 0.6792 | 0.7564 |
| 9 | Inigo | **0.0000** | 0.7504 | 0.7320 |
| 10 | JSONautas | **0.0000** | n/a | 0.5805 |
| 11 | MOLD | **0.0000** | 0.7354 | 0.7115 |

## Per-team best pipeline

| Team | gemma pipeline | gemma F1 | mode | qwen pipeline | qwen F1 | mode |
|:--|:--|--:|:--|:--|--:|:--|
| DRILLER | mixed-extractors-self-consistency-rag | 0.8285 | think | enriched-schema-rag | 0.8346 | nothink |
| Krishan | schema_dynamic | 0.8142 | nothink | schema_dynamic | 0.8076 | nothink |
| CodeStrange | combo_guard_react_repair_ref | 0.8090 | nothink | combo_guard_react_repair_ref | 0.8039 | nothink |
| FranRodrigo | cot | 0.7812 | nothink | cot | 0.8081 | nothink |
| SEsml | baseline | 0.7746 | nothink | adaptive | 0.8017 | think |
| VerbaNex | — | n/a | — | precision_master | 0.8013 | nothink |
| UC3M | prompted | 0.7984 | nothink | prompted | 0.3629 | nothink |
| GRADIANT | limited | 0.6792 | nothink | limited | 0.7564 | nothink |
| Inigo | e232 | 0.7504 | nothink | e232 | 0.7320 | nothink |
| JSONautas | — | n/a | — | baseline | 0.5805 | nothink |
| MOLD | vigil | 0.7354 | nothink | arcane | 0.7115 | nothink |

## Downloads

Machine-readable results, compacted per team.

- **[General results (JSON)](results/gensie-results.json)** — the full leaderboard plus per-team detail (all pipeline × model runs, precision/recall/F1, gap-closed, baselines, and methodology) in a single file.

Per-team bundles — each `.zip` contains a `summary.json` (that team's slice of the leaderboard: every scored pipeline × model run) plus the raw evaluation reports that back it:

| Team | Bundle |
|:--|:--|
| DRILLER | [DRILLER.zip](results/DRILLER.zip) |
| Krishan | [Krishan.zip](results/Krishan.zip) |
| CodeStrange | [CodeStrange.zip](results/CodeStrange.zip) |
| FranRodrigo | [FranRodrigo.zip](results/FranRodrigo.zip) |
| SEsml | [SEsml.zip](results/SEsml.zip) |
| VerbaNex | [VerbaNex.zip](results/VerbaNex.zip) |
| UC3M | [UC3M.zip](results/UC3M.zip) |
| GRADIANT | [GRADIANT.zip](results/GRADIANT.zip) |
| Inigo | [Inigo.zip](results/Inigo.zip) |
| JSONautas | [JSONautas.zip](results/JSONautas.zip) |
| MOLD | [MOLD.zip](results/MOLD.zip) |

## Methodology

20 of the 145 test instances were **excluded uniformly** from all systems and both baselines because their JSON schemas could not be reliably evaluated on the inference stack:

- **`medical_trials` (8) + `cultural_monuments` (2):** their `$defs`/`$ref` schemas cause a llama.cpp GBNF grammar-compilation failure (`400 failed to parse grammar`) — structurally unscoreable for any constrained-decoding system, including the official baseline.
- **`stem_biology` (10):** a **recursive** `$ref` taxonomic-tree schema (arbitrarily deep) that triggers parser recursion failures in participant systems (e.g. DRILLER's `maximum recursion depth exceeded`).

Scoring is on the remaining **125 instances**. The exclusion is uniform, so gap-closed comparisons remain fair.

### Caveats

- **Modes:** `nothink` = spec config (thinking off); `think` = June formal run (thinking on). Each team's best *reported* 125-instance result is used regardless of mode.
- **VerbaNex gemma:** all three VerbaNex gemma pipelines reliably destabilise the shared gemma backend over a full run (they complete in isolation, smoke F1 ≈ 0.76). VerbaNex therefore receives gap 0 on gemma and is carried by its strong qwen result; its true gemma standing is likely higher.
- **Engine stability:** the gemma backend (llama.cpp) crashes under sustained load; results were repaired per-instance on a restarted engine.
- Third-party self-eval data (CodeStrange, JSONautas) and June think-mode data are used where a team lacked a clean spec-config run of its own.
