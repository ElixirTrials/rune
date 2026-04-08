# Math Training Datasets — Column Reference

Quick reference for the four core math training/eval datasets.
All are stored as Arrow datasets under `libs/evaluation/src/evaluation/data/`.

---

## `numina_tir` — NuminaMath Tool-Integrated Reasoning

**Source:** `AI-MO/NuminaMath-TIR` · **Split:** train (72,441) + test (99)  
**Local path:** `data/numina_tir/train`

### Columns

| Column | Type | Description |
|---|---|---|
| `problem` | `str` | The question, LaTeX-formatted |
| `solution` | `str` | Full TIR solution (reasoning + Python + output + boxed answer) |
| `messages` | `list[{role, content}]` | 2-turn chat: `[user: problem, assistant: solution]` |

### Notes

- **No category or difficulty labels.** There is no subject, level, or source field.
- `messages[1]["content"]` is identical to `solution` — the two fields are redundant. Use whichever is more convenient for your training loop.
- **Solution structure** is always: natural-language reasoning → one or more ```` ```python ``` ``` blocks → ```` ```output ``` ```` block → continued reasoning → `\boxed{answer}` at the end.

```
...natural-language setup...

```python
from sympy import ...
# compute the answer
```
```output
63/400
```

Therefore the coefficient is \boxed{\frac{63}{400}}.
```

- On average **1.23 Python blocks per solution**; some solutions show self-correction (code errors → fixed code → re-run).
- **100 %** of solutions contain `\boxed{}`.
- **100 %** of solutions contain at least one ` ```python ` block.
- There is no separate "thinking" vs "answer" field — the solution is the full trajectory.

---

## `numina_cot` — NuminaMath Chain-of-Thought

**Source:** `AI-MO/NuminaMath-CoT` · **Split:** train (859,494)  
**Local path:** `data/numina_cot/train`

### Columns

| Column | Type | Description |
|---|---|---|
| `source` | `str` | Origin dataset / category (see below) |
| `problem` | `str` | The question, LaTeX-formatted |
| `solution` | `str` | Full prose CoT solution ending with `\boxed{answer}` |
| `messages` | `list[{role, content}]` | 2-turn chat: `[user: problem, assistant: solution]` |

### `source` distribution

| Source | Count | % | Notes |
|---|---|---|---|
| `cn_k12` | 276,554 | 32.2% | Chinese high-school curriculum |
| `synthetic_math` | 167,874 | 19.5% | Synthetic competition-style |
| `orca_math` | 153,314 | 17.8% | Synthetic word problems |
| `olympiads` | 150,563 | 17.5% | International olympiad problems |
| `synthetic_amc` | 62,108 | 7.2% | Synthetic AMC-style |
| `aops_forum` | 30,192 | 3.5% | AoPS forum posts |
| `math` | 7,477 | 0.9% | Hendrycks MATH dataset |
| `gsm8k` | 7,342 | 0.9% | GSM8K word problems |
| `amc_aime` | 4,070 | 0.5% | AMC/AIME competitions |

### Notes

- **`source` is the closest thing to a category label** — use it to filter by difficulty tier (e.g. `olympiads` and `amc_aime` are hardest; `cn_k12` and `orca_math` are easiest).
- `messages[1]["content"]` is identical to `solution` — same as numina_tir, fields are redundant.
- **Solution structure:** pure prose reasoning — no code blocks, no tool calls. Terminates with `\boxed{answer}`.
- **97.4 %** of solutions contain `\boxed{}`. The 2.6 % without boxed come almost entirely from `aops_forum` and `olympiads` where the answer is a proof / open-ended argument rather than a single value.
- **0 %** Python/tool-call code.
- No separate thinking vs answer field — the solution is one continuous reasoning block.

---

## `competition_math_train` / `competition_math_test` — Hendrycks MATH

**Source:** `EleutherAI/hendrycks_math` (mirror of the original `hendrycks/competition_math`, which was DMCA'd in Jan 2025)  
**Splits:** train (7,500) / test (5,000)  
**Local paths:** `data/competition_math/<subject>/<split>/` — **7 subject directories**, one Arrow dataset each.

### Columns

| Column | Type | Description |
|---|---|---|
| `problem` | `str` | The question, LaTeX-formatted |
| `level` | `str` | Difficulty: `"Level 1"` (easiest) – `"Level 5"` (hardest) |
| `type` | `str` | Subject / category (see below) |
| `solution` | `str` | Full proof-style solution with `\boxed{answer}` embedded |

### Level distribution (train)

| Level | Count |
|---|---|
| Level 1 | 564 |
| Level 2 | 1,348 |
| Level 3 | 1,592 |
| Level 4 | 1,690 |
| Level 5 | 2,304 |
| Level ? | 2 |

→ **Skewed towards harder problems** (Level 5 is the single largest group).

### `type` / Subject distribution (train)

| Type | Count |
|---|---|
| Algebra | 1,744 |
| Intermediate Algebra | 1,295 |
| Prealgebra | 1,205 |
| Number Theory | 869 |
| Geometry | 870 |
| Counting & Probability | 771 |
| Precalculus | 746 |

### Notes

- **`level` and `type` are the ranking/category labels** — this is the richest metadata of all four datasets.
- **Solution structure:** proof-style prose. `\boxed{answer}` typically appears at the end, but can appear mid-solution as the conclusion of a sub-argument. The `\boxed{}` encloses *only the final answer*, not the reasoning.
- **100 %** of solutions contain `\boxed{}`.
- **No code.** No tool calls, no Python blocks.
- **No `messages` column** — only `problem`, `level`, `type`, `solution`.
- No separate thinking / answer field. The solution is one prose block.
- The dataset is multi-directory: each of the 7 subjects is its own Arrow dataset on disk. The analysis script concatenates them automatically; use `concatenate_datasets()` in code.

---

## At a Glance

| | `numina_tir` | `numina_cot` | `competition_math` |
|---|---|---|---|
| **N (train)** | 72,441 | 859,494 | 7,500 |
| **Category label** | ✗ | `source` (9 values) | `type` (7 subjects) |
| **Difficulty label** | ✗ | ✗ (proxy: source) | `level` 1–5 |
| **Python code** | ✅ 100% | ✗ | ✗ |
| **`\boxed{}` in solution** | ✅ 100% | ✅ 97.4% | ✅ 100% |
| **`messages` column** | ✅ (mirrors solution) | ✅ (mirrors solution) | ✗ |
| **Thinking vs answer split** | ✗ one blob | ✗ one blob | ✗ one blob |
| **Avg answer length** | ~2,019 chars / ~505 tok | ~1,171 chars / ~293 tok | ~536 chars / ~134 tok |
| **Multi-directory on disk** | ✗ | ✗ | ✅ (7 subjects) |
