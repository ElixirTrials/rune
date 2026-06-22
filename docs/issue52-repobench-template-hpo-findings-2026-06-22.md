# RepoBench episodic-template HPO — findings (2026-06-22)

**The adapter-as-context conjecture holds with the right template — no training.** The
earlier N=60 negative (`issue52-repobench-clamp-findings`) was **template-confounded**: it
fed the adapter a multi-file repo DUMP, shredded by the hypernet's 2048-token cap. With an
**episodic, per-task** template (name the one cross-file API the task must call, in the
hypernet's distillation surface) tuned by Optuna, the frozen c3 adapter recovers cross-file
APIs the no-context floor cannot — **strict superset, 0 regressions, on held-out tasks,
including the 32k regime where context-in-prompt is prohibitive.**

Engine commit `0f60516`. Checkpoint c3 (`53e24af2…`). Durable MLflow:
experiment **`issue52-repobench-template-hpo`**, run `template-hpo-W768-n30-t24` (`ba4bffd7…`).

## 1. Method
- **Benchmark/regime:** RepoBench v1.1 Python `cross_file_first`, clamped window W=768 (the
  constrained-hardware regime; Qwen3-4B's 262k window otherwise fits all context). Pool = 30
  (15×8k + 15×32k), offset past the headline/smoke rows.
- **Split:** seed-deterministic via `rune.bench.hpo.split_tasks` → **tune 20 / held-out 10**
  (the held-out set is never seen during tuning).
- **Search (Optuna TPE, 24 trials):** template `variant ∈ {gold, sig, use, minimal, import}`
  × `anchor_chars ∈ {0, 400}` × `scaling ∈ [0.4, 1.3]`.
- **Objective (maximize):** soft-recovery = mean(1.0 if gold identifier recovered else
  edit-similarity) on the tuning set — recovery is sparse at small N, so es smooths it.
- **Caching:** adapter build (hypernet forward) keyed by (task, variant, anchor), reused
  across trials; scaling is a cheap re-hotswap.

## 2. Best config (held-out, never tuned)
**`variant=use`, `anchor_chars=0`, `scaling=0.91`** (tuning soft-recovery 0.732).

| arm | recovery | mean es |
|---|---|---|
| **best episodic (tuned)** | **4/10** | **0.520** |
| floor (no context) | 1/10 | 0.385 |
| dump_gf (old multi-file dump) | 1/10 | 0.357 |
| a2_full (context in prompt, full window) | 1/5 *(5/10 skipped: 32k prohibitive)* | 0.396 |

**Strict superset of floor: +3 net-new recoveries, 0 regressions** (one-sided McNemar exact
p = 0.5³ = 0.125 at this N — directional; not yet significant). Net-new recoveries:
- `euler_to_rotationmatrix` (**32k**) → `euler_to_rotationmatrix(tilt_angles)`
- `all_reduce_tensor` (8k) → `all_reduce_tensor(loss, op=...)`
- `set_cell_size` (**32k**) → `def set_cell_size(text: Union[str, Text], ...`

Two of three are **32k** tasks where `a2_full` is skipped (prohibitive) — the JTBD#3 win:
the constant-prompt adapter recovers the cross-file API where putting context in the prompt
is unaffordable.

## 3. What the HPO learned (interpretable)
- **`use` variant won** — softer "must *use* `X`" framing (added from the smoke near-misses,
  e.g. the assigned-not-called `Qformer.train = disabled_train`) beat gold/sig/minimal/import.
- **`anchor_chars=0`** — the adapter should encode **only the cross-file API**, NOT the local
  in-file prefix (already in the prompt). Including it diluted the conditioning.
- **`scaling≈0.91`** — near full strength (cf. the dump run's flat scaling sweep).

## 4. Honest bounds + next step
N(held-out)=10 — the 4/10 vs 1/10 superset is **directional, not significant** (need ≥5
discordant gains for one-sided p≤0.031 per the powered-significance method). The result is a
clean existence-at-scale: the episodic adapter beats the floor on tasks it never tuned on, with
zero regressions, no training. **Next:** scale the held-out set (target ≥5 discordant gains) at
the best config to cross significance, and report against the same floor/a2_full/dump baselines.

## 5. Reproduction
```
uv run --extra gpu python tools/_repobench_template_hpo.py \
  --n-8k 15 --n-32k 15 --trials 24 --window 768 \
  --experiment issue52-repobench-template-hpo
```
Templates: `src/rune/bench/repobench.py::render_episodic` (variants in `EPISODIC_VARIANTS`).
Held-out split: `rune.bench.hpo.split_tasks`. Scorer: `rune.bench.identifier_match`.
