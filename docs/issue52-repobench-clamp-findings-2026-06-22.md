# RepoBench clamped-window (adapter-as-context) — findings (2026-06-22)

> **⚠️ CORRECTION (superseded in part).** The negative below used a *multi-file repo
> DUMP* as the adapter conditioning (`render_xfile_adapter 'structured'`), which is the
> WRONG template for an episodic per-task adapter. A follow-up HPO over an **episodic**
> template (name the one cross-file API the task must call, in the hypernet's training
> surface) **reverses the result**: tuned by Optuna (held-out 4/10 vs floor 1/10) and
> **confirmed at N=60 fresh rows — episodic 31/60 (0.517) vs floor 9/60 (0.150), McNemar
> p=3.0e-06**, including 32k tasks where context-in-prompt is prohibitive (skipped 30/30).
> See `docs/issue52-repobench-template-hpo-findings-2026-06-22.md`. The §2–§6 negative stands
> only as "the *dump* template fails"; "frozen c3 can't carry context / needs distillation"
> is **withdrawn** — it carries context fine with the right episodic conditioning, no training.

Tests PRODUCT.md's current bet — *"adapters provide unbounded context at constant
prompt length"* (JTBD #3) — on a real cross-file benchmark. **Result (DUMP template):
the conjecture does NOT hold.** Context delivered via the prompt doubles cross-file
recovery; context delivered via the *dumped* adapter is statistically indistinguishable
from no context. (The episodic template, tuned, does hold — see the correction banner.)

Engine commit `bae71d1`. Checkpoint c3 (`c3_t07_lp2_lg1.pt`, sha256 `53e24af2…`).
Durable MLflow: experiment **`issue52-repobench-clamp`**, run `clamp-W768-8k_32k-n60-seed0`
(`9ea65a3b…`) — params + per-task JSONL artifact + 52 metrics.

## 1. Setup (pinned)
- **Benchmark:** RepoBench v1.1 Python (`tianyang/repobench_python_v1.1`), split
  `cross_file_first` — the next line uses a cross-file API *for the first time*, so it
  is unsolvable without the cross-file context by construction. Dedup'd vs Stack v2.
- **The window wall (why a clamp is needed):** Qwen3-4B-Instruct has a **262,144-token
  context window**; RepoBench's hardest level is 128k. Context *always* fits the prompt,
  so the literal "context doesn't fit" conjecture is untestable at full window. We
  **impose a window budget `W`** (constrained-hardware regime = JTBD #3): the prompt is
  truncated to its last `W` tokens (cursor-adjacent code kept; front-loaded context evicted).
- **Arms** (per row; only the *delivery of cross-file context* varies):
  - `floor` — no context; prompt = clamp(prefix, W).
  - `a2_clamp` — context in prompt, clamped to W (front-loaded context evicted).
  - `a2_full` — context in prompt at FULL window (ceiling; skipped when the forward
    exceeds 12k tokens — that skip is the cost argument).
  - `nat@1.0` — context in the **adapter**, natural snippet order; prompt = clamp(prefix, W).
  - `gf@1.0` — context in the **adapter**, gold-snippet-first (gold def guaranteed within
    the hypernet's 2048-token conditioning budget).
- **Metric:** gold cross-file **identifier recovery** (did the needed API name appear in
  the completion) — primary; EM + edit-similarity secondary, no sandbox. Recovery is the
  honest metric: edit-similarity overcredits structural plausibility (a generic
  same-shape line scores high without recovering the API).
- **Params:** W=768, seed 0, temperature 0.0, max_new 48, N=60 (8k×30 + 32k×30).

## 2. Headline (N=60, W=768) — recovery rate
| arm | ALL (60) | 8k (30) | 32k (30) |
|---|---|---|---|
| floor (no context) | 0.233 | 0.200 | 0.267 |
| a2_clamp (ctx in prompt, truncated) | 0.200 | 0.167 | 0.233 |
| **a2_full (ctx in prompt, full window)** | **0.533** | **0.533** | *skipped 30/30 (prohibitive)* |
| nat@1.0 (ctx in adapter) | 0.233 | 0.167 | 0.300 |
| gf@1.0 (ctx in adapter, gold-first) | 0.217 | 0.167 | 0.267 |

**McNemar floor vs gf@1.0:** adapter-only=4, floor-only=5, **p=1.0** — the adapter is
statistically indistinguishable from (marginally below) the no-context floor.
Zero OOM across all 30 32k rows (alloc guards worked).

## 3. What it means (three robust conclusions)
1. **The frozen-c3 adapter adds nothing at scale.** `gf` (0.217) ≈ `nat` (0.233) ≈ `floor`
   (0.233). The adapter *uniquely* recovers 4 cross-file APIs the truncated prompt cannot
   (e.g. `SelfAttentionPooling`@8k, `Console`/`QiCell`@32k) — but **destroys 5** the base
   would have gotten (e.g. `raise PlayStatonOnStandbyError()` → `self._cancellation_token =
   asyncio.Event()`; `down_block = self.get_down_block(` → a wrong `getattr` form). It is a
   *noisy* channel: gains ≈ losses, net zero.
2. **Context works — through the prompt.** `a2_full` more than doubles recovery
   (0.20 → 0.53 at 8k): `PILtoTorch` floor guesses `image.resize(...)`, full-context
   recovers `PILtoTorch(image, None)` exactly; same for `mask_iou_loss`, `UNet`. This
   **isolates the failure to the adapter channel, not the task** — the gold IDs *are*
   recoverable from the context when the model can see it.
3. **The cost argument is real and quantified.** `a2_full` was skipped on **30/30** 32k
   rows because the full-context forward exceeds 12k tokens. So the prompt channel that
   *works* is exactly the one you *cannot afford* at length — which is precisely the
   motivation for an adapter channel that the frozen hypernet does not yet deliver.

## 4. The N=6 "existence proof" did not survive scaling
An earlier N=6 clamp probe found the adapter recovering `SelfAttentionPooling` where the
prompt could not, at 6× fewer prompt tokens — a tempting existence proof. At **N=60 it does
not hold**: that single recovery is real but is offset by the adapter's losses, and the
aggregate is a tie (p=1.0). Reported as a **negative**, not oversold.

## 5. Cheap levers exhausted (no-training)
Before scaling, a paired probe swept the no-training levers at W=768 (8k, N=6):
gold-first ordering, conditioning-cap lift (2048→4096), and adapter scaling (1.0/0.5).
**None strengthened the channel** (recovery stayed 1/6; scaling 0.5 lost it). The weakness
is the hypernet's *encoding* of repo context, not snippet ordering or truncation — c3 was
distilled on code-gen-from-spec trajectories (`## Task / ## Current Code / ## Review
Feedback`), a different distribution from repo-context→completion.

## 6. Honest conclusion + the lever
On a 262k-window base, the unbounded-context bet is only meaningful under an imposed
window budget. In that regime, **frozen c3 is insufficient**: the adapter neither beats the
no-context floor (p=1.0) nor approaches the context-in-prompt ceiling (0.22 vs 0.53). The
mechanism (clamp + escalate + adapter-as-context) is validated end-to-end and the ceiling
shows real headroom — but realizing it requires **distilling the hypernetwork on
repo-context→next-line pairs** (un-freezing c3), the pre-registered future work. This bounds
the adapter-as-memory claim with a second, independent benchmark (cf. the mutated-spec
pointer/content confound in the LCB findings).

## 7. Reproduction
```
uv run --extra gpu python tools/_repobench_clamp_run.py \
  --levels 8k,32k --per-level 30 --window 768 \
  --experiment issue52-repobench-clamp --out /tmp/rb_clamp_run.json
```
Template selection: `tools/_repobench_template_hpo.py`. Scorer + loader:
`src/rune/bench/identifier_match.py`, `src/rune/bench/repobench.py`. Window clamp:
`ModelWrapper.clamp_to_window`. Design spec:
`docs/superpowers/specs/2026-06-22-crossfile-context-adapter-benchmark-design.md`.
