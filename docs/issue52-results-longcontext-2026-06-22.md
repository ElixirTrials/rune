# Results: adapter-carried context under a prompt-length budget (2026-06-22)

**Status:** publication-ready Results prose for `paper_v9.tex` §4. Fills the gap the
results-section guide flagged as un-measured (§4.6: "constant-prompt advantage… requires a
step-indexed eval with a fixed prompt budget and a deliberate context-window stressor on the
baseline arm — not yet measured in Rune") and corrects the HumanEval+ "difficulty-dependent"
limit. All numbers durable in MLflow (`issue52-repobench-clamp`, `issue52-repobench-template-hpo`,
`issue52-humanevalplus`), engine commit `efa7b9e`, checkpoint c3 `53e24af2…`, seed 0.

---

## 4.x  Adapter-carried context versus in-prompt context under a token budget

### Motivation

The central tension of this work (§2) is that in-context learning pays for task information in
prompt tokens on every query, whereas a hypernetwork-generated adapter amortises that information
into fixed-size weights at constant prompt length [1,4,5]. A fair test of the adapter-as-context
claim must therefore (i) require information that lives outside the local prompt, (ii) impose a
prompt-length budget under which in-prompt delivery of that information becomes lossy or
infeasible, and (iii) compare delivering the same information through the prompt versus through the
adapter. We construct exactly this test on real cross-file code completion.

### Experimental design

**Task.** RepoBench v1.1 (Python), split `cross_file_first` [next-line completion where the target
line invokes a cross-file API *for the first time*, so the completion is unsolvable without the
cross-file definition]. We sample N = 60 problems stratified across two context-length strata (30
at the 8k level, 30 at 32k), drawn from a held-out offset disjoint from all tuning data.

**Prompt-length stressor.** The base model (Qwen3-4B-Instruct) has a 262k-token window, so on this
benchmark the cross-file context always fits in the prompt and the constant-prompt question is
vacuous at full window. We therefore impose a deployment-realistic **token budget W = 768**: the
model prompt is truncated to its last W tokens (cursor-adjacent code retained; earlier content,
including the prepended cross-file context, evicted first). This simulates the memory-constrained
regime the system targets (PRODUCT.md JTBD-3) and creates the regime in which the in-prompt and
in-adapter channels can be separated.

**Arms.** Identical base model and decoding (greedy, T = 0); only the delivery of the cross-file
context varies:

| Arm | Cross-file context delivered via | Model prompt length |
|---|---|---|
| `floor` | none | ≤ W (clamped) |
| `a2_clamp` | prepended to the prompt, then clamped to W | ≤ W (context evicted on overflow) |
| `a2_full` | prepended to the prompt at full window | grows with context (unbudgeted ceiling) |
| **`episodic`** | the LoRA adapter (hypernetwork conditioning) | **≤ W (constant; context is not in the prompt)** |

The adapter arm conditions the hypernetwork on an **episodic, per-task** surface — a single block
naming the one cross-file symbol the task must use and its definition, rendered in the network's
distillation format (`## Task / ## Current Code / ## Review Feedback`). The conditioning text
averages 124 tokens (median 52); the model prompt remains the clamped W = 768.

**Metric.** Gold cross-file **identifier recovery**: whether the predicted line contains the
required cross-file symbol (the dataset's gold dependency). This isolates the memory question —
did the channel deliver the needed cross-file API — independent of full-program execution, which
next-line completion does not define. We report Wilson 95% intervals and paired McNemar exact
tests.

### Result 1 — the adapter matches full-context prompting at a fraction of the prompt length

| Arm | Recovery | Wilson 95% CI |
|---|---|---|
| `floor` (no context) | 9/60 = 0.150 | [0.081, 0.261] |
| `a2_clamp` (context in prompt, budgeted) | 11/60 = 0.183 | [0.106, 0.299] |
| **`episodic` (context in adapter, budgeted)** | **31/60 = 0.517** | **[0.393, 0.638]** |
| `a2_full` (context in prompt, unbudgeted ceiling) | 17/30 = 0.567 | [0.392, 0.726] |

Under the W = 768 budget, delivering the context through the prompt barely improves over no
context (`a2_clamp` 0.183 vs `floor` 0.150; CIs overlap heavily) because the context is the first
content evicted on overflow. Delivering the *same* context through the adapter raises recovery to
**0.517** — a 3.4× increase over the floor, with non-overlapping confidence intervals — at the same
clamped prompt length. The paired test is unambiguous: of the 24 problems on which the adapter and
the floor disagree, the adapter recovers the symbol on **23** and the floor on **1** (McNemar exact
**p = 3.0 × 10⁻⁶**). Critically, the budgeted adapter arm is **statistically indistinguishable from
the unbudgeted full-context ceiling** (0.517 [0.393, 0.638] vs 0.567 [0.392, 0.726]) while using a
constant 768-token prompt against a full-context prompt averaging 12,836 tokens — a **16.7× prompt-
length reduction** at parity recovery. Twenty-two of the 31 adapter recoveries are *beyond the
clamped prompt* (the gold symbol is absent from the retained 768-token window and is supplied only
by the adapter).

### Result 2 — the advantage is largest exactly where the prompt cannot hold the context

Stratifying by context length isolates the mechanism (Wilson 95% CIs):

| Stratum | `floor` | `episodic` | `a2_full` (ceiling) |
|---|---|---|---|
| 8k (n = 30) | 0.167 [0.073, 0.336] | **0.600** [0.423, 0.754] | 0.567 [0.392, 0.726] |
| 32k (n = 30) | 0.133 [0.053, 0.297] | **0.433** [0.274, 0.608] | **undefined — prompt prohibitive** |

At 8k the adapter already meets or exceeds the full-context ceiling (0.600 vs 0.567). At 32k the
full-context prompt is **prohibitive on every problem** (context exceeds the 12k-token grading
budget; the `a2_full` arm has no answer on 30/30), yet the adapter still recovers **13/30 = 0.433**
at the constant 768-token prompt — 3.3× the floor. This is the constant-prompt claim demonstrated
directly: the adapter supplies cross-file context precisely in the regime where the prompt cannot
fit it. Against the 32k context the prompt-length reduction is **26.8×**.

### Result 3 — the conditioning format, not the adapter mechanism, is the controlling variable

The same adapter mechanism conditioned on a naïve **multi-file dump** of the surrounding repository
(all candidate snippets concatenated) is a null: in a matched N = 60 run it scored 0.217, within
noise of its floor of 0.233 (McNemar p = 1.0), because the hypernetwork's 2048-token conditioning
limit shreds an undifferentiated dump and the relevant definition is diluted or truncated. Holding
the mechanism, checkpoint, and prompt budget fixed and changing **only** the conditioning to the
episodic per-task surface moves recovery from floor-level (0.183, `dump`) to 0.517 (`episodic`).
The episodic configuration was selected by an Optuna study (24 trials over template variant × in-
file anchor × adapter scaling) on a tuning split and validated on a disjoint held-out split (4/10
vs floor 1/10) *before* the N = 60 confirmation reported above, so the reported rows were never
tuned on. The selected configuration is interpretable: the conditioning should name the single
required cross-file symbol and **exclude** the local in-file prefix (which is already in the
prompt), confirming that the adapter's value here is as a *context channel*, not a generic
output bias. No adapter weights were trained — this is a conditioning-format result on the frozen
checkpoint.

### Result 4 — the HumanEval+ "regression" was a grading artifact; the engine is a strict superset

A prior result reported the full engine (escalate, adapter on) *underperforming* the base model on
HumanEval+ (100/164 vs 116/164, −16) and concluded the system was "difficulty-dependent, hurting
easy tasks." Auditing traced this entirely to two evaluation defects, not a capability regression:
(a) the harness graded the generated function body without the prompt's imports, so a correct
solution whose body did not re-emit `from typing import …` failed with a definition-time
`NameError` on its signature annotation (19/164 tasks carry such a signature); and (b) the engine's
escalation floor was entered on an untrusted model-authored check, discarding correct zero-shot
solutions. With both fixed (graded program = prompt imports + solution + held-out tests; entry
gated only by trusted public examples), re-running both arms at commit `efa7b9e`:

| Arm | Pre-fix | Post-fix |
|---|---|---|
| base (single-shot) | 116/164 = 0.707 | 134/164 = 0.817 [0.751, 0.869] |
| c3 (engine: escalate, adapter @ 1.0) | 100/164 = 0.610 | 135/164 = 0.823 [0.758, 0.874] |

The base baseline rises +18, entirely within the 19 import-vulnerable tasks — a grading correction,
not a model gain. The engine is then a **strict superset** of the corrected base (135 ⊇ 134: one
gain, `HumanEval/10`, a multi-step constructive routine the zero-shot gets subtly wrong and the
repair loop fixes; **zero regressions**). The "difficulty-dependent / hurts easy tasks" claim is
**retracted**: on a saturated, easy benchmark the engine is a no-harm, marginally-positive superset
once the grading confound is removed — the expected null result, and consistent with the cross-file
finding that adapter gains concentrate where the base genuinely lacks the needed context.

### What this establishes, and its limits

These results establish, under a controlled prompt-length budget on a real cross-file benchmark,
that a frozen hypernetwork adapter delivers out-of-prompt context **at parity with full in-prompt
context** while using an order of magnitude fewer prompt tokens, and that it remains effective in
the long-context regime where in-prompt delivery is infeasible (32k: adapter 0.433, prompt
undefined; McNemar p = 3 × 10⁻⁶ overall). This is the constant-prompt, beyond-budget evidence the
prior write-up listed as required-but-unmeasured, and it is directionally consistent with the
Doc-to-LoRA and SHINE long-context results in the literature [1,2], here demonstrated for *coding*
trajectories rather than QA.

Bounds, stated explicitly. (1) The metric is cross-file **identifier recovery**, not functional
pass@1; next-line completion defines no hidden tests, so this measures whether the memory channel
surfaces the required API, not end-to-end task solving. (2) `a2_full` is a **partial ceiling**,
defined on 30/60 problems and undefined at 32k by construction. (3) The adapter is a near-, not
strict-, superset of the floor (23 gains, **1 regression** at 32k where it substituted a related
but wrong call). (4) The conditioning format is decisive and was tuned (held-out validated); the
result is a property of the episodic surface on this checkpoint, not of the adapter mechanism in
isolation. (5) This is a (v)-Rune-vs-context-channel comparison, **not** a pre-registered Gate
verdict (Gate 1 still requires the direct-PEFT comparator).

---

### References
[1] Doc-to-LoRA (Charakorn et al., 2026). [2] SHINE (Liu et al., 2026). [4] Liu et al., 2022
(ICL vs PEFT). [5] LoRA as Knowledge Memory (2026). Numbering follows
`issue52-results-section-guide.md` §9.

**Reproduction.** `tools/_repobench_clamp_run.py --levels 8k,32k --per-level 30 --offset 100
--window 768`; episodic template `src/rune/bench/repobench.py::render_episodic` (variant `use`,
anchor 0, scaling 0.91, selected by `tools/_repobench_template_hpo.py`); HumanEval+
`tools/_he_run.py --arm {base,c3}`. Per-task artifacts attached to each MLflow run.
