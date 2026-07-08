# Engine Architecture

The engine is a single-node LangGraph `StateGraph` (`rune.engine.graph.create_engine`): one
`step` node, entry point `step`, and a conditional self-edge `step → {continue: step,
done: END}`. Iteration is driven entirely by the edge condition; the node itself is stateless
beyond the `RunState` TypedDict (`rune.engine.state`) threaded through it.

## Per-iteration: `step_node`

Each `step_node` invocation performs one full cycle:

1. **Action selection.** `select_action` (`rune.engine.policy`) deterministically returns the
   list of `Action`s for this step — possibly several sibling actions targeting independent
   subtasks in the same execution layer. An empty list signals termination.
2. **Per-action generation** (sequential over the returned actions):
   - `state_to_ctx` builds the Jinja2 context (subtask, plan, existing code, error summary,
     diagnosis-derived `fix_guidance`, capped `repair_history` and `code_trajectory`).
   - The adapter conditioning (trajectory text) is built by a Python renderer selected by the
     run-config `prompt_mode`: `render_training_format_trajectory` (default `full` mode —
     `## Task` / `## Current Code` / `## Review Feedback`, plus a `## Previous Attempts`
     section when there is history), `render_episode_adapter` (`episodic` / `escalate`
     modes), or `render_reference_adapter` (`reference_*` modes). Only the prompt is
     Jinja-rendered via `render_template` (`rune.engine.parse`), with per-mode overrides such
     as the `prompt_episodic_*` templates and `prompt_zeroshot`.
   - `model.generate_adapter(trajectory_text)` synthesizes a LoRA adapter from the trajectory
     text via the HyperLoRA hypernetwork (`rune.model`).
   - `scale_lora_b` (`rune.model.adapter`) scales the adapter's `lora_B` weights by
     `adapter_scaling`, then `model.hotswap_adapter` swaps the scaled state dict into the base
     model.
   - `model.generate` decodes against the action's `output_schema` when one is set: only
     `decompose`, `plan`, and `diagnose` (and the judges) are xgrammar-constrained. `code`,
     `repair`, and `integrate` have `output_schema=None` and emit freeform Python in a
     ` ```python ` fence, de-fenced by `extract_partial_code` — never JSON. The thinking
     phase runs only when `thinking_budget > 0` (the `PipelineConfig` default is 0;
     non-thinking is required for the Instruct base).
3. **Sandbox execution.** Code-bearing actions (`Action.executes_code`) have their output
   passed once through `extract_partial_code`; what is dispatched is not the bare extracted
   source but a probe built by `build_code_probe`: self-tests stripped (`strip_self_tests`)
   and the resolved public/acceptance checks appended. Probes run across all sibling code
   actions concurrently via `asyncio.gather` over `asyncio.to_thread(run_in_sandbox, …)`
   (`rune.sandbox.executor`), and `apply_oracle_fail_closed` forces a failing result when
   checks were configured but the probe did not fire, yielding per-subtask `Feedback`.
4. **Parse and state update.** `parse_output` is applied per action against an accumulating
   `running` snapshot (each sibling sees the prior sibling's applied change, avoiding
   stale-snapshot clobbering). The merged updates append `StepRecord`s to `trajectory`,
   advance `step`, and decrement `budget_remaining` by one.

## Deterministic action sequence

`select_action` realizes the sequence `decompose → plan → code → [diagnose → repair]* →
integrate` as a priority cascade over `RunState`:

- **No subtasks** → `decompose`.
- **Unplanned subtasks** → `plan` for every subtask in the first ready execution layer.
- **Failing subtasks** (`not code_passed[name] and not code_solved[name]`) → for each subtask
  in the first ready layer whose dependencies all pass:
  - flagged in `replan_targets` (with its plan cleared) → `plan` again before any further
    code/repair;
  - no code yet, or `retries ≥ MAX_REPAIRS (4)` → `code` (regenerate from scratch);
  - else if a diagnosis exists **or** a non-empty deterministic `repair_brief` already
    carries the structured failure signal → `repair` (the `diagnose` step is skipped);
  - else → `diagnose`.
- **All subtasks pass** → if exactly one subtask and its sandbox passed, set
  `integrated_code` from that subtask’s code and return `[]` (done; no integrate). If
  multiple subtasks, `integrate`; if integration already failed, `diagnose` then re-
  `integrate`; once `integrated_code` is set, return `[]` (done).

Repair budgeting: `MAX_REPAIRS=4`, `MAX_RETRIES=2·MAX_REPAIRS=8`. A subtask at `MAX_RETRIES`
is exhausted and dropped from selection. If all repairable work is exhausted and integration
is still failing, `select_action` returns `[]` to stop rather than burn the budget on a
non-converging `integrate ↔ diagnose` loop. One extra early-stop: a single benchmark subtask
(name `== entry_point`) with public checks and a retained `best_code` ships `best_code` and
returns `[]` rather than integrating.

## DAG execution layers

`build_execution_layers` uses `graphlib.TopologicalSorter` over `Subtask.depends_on`. Edges
are restricted to known subtask names so a phantom dependency cannot inject a node or block
readiness. `get_ready` yields each topological layer; `select_action` always operates on
`layers[0]` (the current ready frontier) and emits one action per subtask in it, which the
caller dispatches as parallel siblings. A `CycleError` degrades gracefully: all subtasks are
treated as independent (empty dependency sets), so the run proceeds rather than deadlocking.

## Continuation sub-loop

When `model.generate` reports a truncated result for a `code`, `repair`, or `integrate`
action, `step_node` enters an in-iteration continuation loop to recover the cut-off code. The
`prompt_code_continue` user prompt is rendered once before the loop and reused every round;
each round rebuilds the adapter conditioning via `render_training_format_trajectory` with the
last 3,500 chars (`_ACCUMULATED_CODE_CAP`) of `accumulated_code` as `## Current Code`,
regenerates an adapter scaled by `adapter_scaling · cont_multiplier (≈1.53)`, and calls
`model.generate_continuation` with the accumulated code as assistant prefix and a code-only
system prompt. The loop bounds itself by `cont_budget (5)`, exits early on
`degeneration_score > 0.5` (`rune.engine.continuation`), two consecutive empty rounds, a
`validate_syntax` pass, or a non-truncated round. The stitched `accumulated_code` becomes the
action's raw output and flows through `extract_partial_code` like any other code action
(freeform, never JSON-wrapped). See [Model](model.md) for adapter synthesis and continuation
scaling.

## Termination

`should_continue` returns `done` when `actions` is empty (policy signals no remaining work) or
`budget_remaining ≤ 0`, routing the conditional edge to `END`; otherwise `continue` re-enters
`step`.

See [API Reference](../api/engine/graph.md) for implementation details.
