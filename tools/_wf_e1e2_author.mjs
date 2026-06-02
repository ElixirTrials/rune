export const meta = {
  name: 'issue52-e1e2-author',
  description: 'CPU-only authoring of E1/E2 harness code+data against the FROZEN spec (NO GPU work)',
  whenToUse: 'Overlap E1/E2 code authoring with the running T0 GPU job; outputs are drafts for main-loop review.',
  phases: [{ title: 'Author', detail: '4 parallel agents, each owns one file/deliverable (no model load)' }],
}

// HARD CONSTRAINT: every agent is CPU-only. The single GPU is occupied by the running T0 eval.
// NO agent may load a model, run training/eval, or launch a GPU job. Read code + the frozen spec,
// then WRITE code/data files. Each agent owns a DISTINCT file (no concurrent edits to one file).
// Outputs are DRAFTS — the main loop reviews against the contract before any GPU run.

const REPO = '/workspaces/rune-gpu'
const SPEC = `${REPO}/docs/issue52-predeclared-spec-T0-E1-E2-2026-06-02.md`

const COMMON = `
You are authoring code/data for the issue #52 experimentation phase in ${REPO}.
READ FIRST (mandatory): the FROZEN predeclared spec at ${SPEC} — its scoring rules, masks,
negatives, truncation, and adapter-contract numbers are FROZEN and authoritative. Do NOT
deviate from them; if something is underspecified, follow the spec's intent and note the
assumption in open_questions.

CONTRACT FACTS (from the spec; get these EXACTLY right):
- Adapter contract: effective_scaling = lora_alpha = 45.2548 applied UN-DIVIDED (NOT alpha/r).
  Hypernet/functional path: rune.model.adapter_contract (effective_scaling, assemble_adapter).
- Hypernet geometry: r=8, target_modules={'down_proj'}, 36 layers (Qwen3-4B). layer_indices from
  hyp.config.layer_indices.
- An ORACLE PEFT LoRA that must match this substrate uses LoraConfig(r=8,
  target_modules=['down_proj'], lora_alpha=8*45.2548, lora_dropout=0.0) — because PEFT applies
  alpha/r while the functional path applies lora_alpha un-divided, so 8*45.2548 / 8 = 45.2548.
- BODY span scoring (E1): score answer tokens [hi, len) where [lo,hi) is the def-<entry_point>(
  signature line. NEVER score signature or full-span (signature m-mismatch +3.84 dwarfs body +0.14).
- Base load: 4-bit nf4, bf16 compute, double-quant; flash_attention_2; device_map {'':'cuda'}.

CRITICAL: CPU-only. DO NOT import torch-on-GPU to RUN anything, DO NOT load any model, DO NOT
execute training/eval/bench, DO NOT launch a GPU job (the GPU is busy with T0). You MAY read code,
grep, and write files. Verify your file PARSES (python -c "import ast; ast.parse(open(p).read())")
but do NOT run it. Return concrete file paths and a self-check against the contract above.
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['deliverable', 'files_written', 'contract_self_check', 'how_main_loop_runs_it', 'open_questions'],
  properties: {
    deliverable: { type: 'string' },
    files_written: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['path', 'summary'],
        properties: { path: { type: 'string' }, summary: { type: 'string' } },
      },
    },
    contract_self_check: {
      type: 'array', items: { type: 'string' },
      description: 'explicit confirmation each frozen-contract item is honored (lora_alpha, target_modules, BODY span, truncation, no-GPU)',
    },
    how_main_loop_runs_it: { type: 'string', description: 'exact command the main loop will run on GPU under tools/run_guarded.sh' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

phase('Author')

const tasks = [
  {
    label: 'author:E1-body-mask',
    prompt: `${COMMON}

DELIVERABLE (E1 mask freeze): Harden ${REPO}/tools/_specificity_probe.py.
- Find span_bounds (~line 133-150) and its (0,0) missing-marker fallback (~line 144). Per the spec,
  the (0,0) fallback silently makes BODY = full answer (incl. signature), collapsing the body-vs-
  signature discriminator. CHANGE: raise a clear ValueError if the def-<entry_point>( marker is not
  found (j < 0), so an episode is NEVER scored under the (0,0) fallback. Callers must exclude such
  episodes with an explicit reason rather than silently mis-score.
- Confirm (in code comments + your self-check) that the BODY span scored is [hi, len) and the
  SIGNATURE span is [lo, hi). Do NOT change the scoring math (mean_gold_logprob).
- Keep the change minimal and diff-style. Ensure the file still parses.`,
  },
  {
    label: 'author:E1-oracle',
    prompt: `${COMMON}

DELIVERABLE (E1 lead discriminator): Author NEW file ${REPO}/tools/_e1_oracle.py — an oracle
per-episode LoRA capacity probe, scored on the BODY span, comparable to the hypernet.
- For each of the frozen MBPP episodes used by _specificity_probe.py (hidden/absent-template
  regime: render via render_training_format_trajectory with current_code+feedback empty), fit an
  ORACLE PEFT LoRA by CE on the episode ANSWER span ONLY (answer-preserving truncation via
  rune.training.hypernet_distill._prepare_ids at max_seq_length=768). Use
  peft.get_peft_model(base, LoraConfig(r=RANK, target_modules=['down_proj'],
  lora_alpha=RANK*45.2548, lora_dropout=0.0)). --rank arg (default 8); also support 16/32 for the
  capacity branch. gradient_checkpointing may be True for the oracle (PEFT-native).
- Then SCORE the trained oracle on the BODY span [hi,len) through the IDENTICAL mask + math as the
  hypernet (reuse _specificity_probe.py's span_bounds + scoring_core.mean_gold_logprob; import them).
  Report matched / mismatch(derangement) / zero on the BODY span, per episode + mean, to a JSONL --out.
- Mirror the 4-bit base load + PEFT setup from ${REPO}/tools/diag_pre_corpus_gate.py (~lines 137-162).
- This file TRAINS on GPU when the main loop runs it — but YOU must NOT run it. Author + ast-parse only.
- Provide how_main_loop_runs_it = the exact tools/run_guarded.sh command (per-rank).`,
  },
  {
    label: 'author:ceiling-arm',
    prompt: `${COMMON}

DELIVERABLE (E1/E2 in-context CEILING, ~10-line add): There is NO in-context ceiling arm on disk
(doc-prefix-in-prompt, no adapter, same scored span) for goal/file/diff/body. Add one.
- Edit ${REPO}/third_party/doc-to-lora/rune_episode_recall.py: add a logits_ceiling(model, ...)
  path that builds the prompt with the DOC TEXT prepended (in-prompt), uses the base model with
  NO adapter, and scores the SAME answer span as the matched/mismatch arms. Add a per-target
  '*_ceil' column to the output so each facet reports matched / mismatch / zero / CEIL.
- If the tail/continuation facet has its own harness (rune_continuation.py), mirror the same
  ceiling add there; otherwise note it.
- Keep the existing matched (own doc) / mismatch (ctxs[(i+1)%len], a structured negative) / zero
  arms UNCHANGED. Minimal diff. Ensure files still parse.
- Read the file carefully first to match its existing scoring + adapter-application conventions
  (native lora_alpha=45.2548, r=8, down_proj; model.patch_lora_forward).`,
  },
  {
    label: 'author:E2-counterfactuals',
    prompt: `${COMMON}

DELIVERABLE (E2 directionality data): Construct the counterfactual dataset + controls as a data
file ${REPO}/tools/_e2_counterfactuals.json (or .jsonl) plus a short builder/loader
${REPO}/tools/_e2_build.py that documents construction.
- Per the spec: MINIMALLY-EDITED counterfactuals that preserve tokens/local-code and change ONLY
  the causal arrow / next-action implication. FORBIDDEN: bare time-reversal or were<->heading text
  swaps (lexical artifacts). For each matched episode provide: (1) the counterfactual doc (direction
  flipped), (2) a SAME-BAG-OF-EVENTS control (same events, no directional flip / neutral reorder).
- Include ONE curated POSITIVE-CONTROL episode where flipping the causal direction provably changes
  the correct next action (e.g. 'test added then code written' vs 'code written then test added' =>
  different correct next edit), with the expected next-step action tokens labeled, so the ceiling
  arm can confirm matched/ceiling >> counterfactual.
- Scoring target = NEXT-STEP ACTION/code tokens (the consequence direction determines), NOT
  'what happened first?' recall. Document the scored span for each item.
- Draft 4-8 episodes is enough for a first pass; mark which are synthetic vs adapted from real
  trajectories. This is DATA + design; no model is loaded. Ensure JSON/py parse.`,
  },
]

const out = await parallel(
  tasks.map((t) => () => agent(t.prompt, { label: t.label, phase: 'Author', schema: SCHEMA })
    .then((r) => (r ? { ...r, _label: t.label } : null))),
)

return { authored: out.filter(Boolean) }
