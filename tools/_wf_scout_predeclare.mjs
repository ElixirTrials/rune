export const meta = {
  name: 'issue52-scout-predeclare',
  description: 'CPU-only scout of T0/E1/E2 harnesses + predeclared experiment spec (NO GPU work)',
  whenToUse: 'Issue #52 experimentation phase: orient on existing probe harnesses and freeze scoring rules before any GPU run.',
  phases: [
    { title: 'Scout', detail: 'parallel read-only deep-reads of the 4 key harnesses/infra (no model load)' },
    { title: 'Synthesize', detail: 'one agent merges findings into a predeclared T0/E1/E2 spec' },
  ],
}

// HARD CONSTRAINT (stated for any resume): every agent below is CPU-only — read files,
// grep, inspect existing logs/outputs, draft code, write specs. NONE may load the base
// model or run a GPU job. There is one 22GB GPU; a second concurrent base-model load OOMs
// the VM. All actual GPU execution is serialized by the main loop, not this workflow.

const REPO = '/workspaces/rune-gpu'

const SCOUT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['target', 'summary', 'key_findings', 'reuse_or_changes', 'existing_outputs', 'gotchas'],
  properties: {
    target: { type: 'string' },
    summary: { type: 'string', description: 'what this file/infra does, 2-4 sentences' },
    key_findings: {
      type: 'array', items: { type: 'string' },
      description: 'concrete facts: function names, args, masks, row-selection logic, truncation/seq settings, file:line anchors',
    },
    reuse_or_changes: {
      type: 'string',
      description: 'exactly how to reuse for the assigned experiment, or the precise code changes needed (diff-level intent + file:line)',
    },
    existing_outputs: {
      type: 'array', items: { type: 'string' },
      description: 'any already-produced result files, MLflow run IDs, or logged numbers found on disk that could avoid fresh GPU; empty if none',
    },
    gotchas: {
      type: 'array', items: { type: 'string' },
      description: 'leakage risks, truncation/seq mismatches, confounds, scaler_B / contract pitfalls relevant to the assigned experiment',
    },
  },
}

const COMMON = `
You are scouting code in the rune-gpu repo (${REPO}) for the issue #52 experimentation phase.
CONTEXT: a ctx-to-lora HyperLoRA hypernetwork (warm-start qwen_4b_d2l, r=8, lora_alpha=45.25,
target_modules={down_proj}) generates a LoRA adapter from trajectory text; the frozen 4-bit
Qwen3-4B base must then USE the embedded code/context at the next step. The product goal is the
adapter-as-substrate: code+context accessible WITHOUT the prompt. Known empirical state: warm-start
binds LABELS/signatures well (signature m-mismatch +3.84) but CODE BODIES barely (body +0.14);
feedback->edit on external_codereview is weak (held-out matched-swap +0.0687, frac 0.65; warm-start
baseline +0.0185, frac 0.48); calibration: hidden-task specificity +1.17, NIAH +7.7.

CRITICAL CONSTRAINT: you are CPU-only and READ-ONLY for analysis. DO NOT load the base model, DO NOT
run any GPU job, DO NOT execute training/eval. Use Read/Grep/Bash(ls,grep,find,cat-of-small-files) to
inspect code and look for EXISTING result files / logs / MLflow run dirs on disk. Report file:line
anchors. Be concrete and exact — your output is consumed by a synthesis step that must predeclare
frozen scoring rules, so precision about masks, negatives, row-selection, and truncation matters.
`

phase('Scout')

const scouts = [
  {
    label: 'scout:feedback_swap_eval',
    target: 'tools/_feedback_swap_eval.py',
    task: `Deep-read ${REPO}/tools/_feedback_swap_eval.py (the T0 harness). Report EXACTLY:
- CLI args (--ckpt, --n, etc.) and what they control.
- How it selects the eval rows (the 60 val episodes): which corpus file, ordering, seed, slice. Is row selection deterministic and reproducible across two checkpoints?
- How matched / swap / zero margins are computed; what tokens are scored (edit-local mask?) and how that mask is built.
- The truncation / max_seq_length the EVAL path uses. CRITICAL: the historical baseline +0.0185 was measured on a 2048 eval path; the smoke +0.0687 under a 768 fix. Determine what truncation this script currently applies and whether both checkpoints can be evaluated under IDENTICAL truncation on BYTE-IDENTICAL rows.
- The precise code changes to: (a) dump PER-EPISODE matched/swap/zero (row id + values) to JSONL; (b) add a SECOND-checkpoint arm so warm-start and trained are scored on the same rows in one run; (c) assert byte-identical rows + identical truncation across both arms.
Identify leakage/confound gotchas (length-regime shift, row drift, scaler_B contract).`,
  },
  {
    label: 'scout:episode_recall',
    target: 'third_party/doc-to-lora/rune_episode_recall.py',
    task: `Deep-read ${REPO}/third_party/doc-to-lora/rune_episode_recall.py — the positive-control recall harness. HIGHEST-LEVERAGE QUESTION: does it ALREADY produce continuation matched / mismatch / CEILING (prefix-in-prompt) numbers, or only matched-zero (+2.01 is the only logged continuation number so far)? Report:
- What facets/metrics it computes (goal, file, diff, tail/continuation, signature/body) and at what scale/scaling.
- Whether an in-context CEILING arm (prefix-in-prompt) exists or is one flag away.
- Search the repo + /tmp + third_party for EXISTING output files, JSON dumps, logs, or MLflow runs it produced (ls/grep/find). List every existing continuation/ceiling number you can find ON DISK with its source path. This may avoid fresh GPU for E1/E2 — be exhaustive.
- How it loads the warm-start vs other checkpoints; whether it could score E1 (oracle-vs-hypernet) or E2 (direction counterfactuals) with small changes.`,
  },
  {
    label: 'scout:specificity_probe',
    target: 'tools/_specificity_probe.py',
    task: `Deep-read ${REPO}/tools/_specificity_probe.py (matched/mismatch/zero + signature/body span split, task-in-prompt vs hidden). Report:
- How it builds the matched vs MISMATCH (derangement) negatives, and the task-hidden regime (the +1.17 regime).
- How the SIGNATURE span vs BODY span masks are constructed (this is the key reusable asset: E1 must score on BODY/code-content tokens, never signature, or the +3.84 shortcut inflates results).
- How adapters are generated/applied and at what scaling (the contract = lora_alpha=45.25).
- Exactly how this harness can be reused/extended for E1 (oracle-vs-hypernet @ matched rank, scored on body tokens) and E2 (minimal-edit counterfactuals scored on action/next-step tokens). What's already there vs what must be added.`,
  },
  {
    label: 'scout:oracle_train_infra',
    target: 'src/rune/training + tools/_distill_entry.py',
    task: `Scout the training infra for E1's oracle path + cross-over control. Read ${REPO}/src/rune/training/orchestrator.py (oracle QLoRA stage — handoff says it's a stub), ${REPO}/tools/_distill_entry.py, ${REPO}/src/rune/model/hypernetwork.py (adapter generation), and ${REPO}/configs/issue52_recipe_mvc_4b.yaml. Report:
- Is there an existing path to fit an ORACLE per-episode LoRA (standard PEFT LoRA trained by gradient on one episode's hidden-code facts) at r=8 and at higher rank? If the oracle stage is a stub, what's the minimal way to fit one (PEFT API already in deps)?
- How the hypernet generates an adapter for a given context (for the matched-rank comparison and the cross-over tiny-finetune control).
- The warm-start checkpoint path, rank/alpha/target_modules, and how to do a TINY hypernet fine-tune on a handful of exact facts (cross-over control: does a few updates move the hypernet where warm-start didn't?).
- RAM/GPU/seq constraints already wired (run_guarded.sh, max_seq_length=768, offload_base=False, 4-bit). These bound any GPU run the main loop will later launch.`,
  },
]

const findings = await parallel(
  scouts.map((s) => () =>
    agent(`${COMMON}\n\nYOUR TARGET: ${s.target}\n\n${s.task}`, {
      label: s.label,
      phase: 'Scout',
      schema: SCOUT_SCHEMA,
    }).then((r) => (r ? { ...r, _label: s.label } : null)),
  ),
)

const ok = findings.filter(Boolean)
log(`Scout returned ${ok.length}/${scouts.length} findings`)

phase('Synthesize')

const synthPrompt = `${COMMON}

You are the SYNTHESIS step. Below are structured scout findings on the four key harnesses/infra.
Produce a PREDECLARED EXPERIMENT SPEC (markdown) for T0, E1, E2 that the main loop will write
durably and freeze BEFORE running any trained-checkpoint delta (leakage rule). The spec must be
concrete enough to implement directly. Required sections:

## Existing-outputs triage (do this FIRST)
- List every existing on-disk continuation/ceiling/recall number the scouts found, with source path.
- VERDICT: does rune_episode_recall.py (or any dump) ALREADY answer E1/E2's ceiling question, so we
  can avoid fresh GPU? Be explicit about what is answered vs what still needs a run.

## T0 — paired significance (cheap rigor closure, NOT the decision)
- Exact code changes to _feedback_swap_eval.py: per-episode JSONL dump + second-checkpoint arm.
- The truncation-alignment check: re-run BOTH warm-start and trained smoke under IDENTICAL
  max_seq_length on byte-identical rows (do NOT reuse the historical 2048-path +0.0185). Assert
  byte-identical rows after truncation. State the assertion mechanism.
- Stats: paired bootstrap CI + sign test + row-level scatter (heavy-tailed; not t-test alone).
- Predeclared go/no-go interpretation in calibration-ladder units (do NOT anchor to NIAH +7.7).

## E1 — capacity vs representation (oracle-vs-hypernet @ matched rank; the lead discriminator)
- Frozen scoring rule: BODY/code-content token mask only (never signature). Define the mask precisely
  from the specificity-probe span logic.
- Same hidden-code facts, same masks/negatives/prompts for oracle and hypernet (oracle = UPPER BOUND,
  not proof the hypernet objective is wrong).
- Cross-over control: tiny hypernet fine-tune on the exact facts oracle succeeds on.
- Decision table: oracle good@r8 + hypernet bad@r8 -> representation wall (fine-tune); oracle bad@r8
  good higher -> capacity (rank/chunks); both bad high rank -> data/architecture.
- Minimal GPU plan + the implementation path from the oracle/infra scout.

## E2 — directionality (minimal-edit counterfactuals; scored on action-consequences)
- Counterfactual construction: minimally-edited, token/local-code preserving, change ONLY the causal
  arrow / next-action implication. Include a same-bag-of-events control. NO bare time-reversal /
  were<->heading text swaps (lexical artifacts).
- Score on next-step ACTION/code tokens vs the in-prompt CEILING, NOT "what happened first?" recall.
- Positive control: a curated episode where direction clearly changes the correct next action.

## Cross-cutting predeclared gates (write before any run)
- Calibration ladder thresholds; retention gate (NIAH/QA/tail recall preserved); generation-stability
  gate (xgrammar pass@1 not degraded). Positive control per experiment so a null distinguishes
  weak-signal from broken-harness.

## Implementation order + GPU budget
- Ordered checklist (which is pure-CPU code work vs which needs a serialized GPU run, and rough cost
  in "smoke-units"). Flag anything the existing outputs already cover.

SCOUT FINDINGS (JSON):
${JSON.stringify(ok, null, 2)}
`

const spec = await agent(synthPrompt, { label: 'synthesize:predeclared-spec', phase: 'Synthesize' })

return { scoutFindings: ok, predeclaredSpec: spec }
