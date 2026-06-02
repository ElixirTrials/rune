## Findings (ordered by risk)

- **High — Task 8 has a hidden behavior break (`project_label` removal).**  
  The plan says to remove `project_label`, but current templates still depend on it in both code and diagnosis prompts. If removed without coordinated template edits, renders will fail under `StrictUndefined` and prompt behavior changes unexpectedly.
  
```1:3:src/rune/templates/prompt_code.j2
You are implementing the subtask: {{ subtask_name }}
Project: {{ project_label }}
Follow the architecture plan in your context.
```

```7:10:src/rune/templates/prompt_diagnose.j2
Integration is failing.
Project: {{ project_label }}
Integration error: {{ integration_error[:300] }}
Subtasks (use these exact names in subtask_name):
```

- **High — Task 4d’s “join all steps into one completion” likely harms SFT quality.**  
  You’re moving to a `prompt`/`completion` contract (good), but the proposed `_join_field(..., "output")` can merge multiple retries/repairs into one completion string with separators. TRL expects each training example to be one coherent `prompt` + `completion` pair; concatenating multiple completions creates noisy targets and blurred supervision.  
  Substantiation: TRL docs explicitly define one record as either `text` or `(prompt, completion)`, and completion-only loss is applied to that completion span ([TRL SFT Trainer docs](https://huggingface.co/docs/trl/en/sft_trainer), [dataset formats](https://github.com/huggingface/trl/blob/main/docs/source/dataset_formats.md)).

- **Medium — Smoke gate criteria are too brittle for stochastic decoding.**  
  The merge gate currently requires strict outcomes (e.g., zero warnings) from single runs, but inference is sampled (`temperature` in run config). This can cause flaky pass/fail decisions unrelated to code quality.  
  Substantiation: deterministic generation in Transformers is achieved with `do_sample=False` ([Transformers testing guidance](https://github.com/huggingface/transformers/blob/main/docs/source/en/testing.md)).  
  Suggested fix: either run deterministic smoke settings for gating, or require N repeated runs with tolerance.

- **Medium — Migration safety for corpus format is under-specified.**  
  The plan changes corpus schema (`input/output` → `trajectory/prompt/completion`) but doesn’t enforce a schema/version guard. Mixed old/new sessions can silently degrade training (e.g., all rows dropped or malformed mapping).  
  I’d add `schema_version` to `metadata.json` in `write_session`, validate in miner, and fail fast if records don’t satisfy the contract.

- **Medium — Injection hardening is only partial.**  
  You add delimiters and one cap (`integration_doc`), but untrusted user-controlled fields can still be long (notably task text used in prompts). Delimiters help but don’t replace bounded input policy.  
  Substantiation: OWASP LLM guidance recommends both separation/delimitation and input validation/length limits ([OWASP Prompt Injection Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html), [LLMSVS](https://owasp.org/www-project-llm-verification-standard/LLMSVS-v2.0-en.html)).

- **Low — plan command examples violate your own repo run conventions in places.**  
  A few examples use `python` and `grep` directly. Your repo conventions in `CLAUDE.md` prefer `uv run` and your agent tooling strongly prefers `rg`. Not a design flaw, but worth cleaning for consistency/reproducibility.

---

## What I would change

- Keep **Phase ordering** and core intent (P0 smoke-gated train/serve alignment) — that part is strong.
- Update **Task 8** to either:
  - keep `project_label` as a separate capped field, or  
  - explicitly migrate all templates that use it to a new key in the same task.
- Change **Task 4d extraction granularity**: emit one training record per actionable step (or at least per retry), not one concatenated mega-record per `(action, target)`.
- Add **schema versioning + strict validation** for `session.jsonl` and miner ingestion before training.
- Make smoke gate **deterministic or multi-run tolerant**, not single-run absolute.
- Extend injection hardening with **caps on task/user fields** entering prompts, not only `integration_doc`.

---

## Open questions / assumptions

- Is the intended training objective to learn from **all attempts** (including failed ones) or only terminal successful outputs? That should drive Task 4d record design.
- Do you want prompt compaction optimized for token budget (`project_label`) or a single canonical field (`project`)? Right now Task 8 implies both.
- Should benchmark-only corpus production remain the sole source, or do you eventually want `rune run` sessions to opt-in emit compatible session logs?