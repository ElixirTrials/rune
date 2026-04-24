## Methods

Building on the theoretical foundations established in the Background section — LoRA decomposition, hypernetwork architectures, PBB's procedural encoding findings, and the trajectory modality argument — this section specifies Rune's concrete architecture, algorithms, and design decisions. Each subsection identifies the components, their interactions, and the claim tier for every non-trivial design choice. Where Background introduced a concept, Methods references it without re-derivation. Where Methods introduces new formalism, it is marked as such.

---

### System Architecture Overview

Rune is structured as five services collaborating around a central adapter registry. Each service owns a single responsibility; inter-service communication occurs via HTTP APIs and a shared filesystem-backed adapter store.

**rune-agent** is the core execution service. It implements a five-phase pipeline — decompose, plan, code, integrate, diagnose/repair — with DAG-ordered subtask execution and a two-step error recovery mechanism. The agent receives a task description, decomposes it into dependency-ordered subtasks, plans each subtask, generates code in topological order (publishing interfaces to a typed blackboard for downstream subtasks), integrates the results, and applies diagnose-then-repair when code fails. The fifth phase (diagnose/repair) addresses the prompt-adapter tension described in [Background](background.md#the-prompt-adapter-tension-in-error-recovery): diagnosis places error context in the prompt and code in the adapter to produce a concise fix instruction; repair then uses that diagnosis as prompt guidance while the adapter retains domain context. This separation avoids forcing both error details and domain knowledge through the same channel. The pipeline is orchestrated by 18 Jinja2 templates (code.j2, code_continue.j2, code_repair.j2, code_retry.j2, decompose.j2, diagnose.j2, integrate.j2, plan.j2, prompt_code.j2, prompt_code_continue.j2, prompt_code_repair.j2, prompt_code_retry.j2, prompt_decompose.j2, prompt_decompose_concise.j2, prompt_diagnose.j2, prompt_integrate.j2, prompt_integrate_retry.j2, prompt_plan.j2), where each phase has both a trajectory template (adapter context) and a prompt template (model orientation). The pipeline structure is **specified** — it is implemented in the codebase and tested (776+ tests passing).

**lora-server** wraps vLLM with optional pipeline parallelism and dynamic adapter loading following the S-LoRA unified paging pattern.[^sheng2023slora] It serves the base model with QLoRA quantization and exposes an OpenAI-compatible API that accepts adapter selection metadata per request. Development and benchmarking use Gemma 2 2B (google/gemma-2-2b-it) with a Sakana "gemma_demo" checkpoint variant; the production target is Qwen/Qwen3.5-9B. The serving configuration is discussed in detail in the Training Data Strategy and Serving Architecture subsection.

**training-svc** manages adapter production and is the most complete service implementation. It runs direct LoRA fine-tuning on trajectory data using PEFT utilities and hosts the hypernetwork that replaces gradient descent with a single forward pass. A model registry with DeltaCoder warm-start capability enables incremental adapter training from existing checkpoints rather than cold-starting each training run. Training jobs acquire a GPU lease from lora-server, which yields one GPU for the duration of the training job.

**evolution-svc** manages the adapter lifecycle — consolidation, update, archival, and experimental merging — using fitness-driven selection rather than naive composition. The service layer contains stubs; the real evolution logic (TIES/DARE merging, pruning, lineage tracking) is implemented in the scripts layer (`scripts/swarm_evolution.py`). Operations are detailed in the Evolution Operator subsection.

**api-service** provides REST API stubs for external integration. The primary execution path bypasses the service layer: `scripts/rune_runner.py` runs the five-phase pipeline directly, and `scripts/swarm.py` orchestrates multi-agent swarm execution with training pool management, evolution workers, and watchdog supervision.

**adapter-registry** is the shared dependency root: a SQLite-backed metadata store paired with safetensors adapter files on the local filesystem. Every service reads from or writes to the registry. Adapters are write-once and immutable after creation; metadata fields (fitness score, archive status) are mutable. The three-level hierarchy (project, domain, task) is encoded in both the filesystem path convention and the SQLite schema.

**Claim tiers:** The five-service architecture and five-phase pipeline are **specified** — they are implemented in the codebase with 776+ tests passing. Phase 0 and Phase 1 empirical performance on coding tasks is **TBD**.

---

### Doc-to-LoRA Hypernetwork Adaptation

The Background section established that Doc-to-LoRA validates the hypernetwork-to-LoRA mechanism: a hypernetwork takes a document as input and produces LoRA adapter weights in a single forward pass, with no gradient-based fine-tuning at inference time.[^charakorn2026doc2lora] The generated adapter encodes the document's content into the base model's parameter space via the low-rank decomposition \(\Delta W = BA\) established in Background.[^hu2021lora]

Doc-to-LoRA was validated on document QA tasks — retrieving specific facts from long textual inputs. It was not validated on coding trajectories, procedural knowledge, or code execution traces. Rune proposes extending the hypernetwork mechanism to a structurally different input modality: code execution trajectories containing sequential (code, execution result, reflection) triples. This extension is **proposed** and requires Phase 1 empirical validation.

Rune's hypernetwork defines the following formal mapping (new equation):

\[
H: \mathcal{T} \rightarrow (B, A), \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}
\]

where \(\mathcal{T}\) denotes the trajectory space — a sequence of (code, execution result, reflection) triples produced by the recursive loop. The hypernetwork \(H\) takes a complete trajectory and produces the LoRA weight matrices \(B\) and \(A\) that, when applied to the base model via the \(\Delta W = BA\) decomposition from Background, encode the procedural knowledge from that trajectory into the model's parameter space. The hypernetwork produces per-layer \((B, A)\) pairs — one for each targeted transformer layer — though the equation above shows a single pair for notational clarity.

The architecture of \(H\) consists of two stages. First, a variable-length trajectory encoder processes the trajectory into a fixed-size latent representation. The encoder must handle trajectories of varying length (different numbers of attempts, different code lengths per attempt). The proposed encoder design uses a Perceiver-based cross-attention pooling mechanism:[^ha2016hypernetworks] a set of learned latent queries attend to the trajectory token sequence, producing a fixed-dimensional representation regardless of input length. This Perceiver-based encoder is a **proposed architecture choice and Phase 1 ablation target** — alternative encoders (mean pooling, hierarchical attention, recurrent summarization) will be compared during Phase 1 hypernetwork development.

Second, per-layer output heads map the fixed-size latent representation to \((B, A)\) matrices for each targeted transformer layer. Each output head is a linear projection from the latent dimension to the flattened \(B\) and \(A\) matrices for one layer.

For trajectories exceeding the encoder's context limit, the trajectory is chunked at attempt boundaries (each attempt is one chunk), and chunk representations are averaged before being passed to the output heads. This chunking mechanism preserves the self-contained structure of individual attempts while allowing arbitrarily long trajectories to be processed.

The three structural properties of code execution trajectories identified in Background — temporal ordering, procedural abstraction, and self-correcting structure — motivate the choice of trajectories as input modality. These properties distinguish trajectories from documents (unordered factual content) and from few-shot examples (static input-output pairs), and are the basis for expecting the hypernetwork mechanism to transfer from documents to trajectories.

**Claim tiers:** The hypernetwork-to-LoRA mechanism is **expected** to extend from documents to trajectories based on the structural analogy established in Background. Trajectory-to-adapter quality on coding tasks is **proposed** — it is the central empirical hypothesis requiring Phase 1 validation. The Perceiver-based encoder architecture is an **ablation target** — it will be compared against alternatives during Phase 1 development.

---

### Adapter Distillation Pipeline

The distillation pipeline has two implementation paths depending on the development phase. Phase 3 uses direct LoRA fine-tuning as a bootstrapping path — a validated technique that establishes the training infrastructure and produces the first adapter corpus. Phase 4 replaces gradient descent with the hypernetwork forward pass described above — the target architecture that eliminates per-adapter training cost at inference time. The pseudocode below covers both paths.

```text
Algorithm: Rune Recursive Loop
-------------------------------------------------------------
Input:
  task_description : str            (RuneState.task_description)
  test_suite       : str            (RuneState.test_suite)
  adapter_ids      : list[str]      (RuneState.adapter_ids)
  max_attempts     : int            (RuneState.max_attempts)

Output:
  trajectory       : list[AttemptRecord]  (RuneState.trajectory)
  outcome          : 'success' | 'exhausted'  (RuneState.outcome)

Procedure:
  attempt_count <- 0
  trajectory <- []

  loop:
    -- generate node --------------------------------------------------
    generated_code <- generate(task_description, trajectory, adapter_ids)

    -- execute node ----------------------------------------------------
    stdout, stderr, exit_code, tests_passed <- execute(generated_code, test_suite)

    -- reflect node ----------------------------------------------------
    record <- AttemptRecord(attempt_count + 1, generated_code,
                           stdout, stderr, exit_code, tests_passed)
    trajectory.append(record)
    attempt_count <- attempt_count + 1

    -- should_retry routing --------------------------------------------
    if tests_passed OR attempt_count >= max_attempts:
      break  -> save_trajectory
    else:
      continue  -> generate

  -- save_trajectory node ----------------------------------------------
  outcome <- 'success' if tests_passed else 'exhausted'

  -- distillation path (phase-dependent) -------------------------------
  if Phase == 3:  // bootstrapping implementation (specified)
    adapter <- direct_lora_finetune(trajectory)
  elif Phase == 4:  // target implementation (proposed)
    adapter <- H(trajectory)  // hypernetwork forward pass

  adapter_registry.store(adapter, outcome, trajectory)
  return trajectory, outcome
-------------------------------------------------------------
```

Each node in the pseudocode maps directly to the LangGraph `StateGraph` implementation:

**`generate`** receives the task description, the trajectory accumulated so far (empty on the first attempt), and the selected adapter IDs. It calls the base model (served by lora-server with the specified adapters loaded) to produce a candidate code solution. On subsequent attempts, the full trajectory — prior code, execution outputs, error messages — is included in the prompt, providing the model with debugging context. The `generated_code` field in `RuneState` stores the output.

**`execute`** runs the generated code against the test suite in an isolated Docker sandbox. The sandbox is network-isolated, memory-limited, and CPU-limited. Execution produces four fields in `RuneState`: `stdout`, `stderr`, `exit_code`, and `tests_passed`.

**`reflect`** appends the current attempt's data to the trajectory and increments `attempt_count`. Each `AttemptRecord` in the `RuneState.trajectory` list contains the attempt number, generated code, execution output, and pass/fail status.

**`should_retry`** is the conditional routing function: if `tests_passed` is true, the loop terminates via `save_trajectory` (success). If `attempt_count >= max_attempts`, the loop terminates via `save_trajectory` (exhausted). Otherwise, the loop returns to `generate` for another attempt.

**`save_trajectory`** records the terminal `outcome` ('success' or 'exhausted') and persists the complete trajectory to the adapter registry for downstream distillation.

The Phase 3 distillation path (`direct_lora_finetune`) is **specified** — it uses standard PEFT fine-tuning on trajectory data, a validated technique that will be implemented first to establish the training pipeline and produce the initial adapter corpus. The Phase 4 distillation path (`H(trajectory)`) is **proposed** — it replaces per-adapter gradient descent with the hypernetwork forward pass, eliminating training cost at inference time. The Phase 4 path depends on successful hypernetwork training in Phase 1.

---

### Adapter Scaling and Parameter Optimization

Applying a hypernetwork-generated adapter to a base model requires controlling the adapter's *influence strength* — the magnitude of the weight delta relative to the base model's existing parameters. This is governed by the LoRA scaling factor: PEFT computes `scaling = lora_alpha / r` and applies `ΔW = scaling * BA` to each targeted weight matrix. Sakana's original Doc-to-LoRA uses a large `lora_alpha` value that produces an effective scaling of approximately 45.25x, which achieves near-perfect factual recall on document QA tasks by strongly overriding the base model's behavior with document-specific knowledge.

For coding trajectories, this full scaling is destructive. A 200-trial Bayesian optimization search (Optuna TPE sampler across 5 diverse coding tasks) found that **adapter influence must be attenuated to approximately 0.16x of the raw hypernetwork output** — roughly 280x weaker than Sakana's default. At full scaling, the adapter overwhelms the base model's code generation capabilities, producing degenerate repetition and syntactically invalid output. At the optimal attenuation, the adapter provides a contextual nudge — injecting domain knowledge (project conventions, subtask dependencies, prior execution traces) without displacing the base model's learned coding competence.

This finding has a theoretical interpretation: the adapter's role in coding is qualitatively different from its role in document QA. In factual recall, the adapter must override the base model's default response with document-specific facts — strong influence is necessary. In code generation, the base model already knows how to write code; the adapter's job is to steer that existing competence toward the specific patterns relevant to the current task. The adapter is a contextual signal, not a replacement for the model's knowledge. The optimal scaling reflects this asymmetry.

Three bugs were discovered and fixed in the Sakana D2L → PEFT conversion path that were masked at full scaling but became visible at the correct attenuated scale: (1) `combine_lora` bias concatenation assumed single-chunk mode, (2) the alpha-to-PEFT scaling formula omitted rank compensation, and (3) module path prefixes were hardcoded without validation against the base model's architecture. These bugs are documented because they illustrate a general risk: high adapter scaling can mask conversion errors by dominating the output regardless of whether the adapter weights are correctly structured.

Additional optimization findings from the Bayesian search (200 trials, 5 diverse coding tasks):

- **Per-task calibration** improves results. A 5-trial scaling sweep (0.5x–1.5x around the base scaling) before each task adapts the influence strength to task complexity. The pipeline implements this as a `CalibrationConfig` with configurable trial count and scaling range.
- **Generation temperature 0.25** with repetition penalty 1.04 balances consistency with diversity. Lower temperatures produce more deterministic but less creative output; the mild repetition penalty prevents the degenerate loops that appear at higher adapter influence.
- **Code-first prompt styles** (skeleton, must-include) outperform open-ended prompts. The model performs better when given structural constraints rather than free-form instructions.
- **Full-context trajectories** (including error traces and corrections) produce better adapters than minimal summaries, supporting the recursive refinement hypothesis that correction steps are signal, not noise.

**HPO fitness metrics.** The hyperparameter optimization objective couples multiple diff-aware signals: `hunk_loss` (token loss restricted to changed hunks), `hunk_accuracy` (per-token accuracy on hunk tokens), `adapter_improvement` (Pass@1 delta vs. the base model without adapter), and `hunk_entropy` (diversity of the model's hunk-level distribution). The heldout evaluator that computes these metrics loads the base model with 4-bit NF4 `BitsAndBytesConfig` + `device_map="auto"` + `torch_dtype=torch.bfloat16` and threads `attention_mask` through every forward call, making it viable on 9B-scale models. Task-level heldout splits use either `step_index` or `random` strategies with no pair-level leakage. The optimizer is Optuna with a Hyperband pruner.

**Kill-switch.** `train_d2l_qwen3` accepts a `kill_switch_evaluate_fn` kwarg (default-disabled; opt-in via `config.kill_switch_enabled=True`). The switch triggers when Pass@1 regresses by ≥ 5% relative to baseline on HumanEval across 20–30 held-out tasks (k=5 completions per task), halting training before further weight corruption.

**Claim tiers:** The adapter scaling finding (attenuation to ~0.16x) is **empirical** — observed across 200 optimization trials on 5 tasks using Gemma 2 2B with the Sakana gemma_demo checkpoint. Per-task calibration is **specified** and tested. Whether the optimal scaling transfers to other base models (e.g., Qwen 2.5 Coder 7B) is **TBD** — it will be validated during production-scale evaluation.

---

### Three-Stage Training Pipeline

The adapter pipeline has three complementary training stages that serve distinct roles in the system lifecycle:

**Stage 1: QLoRA Bootstrapping.** Standard PEFT fine-tuning on NF4-quantized base model using coding trajectories formatted as SFT chat messages. This path is slow (minutes per adapter) but produces high-quality, gradient-optimized adapters. Stage 1 uses a DeltaCoder warm-start (`danielcherubini/Qwen3.5-DeltaCoder-9B`) which provides three advantages: inherited GDN-aware target module coverage across all 12 module types, convergence acceleration from pre-trained weights (DeltaCoder was trained on 50K CoderForge tool-call examples), and preserved DPO alignment on self-correction pairs that maps directly to the Phase 5 diagnose/repair loop. Stage 1 produces the adapter corpus that trains Stage 2's hypernetwork.

Stage 1 training uses `DiffAwareSFTTrainer`, a subclass of `trl.SFTTrainer` that applies hunk-weighted token loss. The associated `DiffWeightedDataCollator` wraps `trl.DataCollatorForCompletionOnlyLM` and emits a `loss_weights` tensor per batch. When `pre_code` / `post_code` side-channels are present and a tokenizer is configured, the collator computes per-line hunk ranges and assigns `changed_weight` to hunk tokens and `unchanged_weight` to context tokens — concentrating the training signal on the diff rather than boilerplate context. Under identity weights (`changed_weight == unchanged_weight == 1.0`) the trainer is a regression-guarded no-op, identical to standard SFT loss. When side-channels or the tokenizer are missing, the collator falls back to identity weights (1.0 for labeled tokens, 0.0 for `IGNORE_INDEX=-100`) rather than a silent uniform rescale, ensuring the fallback path never silently biases the objective.

**Stage 2: Hypernetwork Production.** Single forward pass through `DocToLoraHypernetwork` generates rank-8 LoRA weights in milliseconds. This is the production path — used within the pipeline retry loop to inject per-subtask context without prompt stuffing. Once trained, the hypernetwork amortizes the cost of adapter generation, replacing minutes of gradient descent with a sub-second forward pass.

Once the hypernetwork is trained, QLoRA's ongoing role shrinks to producing occasional high-value adapters for tasks where the hypernetwork's approximation is insufficient. Those adapters feed back into periodic hypernetwork retraining, creating a self-improving feedback loop.

**Stage 3: Round-2 Distillation.** After the Stage 2 hypernetwork has been trained against the bare base model, Stage 3 re-trains the hypernetwork using **per-bin oracle adapters as teacher signals** instead of the base model. This introduces a richer supervisory signal: the oracle adapters have already been gradient-optimized for specific corpus bins, so the hypernetwork learns to reproduce their behavior rather than approximating the un-adapted base.

The oracle set consists of 25 adapters, one per corpus bin: four pipeline phases (decompose, plan, code, integrate) × six evaluation benchmarks (HumanEval, MBPP, APPS, BigCodeBench, DS-1000, LiveCodeBench), plus one pooled diagnose adapter (`diagnose_pooled`). Oracle adapter IDs follow the scheme `oracle_<bin_key>` where `bin_key` is `<phase>_<benchmark>` or `diagnose_pooled`, set by `libs/corpus-producer/src/corpus_producer/trainer_bridge.py`.

**Functional-LoRA teacher mechanism.** Oracle adapters are applied to the base model via the `apply_functional_lora` context manager (`round2_train.py`). This context manager modifies the base model's forward pass functionally — the base model is **never structurally mutated** (no `PeftModel` wrappers, no `LoraLayer` hook replacements). The oracle cache (`OracleAdapterCache`) stores each oracle as a `LoraDict` tensor dict (`{module: {"A": Tensor[L,r,in], "B": Tensor[L,r,out]}}`), with an LRU cap of four loaded oracles. The functional approach eliminates PEFT hook-leakage risk between teacher and student forward passes within the same training step.

**KL + CE distillation loss.** Each training step performs two forward passes: one through the base model with the oracle adapter applied (teacher), and one through the student hypernetwork. The training loss is a weighted combination of KL divergence between student and teacher output distributions and cross-entropy against the ground-truth labels (`_compute_kl_ce_loss` in `round2_train.py`).

**Startup guard.** If the fraction of training records whose bin has a registered oracle falls below `min_oracle_coverage=0.8`, the training loop raises `RuntimeError` before loading any model. `dry_run` mode surfaces this gate without paying the full training cost. Records whose bin has no oracle are dropped when `oracle_fallback="skip"` (the default); `oracle_fallback="base_model"` is an ablation mode that substitutes the unadorned base model as teacher.

**Strict success gate.** After training, `round2_gate.evaluate_round2_gate` applies a binary success criterion before the adapter is promoted: ≥ 4 of the 6 benchmarks must show ≥ 2.0% Pass@1 improvement over the Stage 2 baseline, **and** no single benchmark may regress by more than 1.0%. `scripts/evaluate_round2.py` exits 0 on PASS and 1 on FAIL, enabling CI gating. The six required benchmarks are HumanEval, MBPP, APPS, BigCodeBench, DS-1000, and LiveCodeBench.

**Round-2 adapter lineage.** Promoted adapters receive ID `round2_<uuid[:8]>`, `task_type="round2_hypernet"`, `generation=2`, and `parent_ids = json.dumps(sorted(oracle_ids))` for full lineage tracking in the adapter registry.

**Claim tiers:** The three-stage pipeline architecture is **specified** — all three stages are implemented and tested. The Stage 3 functional-LoRA teacher mechanism is **specified** and tested. Empirical Pass@1 gains from oracle-teacher distillation vs. Stage 2 are **TBD** — Stage 3 requires production oracle adapters and GPU runs to validate.

---

### Evolution Operator

The Background section established that LoRA Soups' CAT method can outperform data mixing for binary skill composition tasks,[^prabhakar2024lorasoups] but also that orthogonality between merged LoRA modules does not guarantee semantic disentanglement[^zhang2025orthogonality] and that adapter merging can reactivate latent reasoning traces.[^zou2026merging] The Evolution Operator is Rune's approach to fitness-driven adapter lifecycle management — testing adapter quality empirically before promotion rather than assuming composition is safe by default.

The Evolution Operator defines four operations on the adapter registry:

1. **Consolidate** (**specified**): Merge related adapters across task types into a generalized adapter when similarity exceeds a configurable threshold. The output is a new registry entry at a higher hierarchy level (e.g., task-level adapters consolidated into a domain-level adapter). The input adapters are archived — not deleted, per the write-once policy — and remain queryable but ineligible for retrieval routing.

2. **Update** (**specified**): When a new trajectory produces an adapter with higher fitness than an existing adapter of the same task type, the old adapter is archived and the new adapter is registered. This enforces the write-once, archive-not-delete policy: no adapter is overwritten, but the registry always routes to the highest-fitness entry for each task type.

3. **Forget** (**specified**): The `archive_adapter()` function sets `is_archived = 1` in the SQLite registry. Archived adapters are never deleted from disk — they become ineligible for retrieval routing but remain available for historical analysis, lineage tracking, and potential reactivation.

4. **Merge** (**ablation target**): Combine adapter weight matrices from two adapters into a single composite entry. This operation is labeled as an **ablation target** due to the orthogonality and latent trace concerns identified by Zhang et al.[^zhang2025orthogonality] and Zou.[^zou2026merging] Merge will be tested empirically — if composed adapters produce degraded or incoherent behavior, the operation will be disabled in favor of single-adapter selection.

The Evolution Operator uses a fitness criterion to drive all lifecycle decisions. The fitness function for an adapter \(a\) is defined as (new equation):

\[
\phi(a) = \alpha \cdot \text{pass\_rate}(a) + (1 - \alpha) \cdot \text{generalization}(a)
\]

where \(\text{pass\_rate}(a)\) is the adapter's test-passing rate on held-out tasks of the same type, and \(\text{generalization}(a)\) is a cross-task coverage metric measuring the adapter's performance on tasks outside its original training domain. The weighting parameter \(\alpha\) controls the trade-off between specialization (high pass rate on similar tasks) and generalization (broad utility across task types). The default value is \(\alpha = 0.7\), reflecting a preference for specialization — an adapter that reliably solves its own task type is more valuable than one that partially solves many types. The \(\alpha\) parameter is an **ablation target** (default 0.7, pending Phase 4 tuning). The exact generalization metric is **TBD** — it will be defined during Phase 4 when sufficient adapter diversity exists to measure cross-task performance. The `fitness_score` column in the SQLite `adapters` table stores the computed \(\phi(a)\) value.

**Contrast with LoRA Soups:** LoRA Soups' CAT method combines adapters at the data-mixing level — concatenating adapter matrices — without testing whether the resulting combination produces better behavior on downstream tasks. The combination is assumed to be beneficial based on orthogonality properties of the adapter subspaces. The Evolution Operator takes a fundamentally different approach: it tests fitness empirically before promoting any adapter to the registry, and uses fitness-driven selection to determine which adapter combinations (if any) produce beneficial behavior. This directly addresses the orthogonality gap identified by Zhang et al.:[^zhang2025orthogonality] orthogonal subspaces do not guarantee semantic coherence, so Rune tests behavioral coherence rather than assuming it. The merge operation itself is an **ablation target** — it will only be retained if empirical evaluation shows that merged adapters achieve higher fitness than their individual components.

**Claim tiers:** The four operations (consolidate, update, forget, merge) are **specified** in the architecture. The fitness formula weighting \(\alpha\) is an **ablation target**. The merge operation's semantic coherence is **TBD** pending empirical validation of the orthogonality and latent trace concerns from the literature.

---

### Memory Composition Strategy

Rune organizes adapters into a three-level hierarchy that mirrors the granularity of coding knowledge:

- **Project level**: Adapters encoding broad patterns for a specific codebase — its architectural conventions, API design patterns, testing practices, and domain-specific idioms. A project-level adapter captures knowledge that is useful across all tasks within that project.

- **Domain level**: Adapters encoding patterns for a category of tasks that spans projects — API design, database query optimization, algorithm implementation, error handling patterns. A domain-level adapter captures knowledge that transfers across projects for tasks of the same type.

- **Task level**: Fine-grained adapters encoding patterns for a specific task type within a specific context. A task-level adapter captures the most specific knowledge — the particular patterns discovered while solving a particular class of problem.

The filesystem path convention encodes this hierarchy: `~/.rune/adapters/{project|domain|task}/...`. The SQLite `adapters` table tracks `hierarchy_level`, `domain`, and `project_id` columns, enabling queries that filter by any combination of hierarchy and metadata.

At inference time, the adapter router queries the registry by task metadata (`task_type`, `domain`, `project_id`) and selects the highest-fitness adapter at each applicable hierarchy level using the \(\phi(a)\) fitness criterion defined in the Evolution Operator section.

**Default behavior** (**specified**): Single-adapter selection — the most relevant adapter at the most specific applicable level, loaded alone. If a task-level adapter exists with sufficient fitness, it is used. Otherwise, the router falls back to domain level, then project level. The rationale for single-adapter default is the interference risk established in Background: orthogonality between adapter subspaces does not guarantee semantic coherence when multiple adapters are applied simultaneously.

**Multi-adapter composition** (**ablation target**): Additive accumulation across hierarchy levels, where the full weight delta applied to the base model is:

\[
\Delta W_{\text{composite}} = \Delta W_{\text{project}} + \Delta W_{\text{domain}} + \Delta W_{\text{task}}
\]

This compositional mode loads one adapter from each applicable hierarchy level and applies their weight deltas additively. It is the **ablation target** for Phase 5 and beyond. Multi-adapter composition will be tested against single-adapter selection once sufficient adapter diversity exists in the registry. The default remains single-adapter selection until empirical evidence supports multi-level composition.

**Claim tiers:** The three-level hierarchy is **specified** — it is implemented in the adapter registry schema and filesystem layout. Single-adapter selection as the default retrieval mode is **specified**. Compositional accumulation across hierarchy levels is **proposed** and requires empirical validation of behavioral coherence. Multi-adapter composition semantics are an **ablation target** for Phase 5+.

---

### GitHub Mining Pipeline

Training a hypernetwork on coding trajectories requires a corpus of real-world coding sessions with sufficient diversity in task type, failure mode, and correction pattern. Rune includes a GitHub mining pipeline that collects training data from public repositories by mining pull requests, issues, and commit histories. The pipeline extracts structured coding trajectories from PR review cycles — where a PR undergoes review feedback, revision, and re-review — capturing the same attempt-error-correction structure that the agent's own recursive loop produces. Mined trajectories are normalized into the same schema used by the agent's trajectory store, enabling the hypernetwork to train on both self-generated and externally-mined trajectories without schema conversion.

The mining pipeline is **specified** — it is implemented in the codebase. The quality and diversity of mined trajectories relative to self-generated trajectories is an empirical question that will be evaluated during hypernetwork training.

---

### Benchmark Evaluation Framework

Rune includes a coding benchmark evaluation framework for systematic measurement of adapter quality across standard code generation benchmarks. The framework supports six benchmark suites:

- **HumanEval** (EvalPlus): Extended HumanEval with additional test cases for more rigorous evaluation of functional correctness.
- **MBPP** (EvalPlus): Extended Mostly Basic Python Programming benchmark with augmented test suites.
- **APPS**: Competitive programming problems across introductory, interview, and competition difficulty tiers.
- **BigCodeBench**: A more challenging benchmark targeting complex, multi-step coding tasks that require API usage and library knowledge.
- **DS-1000**: Data-science-oriented tasks drawn from Stack Overflow questions across seven libraries (NumPy, Pandas, SciPy, Matplotlib, sklearn, PyTorch, TensorFlow).
- **LiveCodeBench**: A contamination-resistant benchmark sourced from live competitive programming contests after the model's training cutoff.

These six suites also define the structure of the Stage 3 strict success gate: a round-2 adapter is promoted only when ≥ 4 of the 6 benchmarks show ≥ 2.0% Pass@1 improvement with no single benchmark regressing by more than 1.0%.

Execution is organized into three tiers to support different evaluation contexts:

- **Smoke** (~5 minutes): A minimal subset for rapid iteration during development. Sufficient to detect regressions and verify that the evaluation pipeline functions correctly.
- **Mini** (~30 minutes): A medium-sized subset that provides statistically meaningful signal without requiring a full benchmark run. Used for comparing adapter variants and hyperparameter sweeps.
- **Full** (~2 hours): The complete benchmark suite for final evaluation and publication-quality measurements.

The development evaluation target is Gemma 2 2B (google/gemma-2-2b-it), which fits within consumer GPU memory constraints and enables rapid iteration. Production-scale evaluation will target Qwen/Qwen3.5-9B. <!-- TODO(benchmarks-pending): production-scale evaluation numbers not yet available --> The tiered execution structure allows the Phase 1 kill-switch evaluation to use the smoke tier for fast hypothesis testing and the full tier for definitive measurements.

The benchmark framework is **specified** — it is implemented in the codebase and tested. Benchmark results on adapter-augmented models are **TBD** pending GPU validation runs.

---

### Training Data Strategy and Serving Architecture

#### Training Data Strategy

The Background section established Cook et al.'s finding that declarative instructions — including code — can substitute for up to 100 execution examples when fine-tuning LLMs.[^cook2025pbb] This result demonstrates that the format of training signal matters: procedural, instruction-like representations are more information-dense than equivalent example sets. Rune's training data strategy is grounded in this finding: code execution trajectories are the training signal, not (input, output) pairs.

Each trajectory record in the SQLite `trajectories` table contains:

- `task_description`: the natural language specification of the coding task
- `test_suite`: the test code used for evaluation
- A sequence of attempts, each containing `generated_code`, `stdout`, `stderr`, `exit_code`, and `tests_passed`
- A terminal `outcome`: 'success' (tests passed) or 'exhausted' (max attempts reached without passing)

Both successful and exhausted trajectories are training candidates. Successful trajectories encode complete solution patterns — from initial approach through iterative refinement to a passing solution. Exhausted trajectories encode debugging patterns, failure recognition, and partial solutions — knowledge about what does not work and why, which is valuable for avoiding similar failure modes in future tasks.

PBB found that in-context instruction execution remains more reliable than weight-based procedural knowledge for the tasks they tested. This caveat applies directly to Rune: there is a gap between what can be encoded in weights via backpropagation and what can be reliably executed from those weights at inference time. Training on trajectories is **expected** to produce useful adapter representations based on PBB's findings about procedural encoding efficiency — instructions are information-dense, and trajectories are the richest procedural representation available. However, whether trajectory-trained adapters achieve reliable weight-based execution on coding tasks specifically is **proposed** and requires Phase 1 empirical validation. The gap between encoding efficiency and reliable execution from weights is the central empirical question for Rune's training data strategy.

**Claim tiers:** Training on trajectories without input-output pairs is **expected** — grounded in PBB's demonstration that procedural representations are efficient for weight-based learning. Adapter quality on coding tasks specifically is **proposed** — this is Rune-specific and unvalidated.

#### Serving Architecture

The intended serving configuration is vLLM with QLoRA (NF4 quantized base model with bfloat16 adapter matrices) and optionally pipeline parallelism for multi-GPU setups. This is the **intended configuration pending Phase 0 empirical validation** — not an assertion of confirmed compatibility.

**Pipeline parallelism rationale for multi-GPU setups:** For GPUs connected via PCIe (most consumer setups), tensor parallelism (TP) is not recommended. TP requires all-reduce synchronization at every transformer layer. Over PCIe bandwidth (~32 GB/s bidirectional), the all-reduce at every layer becomes a significant bottleneck compared to NVLink (~112 GB/s per direction). Pipeline parallelism splits the transformer layers across GPUs and transfers activations once at each layer boundary — a single point-to-point transfer per GPU boundary, respecting the PCIe bandwidth constraint. Additionally, vLLM issue #21471 documents TP + LoRA corruption on consumer GPUs without NVLink, which is absent under pipeline parallelism.

**QLoRA serves the VRAM constraint:** NF4 quantization reduces the 7B-parameter base model from approximately 14 GB (bfloat16) to approximately 4 GB, leaving headroom for adapter weights and KV cache on consumer GPUs.[^dettmers2023qlora] S-LoRA unified paging manages adapter weights alongside KV cache in a shared GPU memory pool, enabling concurrent serving of multiple adapters without per-adapter VRAM reservation.[^sheng2023slora]

**GPU lease mechanism:** When using multiple GPUs, lora-server can yield a secondary GPU to training-svc during training jobs. The serving process reconfigures to the remaining GPU(s) (reduced capacity but still available), and training-svc runs on the leased GPU with full VRAM access. When the training job completes, training-svc releases the lease, and lora-server restores its full configuration. Serving pauses briefly during reconfiguration but is not fully offline during training.

**Claim tiers:** The pipeline parallelism architecture is **recommended** for multi-GPU PCIe setups based on bandwidth constraints and the vLLM TP+LoRA bug. Pipeline parallelism + QLoRA + vLLM LoRA compatibility is **expected** — it is the primary empirical question for Phase 0 environment validation. The GPU lease mechanism is **specified** in the architecture but **untested** at the implementation level.

---

[^hu2021lora]: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*. [Full entry](references.md#hu2021lora)

[^ha2016hypernetworks]: Ha, D., Dai, A., & Le, Q. V. (2016). HyperNetworks. *arXiv:1609.09106*. [Full entry](references.md#ha2016hypernetworks)

[^charakorn2026doc2lora]: Charakorn, R., et al. (2026). Doc-to-LoRA: Learning to Instantly Internalize Contexts. *arXiv:2602.15902*. [Full entry](references.md#charakorn2026doc2lora)

[^prabhakar2024lorasoups]: Prabhakar, A., et al. (2024). LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks. *arXiv:2410.13025*. [Full entry](references.md#prabhakar2024lorasoups)

[^cook2025pbb]: Cook, J., et al. (2025). Programming by Backprop: An Instruction is Worth 100 Examples When Finetuning LLMs. *arXiv:2506.18777*. [Full entry](references.md#cook2025pbb)

[^zhang2025orthogonality]: Zhang, A., et al. (2025). Rethinking Inter-LoRA Orthogonality in Adapter Merging. *arXiv:2510.03262*. [Full entry](references.md#zhang2025orthogonality)

[^zou2026merging]: Zou, J. (2026). Adapter Merging Reactivates Latent Reasoning Traces. *arXiv:2601.18350*. [Full entry](references.md#zou2026merging)

[^dettmers2023qlora]: Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv:2305.14314*. [Full entry](references.md#dettmers2023qlora)

[^sheng2023slora]: Sheng, Y., et al. (2023). S-LoRA: Serving Thousands of Concurrent LoRA Adapters. *arXiv:2311.03285*. [Full entry](references.md#sheng2023slora)
