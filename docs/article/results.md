## Results

!!! warning "Research Status: Pre-Validation"
    The evaluation infrastructure is built and tested (433+ tests passing,
    benchmark framework implemented), but no GPU training runs or adapter
    evaluations have been conducted. This section presents the planned
    experimental design for Phase 1 hypothesis validation. All tables, figures,
    and metrics below describe proposed measurements, not observed results.

---

### Phase 1 Kill-Switch Experimental Design

Phase 1 is a minimal hypothesis test that gates all subsequent infrastructure investment. The kill-switch structure is binary: only if the primary metric exceeds the specified threshold (H\(_1\) confirmed) does development proceed to Phase 2. If the null hypothesis holds (H\(_0\) not rejected), infrastructure development terminates and the team reassesses. No partial credit is awarded — the kill-switch is a go/no-go decision, not a gradient.

#### Evaluation Benchmark

Rune's benchmark evaluation framework (described in [Methods](methods.md#benchmark-evaluation-framework)) provides three benchmark suites — HumanEval+, MBPP+, and BigCodeBench — organized into three execution tiers (smoke ~5 min, mini ~30 min, full ~2 hr). The Phase 1 kill-switch evaluation uses this framework as follows:

The **primary** evaluation benchmark is HumanEval+ (EvalPlus), an extended version of HumanEval (Chen et al., 2021) with additional test cases that reduce false positives from undertested solutions. Phase 1 uses a held-out subset of 20--30 tasks, not the full benchmark, because Phase 1 is a minimal hypothesis test — statistical power over a small, well-controlled subset is sufficient to detect the **specified** threshold effect. The smoke tier enables rapid iteration during hyperparameter tuning; the full tier provides definitive measurements for the kill-switch decision.

Each task is evaluated with k=5 samples. Tasks in the held-out subset are excluded from the training trajectory corpus so that the evaluation measures generalization to unseen problems, not memorization of training data. The subset is selected to be diverse in task type (string manipulation, arithmetic, data structures, algorithms) to avoid biasing the evaluation toward a narrow skill distribution.

The development evaluation target is Gemma 2 2B (google/gemma-2-2b-it), which fits within consumer GPU memory constraints and enables rapid iteration. The Sakana Doc-to-LoRA checkpoint includes a "gemma_demo" variant compatible with this model. Production-scale evaluation will target Qwen2.5-Coder-7B-Instruct on the full benchmark suite including MBPP+ and BigCodeBench.

**Isolation note:** Phase 1 baseline runs in bfloat16 — NOT QLoRA.[^dettmers2023qlora] This isolation is deliberate: the Phase 1 experiment tests the trajectory-to-adapter hypothesis (does a Doc-to-LoRA hypernetwork[^charakorn2026doc2lora] trained on coding trajectories produce useful adapters?) independently of the quantization variable (does NF4 quantization degrade adapter quality?). QLoRA is introduced in Phase 2 after the bfloat16 baseline passes. Confounding these two variables in a single experiment would make a negative result uninterpretable.

#### Kill-Switch Hypothesis

Pass@1 for a single task is the probability that the first generated sample passes all tests. Averaged over \(N\) tasks with \(k=5\) samples each:

\[
\text{Pass@1} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{task}_i \text{ passes on at least 1 of } k \text{ samples}]
\]

The formal hypothesis pair for the Phase 1 kill-switch gate:

- **H\(_0\) (null):** A Doc-to-LoRA hypernetwork trained on coding trajectories produces adapters that do not improve Pass@1 over the baseline (improvement < 5%).
- **H\(_1\) (alternative):** Trajectory-conditioned adapters improve Pass@1 by \(\geq\) 5% on the 20--30 task HumanEval+ subset compared to the base model (Gemma 2 2B for development evaluation, Qwen2.5-Coder-7B-Instruct for production evaluation) with no adapter.

H\(_1\) passing is necessary but not sufficient to proceed to Phase 2. H\(_0\) acceptance terminates infrastructure development and triggers reassessment. The kill-switch is a research gate, not a quality bar — it tests whether the core hypothesis has empirical support, not whether the system is production-ready.

**Expected range** (from implementation plan): 5--15% improvement signals a real effect. Greater than 15% should be treated skeptically and checked for data leakage — the held-out subset may have leaked into the training trajectory corpus, or the evaluation may be measuring memorization rather than generalization.

#### Training Dataset

The **specified** training dataset protocol for Phase 1:

- **Size:** 50--100 real coding trajectory pairs
- **Source:** HumanEval or SWE-bench-lite, generated via the recursive agent loop or prompted from the base model
- **Structure per trajectory:** task description + attempt sequence (code + error messages + corrections) + final passing code
- **Diversity requirement:** diverse in task type, failure mode, and correction pattern — not variations on the same task, which would induce mode collapse in the hypernetwork (see [Risk Matrix](../appendices/risk-matrix.md))

#### Baseline Comparison Table

*Table 1 (Planned Experiments): Phase 1 Kill-Switch Baseline Conditions*

| Condition | Description | Claim Tier |
|-----------|-------------|-----------|
| Vanilla model (bfloat16) | Gemma 2 2B (dev) / Qwen2.5-Coder-7B-Instruct (prod), no adapter, bfloat16 | Planned baseline |
| RAG baseline | Retrieved trajectory snippets in context (no weight update) | Planned baseline |
| Directly fine-tuned LoRA | Standard PEFT fine-tuning on same trajectory data (no hypernetwork) | Planned baseline |
| Hypernetwork-generated adapter | Rune Phase 1 (Doc-to-LoRA on coding trajectories, gemma_demo checkpoint) | **Kill-switch condition** |

The four baselines isolate distinct variables. The **vanilla model** establishes the zero-adaptation reference — raw model capability on the held-out tasks without any trajectory signal. The **RAG baseline** tests whether in-context retrieval of trajectory snippets alone achieves the threshold, without modifying the model's weights — if RAG suffices, the hypernetwork adds no value over retrieval. The **directly fine-tuned LoRA** tests whether standard gradient-based adaptation on the same trajectory data matches or exceeds the hypernetwork — if direct fine-tuning achieves equivalent quality, the hypernetwork's value is inference-time speed, not adapter quality. Only the **hypernetwork-generated adapter** tests the full Rune hypothesis: that a single forward pass through a trajectory-conditioned hypernetwork produces adapters competitive with gradient-based methods.

**Note:** Configuration details for the RAG baseline (chunk size, retrieval model, top-k) are unspecified in the current implementation plan and will be determined during Phase 1 setup.

#### MLflow Tracking Schema

The **specified** tracking schema records all kill-switch evaluation runs, with adapter weights managed via the S-LoRA unified paging pattern[^sheng2023slora] during serving:

```
run_id | phase | adapter_id | pass_at_1 | training_loss | adapter_cosine_diversity | delta_w_norm
```

The gate decision is recorded as an MLflow run note: `"Gate PASSED"` or `"Gate FAILED — reassessing"`.

---

### Adapter Diversity Metrics

These metrics do not gate the kill-switch decision but are tracked via MLflow as diagnostic signals. They inform the post-experiment written assessment of whether the hypernetwork is learning meaningful structure rather than collapsing to a degenerate solution.

#### Frobenius Norm of Adapter Weights

The Frobenius norm of the adapter weight update \(\Delta W = BA\) for each generated adapter:

\[
\|\Delta W\|_F = \sqrt{\sum_{i,j} (B A)_{ij}^2}
\]

A meaningfully nonzero \(\|\Delta W\|_F\) confirms the adapter is affecting the base model's behavior. If \(\|\Delta W\|_F \approx 0\), the hypernetwork is generating near-zero updates — the adapter is effectively absent and the base model's behavior is unchanged regardless of whether an adapter is loaded. The \(\Delta W = BA\) decomposition is the LoRA reparameterization established in [Background](background.md) and formally defined in [Methods](methods.md).[^hu2021lora]

**Note:** \(\|\Delta W\|_F\) is reported per transformer layer targeted (Q, K, V, O projections and MLP up/down projections), giving a 6--8 dimensional profile rather than a single scalar. This per-layer profile provides richer diagnostic information: a hypernetwork that produces nonzero updates for attention layers but near-zero updates for MLP layers (or vice versa) reveals something about which model components the trajectory signal is reaching.

#### Inter-Adapter Cosine Diversity

For a batch of hypernetwork-generated adapters \(\{\Delta W_i\}_{i=1}^{N}\), the inter-adapter cosine similarity is computed on flattened weight vectors:

\[
\text{sim}(\Delta W_i, \Delta W_j) = \frac{\text{vec}(\Delta W_i) \cdot \text{vec}(\Delta W_j)}{\|\text{vec}(\Delta W_i)\|_2 \, \|\text{vec}(\Delta W_j)\|_2}
\]

Mean pairwise cosine similarity across the batch is the diversity metric.

- **Diversity criterion:** mean pairwise cosine similarity < 0.9 (equivalently, adapter cosine diversity > 0.1)
- **Interpretation:** If adapters cluster (diversity collapses toward 0), the hypernetwork is approaching the degenerate "mean adapter" solution — a known mode collapse failure mode documented in the [risk matrix](../appendices/risk-matrix.md). Healthy diversity indicates the hypernetwork is producing trajectory-specific adapters rather than converging to a single set of weights regardless of input.

#### Planned Metrics Table

*Table 2 (Planned Experiments): Phase 1 Kill-Switch Evaluation Metrics*

| Metric | Symbol | Threshold | Gate Type |
|--------|--------|-----------|-----------|
| Pass@1 improvement | -- | \(\geq\) 5% over baseline | **Kill-switch (primary gate)** |
| Adapter \(\|\Delta W\|_F\) norm | \(\|\Delta W\|_F\) | Meaningfully nonzero | Diagnostic (not a gate) |
| Inter-adapter cosine diversity | \(1 - \bar{\text{sim}}\) | > 0.1 | Diagnostic (not a gate) |
| Training loss convergence | \(\mathcal{L}_{\text{train}}\) | Converges, no early plateau | Diagnostic (not a gate) |

---

### PBB-Inspired Evaluation Criterion (Proposed Secondary Criterion)

This subsection describes a **proposed** secondary evaluation criterion inspired by Programming by Backprop (PBB). It is not part of the kill-switch gate — it is an additional lens for evaluating whether trajectory-conditioned adapters achieve a stronger form of procedural encoding beyond code generation.

PBB (Cook et al., 2025) demonstrates that an LLM trained on source code without input-output examples can still evaluate those programs on new inputs — weight-based procedural encoding enables inference-time program execution.[^cook2025pbb] A secondary evaluation criterion for Phase 1 tests whether Rune's trajectory-conditioned adapters achieve the PBB property: procedural knowledge encoded in adapter weights, not merely pattern-matched from context.

**The PBB test question:** Can a generated adapter enable the model to evaluate a program on held-out inputs without those inputs appearing in the context window?

**Test procedure (four steps):**

1. Generate an adapter from a trajectory that includes a specific algorithmic implementation (e.g., a sorting function, a string manipulation, a recursive algorithm)
2. Load the adapter into the base model
3. Query the model with a natural language request asking it to evaluate the function on held-out inputs — inputs not present in the training trajectory and not provided in context
4. Compare: base model (no adapter) vs adapter-loaded model on the same held-out evaluation queries

**Why this matters:** Pass@1 measures whether the model can write passing code. The PBB criterion measures whether the model has internalized the algorithm itself — a stronger form of procedural encoding. An adapter that enables the model to evaluate a function on novel inputs demonstrates that the trajectory signal has been encoded as procedural knowledge in the model's weights, not merely as a statistical pattern that improves code generation likelihood.

**Grounding statement:** PBB validated this criterion on instruction-following tasks with natural language programs; Rune **proposes** extending it to code execution trajectory-conditioned adapters. The criterion is **proposed** — it has not been validated for the trajectory modality. PBB also found that in-context instruction execution remains more reliable than weight-based procedural knowledge for the tasks they tested, which means this criterion may reveal a gap between what trajectory-conditioned adapters can encode and what they can reliably execute.

This criterion is labeled **Planned Experiments (PBB-Inspired)** and is contingent on Phase 1 infrastructure being in place.

---

### Phase 4 Ablation Structure (Planned)

The following **proposed** experiments are contingent on Phase 1 passing. They are presented here to give the experimental design forward scope — they are not prerequisites for the kill-switch gate. If Phase 1 fails (H\(_0\) not rejected), these experiments are not conducted.

#### Ablation Targets

The ablation targets derive from **specified** architecture choices in [Methods](methods.md) that have been designed with explicit alternatives:

- **Encoder architecture:** Perceiver-based cross-attention (Methods default) vs mean pooling vs hierarchical attention vs recurrent summarization. The Perceiver-based encoder is labeled as an ablation target in Methods — alternative encoders may achieve comparable or superior trajectory compression with different computational trade-offs.
- **Fitness weight \(\alpha\):** Default 0.7 (specified in the evolution operator fitness function \(\phi(a) = \alpha \cdot \text{pass\_rate}(a) + (1 - \alpha) \cdot \text{generalization}(a)\) in [Methods](methods.md)); ablation range TBD. The \(\alpha\) parameter controls specialization vs generalization trade-off — values closer to 1.0 favor task-specific adapters, values closer to 0.5 favor cross-task utility.
- **Multi-adapter composition:** Additive accumulation \(\Delta W_{\text{composite}} = \Delta W_{\text{project}} + \Delta W_{\text{domain}} + \Delta W_{\text{task}}\) (proposed in Methods) vs single-adapter selection (specified default). The composition question is whether combining adapters from different hierarchy levels produces coherent behavior or introduces the interference documented by Zhang et al.[^zhang2025orthogonality] and Zou.[^zou2026merging]
- **Merge operation:** Empirical test of whether merged adapters achieve higher fitness than individual components.[^prabhakar2024lorasoups] The merge operation is an ablation target in the Evolution Operator — it will only be retained if empirical evaluation confirms behavioral coherence.

#### Phase 4 Experiment Parameters

**Planned Experiments (Phase 4, contingent on Phase 1 pass):**

- **Dataset:** Held-out adapter subset from Phase 3 corpus (10--20% withheld from hypernetwork training set)
- **Baseline:** Random adapter (Gaussian weight initialization, same rank) as reconstruction loss reference
- **Metrics:** Reconstruction loss (MSE between hypernetwork-generated and held-out fine-tuned weights); cosine diversity across batch; Pass@1 comparison on held-out HumanEval tasks
- **Expected range:** Reconstruction loss below random baseline confirms the hypernetwork is learning the adapter manifold. Cosine diversity > 0.1 confirms diversity is preserved. Pass@1 50--80% of fine-tuned adapter quality is the **expected** range for a first-pass hypernetwork; parity (\(\pm\)5%) would be a strong result.

---

[^cook2025pbb]: Cook, J., et al. (2025). Programming by Backprop: An Instruction is Worth 100 Examples When Finetuning LLMs. *arXiv:2506.18777*. [Full entry](references.md#cook2025pbb)

[^charakorn2026doc2lora]: Charakorn, R., et al. (2026). Doc-to-LoRA: Hypernetwork Adaptation from Documents. *arXiv:2602.15902*. [Full entry](references.md#charakorn2026doc2lora)

[^hu2021lora]: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*. [Full entry](references.md#hu2021lora)

[^zhang2025orthogonality]: Zhang, Y., et al. (2025). Inter-LoRA Orthogonality and Composition Interference. [Full entry](references.md#zhang2025orthogonality)

[^zou2026merging]: Zou, H., et al. (2026). Latent Trace Reactivation in Merged LoRA Adapters. [Full entry](references.md#zou2026merging)

[^sheng2023slora]: Sheng, Y., et al. (2023). S-LoRA: Serving Thousands of Concurrent LoRA Adapters. *arXiv:2311.03285*. [Full entry](references.md#sheng2023slora)

[^dettmers2023qlora]: Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv:2305.14314*. [Full entry](references.md#dettmers2023qlora)

[^prabhakar2024lorasoups]: Prabhakar, A., et al. (2024). LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks. [Full entry](references.md#prabhakar2024lorasoups)
