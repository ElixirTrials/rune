## Discussion

!!! note "Research Status"
    Rune's infrastructure is built and tested (five-phase pipeline, benchmark
    evaluation framework, adapter registry, model registry with DeltaCoder
    warm-start, 433+ tests passing). GPU training runs and adapter evaluations
    have not been conducted. All claims in this section are qualified as
    **expected** (grounded in prior work) or **proposed** (requiring empirical
    validation). No empirical claims are made.

---

### Expected Contributions

**1. Trajectory modality extension (proposed).** Rune proposes extending the hypernetwork-to-LoRA mechanism from document QA (Doc-to-LoRA)[^charakorn2026doc2lora] to code execution trajectories — a structurally distinct input modality with temporal ordering, procedural abstraction, and self-correcting structure. This is the central novel contribution claim. The extension is **proposed** and requires Phase 1 empirical validation to confirm. No prior work has applied hypernetwork-based adapter generation to code execution traces; the trajectory modality argument established in [Background](background.md) provides the theoretical grounding, but the empirical question — whether trajectory-conditioned adapters produce useful weight updates — is open.

**2. Parametric episodic memory for coding agents (expected).** If the trajectory modality extension holds, Rune provides a composable weight-space memory alternative to context-window-based memory for long-running coding agents, directly addressing the limitation Pink et al. identify as the missing capability for long-term LLM agents.[^pink2025episodic] Each adapter is an episode — a persistent, retrievable, composable unit of procedural knowledge stored in weight space rather than token space. The connection to the episodic memory position is **expected** — it depends on Phase 1 passing.

**3. Fitness-driven adapter lifecycle (specified).** The Evolution Operator's empirical fitness testing before adapter promotion, contrasted with LoRA Soups' assumption-based composition.[^prabhakar2024lorasoups] This is **specified** in the architecture regardless of Phase 1 outcome — it is implemented in evolution-svc (see [Methods](methods.md#evolution-operator)). The four lifecycle operations (consolidate, update, forget, merge) and the fitness criterion \(\phi(a) = \alpha \cdot \text{pass\_rate}(a) + (1 - \alpha) \cdot \text{generalization}(a)\) are architectural commitments. Whether the fitness criterion selects meaningfully better adapters is **proposed** — it requires Phase 2+ empirical validation.

**4. PBB criterion validation for trajectory adapters (proposed).** If Rune's adapters satisfy the PBB program evaluation criterion (see [Results](results.md#pbb-inspired-evaluation-criterion-proposed-secondary-criterion)), this extends Cook et al.'s finding from instruction-following to trajectory-conditioned weight-based procedural encoding.[^cook2025pbb] The PBB criterion tests whether an adapter enables the model to evaluate a program on held-out inputs without those inputs in context — a stronger form of procedural encoding than Pass@1 improvement alone. This is **proposed** — a stronger test than Pass@1 improvement alone, contingent on Phase 1 infrastructure.

---

### Limitations

#### Pre-Validation Status

As of this writing, the system infrastructure is substantially implemented: the five-phase pipeline (decompose, plan, code, integrate, diagnose/repair), adapter registry, model registry with DeltaCoder warm-start, benchmark evaluation framework (HumanEval+, MBPP+, BigCodeBench with smoke/mini/full tiers), GitHub mining pipeline, and swarm orchestration are all built and tested with 433+ tests passing. However, no GPU training runs have been executed, no adapters have been generated from real trajectories, and no Pass@1 measurements have been taken. This article presents a research proposal with substantial infrastructure validation, not a completed study.

Phase 0 environment validation (vLLM + QLoRA compatibility) has not been confirmed. The kill-switch gate outcome is unknown. Every **expected** claim in this article depends on Phase 1 passing. The experimental design in [Results](results.md) describes planned measurements using the implemented benchmark framework; the Discussion interprets those planned measurements in terms of their implications — but neither the measurements nor the implications have been empirically grounded.

#### Adapter Interference Risk

Zhang et al. show that orthogonality between adapter subspaces does not guarantee semantic disentanglement — interference can occur even in mathematically orthogonal subspaces.[^zhang2025orthogonality] Zou demonstrates that additive composition can reactivate latent reasoning traces from individual adapters.[^zou2026merging] These findings establish that multi-adapter composition carries inherent risk that cannot be eliminated through geometric constraints alone.

Rune's default single-adapter retrieval mode mitigates this for the common case. Multi-adapter composition — the ablation target for Phase 5+ — remains unvalidated. The Evolution Operator's fitness-driven selection is designed to detect interference empirically rather than prevent it theoretically (see [Methods](methods.md#evolution-operator)), but this detection mechanism has not been tested. If interference degrades composed adapter quality below single-adapter baselines, the multi-adapter composition mode will be disabled in favor of single-adapter selection — the fallback is architecturally clean because single-adapter selection is the specified default.

#### Hypernetwork Mode Collapse

The hypernetwork may learn the degenerate solution — producing a near-identical "mean adapter" regardless of input trajectory. Diversity regularization in the training loss is the primary mitigation, but the correct regularization weight is unknown before training. The diversity threshold (cosine similarity < 0.9) documented in the [Results](results.md#inter-adapter-cosine-diversity) section is a heuristic, not a theoretically grounded bound.

If mode collapse occurs in Phase 1, the kill-switch gate will fail — but the specific failure mode (loss converges, Pass@1 does not improve, adapters cluster in PCA) will provide diagnostic information about whether targeted regularization can recover. The warning signs documented in the [risk matrix](../appendices/risk-matrix.md) — adapter clustering, cosine similarity approaching 1.0, training loss convergence without Pass@1 improvement — are tracked via MLflow to detect collapse early before the full training budget is exhausted.

#### Cold-Start Corpus Size

The hypernetwork training (Phase 4) requires a corpus of pre-trained adapters from Phase 3. The minimum corpus size for useful hypernetwork training is unknown. The implementation plan sets a target of 50--100 diverse task-adapter pairs, but this is a heuristic estimate. Corpus diversity — over task types, failure modes, and correction patterns — matters more than raw count, but the minimum diversity level is also unknown.

The cold-start problem is two-layered: first, enough trajectories to train Phase 3 direct LoRA adapters; second, enough Phase 3 adapters to train the Phase 4 hypernetwork. If early-phase trajectory collection produces insufficiently diverse corpora, both training stages will be degraded. The Phase 3 direct LoRA bootstrapping path (see [Methods](methods.md#adapter-distillation-pipeline)) is designed to accumulate the initial adapter corpus without requiring the hypernetwork, but the accumulation rate — how many useful adapters per unit of compute time — is an empirical unknown.

---

### Open Research Questions

#### Procedural Encoding in Weight Space

Can code execution trajectories be encoded into LoRA weight deltas such that the base model can execute the encoded procedure on novel inputs without those inputs in context? This is the PBB criterion applied to trajectories.[^cook2025pbb] Background establishes PBB's finding that instructions can substitute for up to 100 examples; the question is whether trajectory-conditioned adapters close the gap between encoding efficiency and reliable weight-based execution.

The question has two levels: (a) does Pass@1 improve — the kill-switch; and (b) does the PBB program evaluation criterion hold — a stronger test. A system could pass (a) by improving solution generation without achieving (b). Level (a) tests whether the trajectory signal reaches the weights at all; level (b) tests whether the encoded knowledge is procedurally executable rather than merely a statistical bias toward correct-looking code.

#### Recursive Refinement Value

The generate-execute-reflect loop produces trajectories of variable depth (1 to `max_attempts` attempts). Does recursive refinement — multiple attempts with error feedback — produce substantially better adapters than single-attempt trajectories? If a successful first-attempt trajectory and a successful 5-attempt trajectory produce similar adapters, the recursive loop adds trajectory collection cost without improving adapter quality. If multi-attempt trajectories produce better adapters (because they contain failure-recovery patterns), the recursive loop is essential to the architecture.

This question is not addressed by the Phase 1 kill-switch, which tests whether any trajectory-conditioned adapter improves Pass@1. It requires a controlled ablation: single-attempt vs multi-attempt trajectories, matched on task, compared on resulting adapter quality. The ablation would need to control for task difficulty — multi-attempt trajectories may correlate with harder tasks, confounding the comparison.

#### Composition Interference

At what point does multi-adapter composition (additive accumulation across hierarchy levels) degrade performance relative to single-adapter selection? Background documents the theoretical concern; Zhang et al.[^zhang2025orthogonality] show orthogonality is insufficient; Zou[^zou2026merging] shows latent trace reactivation. But no empirical measurement exists for Rune's specific adapter distribution.

The question is not whether composition can interfere — it can — but at what combination depth and diversity level interference becomes the dominant effect. The Phase 5+ ablation target for multi-adapter composition (see [Results](results.md#phase-4-ablation-structure-planned)) is designed to answer this question empirically. The three-level hierarchy (project, domain, task) defined in [Methods](methods.md#memory-composition-strategy) creates a natural experimental structure: single-level adapters vs two-level composition vs full three-level composition, compared on the same held-out evaluation tasks.

#### Cold-Start Minimum

What is the minimum corpus size — in terms of trajectory count and diversity — required to train a hypernetwork that generates useful adapters? The implementation plan uses 50--100 as a heuristic. The cold-start minimum determines the practical feasibility of the Phase 4 training approach and whether the Phase 3 accumulation rate is sufficient. If the minimum is substantially higher than 100, the bootstrapping phase becomes a bottleneck that delays the transition from direct LoRA fine-tuning to hypernetwork-based generation.

---

### Future Work

#### QDoRA Integration

QDoRA (Quantized Decomposed LoRA) decomposes weight updates into magnitude and direction components, more closely mimicking full fine-tuning dynamics. It offers faster convergence on complex reasoning tasks and is compatible with 4-bit quantization. QDoRA is not in the current Rune architecture — it is listed as a future enhancement for the Evolution Operator.

In the context of the Discussion, QDoRA integration would replace the QLoRA quantization baseline with a decomposed update scheme that potentially improves adapter quality during the Phase 3 direct LoRA bootstrapping stage. Whether QDoRA's convergence advantages transfer to short-horizon adapter training on coding trajectories is unknown and would require empirical comparison against the QLoRA baseline introduced in Phase 2.[^dettmers2023qlora] The decomposition into magnitude and direction components is theoretically motivated — full fine-tuning updates both, while standard LoRA constrains updates to a low-rank subspace that may not preserve the magnitude-direction decomposition — but the practical benefit for trajectory-conditioned adapters is an open question.

#### Cross-Project Transfer

The three-level adapter hierarchy (project, domain, task levels) is designed to support transfer: domain-level adapters should capture patterns that generalize across projects. Whether domain-level adapters trained on one project's trajectories produce meaningful improvement on a different project's tasks is an open empirical question.

Cross-project transfer testing would require: (a) accumulating adapters from at least two distinct projects; (b) evaluating domain-level adapters trained on project A's trajectories on tasks from project B; (c) comparing against project-specific and vanilla baselines. The domain-level adapter design in [Methods](methods.md#memory-composition-strategy) provides the architectural hook for this experiment, but the experiment itself is contingent on accumulating sufficient adapter diversity across multiple projects — a prerequisite that may take months of production use to satisfy.

#### Online Adaptation

The current architecture assumes a batch training paradigm: collect trajectories, train adapters offline, update the registry. Online adaptation would mean the hypernetwork updates continuously as new trajectories arrive, without discrete training runs. This would require streaming training infrastructure and a mechanism to prevent catastrophic forgetting in the hypernetwork itself — not just in the base model — as new trajectories arrive.[^hu2021lora]

Online adaptation is explicitly out of scope for the current implementation plan but is a natural extension if Phase 4 hypernetwork training proves stable. The transition from batch to online training would also require changes to the adapter registry's write-once semantics — online updates would produce a continuous stream of adapter versions rather than discrete, immutable snapshots.

---

### Broader Implications

Pink et al. argue that episodic memory is the missing capability for long-term LLM agents.[^pink2025episodic] Rune's adapter-per-session design, operating within a five-phase pipeline (decompose, plan, code, integrate, diagnose/repair) with two-step error recovery, is one concrete instantiation of this position — each adapter is an episode, retrieved by task similarity, composable with other episodes. Whether this architecture achieves the episodic memory properties Pink et al. identify (persistence, selective retrieval, compositional combination) is **proposed** — it depends on Phase 1+ empirical validation.

The trajectory modality argument extends PBB's procedural encoding insight:[^cook2025pbb] if instructions are efficient training signal, trajectories — which contain both the procedure and its execution context — may be the most information-dense substrate available for weight-based procedural encoding. This is distinct from the input modalities validated in related work: documents (Doc-to-LoRA[^charakorn2026doc2lora]), in-context examples (SHINE[^liu2026shine]), and text instructions (Text-to-LoRA[^charakorn2025t2l]). Each of these modalities has been validated independently; Rune's contribution is **proposed** as the trajectory extension of the same mechanism.

The kill-switch gate structure is itself a methodological point: explicitly conditioning infrastructure investment on hypothesis validation before building dependent components. If Phase 1 fails, the result is not a negative result about Rune's architecture — it is a null result about whether the trajectory modality extends the Doc-to-LoRA mechanism, which narrows the search space for future approaches. The phased investment structure (Phase 0 hardware validation, Phase 1 kill-switch, Phase 2+ conditional on Phase 1) is designed so that failure at any stage produces useful diagnostic information rather than a wasted investment.

---

[^pink2025episodic]: Pink, O., et al. (2025). The Missing Piece: Episodic Memory for Long-Term LLM Agents. [Full entry](references.md#pink2025episodic)

[^cook2025pbb]: Cook, J., et al. (2025). Programming by Backprop: An Instruction is Worth 100 Examples When Finetuning LLMs. *arXiv:2506.18777*. [Full entry](references.md#cook2025pbb)

[^zhang2025orthogonality]: Zhang, Y., et al. (2025). Inter-LoRA Orthogonality and Composition Interference. [Full entry](references.md#zhang2025orthogonality)

[^zou2026merging]: Zou, H., et al. (2026). Latent Trace Reactivation in Merged LoRA Adapters. [Full entry](references.md#zou2026merging)

[^charakorn2026doc2lora]: Charakorn, R., et al. (2026). Doc-to-LoRA: Hypernetwork Adaptation from Documents. *arXiv:2602.15902*. [Full entry](references.md#charakorn2026doc2lora)

[^charakorn2025t2l]: Charakorn, R., et al. (2025). Text-to-LoRA: Instant Transformer Adaption. [Full entry](references.md#charakorn2025t2l)

[^liu2026shine]: Liu, Y., et al. (2026). SHINE: Scalable Hypernetwork-based In-context Learning. [Full entry](references.md#liu2026shine)

[^prabhakar2024lorasoups]: Prabhakar, A., et al. (2024). LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks. [Full entry](references.md#prabhakar2024lorasoups)

[^hu2021lora]: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*. [Full entry](references.md#hu2021lora)

[^dettmers2023qlora]: Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv:2305.14314*. [Full entry](references.md#dettmers2023qlora)
