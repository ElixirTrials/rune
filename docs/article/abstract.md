## Abstract

<div class="abstract" markdown="1">
**Abstract.** Local coding agents such as Aider, Cursor, and Claude Code operate within fixed context windows and lose procedural knowledge between sessions. Architectural decisions, recurring bug patterns, and project-specific idioms are rediscovered from scratch each time. No mechanism exists to accumulate session-derived knowledge as persistent, composable model parameters.

Rune proposes parametric episodic memory for local coding agents: each coding session produces a write-once LoRA adapter encoding procedural knowledge as parameter deltas in the base model's weight space. Adapters are composable, non-destructive, and reversible, enabling selective retrieval without overwriting previously acquired knowledge. This places Rune in the composable weight-space category of memory strategies, distinct from RAG (token-space) and full fine-tuning (destructive weight-space).

The **proposed** mechanism extends Doc-to-LoRA -- a hypernetwork architecture **validated** on document question-answering -- to code execution trajectories: sequences of generation attempts, execution results, and reflection steps. The hypernetwork maps trajectory inputs to LoRA adapter weights in a single forward pass. This trajectory modality extension has not been empirically validated; Doc-to-LoRA's applicability to code trajectories is the central open research question. An Evolution Operator governs the adapter lifecycle using a fitness function combining pass rate and generalization score, performing consolidate, update, forget, and merge operations.

If the Phase 1 kill-switch gate succeeds, Rune is **expected** to deliver parametric episodic memory for coding agents, addressing the missing capability identified by Pink et al. The fitness-driven adapter lifecycle is **specified** in the architecture and will be evaluated empirically against LoRA Soups' composition-by-assumption approach. The trajectory modality extension and the PBB-inspired procedural behavior criterion are **proposed** and require empirical validation.
</div>
