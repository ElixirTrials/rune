## Background

### Memory Approaches for Language Model Agents

Local coding agents — Aider, Cursor, Claude Code — operate within fixed context windows.
As projects grow, agents lose access to earlier interactions: patterns discovered, bugs
solved, architectural decisions made. Each new session begins without knowledge of prior
sessions. The agent re-encounters the same failure modes, re-derives the same solutions,
and cannot accumulate project-specific expertise over time.

Memory approaches for LLM agents fall into three broad categories that differ in where
and how knowledge is stored:

**Token-space memory** (retrieval-augmented generation, long-context prompting, vector
databases) retrieves text snippets and injects them into the context window before
inference. Retrieval quality determines what the agent can recall, but every retrieved
chunk consumes the same scarce resource — context tokens — that the agent needs for
new instructions and code. Scaling memory capacity in this category means scaling
context window size, which remains bounded and costly.

**Destructive weight-space memory** (model editing via ROME or MEMIT, full fine-tuning)
modifies base model weights directly. Each modification risks overwriting previously
stored knowledge — a phenomenon known as catastrophic forgetting. Knowledge is entangled
in shared parameters: storing new facts can corrupt existing capabilities because the
same weights that encode one concept participate in encoding others.

**Composable weight-space memory** (LoRA adapters, adapter libraries) stores knowledge
as discrete, composable parameter deltas that augment the base model without modifying
it. Each adapter can be loaded, removed, or replaced independently. The base model
remains a stable reference point. Multiple adapters can coexist in a library, each
encoding knowledge from a distinct session or task.

Rune occupies the composable weight-space category. Each coding trajectory is distilled
into a LoRA adapter that encodes the procedural knowledge from that session — patterns,
fixes, architectural decisions — as parameter deltas. Adapters are write-once, immutable,
and independently retrievable by task similarity.

Pink et al. argue that episodic memory — single-shot learning of instance-specific
contexts, retrievable and composable — is the missing capability for long-term LLM
agents.[^pink2025episodic] Rune's adapter-per-session design maps directly onto this
framework: each adapter is an episode, stored independently, and retrieved when a
subsequent task resembles the original.

---

### LoRA: Low-Rank Adaptation

LoRA decomposes the weight update for a pre-trained weight matrix as a product of two
low-rank matrices.[^hu2021lora] Rather than updating the full weight matrix \(W_0 \in
\mathbb{R}^{d \times k}\) directly, LoRA reparameterizes the update as:

\[
\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}
\]

where rank \(r \ll \min(d, k)\). The trainable parameter count drops from \(dk\) to
\(r(d + k)\), which is approximately \(2rd\) when \(d \approx k\). For a 7B-parameter
model with typical hidden dimensions, rank 16–64 LoRA adapters require only a few
million parameters — small enough to store hundreds of adapters on disk and load them
on demand.

During inference, the output becomes \(h = W_0 x + BAx\), where \(W_0\) is frozen.
The adapter's contribution is additive and reversible: removing the adapter (setting
\(\Delta W = 0\)) restores the original model behavior exactly. No modification to
the base model is required.

This reversibility is what makes LoRA viable as a memory substrate rather than merely
a fine-tuning technique. Knowledge stored in one adapter does not interfere with
knowledge stored in other adapters or in the base model — each adapter is a self-contained
parameter delta that augments a stable foundation.

---

### QLoRA: Quantized Low-Rank Adaptation

QLoRA extends LoRA with 4-bit NormalFloat (NF4) quantization of the frozen base model
weights, combined with double quantization (quantizing the quantization constants
themselves) and paged optimizers for memory spill management.[^dettmers2023qlora]

The hardware relevance is significant. NF4 quantization reduces the memory footprint
of a 7B-parameter base model from approximately 14 GB (bfloat16) to approximately 4 GB,
enabling LoRA training and inference on consumer GPUs with 24 GB VRAM. For Rune's
target hardware (dual RTX 4090, 24 GB per GPU), QLoRA makes it feasible to keep the
base model loaded continuously while training or serving LoRA adapters without exceeding
VRAM capacity.

QLoRA is an efficiency layer in Rune's design, not a core algorithmic contribution.
Rune's architecture does not depend on QLoRA specifically — any quantization scheme
that fits the base model into available VRAM while preserving LoRA training quality
would suffice. QLoRA is the practical default given current consumer hardware constraints.

---

### Multi-Adapter Serving: S-LoRA

If each coding session produces an adapter, an agent accumulating hundreds of sessions
needs a mechanism to serve the appropriate adapter at inference time without maintaining
separate model copies for each one.

S-LoRA introduces Unified Paging — a unified memory pool that manages adapter weights
and KV cache tensors together in a shared memory space, enabling thousands of concurrent
LoRA adapters to be served from a single base model instance.[^sheng2023slora] Adapters
are loaded and swapped dynamically without reloading the base model. S-LoRA also
introduces batched LoRA computation that amortizes the overhead of applying different
adapters to different requests in the same batch.

S-LoRA validates a core architectural assumption of Rune: that adapter-per-session
memory is servable at scale from a single model instance. Rune's adapter library design
— write-once adapters retrieved by task similarity — maps directly onto S-LoRA's
concurrent serving model. The base model remains loaded; the appropriate adapter is
swapped in per request based on retrieval results.

---

### Hypernetwork Architectures

Hypernetworks — networks that generate the weights of another network — were introduced
by Ha et al. as a relaxed form of weight-sharing.[^ha2016hypernetworks] Rather than
learning fixed weights, a smaller meta-network learns to produce weights conditioned
on some input signal. This decouples *what* is learned (the target network's behavior)
from *how* it is parameterized (the meta-network's learned generation function).
The paradigm enables instant adaptation: given a new conditioning input, the
hypernetwork produces a new set of weights in a single forward pass.

**Doc-to-LoRA** applies the hypernetwork paradigm to LoRA adapter generation.
A hypernetwork takes a document as input and produces LoRA adapter weights in a single
forward pass — no gradient-based fine-tuning required at inference time. The generated
adapter encodes the document's content into the base model's parameter space, enabling
needle-in-a-haystack question answering that outperforms RAG baselines on long
documents.[^charakorn2026doc2lora]

Doc-to-LoRA validates the hypernetwork-to-LoRA mechanism on textual documents and
document QA tasks — retrieving specific facts from long textual inputs. It was not
validated on coding trajectories, procedural knowledge, or code execution traces.
Whether the same mechanism can encode the richer structure of code execution
trajectories — sequential, causal, procedural — is the hypothesis Rune proposes to
test. The transfer to a structurally different input type is the core open research
question.

**SHINE** is a concurrent hypernetwork system that maps in-context examples (few-shot
demonstrations) to LoRA adapters in a single pass.[^liu2026shine] SHINE's input
modality is demonstration examples — input-output pairs the LLM would otherwise process
in-context. The hypernetwork converts this "in-context knowledge" to "in-parameter
knowledge," eliminating the context cost of few-shot prompting at inference time.
SHINE uses structured examples, not execution traces or documents — a distinction
that becomes central to the trajectory modality argument in the final subsection.

---

### LoRA Composition Methods

LoRA's additive structure (\(\Delta W = BA\)) suggests that multiple adapters could
be combined — summed, concatenated, or interpolated — to merge skills from different
training episodes into a single set of weights.

Prabhakar et al. demonstrate that concatenation of LoRA adapters (the CAT method)
outperforms data mixing for binary skill composition tasks.[^prabhakar2024lorasoups]
This is the first empirical evidence that model merging can exceed training-data-level
composition for skill acquisition — suggesting that adapters encode skills in
structures that are, at least partially, combinable.

However, composition is not straightforward. Zhang et al. show that orthogonality
between merged LoRA modules does NOT guarantee semantic disentanglement.[^zhang2025orthogonality]
Enforcing orthogonal adapter subspaces prevents direct parameter interference, but
cannot ensure that composed adapters produce semantically coherent behavior. Two
adapters can occupy orthogonal subspaces yet still produce conflicting behavioral
outputs when applied together.

Additional evidence from Zou demonstrates that merging adapters can cause unwanted
re-emergence of latent reasoning traces due to partially misaligned update
directions.[^zou2026merging] Composition can reactivate behaviors from individual
adapters that were not intended to surface in the merged result.

These interference results inform Rune's design choice: the default retrieval mode
is single-adapter selection — the most relevant adapter loaded alone, not composed
with others. Multi-adapter composition is an experimental extension. Rune's Evolution
Operator (described in Methods) addresses composition through fitness-driven selection
rather than naive merging, allowing the system to discover which adapter combinations
produce beneficial behavior rather than assuming composition is safe by default.

---

### Programming by Backprop: Code as Procedural Abstraction

Cook et al. demonstrate that declarative instructions — including code — can substitute
for up to 100 execution examples when fine-tuning LLMs.[^cook2025pbb] A single
well-structured instruction achieves comparable weight updates to training on dozens
of input-output pairs. This result establishes that the format of training signal
matters: procedural, instruction-like representations are more information-dense than
equivalent example sets.

This finding provides independent empirical support for Rune's core premise: that code
execution trajectories — structured, procedural, and instruction-like — may be an
efficient substrate for adapter training. If instructions can substitute for examples,
then trajectories, which contain both the instructions (the code) and the procedural
context (execution results, error messages, corrections), may encode knowledge more
efficiently than input-output pair datasets of equivalent size.

PBB also finds, however, that in-context instruction execution remains more reliable
than weight-based procedural knowledge for the tasks they tested. This means the
approach is promising but not guaranteed: there is a gap between what can be encoded
in weights via backpropagation and what can be reliably executed from those weights.
Rune must empirically validate whether trajectory-trained adapters close this gap for
coding tasks specifically.

---

### Concurrent Hypernetwork Work and the Trajectory Modality Argument

Multiple hypernetwork systems now exist for generating LoRA adapters from different
input types. The key differentiator across these systems is the input modality — what
goes in determines what kind of knowledge is encoded. Rune's proposed input modality,
code execution trajectories, is structurally distinct from all existing systems.

| System | Input Modality | Knowledge Encoded |
|--------|---------------|-------------------|
| Text-to-LoRA (Charakorn et al., 2025)[^charakorn2025t2l] | Natural language task descriptions | Declarative task specification — static, instructional |
| Doc-to-LoRA (Charakorn et al., 2026) | Long documents | Factual content — informational, unstructured |
| SHINE (Liu et al., 2026) | In-context examples (few-shot demos) | Demonstrated patterns — exemplary, structured pairs |
| Rune (proposed) | Code execution trajectories | Procedural knowledge — sequential, causal, self-correcting |

Code execution trajectories differ from the other three modalities in three structural
ways:

**Temporal ordering.** Steps in a trajectory have a meaningful causal sequence: code
generation precedes execution, execution precedes error, error precedes diagnosis,
diagnosis precedes fix. This temporal structure carries information about causality
that is absent from documents (which present facts without an ordering obligation)
and from examples (which present correct input-output pairs without the process that
generated them).

**Procedural abstraction.** A trajectory encodes a procedure — how to solve a problem
step by step — rather than a fact (what is true), an instruction (what to do), or an
example (what a correct pair looks like). Procedural knowledge is action-oriented:
it specifies operations, their preconditions, and their effects.

**Self-correcting structure.** Trajectories naturally contain failure-recovery patterns:
attempt → error → diagnosis → fix. These patterns are absent from documents, instructions,
and demonstrations. A trajectory that includes a bug and its correction encodes both
the error pattern and its resolution in the same training signal.

PBB's finding that instructions substitute for examples supports this argument from an
independent direction: if procedural representations are efficient for weight-based
learning, then trajectories — which are the richest procedural representation, containing
both the procedure and its full execution context — may be the most efficient input
modality for hypernetwork-based adapter generation. Whether this theoretical efficiency
advantage holds in practice is a central empirical question for Rune. PBB validated
instructions on a different task distribution; trajectories have not yet been tested
as hypernetwork input.

---

[^pink2025episodic]: Pink, M., et al. (2025). Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents. *arXiv:2502.06975*. [Full entry](references.md#pink2025episodic)

[^hu2021lora]: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*. [Full entry](references.md#hu2021lora)

[^dettmers2023qlora]: Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv:2305.14314*. [Full entry](references.md#dettmers2023qlora)

[^sheng2023slora]: Sheng, Y., et al. (2023). S-LoRA: Serving Thousands of Concurrent LoRA Adapters. *arXiv:2311.03285*. [Full entry](references.md#sheng2023slora)

[^ha2016hypernetworks]: Ha, D., Dai, A., & Le, Q. V. (2016). HyperNetworks. *arXiv:1609.09106*. [Full entry](references.md#ha2016hypernetworks)

[^charakorn2026doc2lora]: Charakorn, R., et al. (2026). Doc-to-LoRA: Learning to Instantly Internalize Contexts. *arXiv:2602.15902*. [Full entry](references.md#charakorn2026doc2lora)

[^liu2026shine]: Liu, Y., et al. (2026). SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass. *arXiv:2602.06358*. [Full entry](references.md#liu2026shine)

[^prabhakar2024lorasoups]: Prabhakar, A., et al. (2024). LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks. *arXiv:2410.13025*. [Full entry](references.md#prabhakar2024lorasoups)

[^zhang2025orthogonality]: Zhang, A., et al. (2025). Rethinking Inter-LoRA Orthogonality in Adapter Merging. *arXiv:2510.03262*. [Full entry](references.md#zhang2025orthogonality)

[^zou2026merging]: Zou, J. (2026). Adapter Merging Reactivates Latent Reasoning Traces. *arXiv:2601.18350*. [Full entry](references.md#zou2026merging)

[^cook2025pbb]: Cook, J., et al. (2025). Programming by Backprop: An Instruction is Worth 100 Examples When Finetuning LLMs. *arXiv:2506.18777*. [Full entry](references.md#cook2025pbb)

[^charakorn2025t2l]: Charakorn, R., et al. (2025). Text-to-LoRA: Instant Transformer Adaption. *arXiv:2506.06105*. [Full entry](references.md#charakorn2025t2l)
