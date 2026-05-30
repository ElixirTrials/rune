# Model Layer

The model layer turns a serialized coding trajectory into a task-specialized
LoRA adapter and runs constrained generation under it. It has three parts:
a HyperLoRA perceiver hypernetwork ([`hypernetwork.py`](../api/model/hypernetwork.md)),
adapter scaling/hot-swap utilities ([`adapter.py`](../api/model/adapter.md)),
and structured inference ([`inference.py`](../api/model/inference.md)).
[`ModelWrapper`](../api/model/wrapper.md) bridges these to the engine.

## ctx-to-lora HyperLoRA perceiver

Adapter generation (`generate_adapter_weights`) is a forward pass, not training:

1. **Trajectory → base activations.** `extract_activations_with_model` tokenizes
   the trajectory text and runs the base model with `output_hidden_states=True`,
   stacking the hidden states at `layer_indices` (the layers the checkpoint was
   trained against) into a `(1, num_layers, seq_len, hidden_dim)` feature tensor
   plus its attention mask.
2. **Activations → LoRA weights.** The features feed `hypernet.generate_weights`,
   an Idefics2 perceiver resampler (`ctx_to_lora`) that compresses the per-layer
   activations into latent queries and emits a nested `{module: {A, B}}` dict of
   per-layer LoRA factors. With `offload_base=True` the base model is moved to CPU
   for this pass to avoid holding both models on the GPU at once.
3. **Weights → PEFT state dict.** `_to_peft_state_dict` flattens the nested output
   into PEFT keys (`base_model.model.model.layers.{i}.{self_attn|mlp}.{module}.lora_{A,B}.weight`),
   keeping only the checkpoint's `target_modules`. `lora_A` is used as emitted;
   **`lora_B` is transposed** (`.t()`) to match PEFT's `(out, rank)` layout.

The hypernetwork is loaded once (`load_hypernetwork`); its `idefics2` perceiver is
patched to eager attention and chunked-MLP forward to cap peak memory. S3 checkpoint
URIs are downloaded and cached under `~/.cache/rune/checkpoints`.

## Hot-swap

`hotswap_adapter` calls PEFT `set_peft_model_state_dict`, which writes the new
LoRA factors into the existing adapter modules **in place** — no base weights are
touched and no module is rebuilt, so each engine step can re-specialize the same
model cheaply. The base model is wrapped with a single `LoraConfig` at load time
(rank and `target_modules` taken from the checkpoint).

## Two-phase inference

`generate` produces structured output in two phases:

- **Phase 1 — thinking (unconstrained).** When `thinking_budget > 0`, the model
  generates freely with `</think>` as the EOS, up to the budget. If the output
  does not end in `</think>`, the tag is appended so the prefix is well-formed.
- **Phase 2 — generation.** Generation continues from that prefix. When an
  `output_schema` is given, an xgrammar `LogitsProcessor` compiled from the JSON
  schema masks the logits to only schema-valid tokens; otherwise generation is
  free (optionally with `no_repeat_ngram_size`).

xgrammar's `TokenizerInfo`/`GrammarCompiler` depend only on the tokenizer, and a
compiled grammar only on its schema, so both are memoized
(`_GRAMMAR_COMPILER_CACHE`, `_COMPILED_GRAMMAR_CACHE`) — recompiling per step over
the ~150k-token vocabulary is the cost this avoids.

### Presence-penalty processor

HF transformers has no native presence penalty, so
`PresencePenaltyLogitsProcessor` adds one: it subtracts a flat penalty from the
logits of every distinct token already generated past the prompt
(`gen_ids[i].unique()`). It is attached when `presence_penalty > 0`, alongside the
grammar processor when present.

### Completion retry (JSON only)

If a *structured* generation is truncated (`truncated and compiled is not None`),
`_try_completion` resumes it without re-prompting: it keeps the same adapter
loaded, continues from the model's own output tensor (`prior_output`), and seeds a
fresh `GrammarMatcher` by replaying the partial JSON via `accept_string`, so the
grammar constraint picks up exactly where it left off. Returns whether the matcher
reached a completed state.

## Continuation scaling (engine sub-loop)

A second, distinct truncation path lives in the engine, not the model layer:
[`step_node`](../api/model/wrapper.md) orchestrates it. When a `code`, `repair`, or
`integrate` action truncates, it enters a continuation sub-loop that, each round,
regenerates the adapter from a `code_continue` trajectory, rescales it by
`cont_scaling = adapter_scaling * cont_multiplier` (default `1.53`), hot-swaps it,
and calls `generate_continuation` — an assistant-prefix continuation with **no
thinking and no grammar** that resumes from the accumulated code. The loop is
bounded by `cont_budget` (default `5`) and stops early on a degeneration score
above `0.5`, on the accumulated code passing a syntax check, or when a round is no
longer truncated. The boosted `cont_multiplier` scaling pushes the adapter harder
toward continuation behavior than the base step.
