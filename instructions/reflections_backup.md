# Reflections

## 2026-05-31 - Pre-corpus robustness gate launch

- The PEFT parity smoke is necessary, but the note that layers are contiguous 0-31 means it does not exercise the positional-vs-absolute layer-indexing failure mode that originally looked dangerous. It can still validate transpose/scaling/export mechanics, but I would not generalize it to arbitrary non-contiguous layer selections unless a non-contiguous parity case is also tested or the production config is locked to contiguous layers.

- The gate is much stronger than the one-value heldout test, but it is still synthetic numeric-token binding. Passing it should justify real-corpus investment, not establish that the adapter will bind edit-relevant semantic facts. I would keep the next real-corpus audit focused on whether the same signal appears for code edits, not just whether numeric needles remain recoverable.

## 2026-05-31 - Scratchpad baseline and data-path review

- The scratchpad says the S3 schema maps directly to D2L (`activation_text` -> context, `teacher_text` -> answer), but the current `hypernet_distill.py` loop reads `record["context"]` and `record["answer"]` directly. Unless the corpus is transformed before training, Stage 1 will either fail on KeyError or silently train on a different preprocessed file than the scratchpad describes. Before any real-corpus run, I would make the mapper explicit and log counts for raw rows, mapped rows, skipped rows, empty contexts, and empty answers.

- The advertised `pre_code`/`post_code` diff coverage is less decisive than it sounds for the current D2L objective. The active D2L mask is teacher-vs-base top-1 disagreement over the answer span; it does not use hunk ranges from `pre_code`/`post_code` the way the legacy diff-aware SFT collator does. So "100% diff coverage" does not guarantee useful D2L gradient. The relevant corpus statistic is per-row diff-token fraction after tokenization and truncation, plus the number of rows with zero teacher/base disagreement.

- I am especially worried about long `activation_text` rows. `_teacher_base_logits` builds `ctx_ids + ans_ids`, then slices to `[:max_length]`, and finally takes the last `ans_len` positions as the "answer" logits. If the context alone exceeds `max_length`, those last positions are context tokens, not answer tokens. That can create a plausible-looking loss on the wrong span. This needs a keep-end or answer-preserving truncation contract before interpreting real-corpus loss curves.

- The scratchpad assumes STaR filtering and row quality are retained, but the D2L loop currently ignores `metadata`, `pass_at_1`, and `quality_score`. That may be fine if the S3 corpus is already perfectly filtered, but it should be verified from a sample rather than assumed. At minimum, inspect the distribution of quality scores, provenance, task/source types, answer lengths, and any failure metadata before treating the corpus as trusted supervision.

- The teacher-quality audit should happen before large-scale training, not only after a disappointing result. D2L compresses frozen-base-with-context behavior; if the teacher is weak, inconsistent, or merely parroting review text on external code-review rows, the hypernetwork can learn that faithfully and still fail pass@1. A small stratified sample comparing teacher outputs/logprobs against `post_code` or edit-local checks would make later failures much easier to attribute.

- The plan discusses contradictory contexts as central evidence, but the real D2L loop appears to train only real context per record. Contradictory contexts are currently a probe/evaluation construct unless negative-context training is added. A future failure of "worse under contradiction" would not necessarily mean the adapter lacks content memory; it may mean the training objective never taught selective anti-conditioning.

## 2026-05-31 - Real-corpus path fixes

- The mapper and answer-preserving truncation fixes address the biggest blockers. The remaining risk is that `teacher_text.startswith(activation_text)` was verified on only a tiny peek. Prefix-stripping is brittle to whitespace, template-version drift, or records where `teacher_text` contains a normalized/re-rendered context. Before trusting corpus-wide training, the mapper stats should include exact-prefix rate, fallback rate, answer length distribution after stripping, and a few sampled fallback examples.

- The open "quality/metadata" item is not just a filtering nicety. The sampled `quality_score` values around 0.28-0.4 suggest the corpus may encode confidence or usefulness gradients that the current D2L loop ignores. If low-score rows dominate, unweighted training may emphasize noisy review edits. I would inspect whether quality correlates with teacher/base disagreement, answer length, and edit size before deciding to drop or use it.

- The teacher-quality audit should include cases where `post_code` changes are large or multi-location. A token-level preference for corrected snippets can look good on small local edits while failing to represent global code-review intent. Stratifying by edit size/source and reporting separate teacher lift would avoid averaging away the hard cases that matter for pass@1.

## 2026-05-31 - Pre-corpus gate failed

- The failed held-out gate is important because it directly falsifies the strongest version of the earlier "reusable binding" interpretation. I agree with the revised read: parity is now a real positive, but the content-binding evidence is train-pair fitting plus weak/unstable extrapolation. The cfg1 trained-value failure is not just variance; it suggests the forced-choice task may be sensitive to token identity, optimization state, or asymmetric priors, so any next synthetic run should report per-value tokenization and base/teacher priors.

- A many-value synthetic disambiguator is reasonable, but it can easily become a new benchmark that teaches numeric lookup rather than code-edit memory. If run, it should vary surface forms and include non-numeric symbolic/code facts, or else be treated narrowly as "does diversity help this toy binding task." Otherwise, a pass may still not transfer to edit-relevant trajectory memory, and a fail may be overly pessimistic about the real 7,670-row corpus.

- Before choosing between "train many synthetic values" and "go to corpus," the key decision is what uncertainty is cheaper to reduce. The real-corpus readiness script already needs to measure teacher quality, diff-token fraction, quality metadata, and edit-relevant token preference. Those measurements may be more decision-relevant than another numeric synthetic run, because they test whether the actual data contains a learnable signal at all.

## 2026-05-31 - Teacher-quality audit redirect

- The advisor redirect is right: real teacher lift is the key kill criterion. But the audit must distinguish "context helps because it contains the answer/template" from "context helps solve the edit." Since `teacher_text` includes an answer block derived from `activation_text`, verify that the teacher prompt contains only pre-answer context and not leaked `## Revision` text. Otherwise, base+context beating base-alone could be a formatting/leakage artifact rather than usable trajectory memory.

- Whole-answer NLL/top-1 is a good first pass, but it may be dominated by boilerplate tokens in `## Revision`. The decisive statistic should emphasize edit-bearing tokens or corrected identifiers/literals, even if edit-local alignment comes later. At minimum, report whole-span lift and a crude non-boilerplate/edit-token lift separately so a high score on markdown scaffolding does not green-light weak supervision.

- The checkpoint provenance question is important enough to log as a separate experimental factor. If the hypernetwork checkpoint is a collapsed Rune-trained artifact rather than a clean Sakana warm-start, then corpus failure has at least two explanations: data/teacher signal failure or damaged initialization. I would avoid drawing method-level conclusions until checkpoint lineage is explicit.

## 2026-05-31 - Teacher-quality audit result

- This is a strong green light that the real corpus contains distillable teacher signal, but it does not prove the hypernetwork can compress that signal. The next corpus run should keep the interpretation split cleanly: teacher audit validates data/teacher signal; training validates compression into adapters. If corpus training fails, do not back-infer that the teacher audit was wrong.

- The edit-local diff-token fraction of 0.143 clears the proposed 0.10 threshold, but it is not a huge margin. With n=120, I would treat this as enough to proceed, not as a stable corpus-wide estimate. The Stage-1 run should log the same teacher/base diff-token fraction online over the actual training rows so we know whether the sampled audit was representative.

- Before launching a long run, define the first-stop criteria. For example: early metrics should show nonzero skipped-safe training, bounded degeneration, student movement toward teacher on diff tokens, and no collapse of agreement-token preservation. Without early stop rules, it will be too easy to rationalize a long expensive run whose loss decreases for the wrong reason.

## 2026-05-31 - Corpus smoke and HPO plan

- The smoke run guardrails are the right shape, but `preservation_agreement >= 0.5` may be too loose if agreement tokens dominate generation quality. A broad perturbation could still pass 0.5 while damaging enough easy syntax or boilerplate tokens to derail completions. I would watch the raw preservation distribution, not just the early-stop threshold, and raise the threshold once the metric's scale is understood.

- The smoke uses `max_seq_len=1024`, while the teacher-quality audit and earlier D2L config discussions centered around 2048. Because long `activation_text` handling was a major risk, the smoke's online `diff_token_frac` should be interpreted at the smoke truncation length, not assumed comparable to the audit's 0.14-0.19 unless the audit used the same max length. If the smoke signal is lower, truncation could be the culprit rather than model compression failure.

- The HPO plan correctly avoids optimizing raw loss, but it should also freeze a held-out split before any tuning. With only 7,670 rows and lots of knobs, repeated short trials can overfit the validation rows or the metric implementation. A small final untouched smoke set would make the eventual HPO result more credible.

## 2026-05-31 - Length generalization plan

- Chunking plus `combine_lora` is a plausible route to longer contexts, but it is not just a length flag. It changes the adapter rank, merge semantics, inference cost, and likely the distribution of generated weights the hypernetwork sees during training. I would avoid treating one-window training success as evidence that the chunked/K-rank regime will work without its own parity and preservation tests.

- Sampling `n_chunks` during training may help length robustness, but it can also introduce a confound: performance changes may come from rank expansion rather than better memory use. Any later length-generalization experiment should compare fixed total rank vs chunk-merged rank where possible, or at least report the cost/quality trade-off explicitly.

- Since current code-review rows mostly fit one 1-2k window, keeping chunking post-baseline is the right prioritization. The baseline should first prove adapter-as-memory on the one-window real edit distribution before adding a second mechanism that expands capacity and complicates attribution.

## 2026-05-31 - Corpus smoke OOM

- The proposed memory fixes are reasonable, but they change the training regime enough that the next smoke should be interpreted as a new configuration, not just a resumed run. In particular, setting the frozen base to `train()` for gradient checkpointing is safe only if dropout and any training-mode stochastic paths are truly absent or disabled. I would verify deterministic teacher/base logits before and after enabling checkpointing.

- Reducing `max_seq_length` to around 768 may solve memory but also changes the corpus signal, especially after the earlier teacher audit was interpreted at 1024. If the smoke succeeds or fails at 768, log online diff-token fraction and answer truncation rates separately; otherwise we may confuse a memory workaround with a scientific result about compression.

- 8-bit Adam is likely necessary, but this is a fragile setting where small optimizer changes could affect whether `scaler_B`, head, and aggregator escape collapse. The smoke should log optimizer coverage and per-component update norms, not only gradients, so an apparent failure is not just quantized optimizer dynamics.

- The OOM itself is a useful correction to the product memory assumption: "base+hypernet fit" is not the same as "base+hypernet+optimizer+autograd fit." Future run plans should state train-time memory budget explicitly, not reuse inference-fit language.

## 2026-05-31 - 4-bit base pivot

- Pivoting to a 4-bit base is a practical memory solution, but it changes the teacher distribution. The previous teacher-quality audit was bf16; if Stage 1 trains against a 4-bit teacher, redo at least a small teacher-vs-base audit in 4-bit and compare diff-token fraction/NLL lift to the bf16 audit. "Both teacher and base are 4-bit" makes the comparison internally consistent, but it does not guarantee the same supervision signal.

- Train/inference precision mismatch is now a major interpretation risk. A LoRA delta learned against quantized base activations may not transfer cleanly to bf16 engine inference, especially with functional LoRA applied inside quantized layers. Before judging pass@1 or tiny bench, run parity/eval in both modes: 4-bit base (training-matched) and bf16 base (target engine), and label results accordingly.

- Rewriting `_functional_lora` around `forward_orig` for bnb layers needs its own equivalence contract. The old PEFT parity smoke does not cover this custom quantized path. I would require a small test that manual base_out + delta matches the intended patched forward numerically on representative quantized modules before interpreting any training signal.

## 2026-05-31 - Smoke #3 healthy

- Smoke #3 is a legitimate green light for the full baseline, but the metric trajectory is still short and batch-1 noisy. The spike to `diff_agreement=0.485` followed by `0.234` should be read as "nonzero learning signal exists," not as a stable performance level. For the full run, monitor rolling medians or windowed averages rather than individual logged points.

- Preservation around 0.90 is encouraging, but preservation can degrade later as the adapter keeps fitting diff tokens. The raised threshold is good; I would also watch for downward drift after the first few hundred steps and not only early-stop failures.

- The full baseline was launched before any held-out edit-relevant eval is wired. That is acceptable as a baseline training run, but its completion should not be described as success unless post-training dual-precision eval and edit-relevant gates are actually run.

## 2026-05-31 - Baseline mid-run snapshot

- The scratchpad now says to build a "frozen split" edit-relevant eval while the baseline is already running for one epoch over about 7,670 rows. If that run is consuming the whole corpus, a post-hoc held-out split is contaminated: every row may already be a training row. Treat any eval on those rows as a train-fit or smoke sanity check, not generalization evidence. For the next judged run, define the split by a stable key before training, preferably by task/source family rather than individual unrolled row, and enforce it in the dataloader so validation and final-test rows are never seen.

## 2026-05-31 - Baseline v2 memory correction

- The correction from "activations caused OOM" to "loss/metric full-vocab fp32 copies were plausibly marginal" is the right epistemic move, but the new `gpu_peak_gb` evidence will only disambiguate if the peak is reset and logged per micro-step or per row. A cumulative `torch.cuda.max_memory_allocated()` high-water mark will saturate after the first worst row and make `corr(gpu_peak_gb, ans_len)` mostly meaningless. Log per-row allocated/reserved deltas, or at least reset peak stats around each measured step and include the row/task id.

- If v2c at seq1024 clears the OOM, that supports the loss-fix hypothesis but does not fully isolate it unless the same long-row configuration that failed before is replayed or the memory trace shows the removed fp32 full-vocab tensors. Avoid upgrading "plausibly marginal OOM cause" into "validated root cause" based only on a successful rerun with different ordering, split usage, or allocator state.

## 2026-05-31 - Length resilience analysis

- Decoupling `max_context_length` from `max_answer_length` is the right next lever, but "context is no_grad, cheap" should be treated as a hypothesis with its own measurement. Activation extraction over 2k-4k context still runs the base model and may retain hidden states/features needed by the hypernetwork, even without backward. Before raising context length broadly, log peak memory and wall-clock separately for context extraction versus answer-span student backward so the new bottleneck is visible.

- The training/eval contract should specify which context length the teacher, base-alone comparator, and hypernetwork each see. If the hypernetwork sees 4k context but the teacher-quality/diff-token audit still uses a shorter or differently truncated teacher prompt, the online diff-token fraction and preservation metrics stop being directly comparable. Decoupling is useful, but only if the truncation policy is explicit and identical wherever a comparison is claimed.

- "Inference context length is not hard-constrained" is directionally true for adapter size, but not literally free. Adapter generation still has tokenizer/model positional limits, perceiver position handling, runtime cost, and out-of-distribution risk beyond trained context lengths. I would phrase the claim as "not constrained by generated adapter size" and keep a long-context parity/preservation smoke before presenting it as hardware-resilient.

## 2026-05-31 - v3b gate and final-training decision

- The clean-val `val_diff_agreement` rise is real evidence that the training loop is no longer inert, but the later real≈contra gate means it should not yet be called content-specific generalization. A safer claim is "unseen-family teacher-matching under the current metric improved"; the metric may still reward a generic code-review adapter until edit-local or harder-contradiction controls separate real from contra.

- Choosing final training over HPO is reasonable because the HPO objective is not specificity-aware yet, but the final run is still optimizing the same objective that just failed to distinguish real from contra. Longer training may amplify generic review-mode behavior as easily as content specificity. Keep periodic checkpoints and run the stronger specificity gate on intermediate/best-val checkpoints, not only the final epoch, so a late overfit or generic peak does not hide the useful point.

- `checkpoint_best.pt` selected by peak `val_diff_agreement` may not be the best specificity checkpoint. Until the gate is strengthened, record enough metadata/checkpoints to choose by a composite after the fact: val_diff_agreement, preservation, real-vs-contra edit-local margin, and train-val gap.

- The bf16 OOM in the smoke gate is not just an eval nuisance if bf16 remains the target engine path. The final result should be reported as 4-bit-train-matched until a memory-safe bf16 gate exists, even if that gate needs smaller `n`, shorter answers, or a streamed/chunked evaluation path.

## 2026-05-31 - Sharper specificity test

- Stopping the overnight final run to test specificity first is the right correction. For the new matched-vs-mismatched logprob gate, report adapter lift over base/zero, not only raw matched and mismatched logprobs. Raw margins can be driven by token difficulty, lexical overlap, or the teacher/base already preferring one edit-local span. The decisive statistic is whether the generated adapter improves matched more than mismatched relative to the same base prompt.

- The mismatched control should be matched on edit-span length, tokenization difficulty, and preferably source/edit type. Otherwise a positive matched>mismatched result may reflect that the mismatched edit is simply less probable or syntactically less compatible with the current file, not that the adapter retrieved the right trajectory fact. If matching is weak, phrase the result as a directional specificity signal and follow it with the completion test before choosing the overnight objective.

- If v3b@200 is the only checkpoint tested, a near-zero margin should not by itself prove that "more-of-same" can never yield specificity. It is strong evidence against launching blind final training, but the branch decision should distinguish "undertrained checkpoint inconclusive" from "objective structurally generic." Negative-context or contrastive training is warranted if stronger checkpoints or a small additional training slice still show matched≈mismatched.

## 2026-05-31 - Specificity result

- The matched≈mismatched result with equal lift over zero is strong evidence for a generic edit-booster at v3b@200. Still, run the checkpoint trajectory gate at more than one adapter scaling if feasible. A single scaling can hide specificity if the generic component dominates at 0.5 while a lower scale exposes a smaller context-specific delta, or if a higher scale causes both adapters to saturate.

- Frequent checkpoints are the right way to test "specificity emerges with training," but choose by the specificity trajectory, not just by `val_diff_agreement`. If matched−mismatched stays flat while matched−zero rises, that is the clean signature that the objective is strengthening a generic adapter rather than trajectory memory.

- If the flat-specificity result persists and the next branch is negative-context or contrastive training, random wrong rows are probably too weak as negatives because they already behave like generic edit prompts. Use hard negatives: same task/source family, same file/edit type, or same row with the edit-bearing feedback removed or contradicted. Otherwise the new objective may again learn "code-review mode" rather than trajectory-specific binding.

## 2026-05-31 - Improving Adapter-As-Memory

- The current objective can be satisfied by a generic "make code-review edits more likely" adapter. To force trajectory memory, the loss needs a relative term: for the same target edit, the matched adapter must beat zero and hard-negative adapters. A practical form is matched KL/CE plus a margin penalty when `score(negative_adapter, target)` approaches or exceeds `score(matched_adapter, target)`. Optimize this on edit-local tokens, not aggregate answer tokens.

- Separate the generic edit prior from the trajectory-specific residual. Estimate a generic adapter from empty, average, or retrieved-irrelevant contexts, then penalize solutions where `W(ctx) - W(generic)` is tiny or non-discriminative. The eval analogue is already clear: matched−zero can rise from generic ability, but adapter-as-memory requires matched−mismatched and centered-delta signals to rise too.

- Improve the data curriculum so the trajectory contains facts that are necessary, not merely stylistic. Rows where many different contexts imply the same obvious revision are useful for a code-edit booster but weak for memory. Prefer or upweight examples with high teacher lift, high matched-vs-hard-negative separation, identifiers/literals/API choices present in the trajectory, and low base/zero solvability. Downweight boilerplate revisions and generic review phrasing.

- Hard negatives should be constructed, not sampled randomly. Best candidates: same row with edit-bearing feedback removed, same row with key identifier/literal contradicted, same repo/source family with a different requested change, or same pre_code with shuffled review comments. These make "use this trajectory" the shortest path to lower loss.

- Batch structure matters. Each minibatch should include matched and hard-negative contexts for the same target edit so the model sees a direct contrast during one optimizer step. Without paired batches, gradient noise may keep rewarding generic review-mode features because they help many examples a little.

- Keep the D2L teacher, but add a specificity gate inside training/validation: matched−zero, matched−mismatched, preservation, and train-val gap. Do not let HPO optimize `val_diff_agreement` alone; it will likely select the best generic adapter. The objective or selection metric should treat matched−mismatched as first-class, even if it is expensive and evaluated less frequently.

- If contrastive training still fails, inspect whether the hypernetwork input actually exposes the discriminative facts. Use trajectory-text ablations: remove the feedback sentence, identifier, or literal and check whether generated weights and edit-local logprobs move. If weights barely change under these ablations, the issue is upstream conditioning or representation, not just the loss.

## 2026-05-31 - Contrastive System Plan

- Building the contrastive path while the current GPU run continues is sensible, but `strip_review_feedback(activation_text)` must preserve the prompt scaffold and distribution as much as possible. If the hard-negative context is visibly shorter, malformed, or missing whole sections, the model can satisfy the margin by detecting "has review text" rather than binding the specific trajectory fact. Prefer masking or replacing only the edit-bearing sentence/span with a neutral placeholder, and log context length/template deltas for matched vs negative.

- The margin loss should use the same edit-local token mask and answer truncation policy as the specificity gate. If training contrast is computed over a broader span than evaluation, it may still learn boilerplate or generic revision style. The implementation should make the edit-local mask a shared helper used by both `contrastive.py` and `gate_trajectory.py`.

- Treat the contrastive weight as a safety-critical knob. Too low will not overcome the generic optimum; too high can sacrifice teacher matching and preservation. Start with a small sweep or log the raw KL and margin-loss magnitudes on a fixed batch before choosing a default, so the combined loss is not dominated accidentally.

## 2026-05-31 - External Research: Adapter Memory Training

- Context7 TRL docs and DPO/CPO literature suggest using a preference-style objective with a reference, not a raw margin alone. For our setting, define the implicit reward as adapter lift over a reference/generic adapter: `r(ctx, y)=logp_adapter(y)-logp_ref_or_generic(y)`. Then optimize matched > hard-negative in this normalized reward space. This should reduce the chance that the loss rewards globally easy edits or a generic code-review prior.

- Hard-negative mining guidance from Sentence Transformers is directly relevant: negatives should be hard but not false negatives. Do not rely on one `strip_feedback` negative. Maintain a candidate pool per row: feedback-masked same row, contradicted identifier/literal, same repo/source-family different edit, and semantically near but nonmatching rows. Select negatives by difficulty bands and margins, skipping candidates that are too close to the positive or obviously unrelated. Log negative type and difficulty so failures are interpretable.

- Doc-to-LoRA's training recipe is a warning about scale and staging. Their reported setup uses staged learning and very large effective packed context-token batches; otherwise training "converges too early." Our small, batch-1/grad-accum regime may be finding the shallow generic optimum because it lacks enough simultaneous context diversity. If hardware allows, prioritize larger effective context-token batches, paired positives/negatives in the same optimizer step, and staged training: first learn stable internalization on one-window examples, then add contrastive/chunked/compositional pressure.

- LoRA-as-memory analyses emphasize that supervision format strongly controls retrievability and that raw text is a weak memory substrate. Convert trajectories into dense memory views, not only `activation_text`: explicit requested change, relevant identifier/literal/API fact, evidence span, target edit span, and QA-style probes. Mixing formats may help the hypernetwork bind the same episode from several views instead of learning generic review style.

- Memory-adapter work suggests specialization can reduce interference. Consider separating "generic edit skill" from "episodic residual": either a frozen learned generic adapter plus a trajectory-residual hypernetwork, or two heads where one predicts shared edit prior and the other is penalized/selected for matched-vs-negative specificity. This matches the observed failure: the current model already learns useful generic editing, but that should not consume the trajectory-memory channel.

- Preference-optimization work on hard negatives warns that negative selection should evolve with the model. Once a cheap gate exists, periodically choose negatives that the current adapter scores highly, rather than fixed random or fixed ablation negatives. Those are the negatives that estimate the missing normalization term and give gradients against the actual current failure mode.

- Episodic-memory benchmarks stress entity/event/time grounding. For coding trajectories, the analogue is task identity, file/function, reviewer request, causal rationale, and concrete edit. If those are implicit inside a long prompt, the hypernetwork may not bind them. A structured "episode card" prepended to the trajectory could be a low-risk way to expose the discriminative facts without changing the engine prompt.

- Treat LoRA memory as capacity-limited and complementary, not magically complete. If specificity remains weak, do not only raise rank or train longer. First check whether the adapter channel is being asked to store too many generic style/format tokens. Push boilerplate back to the base prompt/template and reserve adapter capacity for episode-specific deltas.

## 2026-05-31 - Overnight Campaign Plan

- The campaign should preserve an interpretable baseline for each intervention. If B1, B2, C, and negative-pool changes are chained without a fixed eval set and a shared starting checkpoint/config, any win will be hard to attribute. Prefer one-variable experiments where each candidate starts from the same warm start, uses the same clean gate rows, same scalings, same max steps, and same checkpoint-selection rule.

- "Try everything" is fine operationally, but the morning PR comment should separate exploratory ranking from evidence. A short overnight result can identify the most promising direction; it should not be framed as proving the final training recipe unless it survives rerun or at least a second seed/split slice.

- The pass@1 north star is important, but do not let a generic booster with better pass@1 blur the adapter-as-memory question. Report two lanes: product utility (`adapter > base` on tiny pass@1/edit completion) and research claim (`matched > mismatched` on edit-local trajectory gates). A candidate can be useful while still failing the adapter-as-trajectory-memory bet.

## 2026-05-31 - B1 Contrastive Smoke

- Letting gradients flow through the negative adapter is necessary for the hinge to actually separate matched from wrong-context adapters, but it changes what "success" can mean. A margin can open because matched improves, because the negative is pushed below zero/base on the gold edit, or both. Report matched−zero, neg−zero, and matched−neg separately; a good memory signal should primarily preserve or improve matched while reducing only the inappropriate negative lift.

- The smoke should also watch whether contrastive training harms generic edit utility and preservation. If B1 wins matched−mismatched by making stripped-feedback adapters actively anti-edit, it may pass the memory gate while hurting product utility or destabilizing the generic editing skill. This is another reason the two-lane report matters.

- The scaffold-parity checks are reassuring, but token length parity is not enough. If possible, sample a few matched/negative rendered contexts and verify the masked feedback placeholder does not introduce an obvious lexical cue or unnatural template boundary that the hypernetwork can exploit instead of binding the substantive feedback.

- If the next move is a high-weight re-smoke, add a stop condition beyond preservation: `swapneg−zero` should not become strongly negative while `matched−zero` stays flat. That pattern would mean the margin is being opened by teaching the stripped-feedback adapter to anti-predict edits, not by making the matched trajectory more informative. A useful high-weight run should raise matched−swapneg mainly through matched lift or selective removal of inappropriate negative lift.

- The smoke gate now makes the upstream-conditioning probe the right next step. For that probe, avoid relying on a single global weight cosine; it can hide small but high-leverage deltas or be dominated by the shared generic adapter. Report layerwise relative L2, centered deltas versus neutral/zero context, generated-weight norms, and the induced edit-local logprob deltas. The decisive question is not "are weights numerically different somewhere?" but "does the context-dependent residual move the target edit differently?"

- If generated weights are near-constant across matched/swap/mismatch contexts, treat this as a conditioning or representation failure before changing the loss again. If weights differ but logits do not, the issue may be scale, rank placement, or the generated delta being swallowed by the generic component. Those are different fixes; the probe should separate them.

- Mean-pooled feature comparison is a pragmatic fix for variable sequence lengths, but it can erase sparse edit-bearing feedback tokens. If pooled features look similar, do not conclude extraction ignores feedback without a token-local check: compare pooled features over the feedback span, or ablate/mask the feedback span in the same row and measure generated-weight/logprob movement. The generated-weight comparison remains the stronger end-to-end conditioning signal.

- The v2 probe points to attenuation, not absence: feature variation survives, but the perceiver/head maps it into a tiny residual on top of a large generic adapter. The scaling gate is worth running, but uniform adapter scaling amplifies both generic and residual components. If specificity still does not emerge cleanly, the fix is probably not just larger scaling; it is reducing or factoring out the generic component, e.g. generic-plus-residual heads, centered/residual loss, or an explicit penalty/reward on `W(ctx)-W(neutral)`.

- Layerwise concentration around mid-late layers may be useful. If layers 21-24 carry the largest cross-row residual while feedback-swap residual remains tiny, a targeted scaling/regularization or placement analysis by layer could be more informative than global scaling. Track whether edit-local logprob specificity comes from the same layers that show residual movement.

- The scaling gate strongly disfavors "Sakana-style up-scaling will reveal hidden specificity" for B1 step40. Since matched−swap and matched−mismatch stay at noise across scales while generic lift and preservation move substantially, the residual is either not encoding the relevant feedback or is being applied in the wrong subspace. The next useful fix is architectural/representational: factor generic vs residual, train directly on centered residual movement, or change how trajectory facts reach the weight heads.

- Before declaring this objective-independent, confirm the same flat specificity at the warm start and A checkpoints. If warm start, A, and B1 all show scale-invariant matched≈mismatch while generic lift changes, that is strong evidence of a persistent perceiver-to-weight conditioning bottleneck rather than a contrastive-loss failure.

## 2026-06-01 - Data Archaeology

- The corpus structure changes the interpretation of the whole experiment: the adapter is being asked to store/reconstruct mostly unchanged code, while the feedback-specific edit is a small minority of both context and answer. Full `## Revision` distillation therefore rewards code-body copying and generic edit style much more than trajectory-specific memory. A better memory objective should target edit deltas, patch hunks, or edit-local forced-choice spans, not full revised-code reconstruction.

- The hard-negative design should preserve current code but contrast only the feedback-to-edit mapping. That means same `pre_code`, same scaffold, same target hunk, different or masked feedback. Scoring full-answer logprob will mostly measure whether the adapter can reconstruct the shared code body; scoring edit-local changed tokens is the relevant memory signal.

- Direct recall of full answer can still be a useful diagnostic, but it should not be the primary adapter-as-memory test. If full-answer matched≈mismatched, that is a broad failure. If full-answer matched>mismatched, it may still be driven by shared code-copy effects unless edit-local changed-token lift separates too.

- This points toward a data transformation before more architecture work: train/evaluate on compact edit programs or patches derived from `pre_code -> post_code`, with feedback as the conditioning fact. Keep full-code generation as a downstream integration test, not the core distillation target.

## 2026-06-01 - Confirmation Gate

- The A600 result is not "no memory"; it is weak code-context memory plus strong generic edit boosting. That distinction matters. The adapter appears to encode some row-specific code-body signal after enough training, but the feedback/request signal still does not bind. Frame the finding as "memory channel is dominated by code-copy/generic components, not absent."

- Scaling exposes a small code residual but also destroys preservation. That makes scale a diagnostic, not a production fix. Any centered-residual or layer-targeted follow-up should be judged by whether it increases feedback/edit specificity without the same preservation collapse.

- The next recall probe should report copy, edit, and full spans separately and include the same preservation/generic-lift context. If copy recall separates but edit recall does not, the fix is data/objective reformulation toward patches. If neither separates, the bottleneck is more architectural.

## 2026-06-01 - Recall Probe

- The recall probe weakens the "code-context memory" interpretation: copy tokens do not separate matched from mismatch and are worse than zero, so the adapter is not reliably storing recoverable code-body facts. The positive signal is mostly an edit-token prior, with only faint matched-vs-mismatch separation on edit tokens.

- This makes patch/edit-program supervision the highest-leverage next direction. The adapter should not be trained to carry a full revised file; it should carry the small causal delta from feedback and local code state to edit operation. Full-code likelihood is currently punishing the adapter for not being a verbatim code cache and obscuring whether the episode fact was learned.

- For the PR/update, separate three claims explicitly: no document-style recall, useful generic edit prior, and weak/faint context-conditioned edit signal. Only the third is relevant to adapter-as-trajectory-memory, and it is currently too small to support success.

## 2026-06-01 - Episodic-Memory Framing

- The current corpus is not actually coding trajectories; it is single-turn code-review-to-revision data. That matters because adapter-as-trajectory-memory should preserve ordered agent state across decompose/plan/code/repair/integrate steps, while this data only tests whether one review comment can condition one edit. Treat all conclusions as about single-turn review episodes unless a true multi-step trajectory corpus is used.

- Teacher-forced edit reproduction is not the same as episodic recall. A real episodic-memory test should query facts from the episode that were present in the conditioning context but not simply the supervised output, such as review feedback, file path, task identity, rationale, or prior action. The new QA recall probe is therefore necessary, not optional.

- Patch/edit-program supervision may fix the feedback-to-edit objective, but it still will not prove semi-Markov trajectory memory by itself. The next dataset should include ordered transitions and queries over prior steps, with negatives that preserve local code but alter earlier trajectory facts. Otherwise the system can succeed at patch conditioning while still failing the broader memory bet.

## 2026-06-01 - QA Recall

- The QA recall result is the cleanest negative for episodic memory so far: matched and mismatched adapters are indistinguishable on feedback/file recall, while both beat zero slightly through generic code-review mode. That means the adapter does not expose queryable episode facts, even when the queried facts were in the conditioning text.

- Future training should include explicit queryable-memory supervision, not only edit reconstruction. For each episode, add probes such as "what feedback was given?", "which file/function?", "what was the requested change?", and "what changed from pre to post?" with matched-vs-hard-negative controls. If the adapter cannot answer those, it is not a usable episodic memory substrate.

- Free generation producing near-identical boilerplate across episodes is important evidence of mode collapse into review style. Track diversity and episode-specific lexical overlap in future gates, not just logprob, so generic fluent hallucination is not mistaken for recall.

## 2026-06-01 - Next Scoped PR

- Yes, a Doc2LoRA/Gemma replication is worthwhile, but only as an isolation experiment, not as the main product path. Use their code/checkpoint/base model if available and first reproduce their NIAH or QA fact-recall behavior locally. Then fine-tune or evaluate on a tiny Rune-style patch/query-memory dataset. If their checkpoint recalls facts but our HPO/delta-coder start does not under the same small task, initialization/pretraining is implicated. If both fail on our full-code objective but succeed on patch/query data, data/objective is the main issue.

- Keep that PR narrow: "Can a known-good Doc2LoRA-style hypernetwork produce queryable adapter facts in our environment?" Success criteria should be the recoverability scorecard, especially matched > mismatched for goal/file/change facts. Do not mix this with broad engine training, HPO, or pass@1 work; otherwise the result will be hard to attribute.

- Starting from the current pre-warmed delta-coder remains suspect. It may have a strong generic edit prior and weak context-to-weight gain, exactly matching the observed 25-30x attenuation. Compare at least three starts on the same tiny task: current HPO/delta-coder, fresh/random hypernet head, and Doc2LoRA/Sakana checkpoint if compatible. Log feature residual -> weight residual, matched-mismatch QA recall, and preservation.

- The V1-style oracle/delta-coder is useful if framed as an upper bound: can per-example optimization produce a LoRA that stores/query-recovers the episode at all? If an oracle LoRA cannot pass the recoverability scorecard, the target/data is ill-posed. If oracle passes but hypernetwork fails, the problem is amortization/conditioning. This is a stronger diagnostic than immediately continuing to train the delta-coder.

- I would not spend the next PR simply training the current delta-coder longer on the same full-revision target. The evidence says that objective rewards generic edit boosting and hurts code recall. More of the same may improve a proxy while moving away from adapter-as-memory. Any continued training should use patch/edit-program and explicit QA recoverability targets.

- Recommended order: (1) oracle per-row LoRA on a tiny patch+QA episode set; (2) Doc2LoRA/Gemma reproduction to validate known-good fact recall on our hardware; (3) three-initialization comparison on the same tiny recoverability task; (4) only then scale the winning setup to Rune trajectory data. This separates "can LoRA store it?", "can this hypernetwork architecture amortize it?", and "does our data teach it?"

