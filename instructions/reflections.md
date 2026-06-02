# Reflections

## 2026-06-01 - Monitor Miss and Doc2LoRA-First Plan

- The scratchpad monitor did not fire because `instructions/scratchpad.md` remained empty; the new #52 design existed only in the active terminal transcript. That means the monitoring contract is file-based, not terminal-based: no scratchpad write means no review event. If plans are discussed interactively for several minutes before being logged, I need to review either after the scratchpad write or from the terminal transcript explicitly.

- The proposed Doc2LoRA/Gemma reproduction is a good positive control, and public Doc2LoRA material frames NIAH exactly as context internalization: the base model receives only the query after the document is internalized, and the adapter maintains near-perfect retrieval beyond the native context window ([Sakana](https://pub.sakana.ai/doc-to-lora/), [GitHub](https://github.com/SakanaAI/Doc-to-LoRA)). That validates the scorecard's ability to detect recall when recall exists.

- Pushback on ordering: issue #52's staged diagnostic says oracle per-row LoRA first, then Doc2LoRA positive control. I would not let a Doc2LoRA-first PR become a substitute for the oracle upper bound. Doc2LoRA can kill "our probe is blind," but only the oracle answers "can a LoRA store this Rune episode target at all?" If the order is reversed, state explicitly that this is a probe-validation prelude and not evidence about Rune target well-posedness.

- The shared pure scoring core is the strongest part of the plan. Without sharing the exact span-logprob implementation, the control would validate only the methodology, not the code path used by `tools/diag_recoverability.py`. Add a tiny tensor-level unit test for next-token indexing and span slicing before trusting any m-mismatch number.

- The "tiny Rune-episode bridge" should be interpreted asymmetrically, as the plan says. A pass would be surprising and informative; a fail is mostly OOD and should not be used to condemn the Rune target. Keep that bridge small and non-gating so it does not blur the cleaner oracle diagnostic.

- Tighten the Doc2LoRA pass criterion beyond a bare `m-mismatch > 0`. On a known-good NIAH control, a tiny positive margin could be tokenizer/logprob noise or mismatch sampling luck. Report generation accuracy, per-episode margins, multiple mismatches per episode, and either a confidence interval/standard error or a simple bootstrap. The useful calibration is not just sign; it is effect size at known-good recall.

- Operational caution: the plan mentions HF auth, a sibling checkout, and an isolated env. Keep tokens entirely in the existing environment/HF credential store, do not write secrets into scripts or scratchpad, and avoid committing vendored third-party code, checkpoints, or `.venv` artifacts. Even if Gemma-2B "fits trivially," still follow the repo's GPU rule: `free -g` before model load and GPU runs under `tools/run_guarded.sh`.

## 2026-06-01 - Spec Review: Doc2LoRA Control

- The spec now makes the Doc2LoRA-first reorder explicit and preserves the oracle as the real Rune target well-posedness test. That resolves the main ordering concern.

- Remaining gap: the DoD still says probe validation passes on `m-mismatch > 0 AND m-zero > 0`. For a known-good control, require more than sign: multiple mismatch adapters per episode, per-episode margins, generation accuracy beside logprob margins, and a simple uncertainty estimate. Otherwise a tiny positive average could be over-read as validated recall.

- The spec mentions accepting the Gemma license and downloading checkpoints, but the implementation plan should spell out hygiene: no HF tokens in scripts, docs, scratchpad, or shell history beyond standard credential tooling; no committed third-party checkout, checkpoint, cache, or venv artifacts. This matters because the deliverable deliberately creates `third_party/doc-to-lora/` and downloads gated weights.

## 2026-06-01 - Workflow Launch Review

- The workflow scope split is good: deterministic CPU artifacts can be built autonomously, while gated Gemma download, third-party env setup, and GPU runs remain orchestrated manually. That preserves the isolation/control value without hiding license or hardware failures inside the workflow.

- Add an explicit artifact-hygiene check to the workflow output before declaring CPU phase done: `third_party/doc-to-lora/`, checkpoints, HF cache paths, and `.venv` must be ignored/untracked or absent from `git status`. Otherwise a correct implementation can still leave a messy or risky working tree.

- For `tools/d2l_control/run_scorecard.py`, keep the bridge result visibly non-gating in the emitted JSON/logs, not only in docs. A later reader should not be able to mistake a Gemma-on-code bridge failure for failure of the Rune oracle target.

## 2026-06-01 - GPU Drive / Flash-Attn Pivot

- The isolation decision is being validated by the `ctx_to_lora` and `transformers` version collision. Preserve that as an experimental invariant: results from the Sakana positive control should be attributed to the Sakana repo + its pinned stack, not to Rune's runtime. Log the exact third-party commit, checkpoint path/hash, torch/transformers/flash-attn versions, CUDA version, and any local patches before interpreting NIAH or scorecard numbers.

- Be careful with the torch/flash-attn downgrade path. If a prebuilt flash-attn wheel forces a torch/CUDA stack that differs from the published Sakana setup, the first result should be framed as an environment-reproduction result, not yet a scientific control. The minimum sanity check is: unmodified Sakana NIAH script passes before any custom Rune scorecard conclusions are drawn.

- Local patches to Sakana attention code should remain inert for the reported positive control, as the scratchpad says. I would make this mechanically auditable: keep a patch diff artifact in MLflow and explicitly record whether `D2L_ATTN_IMPL` was unset/default during each run. A positive control with patched attention active would no longer be a clean reproduction.

- MLflow is useful here, but avoid logging huge/gated artifacts such as checkpoints or cached model files. Log JSON metrics, command lines with secrets redacted, env/version manifests, small patch diffs, and result summaries. The checkpoint provenance can be a path/hash, not the model file itself.

## 2026-06-01 - Smoke Pass and Sakana-on-Rune Bridge

- The matched-vs-mismatch smoke margin is decisive enough to calibrate the scorecard scale: known-good recall giving m-mismatch around +7 nats makes #49's +0.0005 goal and +0.075 diff margins effectively negligible. This is strong evidence that the scorecard is not blind.

- The code-recall and Rune-episode bridge results are a real positive: Sakana's Gemma Doc2LoRA checkpoint can bind code facts and can recover goal/file/diff facts from Rune-style episodes. That rules out "LoRA/perceiver-style adapters cannot carry queryable code facts" as a broad objection.

- Tighten the causal claim. These results do not yet fully rule out Rune-specific architectural or implementation issues; they rule out the broader capacity/probe/ill-posed-facts story. The remaining suspects are Rune's training objective/data shape/scale and possibly Rune's specific Qwen hypernetwork implementation. The pending `qwen_4b_d2l` run is important before saying base-family effects are ruled out.

- Keep the unmodified Sakana NIAH reproduction as the anchor before final wording. The custom smoke and bridge are already informative, but the positive control is cleanest only after the published eval path passes under the logged environment.

- The next recommendation should be phrased as a testable hypothesis, not a conclusion: train Rune-style adapters with explicit queryable-memory/patch supervision and compare against the current full-revision objective. The evidence now strongly favors that hypothesis, but "training recipe" includes objective, data format, scale, batch structure, and checkpoint initialization; do not collapse those into a single cause too early.

## 2026-06-01 - NIAH Anchor and Qwen-Family Control

- The unmodified Sakana NIAH reproduction plus the Qwen-4B D2L result clears the two biggest outstanding cautions. The scorecard detects real recall, the public eval path works in this environment, and base-family mismatch is no longer a plausible primary explanation for Rune #49's near-zero margins.

- I agree with the strategic reframing: goal/diff/tail/avoid are query facets over a single remembered episode, not separate mechanisms. The data implication is important: "avoid failure" does not need a special adapter objective, but it does need episodes that actually contain failed attempts and accepted alternatives. Without that coverage, the facet is untestable no matter how good the recall objective is.

- Keep "training recipe" decomposed. The evidence strongly indicts Rune's full-revision edit-reproduction recipe, but objective, query supervision, data format, training scale, batch composition, and initialization are still entangled. The next tiny finetune should be designed as an ablation against the zero-shot Sakana checkpoint: same scorecard, same episodes, before/after finetune, plus retention on NIAH/code recall.

- "Light finetune" should have a hard retention gate, not just a gain gate. A useful specialization improves diff/tail/trajectory facets while preserving broad recall ability; if it gains diff by forgetting NIAH/code facts, it has not solved adapter-as-memory, it has just overfit a new small task.

- The base-model choice is now legitimately open, but separate "best coding base" from "available recall hypernet." A weaker coder with a proven recall hypernet may be the fastest research path; a stronger coder without a recall-capable hypernet may be the better product path but requires paying the Sakana-style training cost first.

## 2026-06-01 - Continuation Facet

- The continuation/tail result is another strong positive: Sakana's adapter improves specific code-state likelihood over both mismatch and zero, exactly where Rune #49 was negative. That makes "drives next step" look like a recall facet that the Doc2LoRA recipe can support.

- Be precise with the scorecard wording. The experiment covers goal/file/diff/continuation-style tail recall, but it does not yet cover `avoid` in the original #52 sense unless the episode includes a failed/rejected attempt and an accepted alternative. Do not say all four original scorecard facets are solved until avoid-failure is tested on failure-bearing episodes.

- The generation drift is not a failure for the current ranking/logprob scorecard, but it is relevant for downstream agent use. Keep ranking recall and verbatim generation as separate metrics: positive logprob memory can drive next-step selection even when greedy reconstruction of a multi-line body is imperfect.

## 2026-06-01 - Light-Finetune Ablation Launch

- The finetune is correctly guarded against the #49 failure mode by tracking matched-vs-mismatch, not only matched-vs-zero. Keep the before/after comparison on exactly the same eval episodes and mismatch construction so any gain is attributable to specialization rather than sampling drift.

- The retention gate should probably be stricter than "NIAH + clean-code m-zero kept >=70%" if the claim is "light specialization preserves recall." A 30% loss in known-good recall is large. At minimum, report the continuous retention ratios and treat 70% as a fail-soft warning threshold, not a clean pass threshold.

- "Light" here means few steps from a recall-capable checkpoint, not few trainable parameters: freezing all but the hypernet still leaves a large trainable module. Log update norms or weight deltas relative to the starting checkpoint so later interpretation can distinguish small specialization from substantial retraining.

- CE on query+answer can improve answer-format familiarity without improving episode specificity. The decisive gain should be per-facet m-mismatch on held-out episodes, especially diff/tail, plus unchanged or improved generation accuracy where relevant. Avoid averaging goal/file gains over a stagnant diff facet.

## 2026-06-01 - Light-Finetune Result

- The result is exactly the distinction the scorecard was built to expose: retention is excellent, m-zero improves, but diff specificity gets worse. This is a clean negative for plain CE as the specialization objective, not a negative for warm-started specialization.

- The contrastive follow-up should be facet-paired. For `diff`, hard negatives must preserve local code and answer format while changing the trajectory fact that determines the hunk; for `goal`, negatives should alter the request while preserving file/code; for `file`, same request/diff with a different file may be needed. A generic "other episode" negative may be too easy or may reward superficial metadata binding.

- Report contrastive results per facet and per negative type. If file improves but diff does not, that is not a partial win for the main coding-memory claim; it means the model learned easy metadata binding while still failing the hard edit-relevant facet.

## 2026-06-01 - Feedback-Swap Diff Collapse

- The feedback-swap result is important: diff m-mismatch dropping from the generic-negative +1.01 to +0.17 shows that much of "diff recall" is local-code/code-output echo rather than binding to the trajectory fact. This is a stronger and cleaner version of the #49 code-driven-not-feedback-driven finding.

- I agree with the memory/action separation: do not train the adapter to store or emit diffs as the primary memory target if that primes generic patch emission. Store the episode state and trajectory facts (goal, tail/current state, tried steps, failure reasons), then test whether the base can use those memories to choose or generate the right edit.

- Be careful not to overstate "diff is a bad target" universally. It is a bad *memory supervision target* under hard negatives that preserve local code and alter the trajectory fact. The diff remains a valid downstream action/evaluation target for the memory-to-edit utility test.

- The proposed eval-only facet-negative test is the right next move. For goal, feedback-swap should produce a large matched-over-swap margin if the adapter really binds the request; for diff, collapse under feedback-swap would confirm it is not request-bound. Report both generic-negative and facet-specific hard-negative margins.

- If contrastive training is retried, fix memory/OOM only after the eval-only negative analysis finishes. Otherwise there is a risk of spending GPU time salvaging a diff objective that the better negative already shows is conceptually misaligned.

## 2026-06-01 - Deliverable 1 Reviewer Sign-Off

- Sign-off: the Deliverable 1 conclusion is supported. The positive control reproduced, the scorecard is calibrated against real recall, Sakana recalls Rune-style facts zero-shot, Qwen-family base mismatch is ruled out as primary cause, and the feedback-swap negative cleanly separates episode-fact memory from diff/code echo.

- The strongest causal wording I would approve is: Rune #49 failed because its recipe did not train queryable episode memory and instead optimized an edit/full-revision emission objective that rewards generic/code-driven behavior. Keep "training recipe/objective" as the cause bucket; do not reduce it to only "objective" without acknowledging data format, query supervision, scale, batch structure, and initialization.

- The memory/action separation is now the right framing for the PR: memory should store goal/state/tried/failure facts; diff remains the downstream action target. The next decisive experiment is memory-to-edit utility, not more diff-as-memory optimization.

- Caveats to carry into the PR: avoid-failure is still untested without failure-bearing trajectories; recall does not yet imply pass@1 or edit utility; and last-N-lines/tail should be recalled as state, not trained as verbatim output.

- Durability before PR: make sure any committed scorecard code can reproduce the key JSON/MLflow metrics from documented commands, and run `git status` to verify third-party checkout, checkpoints, HF cache, and venv artifacts are not staged or tracked.

## 2026-06-01 - Deliverable 2 Utility-Test Design

- The move to a memory-to-edit utility test is correct, and ranking should be primary. Free generation already showed enough drift that it should be a secondary realism check, not the first pass/fail criterion.

- The main risk is construct validity: `external_codereview` is single-turn, so there is no natural next action. A constructed utility task must include controls showing the edit is actually inferable from the hidden goal/state: an in-context upper bound with goal+tail in the prompt, zero/no-adapter lower bound, matched-adapter, mismatch-adapter, and ideally feedback-swap hard negatives on the same file/local code.

- Keep the prompt contract exact. If goal/tail are meant to live only in the adapter, do not leak them through filenames, comments, patch text, or distractor labels. The prompt can contain the local pre-code needed for editing, but not the requested change or prior trajectory fact being tested.

- Ranking distractors must be hard but valid. Same-file feedback-swap edits are better than random wrong edits, because random distractors may be rejected by syntax/local-code compatibility rather than by recalled episode state. Report per-negative-type results so "utility" is not inflated by easy distractors.

- Add an "in-context solves it" sanity gate before interpreting adapter failure. If base+prompted goal/tail cannot rank the correct edit, the constructed task is ill-posed as an edit-utility benchmark rather than evidence against adapter memory.

## 2026-06-01 - Single-Step Avoid Coverage

- The correction is right: a single external code-review row can represent one tried attempt. `pre_code` is the rejected attempt, review feedback is the rejection reason, and `post_code` is the accepted form. That makes a one-step `avoid` facet testable before mined multi-step trajectories.

- The avoid arm needs its own prompt contract, as noted. If rejected `pre_code` is visible as the current code, the model can compare candidates using in-prompt evidence rather than adapter memory. For avoid-as-memory, the rejected form and critique should be internalized, while the scoring prompt should expose only the neutral task/candidate-comparison scaffold needed to rank accepted vs rejected.

- Use zero and mismatch baselines carefully because accepted code may be intrinsically more likely than rejected code. The memory signal is not `accepted > rejected` by itself; it is an improvement in the accepted-minus-rejected preference under matched memory versus zero and versus a mismatch episode with similar local code/edit type.

- Keep one-attempt avoid separate from multi-attempt avoid. Passing the premined review-pair avoid test would show the adapter can remember one rejected form and its critique; it would not yet prove it can preserve ordered exploration history across several failed repairs.

## 2026-06-01 - EOD Checkpoint Scope Expansion

- Strong agreement with the new avoid refinement: do not internalize failed code verbatim. The memory target should be abstract failure facts/critique, because the diff/code-echo evidence says embedding code strings can prime the model to emit them.

- The EOD trained-checkpoint/pass@1 goal is much more ambitious than the validated evidence so far. Sakana proves recall is possible and CE specialization preserves recall, but Rune's own Qwen3.5-9B hypernet has not yet been trained with a recall objective. A pass@1 claim requires Rune engine + Rune base + Rune hypernet, not the Sakana control stack.

- Before spending the day on HPO, define the minimum viable checkpoint precisely: warm-start source, fixed training recipe, max steps, selection metric, and stop criteria. "HPO-optimized by EOD" is likely too broad unless it means a very small sweep over already-known-safe knobs, not a fresh open-ended Optuna campaign.

- Keep the cheap memory-to-edit utility gate running in parallel, but do not let it silently become optional. If the utility gate is negative, training a larger Rune checkpoint may optimize recall without improving action. If the deadline forces proceeding anyway, label the checkpoint run exploratory/product-risky rather than validated.

- Adapter-template changes in `src/rune/` should be treated as part of the experiment contract, not just prompt polish. Log the exact episode serialization used for training and inference; otherwise a pass@1 failure could be a template mismatch rather than a recipe failure.

## 2026-06-01 - Existing Rune Machinery / EOD MVC

- Good update: finding the contrastive/hard-negative machinery already wired makes an EOD minimum viable checkpoint more realistic. Reusing known-good HPO params plus a small guarded sweep is the right interpretation of "HPO-optimized" under the deadline.

- Watch the conceptual mismatch: the existing contrastive objective is on edit-local spans. That can still be useful if the adapter internalizes goal/critique/state and the loss asks whether those memories improve the downstream edit choice. But it should not be described as "embedding diffs" or "diff memory"; it is policy/action supervision conditioned on memory facts.

- The template change is now load-bearing. If training internalizes one episode serialization and engine inference renders another, pass@1 becomes uninterpretable. Add a cheap serialization snapshot/hash to every checkpoint or MLflow run: train template name/version, inference template name/version, and a sampled rendered episode.

- Before pass@1, run the success gate that distinguishes matched vs mismatch on the same checkpoint. A pass@1 change without matched-over-mismatch movement could again be generic edit boosting.

- Baseline first is essential. If the remembered "1.0 post-#50" baseline is a tiny config or contaminated path, the EOD bench should state that plainly and use the same tasks/config/checkpoint-loading path for base, previous best, and new checkpoint.

## 2026-06-01 - Reorient to Fast Pass@1 Loop

- The reorientation is reasonable: the product question is whether corrected Rune training moves pass@1, so Sakana utility should not become a long diagnostic detour. Treat pass@1 as the fast outer loop and matched-vs-mismatch as the explanatory inner probe.

- The fast subset must be frozen before iteration. If the adapter-sensitive subset is chosen or revised after seeing results, pass@1 becomes a tuning target rather than evidence. Keep a tiny iteration subset, a separate holdout mini-bench, and the final full/standard bench distinct.

- Scaling and prompt-architecture sweeps are valid eval-time levers, but they can mask training quality. Report them as a grid over the same checkpoint, and keep at least one fixed canonical setting across checkpoints so progress is not just a different decode/prompt/scaling choice.

- If pass@1 improves while matched-vs-mismatch stays flat, call it a generic utility win, not evidence for episodic memory. That may still be product-useful, but it is a different claim from issue #52's adapter-as-memory bet.

- Conversely, if matched-vs-mismatch improves but pass@1 does not, do not discard it immediately: it may mean memory is present but the engine/prompt/policy cannot exploit it yet. That case should feed prompt/action design, not another blind training run.

## 2026-06-01 - Base/Warm-Start Decision

- I agree with Option B flavor-2 as the best research lane: switch Rune's native engine to the Sakana-compatible `Qwen3-4B-Instruct-2507` base and warm-start a new Rune hypernet from `qwen_4b_d2l`. This directly attacks the heavy recall-install problem by starting from a checkpoint already proven to recall Rune-style facts.

- Keep the wording precise: this is a move to a **recall-compatible Qwen base**, not yet a move to the best available coding model. The trade is lower coding ceiling for a clean, fast test of recall→utility inside Rune. If this lane works, the next product lane is training the same Sakana-style recall recipe on a stronger coding base.

- Two cheap gates should precede any long training: (1) `qwen_4b_d2l` loads cleanly through Rune's `HyperLoRA` path with the checkpoint's own layer indices and target shapes; (2) the warm-started model still shows positive matched-vs-mismatch recall under Rune's activation extraction/generation path, not only under Sakana's repo.

- Pass@1 comparisons must be within-base first. Compare Qwen3-4B zero/no-adapter vs Qwen3-4B + warm-start/adapted hypernet on the same tasks; do not frame lower absolute pass@1 versus Qwen3.5-9B as failure of the memory approach.

- Option A (Sakana-4B oracle teaching DeltaCoder/Qwen3.5-9B) should remain a later product path, not today's critical path. Cross-base KD adds tokenizer/hidden-state/adapter-shape translation risk and could burn days before answering the simpler question: can a recall-capable hypernet improve Rune's edit utility at all?

## 2026-06-01 - Qwen3-4B Coding Adequacy and Compatible Bases

- Qwen3-4B-Instruct-2507 is coding-capable enough for the research lane, though not the best coding base. Its model card reports coding scores including LiveCodeBench v6 35.1, MultiPL-E 76.8, and Aider-Polyglot 12.9, alongside explicit improvements in coding/tool-use. That is sufficient to test whether a recall-capable adapter improves edit utility, but it sets a lower absolute pass@1 ceiling than Qwen3.5-9B/DeltaCoder or a dedicated coder model.

- Among released Sakana-compatible checkpoints, Qwen3-4B is still the best fit for Rune's next research step. Gemma-2B is smaller and weaker for coding; Mistral-7B has a released D2L path but is not clearly better for the Rune/Qwen integration and likely adds more engine drift. Qwen3-4B has the key advantage: `qwen_4b_d2l` already showed positive Rune-fact recall and is same-family with Rune's current Qwen stack.

- Qwen3-4B is trainable. Public infrastructure and adapters show LoRA/QLoRA fine-tuning works for `Qwen/Qwen3-4B-Instruct-2507`, and Sakana's own `qwen_4b_d2l` proves a D2L hypernetwork can be trained for this base. So the blocker is not trainability; it is whether the lower coding ceiling is acceptable for a research proof.

- There is no obvious already-compatible "best coder + Sakana recall hypernet" checkpoint. Stronger coding bases such as Qwen2.5-Coder-7B-Instruct have much stronger public code scores (HumanEval/MBPP around the high 80s/low 80s in the Qwen2.5-Coder report), but no released Sakana D2L warm-start. Using them means paying the recall-hypernet training cost first.

- Recommendation: proceed with Qwen3-4B + `qwen_4b_d2l` as the fastest research lane, but define success within-base: Qwen3-4B no-adapter/base vs Qwen3-4B + recall hypernet. Do not compare absolute pass@1 against Qwen3.5-9B or DeltaCoder as the primary verdict. If the within-base memory utility signal is positive, then invest in the product lane: train a Sakana-style recall hypernet on a stronger coding base, likely Qwen2.5-Coder-7B or a newer Qwen coder model.

- Practical gate before commitment: run the base-only Qwen3-4B pass@1 smoke on the frozen adapter-sensitive subset. If base-only Qwen3-4B cannot solve any tasks or cannot follow the engine prompt, the research lane needs a simpler utility benchmark or a stronger compatible base. If it has nonzero pass@1, it is adequate for measuring adapter lift.

## 2026-06-01 - Cloud Training vs Local Runtime

- Clarification: training hardware and deployment hardware should be decoupled. The product constraint is cheap local Rune inference, not necessarily cheap local training. It is acceptable to pay a one-time cloud training cost for a recall hypernet if the resulting runtime is a quantized local coding base plus small generated LoRA adapters.

- This strengthens the two-lane plan. The Qwen3-4B + `qwen_4b_d2l` lane remains the fastest research proof because it already has recall. The product lane should not be limited by this L4; it should target the strongest coder that can run locally at inference, likely a 7B-class Qwen coder first, then train a Sakana-style recall hypernet for that base on larger cloud GPUs.

- For product-base selection, optimize for local inference pass@1/latency after quantization, not for local trainability. A 7B coder that runs acceptably in Q4/Q5 locally is a better product target than a 4B general instruct model, even if its D2L training requires cloud.

- Do not merge a coding LoRA into the Sakana-compatible 4B base as a shortcut unless we explicitly revalidate recall after the merge. Merging changes the hidden activations the perceiver consumes and may put the recall hypernet out of distribution. It is cleaner either to use the unmerged 4B research lane or deliberately train the recall hypernet on the final coding base.

- Proposed decision rule: prove within-base recall→utility on Qwen3-4B now; if positive, budget a cloud pilot for a 7B coder recall hypernet (2k-5k steps to measure step time, retention, and early m-mismatch) before committing to a full 20k-80k Sakana-scale run.

## 2026-06-01 - Phase 0 Gate Framing

- The Phase 0 ordering is right: environment/load gate before base pass@1 before recall. A flash-attn or checkpoint-load failure is a stack compatibility blocker, not a scientific result about Qwen3-4B or adapter memory.

- Be precise about "base-only" in Rune. If `ModelWrapper` requires a checkpoint and the workaround is `adapter_scaling=0`, log that as **adapter-disabled baseline via scaling=0**, not no-hypernet/no-adapter. Confirm that generated adapter weights are not applied or have zero effect on logits at that setting.

- The base-only pass@1 smoke should be interpreted narrowly: it tests whether Qwen3-4B can follow the Rune engine/prompt on the frozen MBPP subset. If it fails because of wrapper, flash-attn, prompt template, or checkpoint plumbing, do not conclude the base cannot code.

- For the recall gate, require the measurement to go through Rune's exact activation extraction path and checkpoint loader with `qwen_4b_d2l` layer indices. A positive result in the Sakana repo is already known; the new evidence needed is native-Rune-stack recall.

## 2026-06-01 - Qwen4B Bias/Rank Plumbing Finding

- This is correctly classified as plumbing, not a coding/pass@1 result. A rank mismatch from `use_bias=True` means Gate 0 did not test Qwen3-4B's coding ability.

- The train/infer inconsistency is the important discovery: engine PEFT export tries to combine head bias into an expanded rank, while the functional training/diagnostic path currently ignores that bias and uses raw rank-8 weights. Until this is made coherent, pass@1 and training results are not interpretable.

- Run the no-bias/native-functional recall gate first, but treat it as a diagnostic of whether the bias is necessary for recall. If recall survives strongly without bias at the correctly calibrated scaling, disabling bias everywhere is the simplest coherent path. If recall collapses, preserve warm-start fidelity and thread `combine_lora`/head-bias semantics through training, diagnostics, and engine export rather than silently dropping part of the checkpoint.

- Scaling must be stated in the same units for each path. Sakana/PEFT effective scaling includes alpha/r; Rune functional scaling is raw. Any recall or pass@1 comparison must log which convention was used, or a false negative/positive is very plausible.

- Before resuming pass@1, add a tiny parity check: for one generated adapter, functional logits and engine/PEFT logits should agree under the chosen bias/scaling convention. Otherwise the bench could still be measuring an export bug instead of adapter utility.

## 2026-06-01 - Native Rune Recall Gate Negative

- This result blocks Option B as currently wired. The qwen_4b_d2l checkpoint is recall-capable in Sakana's stack, but through Rune's current functional path it produces only noise-level margins. Do not start training or pass@1 until the path mismatch is explained.

- The most likely root cause is not "Qwen4B cannot recall" but implementation/representation mismatch: dropped head bias, scaling convention, feature extraction mismatch, or different episode construction. The disambiguation should compare Sakana internalize and Rune functional on the same exact episode/fact pair with one variable changed at a time.

- First priority: reproduce Sakana-style recall inside Rune with the closest possible semantics: same rendered context, same tokenizer/input IDs, `combine_lora`/head-bias included, and effective alpha/r scaling matched. If that restores +1-ish margins, then the fix is to align Rune train/diag/engine with Sakana semantics.

- If adding head-bias/combine semantics still fails, inspect the feature path. Sakana may be feeding a ctx_encoder/perceiver representation that is not equivalent to Rune's `extract_activations_with_model` hidden states. In that case, warm-starting the perceiver is not valid until Rune uses the same feature interface or retrains the perceiver for Rune's interface.

- Treat this as the new Phase 0 blocker. A base-only pass@1 smoke can still be useful for coding adequacy, but it should not unblock the memory training lane unless native-Rune recall is restored.

## 2026-06-01 - Feature Interface Diagnosis

- The current narrowing is sound: perceiver weights loaded, bias/scaling cannot plausibly explain matched-vs-mismatch collapse by themselves, and the adapter is near-inert vs zero. Feature-interface mismatch is now the leading hypothesis.

- Run the same-episode path A/B before making code changes: Sakana internalize vs Rune functional on identical episodes/facts. This cleanly separates "episode construction changed" from "Rune path changed." If Sakana remains strong and Rune stays near-zero, the path is guilty.

- After that, align one feature-interface variable at a time. Start with the easiest faithful reproduction of Sakana `PerLayerActivations`: hidden-state layer selection/drop-last behavior, dtype/quantization, and masking. Do not combine this with bias/export fixes in the same test, or attribution will be muddy.

- The context-conditioning tensor check is useful as a quick sanity check, but it is weaker than path A/B. Different generated weights do not prove useful recall, and similar weights only confirm inertness. The decisive evidence is whether matched-vs-mismatch recall returns when Rune uses Sakana-equivalent features.

- Keep pass@1 paused for memory claims until this is resolved. If base-only pass@1 is run meanwhile, label it only as a Qwen3-4B coding/prompt smoke.

## 2026-06-01 - Clean Path A/B Correction

- Good correction: the earlier gate2 comparison was not a clean path test because query format differed. Doc2LoRA was trained/evaluated on QA-style prompts; raw teacher-forcing after a markdown header can understate recall even if the adapter contains the fact.

- The restored same-episode/same-query/same-scoring A/B is the right experiment. Do not conclude feature-interface mismatch until the Rune functional twin completes under the same QA episodes and queries.

- If Rune functional is strong under QA while `diag_recoverability` stays weak, the immediate fix is the probe/scorecard prompt format, not feature extraction. The recoverability harness should then use query formats that match the trained recall behavior, or at least report both "neutral header" and "explicit QA" probes.

- If Rune functional remains near-zero under the same QA setup, the path diagnosis returns: feature extraction/application semantics are the blocker.

## 2026-06-01 - Path A/B Result: Application Semantics

- This A/B is decisive: same episodes, same QA queries, same scoring, Sakana strong and Rune functional near-zero. The feature-extraction hypothesis is now weaker; the leading issue is adapter application/assembly in Rune's functional path.

- The key problem is semantic fidelity to Sakana's generated adapter format. For `use_bias=True`, raw `generate_weights` output is not necessarily the final adapter Rune should apply; `combine_lora`, head-bias handling, B orientation, alpha/r scaling, and rank expansion are part of the checkpoint's contract.

- Do not train using the current functional path. If training applies a mis-assembled rank-8 adapter, the Sakana warm-start recall is invisible and the run is effectively trying to relearn recall under Rune's private convention.

- Next gate should be an application-parity gate: one generated adapter, one input, compare logits from (a) Sakana/ctx_to_lora application, (b) Rune functional with combine/head-bias/orientation fixed, and (c) Rune engine/PEFT export after rank fix. Only after these agree should pass@1 or distillation resume.

- Prefer adopting `ctx_to_lora`'s application semantics wholesale if possible, rather than independently re-deriving transpose/rank/bias/scaling details. This is a correctness boundary, not an abstraction preference.

## 2026-06-01 - Reassessing Cross-Path Conclusions

- I agree with the narrowed doubt: the cross-path magnitude comparison is invalid until Rune's application path is made faithful. Sakana-stack +7 or +1.6 and Rune-functional +0.0005 were not measured under equivalent adapter application semantics.

- This does not erase Deliverable 1. Still supported: Sakana can encode/query Rune facts; the score function can detect recall; Qwen-family bases can bind facts; diff-as-memory collapses under feedback-swap in the Sakana-faithful stack. What becomes provisional is any claim about the *magnitude* of Rune-stack recall or the exact ratio between Sakana and Rune margins.

- For #49, the correct revalidation is checkpoint-format-specific. If #49 checkpoints are truly `use_bias=False`, Rune's old functional path may have been faithful for them; if so, the negative still stands. But this must be demonstrated with a parity check, not assumed from architecture history.

- Adopt the `ctx_to_lora` adapter contract as the implementation boundary for all future work: generation, training application, diagnostics, and engine export. Then rerun a small calibration set through the fixed Rune path before using previous margin scales as evidence.

- Until that parity gate passes, pass@1 can only be used for base/prompt smoke, not for adapter-memory claims. Training on the current path would risk optimizing an artifact of Rune's misapplication convention.

## 2026-06-01 - Combined Adapter Gate

- The combined+bias gate rules out the simple "Rune forgot combine_lora/head bias" explanation for the recall collapse. Since m-mismatch stays near zero, the next suspect should move upstream to context encoding / feature generation, especially tokenization and ctx affixes.

- The strongest next test is to perturb only the context encoder in the working Sakana stack: use Rune-style plain tokenization/features while keeping Sakana application/scoring fixed. If recall collapses there, tokenization/feature-interface is causal. Comparing generated A/B tensors from Sakana-features vs Rune-features is a good companion diagnostic.

- Important correction: #49 being `use_bias=True` means my earlier "likely use_bias=False" caveat was wrong. However, #49 was trained and measured under Rune's own convention, so the cross-convention failure of `qwen_4b_d2l` does not automatically rescue #49. It does mean the absolute #49 verdict should remain provisional until the application/feature contract is audited.

- Keep the distinction between self-consistency and faithfulness. A model can train under Rune's self-consistent but non-Sakana feature/application convention and still fail to learn recall because the convention makes the recall problem harder or weaker. That is infrastructure-plus-recipe, not pure objective.

- Do not resume training until the ctx encoding test identifies whether Rune should adopt Sakana `tokenize_ctx_text`/`CTX_AFFIXES` and PerLayerActivations semantics wholesale for warm-start compatibility.

## 2026-06-01 - Feature Isolation Result

- The feature-isolation result overturns the ctx-tokenization/feature-interface hypothesis as the primary cause. If Sakana features and Rune-style features generate nearly the same A/B tensors, then the perceiver warm-start is not being invalidated by Rune's context features.

- The remaining problem is downstream of generated A/B: application magnitude, scaling convention, LoRA placement during scoring, or a scoring/input mismatch. A scaling sweep is reasonable, but if it does not recover m-zero and m-mismatch sharply, stop treating scaling as the answer.

- The next decisive test should be logit-level application parity on the same generated A/B and same query tokens: apply with Sakana's own path versus Rune functional path, then compare logits before scoring spans. This avoids inferring application correctness from A/B tensor similarity alone.

- If Sakana-apply logits move strongly and Rune-functional logits do not for the same A/B, the bug is in application. If logits match but scores differ, the bug is in scoring/query construction. If both logits and scores match but recall is low, revisit the assumption that the same A/B was actually used in the earlier Sakana recall run.

- The #49 conclusion remains provisional in magnitude, but this result reduces the chance that a simple ctx-affix/tokenization fix will rescue it. The infrastructure issue is now narrower: faithful LoRA application/scaling/scoring, not the perceiver feature interface.

## 2026-06-01 - Scaling Sweep Reopens #49

- The scaling sweep is decisive enough to change priorities: a known-recall checkpoint is nearly invisible at alpha/r but becomes increasingly recoverable as raw functional scaling rises. That makes low-scaling measurement a serious candidate explanation for earlier near-zero Rune margins.

- The proposed #49 scaling sweep is now the right next experiment. It directly answers whether #49 lacked recall or whether the diagnostic/application scale hid recall. Run it before making any further recipe claims.

- Keep two questions separate: (1) empirical existence of recall under high scaling, and (2) the principled scaling convention that should be used in training/engine. A high-scale sweep can reveal hidden recall, but it is not by itself the production scaling fix.

- If #49 recall rises with scaling, the #49 "recipe failure" conclusion must be rewritten substantially: the failure may have been application/measurement scale, possibly plus generation instability at the previously tried production scales. If #49 stays flat while qwen_4b_d2l rises, then the recipe-failure conclusion survives.

- Before resuming pass@1, still require a principled application/scale contract and a generation-stability check. High scaling may recover logprob memory while breaking structured generation, so recall and usable pass@1 remain separate gates.

## 2026-06-01 - #49 Scaling Sweep Resolution

- This result resolves the major doubt. Scaling can hide a real recall adapter in Rune's functional path, as shown by qwen_4b_d2l, but it does not rescue #49: #49 remains flat on m-mismatch and becomes strongly anti-QA on m-zero as scale rises.

- The #49 recipe-failure conclusion survives, with a sharper mechanism: the full-revision/edit-emission objective produced an adapter that pushes toward code/edit emission and away from queryable episode recall. It was not merely an under-scaled memory adapter.

- The cross-path calibration caveat still matters for magnitudes, but not for the sign of the #49 verdict. We can now say: qwen_4b_d2l contains recall that Rune under-applies at low scale; #49 does not contain comparable queryable recall even when over-applied.

- Option B remains viable only if two gates pass: a principled scaling/application contract that exposes qwen_4b_d2l recall, and a generation-stability/pass@1 gate showing that the scale needed for recall does not destroy structured coding behavior.

- This is a good point to stop re-litigating #49 and move back to engineering the corrected application/scale path for the recall-capable warm start.

## 2026-06-01 - Sakana Apply Contract

- The scaling finding is now principled, not just empirical: Sakana applies LoRA with `scaling = lora_alpha`, while Rune diagnostics used `alpha/r`. For qwen_4b_d2l this is an 8x factor, exactly the kind of missing convention that can make a recall adapter look inert.

- Replicating Sakana's apply path wholesale is the right DRY fix if the bf16/combine/alpha grid confirms it. This should replace divergent training, diagnostic, and engine application logic rather than adding another parallel implementation.

- Keep precision in the attribution. Scaling likely explains much of the gap, but the scratchpad still notes bf16 vs 4-bit and combine_lora/bias as grid variables. Wait for the grid before saying scaling alone is the root cause.

- The generation-stability gate is not optional. The corrected Sakana scale may recover recall but could reintroduce the structured-generation breakage seen in earlier Rune runs. The final contract must satisfy both: queryable recall and stable engine generation/pass@1.

- Once the apply contract is fixed, rerun three small anchors before training: qwen_4b_d2l recall, #49 recall/anti-QA, and functional-vs-engine parity. That gives a clean new baseline for all later recipe claims.

## 2026-06-01 - Replication Grid Sign-Off

- The replication grid validates the fix direction. Native Rune recall moves from noise to real episode-specific recall when using Sakana's scaling convention plus combined adapter assembly. This is enough to proceed with implementing the Sakana apply contract in Rune.

- Attribution is now strong but not singular: `scaling=lora_alpha` is dominant, `combine_lora` matters at high scale, and 4-bit precision is not the bottleneck. The remaining gap to Sakana-faithful recall is plausibly feature/tokenization fidelity, but it is no longer blocking the application fix.

- Implement the fix as one shared application path, not local patches in each caller. Training, diagnostics, and engine export should all use the same assembled adapter semantics: `combine_lora`, head bias when present, correct B orientation, and `lora_alpha` scaling.

- Keep the post-fix gates in this order: qwen_4b_d2l recall anchor through fixed Rune path, #49 anti-QA/flat-specificity anchor through fixed Rune path, functional-vs-engine parity, then generation-stability/pass@1. Do not skip straight to training.

- If generation stability fails at alpha scale, treat that as a new policy/decoding integration problem, not as evidence against recall. The corrected path has now shown that recall can be exposed; the next question is whether the engine can use it without destabilizing output.

## 2026-06-01 - Adapter Contract Implementation Plan

- The workflow shape is right: map first, implement one shared adapter contract, then run CPU tests before any GPU anchors. Keeping GPU anchors manual is appropriate for this core-path change.

- Be very careful with PEFT scaling when expanding rank for bias. If the desired effective scale is `lora_alpha`, and PEFT applies `lora_alpha_peft / r_peft`, then setting `lora_alpha_peft = lora_alpha * r_peft` is correct only if PEFT has no additional scaling already baked into the assembled tensors. Add a toy numerical test that compares PEFT-export logits to the functional contract, not just shape/rank checks.

- The adapter contract should expose explicit names for the two scales: checkpoint `lora_alpha` and any user/runtime multiplier. Avoid reusing `adapter_scaling` ambiguously for both; past bugs came from alpha vs alpha/r vs runtime scaling being conflated.

- Keep `diag --scaling` as an override for research sweeps, but default diagnostics should use the checkpoint contract. A default that silently reverts to `0.5` or `alpha/r` would recreate the measurement bug.

## 2026-06-01 - Adapter Contract Diff Review

- Sign-off on the CPU-side direction: one shared adapter contract, checkpoint `lora_alpha` as the default effective scale, `combine_lora`/head-bias included, and distinct scale names are the right fixes. The unit/mypy/ruff status is encouraging.

- Do not treat CPU-green as complete. The missing real-model functional-vs-engine logit parity harness is a blocker before pass@1 or training claims. Arithmetic tests are necessary but not sufficient for PEFT export correctness on the actual Qwen/Sakana checkpoint.

- The biggest remaining risk is now explicit: recall wants the Sakana alpha-scale, while prior Rune structured generation reportedly broke at that scale. Generation-stability/MBPP smoke is not a formality; it is the main product gate.

- The corrected path also changes training dynamics by routing bias ranks through autograd. When training resumes, watch scaler_B/head-bias norms and collapse metrics from the first few steps; prior safe hyperparameters may not transfer exactly.

- GPU anchor order looks right: qwen recall anchor, #49 anti-QA anchor, functional-vs-engine parity, then generation-stability/pass@1. If parity fails, stop before generation benchmarks.

## 2026-06-01 - Functional-vs-Engine Parity Pass

- Anchor #3 closes the main application-export correctness blocker. Real-model PEFT hotswap logits match the functional contract closely enough for bf16, with last-token argmax agreement, so engine pass@1 will now measure the corrected adapter contract rather than a PEFT export bug.

- This unlocks the generation-stability/MBPP smoke, but it does not guarantee it will pass. The remaining open question is whether alpha-scale recall can coexist with structured generation and useful code output in Rune's engine loop.

- If generation breaks at the corrected scale, preserve this parity result: the adapter application is correct, so the next work should focus on prompt/decoding/policy integration or runtime multiplier scheduling, not on re-questioning the adapter contract.

## 2026-06-01 - Anchor #4 Baseline Plan

- The Anchor #4 split is right: first reproduce the Sakana free-generation baseline at the same effective scale, then test Rune's xgrammar/MBPP structured path. This separates "adapter can generate coherently under Sakana policy" from "Rune engine can exploit the same adapter contract."

- Keep the comparison honest: Sakana free-form recall success is not a pass@1 result and not equivalent to Rune structured generation. It is a sanity baseline for scale/application, not the product gate.

- When Rune structured generation is tested, log prompt/template, adapter scaling/runtime multiplier, xgrammar settings, and whether the adapter output is used in task-only vs structural prompt mode. If it fails, those are the levers to inspect before touching the adapter contract again.

- The hardened parity harness changes are good. Retain the adapter-determinism assertion and mean drift backstop in committed tests or reproducibility scripts; they guard exactly the kind of silent mismatch that caused this detour.

## 2026-06-01 - Anchor #4 Sakana Baseline Pass

- The Sakana free-generation baseline establishes that `lora_alpha` scale is not inherently degenerative for this checkpoint: recall is strong and generation is coherent when using the Sakana policy.

- Combined with functional-vs-engine parity, this narrows any future Rune structured-generation failure to the Rune engine side: xgrammar constraints, prompt/template policy, runtime multiplier scheduling, or task framing. It should not be read as "alpha scale is invalid" or "the adapter contract is wrong."

- The next product gate is still Rune MBPP/structured generation, not Sakana free-form recall. Report it as such: Sakana proves coherent recall at scale; Rune must prove useful constrained coding behavior at the same or scheduled scale.

## 2026-06-01 - Rune Free-Form Eyeball Pass

- The Rune free-form eyeball is a useful direct control: the corrected engine path can generate fluent multi-token text at `adapter_scaling=1.0`, so alpha-scale degeneration is not a generic PEFT/runtime problem.

- Do not over-read it as memory utility. The reported generations are coherent but factually wrong on some internalized facts, which means the remaining ctx-feature/recall gap can still matter for pass@1 even if decoding stays stable.

- For the xgrammar/MBPP smoke, separate three outcomes in the log: JSON/grammar closure, executable task correctness, and matched-over-zero or matched-over-mismatch memory lift. A pass@1 delta without a specificity signal is a generic utility result, not evidence for episodic memory.

- Include a base or adapter-disabled baseline on the same frozen Phase 0 tasks and keep any runtime-multiplier sweep clearly labeled. If `adapter_scaling=1.0` fails but lower multipliers work, that is a scheduling/decoding tradeoff, not a reason to reopen the adapter-application contract.

## 2026-06-01 - Xgrammar/MBPP Smoke Strong Positive

- This is a real milestone: the feared alpha-scale structured-generation failure did not reproduce. Zero truncation / JSON-close failures across 10 tasks means the corrected adapter contract is compatible with Rune's xgrammar loop, at least on this Phase 0 slice.

- Treat `7/10` versus adapter-disabled `0/10` as a strong within-base product signal, but not yet a mechanistic proof of episodic memory. The next confirmation should show per-task generated code, exact task IDs, prompt/template, seed/settings, and the adapter-disabled outputs on the same path.

- The remaining specificity question is whether the lift comes from the matched episode memory or from a generic coding/style boost induced by `qwen_4b_d2l`. Add a mismatch-adapter or shuffled-context arm on the same 10 tasks if feasible; if matched beats mismatch, the pass@1 gain supports the issue #52 memory claim rather than only a useful adapter prior.

- Because this is an adapter-sensitive 10-task subset, freeze it now and keep it distinct from any holdout/full-bench result. It is valid for fast iteration, but not enough by itself for a broad pass@1 claim.

## 2026-06-01 - Handoff Review

- The handoff is strong on the root-cause chain and correctly names the scaling/application contract as the committed fix. However, it is now stale: anchors #1, #2, #3, the Sakana free-generation baseline, the Rune free-form eyeball, and the first xgrammar/MBPP smoke have all advanced past the "NEXT: GPU validation" framing. A fresh handoff should promote these from planned gates to observed results.

- The core claim "adapter-application bug, NOT architecture/features" is directionally right but a little too absolute. The application bug was the blocker, but the residual Sakana-vs-Rune recall gap remains attributed to ctx feature fidelity, and the current pass@1 lift still needs a mismatch/shuffled-context arm before it can be mechanistically assigned to matched episodic memory rather than a generic adapter prior.

- The forward plan should change from "if anchors pass, train" to "after confirmation rerun and specificity arm, decide whether training is still needed for Deliverable 2." A `7/10` Phase 0 pass@1 smoke from the warm-started qwen checkpoint may already answer part of the utility question; training should now have a sharper purpose, such as improving specificity/holdout pass@1 or adapting facts/critique serialization, not merely proving the fixed contract works.

- The reproducibility gap is still the handoff's weak point. The decisive GPU harnesses and bench tooling are listed as local-only scratch (`tools/_pathab_rune.py`, `_bench_entry.py`, `_parity_engine_vs_functional.py`, etc.). Before final PR or another agent handoff, either commit the minimal reproducibility scripts/tests or write exact artifact paths, command lines, MLflow run IDs, task IDs, seeds, and generated-code dumps into a durable doc.

- Keep the Phase 0 result scoped. The handoff should explicitly distinguish: (1) fixed application contract validated, (2) qwen_4b_d2l warm-start has real recall through Rune, (3) structured generation is stable on a 10-task adapter-sensitive subset, and (4) broad pass@1 / full benchmark / trained corrected-recipe claims remain open.

## 2026-06-01 - Confirmation Rerun Tempering

- The 3-task code-dump rerun is exactly the right correction to the first `7/10` readout. It confirms the harness is producing real Python and that base `scaling=0.0` failures are genuine degeneration, not a benchmark plumbing artifact.

- It also weakens the memory-causality claim for now: on the inspected subset, adapter and base tie `1/3`, and the adapter misses function-name casing on tasks where clean naming matters. The strongest observed effect is generation discipline / anti-degeneration, which may be a generic prior from the D2L adapter rather than matched episode recall.

- The mismatch-adapter arm is now not optional. Run matched, mismatch/shuffled-context, and zero on the same frozen 10 tasks with per-task code dumps. If matched and mismatch both suppress degeneration similarly, the Phase 0 win is still product-useful but not evidence for issue #52's episodic-memory mechanism.

- For interpretation, report pass/fail alongside failure class: JSON/truncation, syntax, wrong function name/API contract, and semantic wrong answer. The current misses are name-contract errors, which suggests the next lever may be prompt/schema enforcement as much as memory training.

## 2026-06-01 - Mismatch Arm Plan Review

- The deranged task-to-adapter mapping is the right specificity test because it preserves an in-distribution adapter context while breaking the task/memory binding. Make the derangement deterministic and log the exact task-ID mapping so a surprising result is auditable.

- The monkeypatch boundary is load-bearing. Verify with dumped rendered trajectories that only the adapter-conditioning trajectory receives the mismatched task, while the visible generation prompt, tests, function signature expectations, and benchmark task remain the original target task.

- Keep matched, mismatch, and zero identical on decode settings, retry budget, seed, timeout, and failure classification. If mismatch runs slower or exhausts retries differently, separate "adapter helps search/discipline" from "adapter remembers the right episode."

- If mismatch is close to matched, do not collapse the whole result to failure. It would still show a useful anti-degeneration adapter prior; it would just move the episodic-memory claim back to the logprob recall gates or require harder task-specific utility probes.

## 2026-06-01 - Probe-First Specificity Plan

- Probe-first is the right use of GPU time. A cheap matched/mismatch/zero logprob check on reference MBPP solutions can tell whether short task descriptions actually move the generated adapter before launching another long generation run.

- Keep the interpretation narrow because the task remains visible in the generation prompt. A positive matched-over-mismatch reference-solution margin supports task-specific adapter utility, but a flat margin does not refute episodic memory for hidden feedback/tried/critique facts.

- Weight-space distance is a sanity check, not an outcome metric. If weights differ but reference logprobs do not, the adapter may encode differences irrelevant to MBPP correctness; if weights barely differ, the deranged generation arm is unlikely to prove specificity.

- The advisor traps are important: no identity fallback, seed keyed to generation task, and apples-to-apples reruns through the same harness. Add a rendered-trajectory diff artifact for at least one matched/mismatch pair so the monkeypatch boundary is auditable.

## 2026-06-01 - Specificity Probe Result

- This is a clean split result. The absent-task regime is strong evidence that `qwen_4b_d2l` encodes short MBPP task descriptions through Rune's path: matched beats deranged on all 10 tasks with a large reference-solution logprob margin.

- The present-task regime answers the Phase 0 pass@1 mechanism more cautiously: when the task is already in the prompt, matched does not add utility over mismatch on this ranking probe, while both adapters beat zero. That supports the "generic anti-degeneration / discipline prior" explanation for the `7/10` smoke more than an episodic-memory explanation.

- Do not let the present-task negative erase the absent-task positive. It says task-memory is redundant for single-turn MBPP prompts, not that adapter memory fails. The next memory-utility test should hide or underspecify the relevant fact: feedback, tried step, critique, prior state, or a task detail not present in the visible prompt.

- I would skip the full 2-hour three-arm generation run unless the team needs a sanity check for functional pass@1. The ranking probe already provides the primary answer; the next higher-value work is a hidden-fact utility probe or a prompt/schema fix for name/API-contract errors.

## 2026-06-01 - Signature/Body Split Probe

- The split is important and should change the language from "task-specific memory" to "mostly signature/name memory" for this MBPP probe. Absent-task signature recall is very strong, while body/algorithm recall is weak, so the adapter is not yet demonstrating deep solution recall.

- This explains both sides of the Phase 0 behavior: the adapter can suppress degeneration and recall surface task identifiers, but at contract scale that surface memory can fight the visible prompt's exact function name/casing.

- For goal 4, do not train or select only on whole-solution logprob if signatures dominate the signal. Track signature and body/algorithm spans separately, and consider downweighting or masking the signature when the question is whether the adapter stores useful solution state.

- The next utility probe should hide a fact that cannot be solved by function-name recall alone, such as a required branch condition, prior failed approach, critique-derived constraint, or state variable invariant. Otherwise the scorecard can pass by learning labels rather than actionable memory.

## 2026-06-01 - Per-Task Body Resolver

- Good correction to the previous wording: body/algorithm recall is not absent. It is real but faint and uneven, with positive margins on discriminative details such as `key=sum` or regex-like structure and flat/negative margins on trivial or generic bodies.

- Keep the stronger conclusion as comparative, not absolute: signature/name recall is much more robust than body recall, but the warm start does contain a weak algorithmic signal worth deepening.

- For any training run, use selection metrics that prevent signature dominance from hiding body progress: report signature, body-discriminative-token, and full-solution spans separately. A checkpoint that improves signatures but leaves body flat has not advanced the memory-to-edit claim.

- The next hidden-fact probe should choose facts with low label leakage and high action relevance. Good candidates are critique-derived constraints, branch conditions, invariants, or rejected-strategy facts that cannot be recovered from the function name alone.

## 2026-06-01 - Signature Enforcement Plan

- Prompt-level signature enforcement via augmented task descriptions is the right cheap first lever, but interpret it as an end-to-end task-format intervention: the augmented text flows into both adapter conditioning and the visible generation prompt.

- If the augmented run improves pass@1, it proves explicit signature instructions can overcome or retrain the name-casing failure mode in this setup; it does not isolate whether the fix came from stronger prompt pressure, changed adapter memory, or both.

- Keep the comparison apples-to-apples with the same frozen 10 tasks, same scaling, same harness, and per-task failure classes. Pay special attention to whether only the previous name/API failures improve or whether unrelated tasks regress from the longer/more directive task text.

- If prompt-level enforcement is flat, do not conclude schema enforcement cannot help. It may mean soft prompt instructions are still weaker than the adapter's recalled signature; a harder output contract that validates or patches the exact function name remains a separate lever.

## 2026-06-01 - Hidden-Fact Probe Orientation

- The feedback-swap hard negative is the right next control because it preserves local code while changing the hidden critique. That directly tests whether the adapter binds feedback facts rather than echoing code.

- For the avoid arm, keep the rejected code and critique out of the visible prompt if the claim is memory. If the prompt shows the rejected attempt, the model can solve by in-context comparison rather than adapter recall.

- Score accepted-vs-rejected as a difference-in-differences against zero and feedback-swap, not as raw accepted preference. Accepted code may be intrinsically more likely even without memory.

- Track facets separately: goal/critique recall, feedback-swap diff specificity, and one-step avoid are related but not interchangeable. A win on critique recall does not automatically prove action utility unless it shifts accepted-vs-rejected or edit choice.

## 2026-06-01 - Signature Enforcement Validity Gate

- The `_is_simple_task` check is a good validity gate: if augmentation does not change the engine path, the comparison is prompt/task-format confounded but not path confounded.

- The missing full per-task baseline code dump matters. Read the signature-enforcement run primarily as a transition test for the known name failures (`mbpp/12`, `mbpp/14`) plus overall well-formedness, not as a clean aggregate `7/10` baseline comparison.

- If aggregate pass@1 changes, report it as suggestive only unless the changed tasks can be tied to known failure classes. The useful claim is narrower: whether explicit signature text fixes the observed name/API-contract failure mode.

- If `mbpp/12` and `mbpp/14` still fail by name/casing, escalate to hard schema or post-generation signature normalization rather than more soft prompt wording.

## 2026-06-01 - Signature Enforcement Result

- Lever A is validated for the narrow failure mode: the two known name/casing failures flipped to pass under explicit signature text, and no visible regression appeared on the inspected frozen slice.

- Keep the aggregate `9/10` framed as suggestive, not a clean controlled pass@1 delta, because the original full per-task baseline dump was missing. The defensible claim is "signature prompting fixed known name-contract errors."

- The remaining `mbpp/57` failure is a different class: return type / semantic contract, not function name. The next cheap product lever is likely explicit return-type/output contract enforcement, but that should stay separate from the hidden-fact memory probe.

- This reinforces the broader lesson: structured prompt/schema constraints can convert the generic anti-degeneration prior into more useful code, but they do not by themselves advance the episodic-memory mechanism claim.

## 2026-06-01 - Avoid Ceiling Gate

- This is the right failure to catch early. If critique in the visible prompt does not improve accepted-over-rejected preference, then an adapter-hidden critique result would be uninterpretable; the constructed avoid task lacks an in-context upper bound.

- Do not build the matched/feedback-swap adapter apparatus on this avoid formulation until the task is repaired. The next step should be either a cleaner directive subset with verified hunk alignment or mined engine trajectories where the critique/failure fact actually determines the next action.

- The bimodality is useful: keep the strong-positive examples as candidates for a curated pilot, but do not average them with non-directive or wrong-hunk rows and call that a corpus-level verdict.

- Keep critique recall and critique utility separate. Existing goal/critique recall can be positive while accepted-vs-rejected action utility is invalid on this corpus.

## 2026-06-01 - Single-Hunk Avoid Ceiling

- The single-hunk rerun corrects the earlier strongest conclusion: the avoid task is not simply dead. Removing wrong-hunk contamination moves the in-context critique effect in the right direction.

- The upper bound is still weak and noisy: mean DiD is positive but frac positive is near coin flip, and high-base-preference rows have little headroom. That makes a full adapter-hidden avoid run likely underpowered unless the subset is improved.

- Avoid stacking filters after looking at outcomes. A single structural filter chosen blind to DiD is defensible; adding more filters now risks tuning the evaluation set to the desired signal.

- Best next scientific path remains purpose-built mined trajectories or a predeclared directive/low-base-pref pilot. If using the current corpus anyway, report it as exploratory and expect wide uncertainty.

## 2026-06-01 - Consensus Literature Framing

- Correction to keep front-and-center: Doc2LoRA's QA/NIAH setup is already a legitimate **memory substrate** demonstration in the retrieval sense. D2L generates a LoRA adapter from an unseen context, then answers later queries without re-consuming that context, with near-perfect needle retrieval beyond the target model's native window [1]. That is not merely "discipline"; it is parametric retrieval from an adapter.

- Hypernetwork literature supports this framing. Hypernets are explicitly networks that generate target-network weights and are used for adaptability, continual learning, transfer, and compression [2]. Task-conditioned hypernetworks in continual learning are described as preserving task-specific weight realizations in memory, with long memory lifetimes in a compressive regime [3]. Fast-weight work likewise treats generated weights as an on-the-fly memory/program that supports associative retrieval [4].

- LoRA-as-memory is also an active framing outside Doc2LoRA: recent work explicitly studies LoRA as modular knowledge memory complementary to RAG/ICL [5], and memory-adapter systems such as MemLoRA distill memory operations into adapters for local/on-device memory workflows [6].

- Therefore, the Rune specificity probe should be worded carefully: **PRESENT-regime MBPP is negative for additive action utility when the task is already visible**, not negative for "adapter as memory." The ABSENT-regime result is the one aligned with Doc2LoRA's QA standard, and it is positive: matched adapters retrieve hidden task information. The remaining weakness is depth/actionability of what is retrieved, not the existence of adapter memory.

- The next scientific standard should be two-stage: first, a hidden-fact QA/logprob retrieval test for critique/tried/state facts, matching the Doc2LoRA memory criterion; second, a harder action-utility test showing those retrieved facts change edit choice or pass@1. Do not require the second before acknowledging the first as memory.

- Practical implication for issue #52 wording: "memory exists but is currently shallow/name-dominated on MBPP; visible-task pass@1 lift is generic discipline; hidden-fact retrieval remains the right adapter-memory test; hidden-fact action utility remains the harder product test."

References:

[1] [Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://consensus.app/papers/details/9cc22e858f1f5e30a9b8063b08c95c86/?utm_source=cursor) (Rujikorn Charakorn et al., 2026, ArXiv)
[2] [A brief review of hypernetworks in deep learning](https://consensus.app/papers/details/d268e79d014d51468a884751c35805fb/?utm_source=cursor) (Vinod Kumar Chauhan et al., 2023, Artificial Intelligence Review)
[3] [Continual learning with hypernetworks](https://consensus.app/papers/details/b643863557a25ad29cb54cdb12faa5e9/?utm_source=cursor) (J. Oswald et al., 2019, ArXiv)
[4] [Gated Fast Weights for On-The-Fly Neural Program Generation](https://consensus.app/papers/details/d444c7639f24579f812336a1856a53ce/?utm_source=cursor) (Imanol Schlag et al., 2017)
[5] [Understanding LoRA as Knowledge Memory: An Empirical Analysis](https://consensus.app/papers/details/eca73c0df6285281a2f5e473b405e562/?utm_source=cursor) (Seung-Heon Back et al., 2026, ArXiv)
[6] [MemLoRA: Distilling Expert Adapters for On-Device Memory Systems](https://consensus.app/papers/details/217d4b1eed4e572b95768e1a46d22d1a/?utm_source=cursor) (Massimo Bini et al., 2025, ArXiv)

## 2026-06-01 - Consensus Guidance on Borderline Avoid Result

- The borderline avoid result fits known code-review automation failure modes. Tufano et al. emphasize that aggregate success rates are not meaningful without characterizing what the reviewer asked for, and their qualitative review found dataset issues in code-review automation benchmarks [7]. This maps directly onto our bimodality: some critique/edit pairs are directive and hunk-aligned; others are non-directive or target a different edit.

- Feedback helps code repair only when it is useful and correctly grounded. Coffee frames open-source code LLM feedback as risky because models can follow superficial feedback formats while receiving misleading guidance [8]. RL4F similarly optimizes critiques against downstream repair performance rather than assuming natural-language feedback is automatically useful [9]. For Rune, raw GitHub review text should not be assumed to define the action target.

- Preference/evaluation protocol matters. Bansal et al. show that ratings and rankings can disagree substantially and that evaluation outcomes depend on the feedback protocol [10]. Our DiD accepted-vs-rejected ranking is therefore the right family of metric, but it must be stratified by headroom and feedback type rather than averaged over heterogeneous rows.

- Recommended repair to the avoid probe:
  1. Keep the in-context ceiling gate mandatory.
  2. Predeclare structural filters before adapter runs: exactly one replace hunk, critique text references the edited symbol/condition/API, and visible scaffold excludes rejected code and critique.
  3. Stratify by no-critique base preference: low/neutral-base-pref rows are the informative subset; high-base-pref rows have ceiling effects and should be reported separately, not mixed into the primary mean.
  4. Add a "normalized critique" arm: rewrite raw review feedback into an explicit actionable constraint, then compare raw vs normalized. If normalized works and raw does not, the bottleneck is review-text actionability, not adapter memory.
  5. Treat mined engine trajectories as the clean target distribution: the failure fact and next accepted action can be made causally aligned by construction.

- Practical verdict: do not spend a long adapter-hidden avoid run on the full external_codereview slice. Either run a small, predeclared, directive/low-headroom pilot as exploratory evidence, or move directly to mining purpose-built failure-bearing trajectories.

References:

[7] [Code Review Automation: Strengths and Weaknesses of the State of the Art](https://consensus.app/papers/details/d70e13fc3ee754f885719f813151f94d/?utm_source=cursor) (Rosalia Tufano et al., 2024, IEEE Transactions on Software Engineering)
[8] [Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback](https://consensus.app/papers/details/93f6635325b45758b44140ca22f02a83/?utm_source=cursor) (Seungjun Moon et al., 2023, ArXiv)
[9] [RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs](https://consensus.app/papers/details/2253911279b55f269b457f3dc83fc084/?utm_source=cursor) (Afra Feyza Akyürek et al., 2023)
[10] [Peering Through Preferences: Unraveling Feedback Acquisition for Aligning Large Language Models](https://consensus.app/papers/details/9f7c42268764502abb9a487fe3776214/?utm_source=cursor) (Hritik Bansal et al., 2023, ArXiv)

## 2026-06-01 - Why Body Recall Is Weak

- Weak body recovery is a good justification for purpose-gated fine-tuning, not a contradiction of the memory-substrate result. The warm-started D2L adapter retrieves salient labels/signatures strongly, while body/algorithm recall is faint and uneven. That is exactly the kind of capability gradient a focused Rune fine-tune should try to shift.

- Likely causes: signatures are short, high-salience, low-entropy labels; body spans contain many generic Python tokens that dilute logprob margins; MBPP tasks allow multiple valid implementations, so a single reference body underestimates functional recall; and `qwen_4b_d2l` was trained for document/QA internalization, not code-body/action-state recall.

- The right training target is therefore not "more name recall." Fine-tuning should mask or downweight signatures and reward body-discriminative tokens, branch conditions, invariants, critique-derived constraints, and accepted-vs-rejected action facts under hard negatives that preserve names/local code.

- This also gives a clean success criterion for goal 4: fine-tuning is useful if matched-over-mismatch improves on body/discriminative-token spans and hidden-fact utility while retaining QA recall and generation stability. It is not useful if gains are only signature/name recall or generic anti-degeneration.

- "Better recovery of embedded code" is a valid intermediate metric, but prefer discriminative code facts or functional equivalence over raw full-body string likelihood, because many correct bodies are not text-identical to the reference.

## 2026-06-01 - Mining Failure-Bearing Trajectories

- This is the right main investment after the avoid ceiling results. The needed dataset is not generic review pairs; it is trajectories where a failed attempt, diagnostic signal, and accepted repair are causally aligned in one session.

- Before a large benchmark harvest, run a tiny mining smoke that answers three structural questions: does `sessions_dir` record every retry/repair step, are failed code outputs and stderr/diagnosis recoverable as distinct fields, and are there enough fail-then-pass episodes to justify scaling?

- Predefine episode validity before mining at scale: same subtask/target, failed output available, later passing repair available, diagnostic/critique text non-empty, and accepted repair actually changes the failed region. Otherwise the corpus can recreate the external_codereview ambiguity.

- Track yield as the first metric: tasks run, sessions written, fail-then-pass chains, valid avoid episodes, and ceiling-gate pass rate. If yield is sparse, switch to deliberately harder tasks or instrument synthetic failure injection rather than burning GPU on easy MBPP.

- Keep the mined corpus dual-use: retrieval QA over hidden failure facts first, then action utility. The QA gate should confirm adapter memory before the more expensive accepted-vs-rejected utility gate.

## 2026-06-01 - Mining Structural Checks

- The repair shard co-location finding is good news: the needed avoid triple is already close to the existing trajectory format, so extraction may be cheap once valid sessions exist.

- The sandbox finding is a major scope correction. If mid-loop feedback only captures syntax/import/load-time failures, mined MBPP avoid episodes will mostly teach structural cleanup, not "tried approach failed semantically, repair the algorithm." That is weaker than the #52 tried/critique facet.

- Before harvesting many sessions, instrument or verify semantic feedback in the retry loop. For MBPP, even one task-provided example assertion or held-out unit test during repair would create much more meaningful failure-bearing trajectories.

- If semantic feedback is not available, label the resulting corpus honestly as structural-error repair memory. It can still be useful, but it should not be used as evidence for semantic avoid/failure-history memory.

## 2026-06-01 - Semantic Mining Plan

- The finalized semantic-signal instrumentation is the right response to the structural-check finding. Using the visible example assert as a mining-only oracle should produce more relevant fail-then-repair episodes than syntax/runtime errors alone.

- Keep the scope tight: this is a yield probe, not a benchmark-quality evaluator. A single visible example assert can miss many semantic bugs and can also encourage overfitting to that example, so use it only to harvest candidate trajectories.

- The monkeypatch guardrail matters. Keep it outside product `src/rune` behavior and log that mined trajectories were generated under an augmented semantic-feedback harness.

- The pre-registered sparse branch is sound. If valid semantic fail-then-pass yield is near zero on the frozen 10, do not scale blindly to full MBPP; move to harder tasks, sampled candidate generation, or purpose-built synthetic failure injection with explicit provenance labels.

## 2026-06-01 - Body-Recall Micro-Finetune Canary

- It remains plausible that code-body / algorithm recall becomes much stronger after focused fine-tuning. The current evidence only says the `qwen_4b_d2l` warm start is name/signature-dominant; it does not prove body facts are unlearnable.

- Before an overnight run, add a small canary: build a tiny body-recall set (`16-32` examples) where the task/body fact is hidden from the scoring prompt, signatures are controlled or masked, and the scored spans are body-discriminative facts such as branch conditions, key operations, invariants, return-type behavior, or critique-derived constraints.

- Run a very short warm-start fine-tune from `qwen_4b_d2l` (`50-200` steps, base frozen, hypernet/perceiver only, `contrastive=True`) with hard negatives that preserve signature/local code but alter the body fact. Evaluate every few steps.

- Go/no-go metrics: body/discriminative-token matched-minus-mismatch should improve; signature matched-minus-mismatch should be tracked separately so it cannot hide flat body progress; D2L QA recall and generation stability should not collapse.

- If body recall moves quickly, overnight goal-4 training is justified. If only signatures improve or body remains flat, change the objective/data before spending the night.

- Optional capacity check: fit a tiny oracle per-episode LoRA on a few body facts. If oracle LoRA can recover them but the hypernet canary cannot, the bottleneck is hypernet/data training rather than representational capacity.

## 2026-06-01 - Training Checkpoint Gate

- The shift from `val_diff_agreement` to matched-over-feedback-swap on edit-local tokens is the right correction. `val_diff_agreement` can reward the generic discipline prior that already explained the present-task MBPP lift, while matched-over-swap is the metric that actually tests whether feedback is bound to the episode.

- The baseline being near coin-flip (`matched-swap +0.0185`, frac positive `0.48`) makes the smoke run interpretable: a useful checkpoint should move this metric visibly above baseline, not merely improve matched-over-zero. If the 60-step smoke shows an active contrastive loss but no matched-over-swap movement, stop and inspect data/negative construction before spending GPU on the full 300-step pilot.

- Periodic saves are important because the built-in best checkpoint is still selected on the wrong proxy. Treat `checkpoint_best.pt` as a convenience artifact only unless it is re-ranked post hoc by feedback-swap specificity plus retention.

- Add a hard retention readout before calling the checkpoint HPO-ready: absent-regime task/signature recall, body/discriminative-token recall if available, and a small generation-stability check. A checkpoint that gains feedback-swap specificity by forgetting the Doc2LoRA recall prior or destabilizing xgrammar is not yet a pass@1-HPO candidate.

- Keep the claim scoped even if the pilot succeeds. The current corpus is still external code-review, not mined multi-step Rune trajectories; success would show feedback-binding specialization from a recall-capable warm start, not full issue #52 avoid-history memory.

## 2026-06-01 - Smoke OOM and Sequence Truncation

- The OOM diagnosis is plausible: contrastive training is graph-retaining in a way the no-grad feedback-swap eval is not, and the fp32 full-vocab log-softmax at 2048 tokens is a real memory cliff on this L4. Dropping to `max_seq_length=768` is a reasonable unblocker if it was the training loop's intended regime.

- But the baseline was measured with the 2048-token eval path, while the smoke/full training now sees 768-token contexts. Recompute or at least report feedback-swap eval under the same 768-token truncation before comparing trained checkpoints to the `+0.0185` baseline; otherwise an apparent gain could be a length/truncation distribution shift.

- Log truncation statistics as a gate, not as an afterthought: fraction of rows truncated, fraction with all edit-local tokens skipped, retained edit-local-token count distribution, and whether skipped rows are systematically longer or different in feedback type. If many hard examples disappear at 768, the smoke can look cleaner without learning the intended binding.

- If 768 still OOMs, prefer enabling gradient checkpointing before shortening further. Reducing length below the median episode risks turning the experiment into "learn from short easy rows" rather than testing issue #52's trajectory-memory target.

- Keep MLflow metadata explicit: `max_seq_length`, `PYTORCH_CUDA_ALLOC_CONF`, truncation/skip counts, and any gradient-checkpointing flag should travel with the checkpoint artifact. This matters because pass@1 HPO later needs to know what training distribution produced the adapter.

## 2026-06-01 - Smoke-Train Mechanical Pass

- The 60-step smoke clears the mechanical gate: contrastive hinge moved in the right direction, scaler values stayed stable, bias gradients flowed, and peak GPU memory was well below the L4 ceiling at `max_seq_length=768`.

- Do not upgrade this to a scientific pass yet. The moving in-train hinge mostly proves the optimizer can reduce the objective on sampled training rows; the issue #52 gate is held-out matched-over-feedback-swap specificity. The next decisive number is the 60-step checkpoint eval against a truncation-aligned warm-start baseline.

- If the held-out eval improves, check whether the gain comes from better matched-vs-swap rather than both matched and swap changing together. A uniform edit-local likelihood boost is still the generic discipline confound in another form.

- If the held-out eval is flat despite the hinge moving, treat that as evidence that the hard-negative construction or corpus is too easy/in-sample, not as evidence that feedback binding is impossible. The immediate fix would be validation-negative auditing and row-level examples, not a longer blind run.

- Full 300-step training is justified only after the 60-step checkpoint beats the aligned baseline or there is a predeclared reason to view the 60-step eval as underpowered. Otherwise the run risks optimizing a visible training loss without advancing the recoverability scorecard.

## 2026-06-01 - Warm-Start Scaler_B Clobber

- The held-out smoke eval correctly invalidates the previous mechanical-pass readout. A moving in-train hinge is not meaningful if warm-start `scaler_B` was overwritten from a learned distribution to a uniform `1.0`; the optimizer was then operating in a badly over-scaled adapter regime rather than testing feedback binding.

- The root cause is a strong implementation finding, not a negative scientific result. The uniform `matched-zero` collapse around `-8.8` and the state-dict diff localizing the change to `scaler_B.down_proj` are exactly the kind of evidence needed to distinguish objective failure from checkpoint-loading damage.

- The proposed fix is right in spirit: only reinitialize `scaler_B` when it is actually in the zero/collapsed basin, and preserve learned warm-start values otherwise. Add a narrow regression test or checkpoint-load smoke that asserts a non-collapsed warm-start `scaler_B` survives initialization unchanged; this bug is too easy to reintroduce.

- Rerun the 60-step smoke from a clean warm-start after the fix. Do not reuse the clobbered checkpoint for any trend, artifact, or MLflow comparison except as a documented failed run.

- In the rerun, log the pre-train and post-train `scaler_B` mean/std alongside matched-swap and matched-zero. The success condition should include "warm-start scale preserved within sane bounds," not only "hinge moved."

## 2026-06-02 - Fixed Smoke Positive, Still Modest

- The fixed rerun clears the immediate implementation concern: `scaler_B` is preserved at the warm-start distribution, the broken over-scale collapse is gone, and held-out matched-over-feedback-swap moves from `+0.0185`/`0.48` to `+0.0687`/`0.65`. That is the first real evidence in this lane that the feedback-swap objective can add episode-specific signal on held-out rows.

- Treat the effect size as a smoke, not a destination. A `+0.05` absolute gain over a weak baseline at `n=60` is encouraging but still small enough that row-level variance, truncation, and sample composition can matter. Report confidence or at least bootstrap/error bars before using it as the basis for a long-duration run.

- The matched-zero jump to `+0.5021` is good sanity evidence that the adapter remains helpful, but it also reintroduces the discipline-prior confound. The key plot/table should decompose matched, swap, and zero separately so a reader can see whether specificity improved because matched rose, swap fell, or both moved.

- Before launching a long run, verify the earlier truncation concern under the fixed path: same `max_seq_length=768` baseline, skip/truncation counts, and retained edit-local-token distribution. The fixed `scaler_B` result is only comparable to a baseline measured under the same length regime.

- A 300-step pilot is reasonable if framed as a bounded scaling test with periodic post-hoc selection on matched-swap plus retention, not as "full Sakana-scale" or an HPO-ready checkpoint by default. The go/no-go after 300 should still require retention and generation stability, not just a higher feedback-swap number.

## 2026-06-02 - Pre-Long-Run Gate Reframe

- The deep-dive reframe is correct: the fixed smoke is better read as a weak feedback-specific signal riding on a much larger generic adapter boost, not as a green light for a long run on the unfiltered corpus. `matched-zero +0.50` versus `matched-swap +0.0687` is the same failure mode the earlier work warned about, only smaller and cleaner.

- T0 should be mandatory before any new GPU training: compute paired per-episode delta between warm-start and 60-step fixed checkpoint on the same validation rows. Aggregate deltas without pairing can hide whether the same episodes improved or whether sample noise/composition produced the mean shift.

- T1 is the highest-value discriminator. If the directive/base-uncertain subset turns the weak `+0.0687` signal into a clear matched-over-swap margin, the main blocker is false negatives and low feedback-edit mutual information in `external_codereview`; then filtered or reweighted data is justified. If T1 is flat, a longer run on the same corpus should be stopped.

- Keep T1 selection rules predeclared and independent of the trained-checkpoint outcome. Filtering on rows that already improve after training would leak the result; filtering on no-critique base preference, critique directiveness, or oracle in-context ceiling is defensible if fixed before the run.

- T2 should be treated as a capacity boundary, not an optimization flourish. If an oracle LoRA at rank 8 cannot bind body/feedback facts while higher rank can, increasing rank is a scientific necessity before HPO. If rank-8 oracle succeeds, the next bottleneck is hypernetwork/data/objective rather than LoRA capacity.

- The go/no-go bar should be written in calibrated units before training resumes: target matched-swap effect size, fraction positive, retention threshold, and generation-stability threshold. Without that, "good metrics" will drift toward whatever improved after the fact.

## 2026-06-02 - Pushback on the Gate Reframe

- I concur with the main caution, but not with every inference. Calling the fixed smoke "#49 in miniature" is directionally useful, yet it risks flattening an actual difference: #49 had near-zero specificity under the corrected questions, while the fixed smoke has at least a small held-out matched-over-swap movement. The right conclusion is "insufficient and confounded," not "same failure already proven."

- The `14% feedback-specific / 86% generic` decomposition is rhetorically helpful but mathematically fragile. `matched-zero` and `matched-swap` do not form an additive partition of one lift; swap adapters may carry some true task/code signal and zero may be a poor denominator. Use it as a warning sign, not as a quantitative attribution.

- T0 should not rely only on a paired t-test. Per-episode margins are likely heavy-tailed and bounded by prompt/token quirks, so report paired bootstrap CI, sign test, and row-level scatter. Also verify the 60 rows are exactly identical across baseline and trained eval after any truncation/filtering changes.

- T1 is decisive only if the subset definition is frozen before looking at trained-checkpoint deltas and is applied separately to train and validation without leakage. A "base-uncertain" filter derived from the same scoring target can select rows where the metric has more headroom, which is useful, but it can also manufacture an easier evaluation distribution. Report both filtered and full-val results.

- Be careful with "external_codereview is the wall." If T1 is flat, it may mean the hard-negative construction is wrong, the feedback text is not normalized enough, or the edit-local span metric misses the action signal. It is evidence against this corpus/objective/metric combination, not against the corpus in all forms.

- The long-run bar should not be anchored to NIAH `+7.7` as though edit-feedback binding should approach needle retrieval. A better calibration ladder is: known-good hidden-task specificity (`+1.17`), feedback-swap diff collapse (`+0.174`), fixed smoke (`+0.0687`), and in-context directive ceiling (`+0.52`). Demand meaningful movement on that ladder, but do not make an unrelated retrieval scale the only standard.

- Add one positive-control row to T1/T2: a synthetic or curated feedback case where the swapped critique clearly changes the correct edit and the in-context ceiling is large. If the training/eval harness cannot move on that, the apparatus is broken; if it can, failures on `external_codereview` are more confidently data-quality failures.

## 2026-06-02 - Body and Directionality Gate Controls

- The doc-QA origin hypothesis is a useful correction: rank/chunk capacity, representation, and directionality are distinct axes. Doc2LoRA's own chunking mechanism supports the capacity part of the argument because it composes per-chunk adapters along the rank dimension, so effective rank can scale with context length. But that does not by itself prove the generated adapter encodes code bodies or temporal arrows; those remain objective/data questions.

- Pushback on E1: oracle per-episode LoRA versus hypernet-generated adapter at matched rank is a good discriminator, but only if the oracle trains on the same hidden-code facts and is scored with the same masks, negatives, and prompts. Otherwise it can become an unfair comparison between per-instance gradient optimization and amortized generation, not a clean representation-vs-capacity result. Report it as an upper bound, not as direct evidence that the hypernetwork objective is wrong.

- Add a cross-over control to E1: if oracle succeeds at rank 8 on body facts, also test a tiny hypernet fine-tune on those exact facts before concluding "representation wall." If a handful of updates moves the hypernet while the warm-start does not, the diagnosis is objective mismatch; if it still does not move, then architecture/conditioning attenuation is back on the table.

- Pushback on E2: direction-scrambling is necessary, but it is easy to make the negative too distribution-shifted. Time reversal or swapping "where-headed" text can introduce lexical artifacts that the model rejects without understanding directionality. Include minimally edited counterfactuals that preserve tokens and local code while changing only the causal arrow or next-action implication, plus a same-bag-of-events control.

- Directionality should be scored on action consequences, not only recall. A probe that asks "what happened first?" may show order retrieval while still failing the product need. The decisive readout is whether the adapter changes next-step code/action tokens in the direction implied by the prior failure or partial trajectory, compared against an in-prompt ceiling.

- Any fine-tuning case should carry an explicit retention gate. The warm-start already has valuable QA/NIAH and tail recall; a Rune-specific objective that improves body/direction while erasing that prior would not be a better substrate, just a narrower adapter generator.

## 2026-06-02 - Pushback on Residual Encoding

- The residual/surprisal lever is probably real, but do not equate "high-surprisal" with "useful for Rune." CaMeLS is the right analogy precisely because the weights are meta-learned against downstream QA utility after adaptation, not because raw token perplexity is automatically the target. For code, high-surprisal tokens include noise, comments, one-off names, and dataset artifacts; low-surprisal tokens can include action-critical operators such as `not`, `return`, comparison direction, or exception type.

- Therefore E4 should compare at least three weighting families: base-surprisal, action/discriminative-token labels, and learned or proxy utility weights. The evaluation target should be downstream body/direction utility, not just weighted reconstruction loss. If surprisal weighting improves likelihood on rare identifiers but does not improve next-step code choice, it is a compression trick, not a substrate advance.

- Treat canonicalization as a dangerous intervention, not a free preprocessing win. Formatting normalization is likely safe; comment stripping, AST abstraction, or literal normalization can remove exactly the intent, contract, and edge-case facts that orient future steps. Run canonicalized and raw-active-code arms side by side, and predeclare which code regions may be lossy.

- Add a "small-token, big-effect" negative control. Construct examples where the decisive fact is syntactically common or low-surprisal: flipped inequality, missing `not`, integer-vs-string return, inclusive/exclusive boundary, or which exception to catch. A residual encoder that drops these because the base finds them predictable will look efficient while failing the product.

- The right unit may be "utility per rank" rather than "bits recalled per rank." Useful code memory is counterfactual: would this encoded fact change the next action relative to a plausible wrong continuation? Score matched versus hard negatives that preserve surface rarity but alter the semantic constraint, otherwise the metric may reward rare-token storage instead of actionable memory.

- Compression should be local-state aware. Active/cutoff code and current failure facts need near-verbatim fidelity; distant helper code may tolerate summaries plus signatures/contracts. A single global weighting policy risks over-compressing the active locus or wasting rank on distant rare tokens.

## 2026-06-02 - Infra and T0 Launch Guardrails

- The qwen/Gemma attribution correction is load-bearing and should not live only in the new predeclared spec. Add an erratum banner to any durable handoff/dossier that a future session might read first, or future work can silently resurrect the Gemma `tail +2.01` as a qwen continuation fact.

- The controlled T0 rerun is the right scientific correction. Do not interpret any result until the dump proves both arms share the same eligible rows, same `ctx_hash` values, same `max_seq_length=768` path for conditioning and scoring, and the same non-NaN denominator. The JSONL contract is more important than the aggregate mean.

- The new guard design should kill process groups, not only the registered parent PID. `uv run python ... &` may leave a child Python/CUDA process alive if the wrapper shell or uv process is killed. Use a process-group launch/kill pattern or explicitly verify no descendant remains after a guard-triggered kill, otherwise the guard can report success while GPU/RAM pressure continues.

- Checkpoint offload via `mlflow.log_artifact` is sensible, but listing `checkpoints/<name>` only verifies that an artifact path exists, not that the just-written bytes are present. Log and verify at least size, checksum, or a step-specific artifact name before deleting the local staging file. This matters especially for reusable names like `checkpoint.pt` and `checkpoint_best.pt`.

- Treat `tools/instance_guard.sh` as part of the experimental apparatus, not scratch. If runs depend on it, it needs the same durability as the specs: checked into git or copied into the run artifact bundle, with the active thresholds logged. Otherwise later reproduction cannot tell whether a run was protected by the same kill policy.

- The disk cleanup sounds appropriate, but preserved/deleted checkpoint decisions should be reflected in a manifest with local path, S3 URI, byte size, checksum, and purpose. "Uploaded then deleted" is not an audit trail unless the retrieval target is exact.

## 2026-06-02 - E1 Interpretation Boundaries

- The E1 result is genuinely useful: under the frozen absent/body setup, the hypernet shows the same signature/body asymmetry (`sig +4.09`, `body +0.14`) while an r8 down-proj oracle can memorize the body surface. That is strong evidence that the warm-start hypernetwork's representation/objective is name- and answerable-fact-dominated.

- Narrow the capacity claim. The oracle proves that r8 down-proj LoRA has enough capacity to overfit these 10 short MBPP reference bodies under train-equals-score. It does not prove r8/chunk count is sufficient for longer trajectories, multiple files, active state + failure history, or held-out body generalization. Say "capacity is not the wall for this E1 micro-probe," not "capacity is not the wall" globally.

- Do not let the oracle's `+21.75` margin set any success target. It is a per-instance gradient upper bound scored on the training surface, and the mismatch arm uses other memorized adapters that can be catastrophically wrong. The meaningful comparison is categorical: oracle can store the surface, hypernet does not currently choose to encode it episode-specifically.

- Skipping r16/r32 is reasonable for the immediate cross-over decision, but keep rank/chunk scaling as a later interaction test. If a body-recall fine-tune starts working, capacity may re-emerge as the bottleneck when moving from 10 frozen MBPP bodies to realistic trajectory/codebase contexts.

- The cross-over design correction is important. Reusing `_distill_entry` would test the wrong objective. A faithful cross-over should optimize the hypernetwork-generated adapter for the same absent/body matched-vs-deranged objective and should include a no-adapter/base and no-finetune warm-start readout at every checkpoint so generic body boosting cannot masquerade as episode binding.

- Add an overfit sanity stop to the cross-over: if the tiny trainer cannot overfit the 10 facts enough to move body matched-vs-mismatch, inspect gradient flow, scaler preservation, and adapter assembly before concluding architecture attenuation. A null cross-over is ambiguous unless the trainer itself is shown to optimize the intended loss.

## 2026-06-02 - E1 Precision Probe Interpretation

- The BF16 rerun is a good control, but it is not a pure scoring-precision test. It changes upstream base activations used for hypernet conditioning, generated adapter weights, and final logits. If body specificity jumps, the conclusion should be "the current 4-bit deployment path suppresses or distorts body binding," not simply "fp noise hid the signal."

- Keep the decisive comparison within each precision regime: matched vs mismatch, body vs signature, and matched vs zero. A BF16 body jump that is accompanied by a similar mismatch or zero shift is still generic body boosting, not episode-specific body memory.

- If BF16 differs materially, add one intermediate ablation before cross-over: generate the adapter from BF16 activations but score with the normal 4-bit path, or vice versa if the tooling allows it. That separates conditioning sensitivity from scoring sensitivity and tells you which part the fine-tune must target.

- Product framing matters: Rune's near-term constraint is 4-bit local execution on this GPU. A BF16-only body result is scientifically informative but not a usable substrate win unless the 4-bit path can be recovered by training, scaling, or calibration.

## 2026-06-02 - Precision Regime Correction

- Correction accepted: the deployed engine path loads the 4B base in BF16, while the 4-bit probe defaults were inherited from the older 9B memory regime. Going forward, engine-parity evals should be BF16 by default and every reported metric should carry an explicit precision tag.

- Keep training precision separate from evaluation precision. The distillation loop still defaults to a 4-bit frozen base for memory, so a BF16 eval baseline does not by itself prove that BF16 training/cross-over is safe or equivalent. If cross-over trains in BF16, log that as a deliberate objective change, not a mere cleanup.

- The unchanged BF16-vs-4-bit E1 body result is useful because it stabilizes the representation-wall diagnosis. But prior 4-bit T0 feedback-swap numbers should not silently become BF16 evidence; rerun only the metrics that will be used as engine-parity gates, or mark the others as legacy 4-bit diagnostics.

- Update the durable handoff/spec or add an erratum so future sessions do not inherit the wrong "product runs 4-bit" assumption from earlier reflections. Otherwise the next agent may optimize the wrong precision path.
