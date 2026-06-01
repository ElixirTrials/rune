# Reproducing the Issue #52 Doc2LoRA positive control

> **This branch is experimentation preserved for posterity.** None of this code is wired into
> Rune. It runs against the external `SakanaAI/doc-to-lora` repo + its pinned stack. The durable
> conclusions live in `docs/issue52-findings-2026-06-01.md` on the working branch; all runs are
> in MLflow `issue52-d2l-control`.

## Layout
- `tools/scoring_core.py` — pure-torch `mean_gold_logprob` / `masked_gold_logprob` (the shared
  scorecard math; torch-only so it imports from the Sakana venv via `sys.path`).
- `tools/d2l_control/episodes.py` — doc-fact episodes + `build_rune_episodes` (reformulate
  `external_codereview` rows → patch+QA episodes carrying goal/file/diff).
- `tools/d2l_control/probes/` — the experiment scripts (run from the Sakana venv):
  - `rune_smoke.py` — load + teacher-forced logits + needle recall (calibration).
  - `rune_code_recall.py` — code-fact recall (matched/mismatch/zero + generation).
  - `rune_episode_recall.py` — Rune episodes through Sakana (goal/file/diff).
  - `rune_continuation.py` — tail/continuation recall.
  - `rune_facet_negatives.py` — goal vs diff under generic AND feedback-swap hard negatives.
  - `rune_finetune_specialize.py` — light-finetune ablation (CE; retention + specificity gates).
  - `rune_finetune_contrastive.py` — contrastive variant (feedback-swap hinge; OOMs at 22GB —
    needs 8-bit Adam + ctx truncation; the eval-only `rune_facet_negatives` already answers it).
- `tools/d2l_control/log_to_mlflow.py` — parse probe logs → MLflow (run in the Rune venv).

## Environment (the version skew is the reason for isolation)
Rune's venv has `ctx_to_lora` + transformers 5.x; Sakana pins transformers 4.51.3 — incompatible.
Use a SEPARATE venv:
```bash
git clone https://github.com/SakanaAI/doc-to-lora third_party/doc-to-lora   # commit baa85db4
cd third_party/doc-to-lora && uv venv
uv pip install --python .venv torch==2.7.0 transformers==4.51.3 accelerate==1.6.0 \
  datasets==3.6.0 einops jaxtyping peft torchmetrics sentencepiece protobuf safetensors \
  hf_transfer bitsandbytes
# flash-attn: prebuilt wheel matching torch2.7/cu12/cp312/abiTRUE
uv pip install --python .venv \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv run hf download SakanaAI/doc-to-lora --local-dir trained_d2l   # ~6.5GB (gemma_demo, qwen_4b_d2l)
```
Gemma-2-2b-it / Qwen3-4B bases download on first load. GPU rule: `free -g`; run under
`tools/run_guarded.sh <logfile> <script>`.

## Run (from `third_party/doc-to-lora`, copy the probes in or symlink the rune tools dir)
```bash
../../tools/run_guarded.sh /tmp/smoke.log rune_smoke.py              # needle m-mismatch ~ +7.7
../../tools/run_guarded.sh /tmp/code.log  rune_code_recall.py        # code facts m-mismatch ~ +7.1
../../tools/run_guarded.sh /tmp/ep.log    rune_episode_recall.py     # rune goal/file/diff +2.3/+1.8/+1.0
../../tools/run_guarded.sh /tmp/fn.log    rune_facet_negatives.py    # goal holds +1.59, diff collapses +0.17
D2L_CKPT=trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
  ../../tools/run_guarded.sh /tmp/qwen.log rune_code_recall.py       # base-family control
# unmodified NIAH (reproduction anchor; needs rouge_score llmlingua tensorboardX wandb):
WANDB_MODE=disabled .venv/bin/python run_eval.py \
  --checkpoint_path trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
  --datasets ctx_magic_number_512_1024 --split test --max_test_samples_per_ds 40 \
  --max_ctx_chunk_len 1024 --eval_batch_size_gen 4                   # rougeL.f1 = 1.0
```
Then `uv run python tools/d2l_control/log_to_mlflow.py` (Rune venv) → MLflow `issue52-d2l-control`.

## Local Sakana patch (kept INERT for all reported numbers)
`aggregator.py`/`idefics2.py` gain an env-gated eager perceiver-attention path (`D2L_ATTN_IMPL`).
Left UNSET (flash) for every reported run, so the patch does not affect results. See
`d2l_patch.diff`. Provenance (versions, commit, checkpoint SHA256s) in `d2l_provenance.json`.
