#!/usr/bin/env bash
# RFT (rejection-sampling fine-tuning) run — REMOVE-BEFORE-MERGE. GPU serial.
# Reuses existing tools (DRY): gen escalate sessions -> mine positives ->
# existing distiller -> eval c3 vs RFT on held-out. Train = MBPP-160 (disjoint
# from the 24 held-out eval), so no leakage.
set -uo pipefail
cd "$(dirname "$0")/.."
RFT=/tmp/rft; mkdir -p "$RFT"
SUM="$RFT/SUMMARY.txt"; : > "$SUM"
C3=/tmp/phase1/ckpt/c3_t07_lp2_lg1.pt
MBPP=benchmarks/mbpp160_tasks.json
HELDOUT=benchmarks/mbpp_heldout_tasks.json

wait_for () { while pgrep -f "$1" >/dev/null; do sleep 20; done; }

# 1. generate new-format escalate sessions on MBPP-160 (the RFT trajectory source)
echo ">>> [1/4] gen escalate(new-format) on MBPP-160" | tee -a "$SUM"
tools/run_guarded.sh "$RFT/gen.log" tools/_goal3_multiturn_probe.py run \
  --arm c3 --tasks "$MBPP" --sessions "$RFT/gen_sessions" --out "$RFT/gen.json" \
  --seed 0 --max-iters 12 --prompt-mode escalate --adapter-scaling 0.627
wait_for "_goal3_multiturn_probe.*$RFT/gen.json"

# 2. mine positives (only tasks that passed held-out tests -> trustworthy targets)
echo ">>> [2/4] mine RFT corpus" | tee -a "$SUM"
uv run python tools/_rft_mine.py --sessions "$RFT/gen_sessions" --held-out-only \
  --out "$RFT/corpus.jsonl" 2>&1 | tee -a "$SUM"
python3 -c "
import json
rows=[l for l in open('$RFT/corpus.jsonl') if l.strip()]
k=max(1,len(rows)//10)
open('$RFT/corpus_train.jsonl','w').writelines(rows[:-k] or rows)
open('$RFT/corpus_val.jsonl','w').writelines(rows[-k:])
print('corpus: train',len(rows)-k,'val',k)
" | tee -a "$SUM"

# 3. distill c3 on the positives (existing distiller; train-scaling matches inference)
echo ">>> [3/4] distill c3 -> RFT ckpt" | tee -a "$SUM"
tools/run_guarded.sh "$RFT/distill.log" tools/run_corpus_distill.py \
  --corpus "$RFT/corpus_train.jsonl" --val-corpus "$RFT/corpus_val.jsonl" \
  --checkpoint "$C3" --out "$RFT/ckpt" --epochs 2 --lr 1e-4 \
  --train-scaling 0.627 --max-seq-length 2048
wait_for "run_corpus_distill.*$RFT/corpus_train"
RFT_CKPT="$RFT/ckpt/checkpoint_best.pt"
[ -f "$RFT_CKPT" ] || RFT_CKPT=$(ls -t "$RFT/ckpt"/checkpoint_step*.pt 2>/dev/null | head -1)
echo "RFT ckpt = $RFT_CKPT" | tee -a "$SUM"

# 4. eval c3 (baseline) vs RFT on the 24 held-out, escalate mode (new format)
for arm in c3 rft; do
  ck=$([ "$arm" = rft ] && echo "$RFT_CKPT" || echo "$C3")
  echo ">>> [4/4] eval $arm on held-out-24 (escalate)" | tee -a "$SUM"
  tools/run_guarded.sh "$RFT/eval_$arm.log" tools/_goal3_multiturn_probe.py run \
    --arm c3 --ckpt "$ck" --tasks "$HELDOUT" --sessions "$RFT/eval_${arm}_sessions" \
    --out "$RFT/eval_$arm.json" --seed 0 --max-iters 12 \
    --prompt-mode escalate --adapter-scaling 0.627
  wait_for "_goal3_multiturn_probe.*eval_$arm.json"
  python3 -c "import json;d=json.load(open('$RFT/eval_$arm.json'));print(f'EVAL $arm held-out-24: {d[\"passed_tasks\"]}/{d[\"total_tasks\"]} = {d[\"pass_at_1\"]:.3f}')" | tee -a "$SUM"
done
echo "=== RFT DONE (c3 vs RFT above = did rejection-sampling help) ===" | tee -a "$SUM"
