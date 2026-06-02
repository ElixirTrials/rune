"""Goal-3 specificity probe (cheap, logprob-level) BEFORE the 2hr generation run.

Question: at the contract scale (adapter_scaling=1.0 == lora_alpha), does the
qwen_4b_d2l adapter conditioned on MBPP task i carry task-SPECIFIC information,
or is the 7/10 xgrammar lift a generic anti-degeneration prior?

Protocol (one model load, the 10 frozen Phase-0 tasks):
  - matched adapter   = hypernet( render_training_format_trajectory(task_i) )
                        i.e. the EXACT engine conditioning surface (graph.py).
  - mismatch adapter  = matched adapter of a DERANGEMENT partner perm(i) (no
                        task is its own partner; raises if derangement invalid).
  - zero              = base, no adapter.

Two metrics:
  (A) weight-space ||A_i - A_perm(i)|| / ||A_i|| over the assembled adapter
      tensors. SANITY only (advisor/reviewer): tiny => deranged gen arm cannot
      prove specificity; large => differences exist but may be MBPP-irrelevant.
  (B) mean gold logprob of a held-constant REFERENCE SOLUTION span under
      matched / mismatch / zero, in TWO prompt regimes:
        - present: task description IS in the prompt (faithful to the bench;
          adapter conditioning is partially redundant -> narrow read).
        - absent : task description NOT in the prompt (NIAH-style; tests whether
          the adapter encodes task-specific solution info AT ALL).
      margin_mm = lp_matched - lp_mismatch ; margin_mz = lp_matched - lp_zero.

DECISIVE READ (pre-registered):
  matched > mismatch (present)  => task-specific adapter UTILITY in the bench
                                   setting -> deranged gen run is worth it.
  matched ~= mismatch (present) but matched > mismatch (absent) => adapter
                                   encodes task info but it is redundant with the
                                   visible prompt (explains discipline-only 7/10).
  matched ~= mismatch (both)    => short MBPP descriptions do not move the adapter
                                   in an MBPP-relevant way; 7/10 is a generic prior.
  NOTE: a flat present-margin does NOT refute #52's episodic-memory bet for hidden
  multi-turn feedback/tried/critique facts (those are not in the prompt at all).

Run in RUNE's venv (bf16, flash-attn): uv run python tools/_specificity_probe.py --bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

RUNE = "/workspaces/rune-gpu"
sys.path.insert(0, f"{RUNE}/tools")
sys.path.insert(0, f"{RUNE}/tools/d2l_control")

import scoring_core  # noqa: E402

CKPT = (
    f"{RUNE}/third_party/doc-to-lora/trained_d2l/"
    "qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
BASE = "Qwen/Qwen3-4B-Instruct-2507"
TASKS_FILE = f"{RUNE}/benchmarks/mbpp_phase0_iter.json"
MAX_ANS_TOK = 96

# Prompt regimes (module-level so tools/_e1_oracle.py imports the BYTE-IDENTICAL
# hidden-regime prompt). present: task desc IS in the prompt; absent: hidden (NIAH).
PRESENT = "Write the following Python function.\n\n{desc}\n\nReturn only the code."
ABSENT = "Write the Python function you have just studied. Return only the code."

# Held-constant idiomatic reference solutions (exact entry_point casing). The
# matched-minus-mismatch margin scores the SAME string under each adapter, so the
# exact phrasing is not load-bearing (advisor): only that it is a plausible target.
REFS: dict[str, str] = {
    "mbpp/11": (
        "def remove_Occ(s, ch):\n"
        "    for i in range(len(s)):\n"
        "        if s[i] == ch:\n"
        "            s = s[0:i] + s[i + 1:]\n"
        "            break\n"
        "    for i in range(len(s) - 1, -1, -1):\n"
        "        if s[i] == ch:\n"
        "            s = s[0:i] + s[i + 1:]\n"
        "            break\n"
        "    return s\n"
    ),
    "mbpp/12": "def sort_matrix(M):\n    return sorted(M, key=sum)\n",
    "mbpp/14": "def find_Volume(l, b, h):\n    return (l * b * h) / 2\n",
    "mbpp/16": (
        "import re\n"
        "def text_lowercase_underscore(text):\n"
        "    return bool(re.search('^[a-z]+_[a-z]+$', text))\n"
    ),
    "mbpp/17": "def square_perimeter(a):\n    return 4 * a\n",
    "mbpp/18": (
        "def remove_dirty_chars(string, second_string):\n"
        "    return ''.join(c for c in string if c not in second_string)\n"
    ),
    "mbpp/19": (
        "def test_duplicate(arraynums):\n"
        "    return len(arraynums) != len(set(arraynums))\n"
    ),
    "mbpp/20": (
        "def is_woodall(x):\n"
        "    if x % 2 == 0:\n"
        "        return False\n"
        "    x = x + 1\n"
        "    p = 0\n"
        "    while x % 2 == 0:\n"
        "        x = x // 2\n"
        "        p = p + 1\n"
        "    return x == p\n"
    ),
    "mbpp/56": (
        "def check(n):\n"
        "    rev = 0\n"
        "    m = n\n"
        "    while m > 0:\n"
        "        rev = rev * 10 + m % 10\n"
        "        m //= 10\n"
        "    return 2 * rev - 1 == n\n"
    ),
    "mbpp/57": (
        "def find_Max_Num(arr):\n"
        "    arr.sort(reverse=True)\n"
        "    return int(''.join(map(str, arr)))\n"
    ),
}


def derangement(n: int) -> list[int]:
    """Deterministic derangement: i -> (i + 1) % n. No fixed points for n >= 2."""
    if n < 2:
        raise ValueError("derangement needs n >= 2")
    perm = [(i + 1) % n for i in range(n)]
    if any(perm[i] == i for i in range(n)):
        raise ValueError("invalid derangement (fixed point)")
    return perm


def span_bounds(tok, ans: str, entry_point: str) -> tuple[int, int]:
    """Token offsets (within the answer) of the `def <entry_point>(...):` line.

    Returns (lo, hi): the SIGNATURE occupies answer-token range [lo, hi); the BODY
    is [hi, len) (E1's only scored span; see spec). Boundaries from prefix
    re-tokenization are approximate at BPE seams, but the matched-vs-mismatch
    comparison scores the SAME span under each adapter, so any fixed boundary is a
    valid discriminator (advisor).

    E1 FROZEN (predeclared spec): the old `(0,0)` missing-marker fallback silently
    set hi=0, making BODY=[0,len) == the FULL answer (signature included), which
    collapses the body-vs-signature discriminator (body +0.14 vs signature +3.84).
    Raise instead so an episode is NEVER scored under the (0,0) fallback; callers
    must exclude such episodes with an explicit reason.
    """
    marker = f"def {entry_point}("
    j = ans.find(marker)
    if j < 0:
        raise ValueError(
            f"def-{entry_point}( signature marker not found in reference answer; "
            "refusing the (0,0) fallback that would score signature as BODY"
        )
    line_end = ans.find("\n", j)
    if line_end < 0:
        line_end = len(ans)
    lo = len(tok(ans[:j], add_special_tokens=False).input_ids[:MAX_ANS_TOK])
    hi = len(tok(ans[: line_end + 1], add_special_tokens=False).input_ids[:MAX_ANS_TOK])
    return (lo, hi)


def build_full(tok, device, prompt: str, answer: str):
    chat = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        chat, add_special_tokens=False, add_generation_prompt=True, return_tensors="pt"
    )
    p = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
    a = tok(answer, add_special_tokens=False).input_ids[:MAX_ANS_TOK]
    a = torch.tensor([a], device=device)
    full = torch.cat([p, a], dim=1)
    return full, p.shape[1], a.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--model-id", type=str, default=BASE)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--out", type=str, default=None,
                    help="per-(regime,span,task) JSONL dump incl. raw lp_m/lp_x/lp_z")
    a = ap.parse_args()

    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from rune.engine.graph import render_training_format_trajectory  # noqa: PLC0415
    from rune.model.adapter_contract import effective_scaling  # noqa: PLC0415
    from rune.model.hypernetwork import (  # noqa: PLC0415
        HypernetworkConfig,
        extract_activations_with_model,
        load_hypernetwork,
    )
    from rune.training.hypernet_distill import _functional_lora  # noqa: PLC0415

    tasks = json.loads(Path(TASKS_FILE).read_text())
    n = len(tasks)
    perm = derangement(n)
    for t in tasks:
        if t["task_id"] not in REFS:
            raise KeyError(f"no reference solution for {t['task_id']}")  # trap (a)

    print("loading base + hypernet...", flush=True)
    load_kw = dict(
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
    )
    if not a.bf16:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    print(f"  base dtype = {'bf16' if a.bf16 else '4bit-nf4'}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(a.model_id, **load_kw).eval()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    hyp = load_hypernetwork(HypernetworkConfig(checkpoint_path=a.ckpt), device="cuda")
    hyp.eval()
    scaling = effective_scaling(hyp)
    li = [int(x) for x in hyp.config.layer_indices]
    hyp_device = next(hyp.parameters()).device
    hyp_dtype = next(hyp.parameters()).dtype
    device = next(base.parameters()).device

    from ctx_to_lora.modeling.lora_merger import combine_lora  # noqa: PLC0415

    n_chunks = torch.tensor([1], device=device)
    n_qs = torch.tensor([1], device=device)
    head_bias = hyp.get_head_bias() if getattr(hyp.config, "use_bias", False) else None
    print(
        f"  scaling(effective)={scaling}  use_bias={head_bias is not None}  "
        f"n_layers={len(li)}",
        flush=True,
    )

    # Conditioning trajectory == the EXACT engine surface (graph.py render).
    trajs = [
        render_training_format_trajectory(task=t["description"]) for t in tasks
    ]
    # Audit artifact: one matched/mismatch rendered-trajectory pair (reviewer ask).
    print(
        "\n[AUDIT] task0 matched-traj vs mismatch-traj (partner):\n"
        f"  MATCHED (task {tasks[0]['task_id']}):\n{trajs[0]!r}\n"
        f"  MISMATCH (partner {tasks[perm[0]]['task_id']}):\n{trajs[perm[0]]!r}",
        flush=True,
    )

    def assemble(doc: str):
        feats, am = extract_activations_with_model(doc, base, tok, li, a.max_seq_length)
        feats = feats.to(device=hyp_device, dtype=hyp_dtype)
        am = am.to(hyp_device)
        with torch.no_grad():
            ld, _ = hyp.generate_weights(feats, am, None)
        return combine_lora(ld, n_chunks, lora_bias=head_bias)

    print("\nassembling 10 matched adapters...", flush=True)
    adapters = [assemble(tr) for tr in trajs]

    # (A) weight-space ||A_i - A_perm(i)|| / ||A_i|| over assembled tensors.
    def rel_dist(da, db) -> float:
        # lora_dict = {module_name: {"A": tensor, "B": tensor}} (combine_lora shape).
        num = 0.0
        den = 0.0
        for k in da:
            for sub in ("A", "B"):
                x = da[k][sub].float()
                y = db[k][sub].float()
                num += float(((x - y) ** 2).sum())
                den += float((x**2).sum())
        return (num**0.5) / (den**0.5 + 1e-12)

    print("\n=== (A) WEIGHT-SPACE DERANGEMENT DISTANCE (sanity) ===", flush=True)
    rels = []
    for i, t in enumerate(tasks):
        rd = rel_dist(adapters[i], adapters[perm[i]])
        rels.append(rd)
        print(f"  {t['task_id']:8s} <- partner {tasks[perm[i]]['task_id']:8s}"
              f"  ||dA||/||A||={rd:.4f}", flush=True)
    print(f"  MEAN rel-dist = {sum(rels) / len(rels):.4f}", flush=True)

    # (B) reference-solution logprob under matched/mismatch/zero, two prompt regimes.
    def logits_with(ld, full):
        with torch.no_grad(), _functional_lora(base, li, ld, scaling, n_qs):
            return base(full, use_cache=False).logits[0]

    # PRESENT/ABSENT are module-level constants (shared with tools/_e1_oracle.py).
    # Three scored spans per task: full answer, the def-signature line, the body.
    # The signature span is the name-contract discriminator (advisor): does matched
    # memory assign LOWER logprob to the correct `def <name>(...)` than mismatch,
    # i.e. fight the in-prompt name (cf. the sorted_matrix != sort_matrix miss)?
    dump: list[dict] = []
    for regime, tmpl in (("present", PRESENT), ("absent", ABSENT)):
        print(f"\n=== (B) REFERENCE-SOLUTION LOGPROB  regime={regime} ===", flush=True)
        # spans -> list of (task_id, m-mismatch, m-zero)
        agg: dict[str, list[tuple[str, float, float]]] = {
            "full": [],
            "sig": [],
            "body": [],
        }
        for i, t in enumerate(tasks):
            prompt = tmpl.format(desc=t["description"]) if "{desc}" in tmpl else tmpl
            ans = REFS[t["task_id"]]
            full, start, length = build_full(tok, device, prompt, ans)
            if length < 1:
                continue
            ids = full[0]
            try:
                lo, hi = span_bounds(tok, ans, t["entry_point"])
            except ValueError as exc:
                print(f"  [EXCLUDED] {t['task_id']}: {exc}", flush=True)
                continue  # marker missing -> never score (would contaminate body)
            spans = {"full": (start, length)}
            if hi > lo:
                spans["sig"] = (start + lo, hi - lo)
            if length > hi:
                spans["body"] = (start + hi, length - hi)

            lg_m = logits_with(adapters[i], full)
            lg_x = logits_with(adapters[perm[i]], full)
            with torch.no_grad():
                lg_z = base(full, use_cache=False).logits[0]
            for name, (s, ln) in spans.items():
                lp_m = scoring_core.mean_gold_logprob(lg_m, ids, s, ln)
                lp_x = scoring_core.mean_gold_logprob(lg_x, ids, s, ln)
                lp_z = scoring_core.mean_gold_logprob(lg_z, ids, s, ln)
                agg[name].append((t["task_id"], lp_m - lp_x, lp_m - lp_z))
                dump.append({"regime": regime, "span": name, "task_id": t["task_id"],
                             "neg_task_id": tasks[perm[i]]["task_id"],
                             "lp_m": lp_m, "lp_x": lp_x, "lp_z": lp_z,
                             "m_mismatch": lp_m - lp_x, "m_zero": lp_m - lp_z})

        # Per-task sig/body (advisor resolver): short-body tasks (sort_matrix,
        # square_perimeter, test_duplicate) have minimal generic-token dilution, so a
        # high body margin THERE = algorithm genuinely recalled, not just the name.
        for name in ("full", "sig", "body"):
            rows = agg[name]
            if not rows:
                continue
            for tid, mm_i, mz_i in rows:
                print(
                    f"    [{name:4s}] {tid:8s} m-mismatch={mm_i:+.4f}  m-zero={mz_i:+.4f}",
                    flush=True,
                )
            mm = sum(r[1] for r in rows) / len(rows)
            mz = sum(r[2] for r in rows) / len(rows)
            frac = sum(1 for r in rows if r[1] > 0) / len(rows)
            print(
                f"  [{name:4s}] MEAN n={len(rows):2d} m-mismatch={mm:+.4f}"
                f"  m-zero={mz:+.4f}  frac(m-mis>0)={frac:.2f}",
                flush=True,
            )

    if a.out:
        with open(a.out, "w") as f:
            for r in dump:
                f.write(json.dumps(r) + "\n")
        print(f"\n[dump] {len(dump)} rows -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
