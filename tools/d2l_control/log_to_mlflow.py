"""Issue #52 — log the Doc2LoRA positive-control results to MLflow (localhost:5000).

Runs in the RUNE venv (has mlflow). Parses the probe logs in /tmp + the NIAH CSV +
the provenance manifest, and logs one MLflow run per experiment under experiment
'issue52-d2l-control'. Per the reviewer: log JSON metrics, version/provenance manifest,
the (inert) patch diff, and result summaries as ARTIFACTS — never checkpoints/model files.
Checkpoint provenance is path+sha256 only.

Usage:  uv run python tools/d2l_control/log_to_mlflow.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mlflow

TRACKING_URI = "http://localhost:5000"
EXPERIMENT = "issue52-d2l-control"
PROV = "/tmp/d2l_provenance.json"
PATCH = "/tmp/d2l_patch.diff"


def _f(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_smoke(p="/tmp/d2l_smoke.log") -> dict | None:
    t = _read(p)
    if not t or "SMOKE PASS" not in t:
        return None
    return {
        "needle_m_mismatch": _f(r"m-mismatch=\+?(-?[\d.]+)", t),
        "needle_m_zero": _f(r"m-zero=\+?(-?[\d.]+)", t),
        "gen_hit": 1.0 if "contains '4417': True" in t else 0.0,
    }


def parse_code(p: str) -> dict | None:
    t = _read(p)
    if not t or "CODE RECALL SUMMARY" not in t:
        return None
    return {
        "code_m_mismatch_mean": _f(r"mean m-mismatch=\+?(-?[\d.]+)", t),
        "code_m_zero_mean": _f(r"mean m-zero=\+?(-?[\d.]+)", t),
        "code_gen_accuracy": _f(r"gen_accuracy=([\d.]+)", t),
        "code_frac_specific": _f(r"frac\(m-mismatch>0\)=([\d.]+)", t),
    }


def parse_episodes(p: str) -> dict | None:
    t = _read(p)
    if not t or "RUNE-EPISODES-THROUGH-SAKANA" not in t:
        return None
    out = {}
    for tgt in ("goal", "file", "diff"):
        m = re.search(rf"{tgt}\s+n=\d+\s+mean m-mismatch=\+?(-?[\d.]+)\s+mean m-zero=\+?(-?[\d.]+)\s+frac\(m-mis>0\)=([\d.]+)", t)
        if m:
            out[f"{tgt}_m_mismatch"] = float(m.group(1))
            out[f"{tgt}_m_zero"] = float(m.group(2))
            out[f"{tgt}_frac_specific"] = float(m.group(3))
    o = _f(r"OVERALL mean m-mismatch=\+?(-?[\d.]+)", t)
    if o is not None:
        out["overall_m_mismatch"] = o
    return out or None


def parse_niah_csv() -> dict | None:
    import csv
    hits = list(Path("third_party/doc-to-lora/trained_d2l/gemma_demo").glob("eval-results-*/**/evaluation_results_generation.csv"))
    if not hits:
        return None
    rows = list(csv.DictReader(open(sorted(hits)[-1])))
    vals = [float(r["rougeL.f1"]) for r in rows if r.get("rougeL.f1") not in (None, "", "None") and int(r["num_samples"]) > 0]
    return {"niah_rougeL_f1": max(vals) if vals else None, "niah_csv": str(sorted(hits)[-1])} if vals else None


def _read(p: str) -> str | None:
    fp = Path(p)
    return fp.read_text() if fp.exists() else None


# #49 reference (handoff §4.7, A600) — Rune's OWN checkpoint on the same targets.
RUNE_49 = {"goal_m_mismatch": 0.0005, "file_m_mismatch": 0.011, "diff_m_mismatch": 0.075}


def main():
    prov = json.load(open(PROV)) if Path(PROV).exists() else {}
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    common_params = {
        "sakana_git_commit": prov.get("sakana_git_commit"),
        "cuda": prov.get("cuda"),
        "D2L_ATTN_IMPL": prov.get("D2L_ATTN_IMPL"),
        **{f"ver_{k}": v for k, v in (prov.get("versions") or {}).items()},
    }

    runs = [
        ("niah_repro_gemma", {"stage": "reproduction-anchor", "base": "google/gemma-2-2b-it",
                              "dataset": "ctx_magic_number_512_1024", "n": 40}, parse_niah_csv()),
        ("scorecard_calibration_gemma", {"stage": "calibration", "base": "google/gemma-2-2b-it"}, parse_smoke()),
        ("code_recall_gemma", {"stage": "code-recall", "base": "google/gemma-2-2b-it"}, parse_code("/tmp/d2l_coderecall.log")),
        ("rune_bridge_gemma", {"stage": "bridge", "non_gating": True, "base": "google/gemma-2-2b-it"}, parse_episodes("/tmp/d2l_runeep.log")),
        ("code_recall_qwen4b", {"stage": "code-recall", "base": "Qwen/Qwen3-4B-Instruct-2507"}, parse_code("/tmp/d2l_qwen_code.log")),
        ("rune_bridge_qwen4b", {"stage": "bridge", "non_gating": True, "base": "Qwen/Qwen3-4B-Instruct-2507"}, parse_episodes("/tmp/d2l_qwen_ep.log")),
        ("rune_reference_issue49", {"stage": "reference", "note": "Rune OWN checkpoint, handoff 4.7 A600"}, RUNE_49),
    ]

    logged = 0
    for name, tags, metrics in runs:
        if not metrics:
            print(f"skip {name}: no results yet")
            continue
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"issue": "52", "deliverable": "doc2lora-positive-control", **tags})
            mlflow.log_params(common_params)
            ck = (prov.get("checkpoints") or {}).get("gemma_demo" if "gemma" in name else "qwen_4b_d2l")
            if ck:
                mlflow.log_params({"ckpt_path": ck["path"], "ckpt_sha256": ck["sha256"][:16]})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            for art in (PROV, PATCH):
                if Path(art).exists():
                    mlflow.log_artifact(art)
            csv = metrics.get("niah_csv")
            if csv and Path(csv).exists():
                mlflow.log_artifact(csv)
        logged += 1
        print(f"logged {name}: { {k: v for k, v in metrics.items() if isinstance(v,(int,float))} }")
    print(f"\n{logged} runs logged to {TRACKING_URI} / {EXPERIMENT}")


if __name__ == "__main__":
    main()
