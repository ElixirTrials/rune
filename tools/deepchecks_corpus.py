"""One-time DeepChecks data + split QC for the issue #49 corpus (CPU).

Complements (does NOT replace) the family-keyed split, `_corpus_stats`, and the
teacher-quality audit. Validates the DATA layer only:
  - data-integrity on train (duplicates, conflicting/empty samples, property outliers);
  - train-test-validation train-vs-val and train-vs-test (text-property drift +
    embeddings drift = an independent representativeness + near-duplicate-leakage
    check the key-based family split cannot catch).

NOT a training-dynamics monitor (that's MLflow) and NOT a model-quality gate (those
are the bespoke retrieval/forced-choice/edit-local diagnostics). Run ad-hoc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _contexts(path: str) -> list[str]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(str(r.get("activation_text") or r.get("context") or ""))
    return out


def _summarize(result: object, name: str) -> dict:
    """Condition pass/fail summary from a SuiteResult."""
    passed = failed = warn = 0
    fails = []
    for cr in getattr(result, "results", []):
        for cond in getattr(cr, "conditions_results", []) or []:
            cat = getattr(cond.category, "name", str(cond.category))
            if cat == "PASS":
                passed += 1
            elif cat in ("WARN",):
                warn += 1
            else:
                failed += 1
                fails.append(f"{getattr(cr.check,'name',lambda:'?')()}: {cond.details}")
    return {"suite": name, "passed": passed, "warn": warn, "failed": failed,
            "failures": fails[:10]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/rune-corpus")
    ap.add_argument("--prefix", default="external_codereview")
    ap.add_argument("--out-dir", default="docs/superpowers/artifacts")
    ap.add_argument("--embeddings", action="store_true",
                    help="also compute embeddings drift (downloads a model)")
    a = ap.parse_args()

    from deepchecks.nlp import TextData
    from deepchecks.nlp.suites import data_integrity, train_test_validation

    splits = {s: f"{a.dir}/{a.prefix}.{s}.jsonl" for s in ("train", "val", "test")}
    tds = {}
    for name, path in splits.items():
        if not Path(path).exists():
            print(f"missing split: {path}")
            continue
        td = TextData(_contexts(path), task_type="text_classification")
        td.calculate_builtin_properties(include_long_calculation_properties=a.embeddings)
        if a.embeddings:
            td.calculate_builtin_embeddings()
        tds[name] = td
        print(f"loaded {name}: {td.n_samples} samples")

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    summary = {}

    di = data_integrity().run(tds["train"])
    di.save_as_html(f"{a.out_dir}/deepchecks_train_integrity.html")
    summary["train_integrity"] = _summarize(di, "data_integrity(train)")

    for split in ("val", "test"):
        if split not in tds:
            continue
        ttv = train_test_validation().run(tds["train"], tds[split])
        ttv.save_as_html(f"{a.out_dir}/deepchecks_train_{split}.html")
        summary[f"train_vs_{split}"] = _summarize(ttv, f"train_test_validation(train,{split})")

    Path(f"{a.out_dir}/deepchecks_corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
