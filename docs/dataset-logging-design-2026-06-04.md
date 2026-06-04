# Dataset logging for reproducible experiments (MLflow `mlflow.data`)

**Why this doc exists:** the `external_codereview.val.clean.jsonl` corpus that gate-1 retention
needs was a **derived split that was never logged anywhere** — only in ephemeral `/tmp/rune-corpus/`,
never an MLflow artifact, and `run.inputs.dataset_inputs == []` on every run that used it (dataset
tracking was never wired up). It was recoverable only because the *raw* upstream
`external_codereview.unrolled.jsonl` happened to survive on S3
(`s3://…us-east-1…/training-data/github-pairs/`) — staged there by hand, not by any logging
discipline — from which the splits regenerate deterministically (`split_corpus.py` →
`corpus_split_qc.py`). **Recovered + made durable 2026-06-04** (splits now at
`…/github-pairs/splits/`). This is a process gap, not a tooling limit, and the next dataset may not
have a surviving raw source. MLflow 3.13 (installed) has everything we need to make every dataset
reproducible by lineage rather than by luck.

## What MLflow gives us (verified against installed 3.13.0)

Two complementary layers:

### 1. Classic `mlflow.data` + `log_input` — for TRAINING/EVAL corpora (the one we need now)
Each run records its input datasets as first-class lineage: name, **content digest**, source URI,
schema, profile, and a `context` ("training"/"validation"/"test"). Retrievable + reloadable later.

```python
import mlflow, mlflow.data, pandas as pd
df = pd.read_json("external_codereview.val.clean.jsonl", lines=True)
ds = mlflow.data.from_pandas(
    df,
    source="s3://elixirtrials-…-artifacts/datasets/external_codereview/<sha>.jsonl",  # durable, not /tmp
    name="external_codereview.val",
    digest="<sha256-of-bytes>",        # explicit → identical file always yields identical digest
)
with mlflow.start_run():
    mlflow.log_input(ds, context="validation")

# later, on any fresh instance:
run = mlflow.get_run(run_id)
logged = run.inputs.dataset_inputs[0].dataset       # name, digest, source recorded
src   = mlflow.data.get_source(logged)
local = src.load()                                   # downloads the bytes back from S3
```
`from_pandas(df, source, targets=None, name=None, digest=None, predictions=None)` — `source` accepts
any URI string (`s3://`, `http(s)://`, local path) or a `DatasetSource`. `mlflow.data.get_source(ds)`
+ `.load()` round-trips the bytes.

### 2. `mlflow.genai.datasets.EvaluationDataset` — for GOLDEN EVAL SETS (the multi-turn engine eval)
Versioned datasets attached to an **experiment** (not a single run), SQL-backed, built by
`create_dataset()` / `get_dataset()` / `merge_records()` (from traces, dicts, or DataFrames). Right
tool for the held-out engine-eval golden tasks and regression "golden sets" — curate once, reuse
across app/checkpoint versions. Second phase; not needed to unblock gate-1.

## The reproducibility traps and how we handle them

1. **`log_input` records a POINTER, not the bytes.** The source is an `s3://` URI; if that object is
   later deleted/moved, `mlflow.data.get_source(ds).load()` fails — the failure mode we just hit.
   Mitigation: keep the canonical corpus on a durable S3 prefix and **prefer content-addressed /
   immutable paths** so a logged URI never silently changes underneath a run. (We deliberately do
   *not* `log_artifact` the bytes on every run — that would duplicate a ~90MB corpus into S3 per HPO
   trial; reserve byte-attachment for small one-off eval sets if ever needed.)
2. **Digest choice: use MLflow's built-in `MetaDataset` digest.** It is derived from the source URI
   + name (not the file bytes), so it is stable per-URI across machines and requires reading nothing
   — correct for large corpora. The tradeoff: content changing at the *same* path does not change
   the digest, which is exactly why trap #1's content-addressed-path guidance matters. (Earlier
   drafts hand-rolled a sha256-of-bytes digest; dropped per KISS/DRY — don't write custom code where
   MLflow's own digest suffices.)
3. **Derived-split determinism is library-version-dependent.** `corpus_split_qc.py`'s near-dup
   "clean" step uses scikit-learn TF-IDF; its defaults can change across sklearn releases. **Record
   the sklearn version (and the source `unrolled` digest) alongside each derived digest** so a
   future regen is verifiably identical. Reference for the current recovery: source
   `external_codereview.unrolled.jsonl` sha256 `4931fe03…` (7,670 rows) + sklearn **1.9.0** →
   `val.clean` 323 rows sha256 `7e3692df…`, `test.clean` 343 rows sha256 `744715a6…`.

## Implemented wiring (minimal — thin wrapper over MLflow, KISS/DRY)

Every entrypoint already funnels through `src/rune/tracking.py` (`configure_mlflow` + `tracked_run`).
The implementation is one **thin** helper that delegates entirely to MLflow's own dataset objects —
no hand-rolled digest, no byte-reading, no directory walking (those would be custom code where
MLflow suffices). `MetaDataset` is the documented metadata-only primitive: it records the source +
**MLflow's own content digest by reference**, so it never reads the data and is safe for the ~90MB
train corpus.

**`src/rune/tracking.py` — `log_dataset` (shipped):**
```python
def log_dataset(uri: str | Path, *, name: str, context: str) -> str:
    """Log a dataset (file/dir/S3 URI) as an MLflow input so the run is reproducible.
    A metadata-only MetaDataset records the source + MLflow's content digest by
    reference — it does not read the data, so it is safe for large corpora."""
    dataset = MetaDataset(source=resolve_dataset_source(str(uri)), name=name)
    mlflow.log_input(dataset, context=context)
    return dataset.digest
```
Digest decision: **use MLflow's built-in `MetaDataset` digest** (derived from source URI + name),
not a hand-computed sha256. It is stable per-URI across machines; content-versioning is the URI's
job (prefer content-addressed/immutable paths). Tested in `tests/unit/test_tracking.py` (real
round-trip on a sqlite backend: asserts name, `source_type`, source, returned digest, and the
`mlflow.data.context` tag).

**Call sites (shipped, one line each inside the existing `tracked_run`):**
- `train` → `log_dataset(corpus_dir, name=corpus_dir.name, context="training")`
- `bench` (both the `bench-hpo` and plain `bench` runs) →
  `log_dataset(tasks_file, name=tasks_file.name, context="test")`

**Backfill (done 2026-06-04):** MLflow experiment `corpus-registry`, run
`register-external_codereview-2026-06-04` (`ea4f3c43…`) logs the recovered splits by their durable
us-east-1 S3 URIs — `unrolled`, `train` (training), `val.clean` (validation), `test.clean` (test) —
so the gate-1 corpus has lineage and is never floating only in `/tmp` again.

**Convention going forward:** no run trains/evals on a dataset that isn't `log_input`-ed; pass the
durable S3 URI (not `/tmp/rune-corpus/…`) as the `uri`. The `diag_*`/`gate_*` tools should default
their `--val` to the canonical S3 URI so a fresh instance reproduces with zero manual staging.

## Net
- **Lineage:** every run answers "exactly which data" — `run.inputs.dataset_inputs` (name, source,
  digest, context).
- **Recovery:** `mlflow.data.get_source(ds).load()` pulls the bytes back from the S3 source URI.
- **Version key:** MLflow's `MetaDataset` digest, stable per-URI across machines.
- **Eval golden sets:** `mlflow.genai.datasets` for the multi-turn engine eval (phase 2) — separate,
  not needed here.
