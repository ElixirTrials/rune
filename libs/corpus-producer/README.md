# corpus-producer

Self-distillation pipeline that produces the 25-bin oracle training corpus used by the round-2 hypernetwork distillation loop.

## Purpose

Runs coding problems through the full 5-phase Rune pipeline (`scripts/rune_runner.py`) in subprocess mode, filters for successful completions, rationalizes phase artifacts into training records, and bins them by `(phase, benchmark)` into 25 oracle bins:

- `<phase>_<benchmark>` for 4 phases × 6 benchmarks (e.g., `code_humaneval`, `plan_mbpp`)
- `diagnose_pooled` for all diagnose-phase artifacts across benchmarks

Each bin yields one oracle adapter, registered in the `AdapterRegistry` with `id = "oracle_<bin_key>"` by `trainer_bridge.invoke_bin_training`. These oracle adapters are the teacher signals for round-2 hypernetwork distillation (see `libs/model-training` round-2 modules).

## Key Modules

| Module | Purpose |
|--------|---------|
| `pipeline_runner.py` | Subprocess wrapper around `scripts/rune_runner.py`; runs one `(benchmark, problem)` pair end-to-end and parses per-phase artifacts |
| `success_filter.py` | Filters phase artifacts by per-phase success criteria before training |
| `rationalization.py` | Converts phase artifacts into training-ready (prompt, completion) records |
| `binning.py` | `bin_artifacts` — groups artifacts into `<phase>_<benchmark>` / `diagnose_pooled` bins |
| `manifest.py` | Writes JSONL manifests per bin for downstream trainer consumption |
| `trainer_bridge.py` | `invoke_bin_training(bin_key, manifest_path)` — trains one oracle adapter with DeltaCoder warm-start defaults (rank=64, alpha=32, lr=2e-4, constant LR, diff-aware loss); sets the `oracle_<bin_key>` adapter ID |
| `s3_uploader.py` | `build_s3_key`, `upload_manifest` — optional S3 manifest upload; lazy `boto3` import; graceful degradation on missing credentials |
| `progress_db.py` | SQLite-backed progress tracker; shared across GPU shards so restarts are safe |
| `models.py` | `PhaseArtifact` dataclass — cross-module data contract |

## CLI Entry Point

`scripts/phase_corpus_producer.py`

### Flags

- `--out-dir PATH` — manifest output directory.
- `--shard IDX/TOTAL` — round-robin slice of problems for multi-GPU parallelism. Enabled by the `_parse_shard` + `apply_shard` helpers in the script.
- `--cuda-visible-devices DEVICES` — sets `CUDA_VISIBLE_DEVICES` in each subprocess pipeline run.
- `--s3-bucket NAME` / `--s3-prefix PREFIX` — optional S3 manifest upload (local manifest remains source of truth; S3 is a pure add-on).

### Multi-GPU example

```bash
for i in 0 1 2 3; do
    uv run scripts/phase_corpus_producer.py \
        --shard $i/4 --cuda-visible-devices $i \
        --out-dir data/phase_corpus &
done
wait
```

The shared `progress_db.py` tracker ensures shards do not duplicate work and restarts resume cleanly.

## Oracle Adapter ID Scheme

Adapter IDs are deterministic: `oracle_<bin_key>`. Examples:

- `oracle_decompose_humaneval`
- `oracle_plan_mbpp`
- `oracle_code_apps`
- `oracle_diagnose_pooled`

See `libs/adapter-registry/README.md` for the full adapter ID convention table.
