# Persistent MLflow + Adapter Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Survive dev-pod termination by replicating MLflow's `mlflow.db`, the adapter-registry's `rune.db`, and MLflow artifacts to S3 via Litestream + MLflow's `--default-artifact-root`.

**Architecture:** Run a single MLflow server (Docker) as the sole writer to `mlflow.db`. Bind-mount `${HOME}/.rune` (the HPO trainer's existing rune.db location) into a Litestream sidecar so the host trainer and the API/lora-server containers all touch the same SQLite file. Litestream replicates both DBs to `s3://elixirtrials-949678234935-eu-west-2-artifacts/`. MLflow artifacts go straight to S3 via `--default-artifact-root` + `--serve-artifacts`. Pod startup runs `litestream restore -if-replica-exists` once before MLflow boots; the trainer talks `http://localhost:5000`, never sqlite.

**Tech Stack:** Litestream 0.3.13, MLflow 2.16.2, Docker Compose v2, SQLite WAL, AWS S3 (eu-west-2), uv/Python 3.12.

---

## Revisions vs. original `instructions/persistent_mlflow.md`

1. **Bind-mount `${HOME}/.rune` instead of using a docker-managed volume for rune.db** — keeps host HPO and containerised services pointing at the same file. Original plan's `/data/rune/rune.db` only exists inside containers; HPO runs on host.
2. **Mount `~/.aws:/root/.aws:ro` into mlflow + litestream containers** — required because MLflow writes artifacts to S3 itself (`--serve-artifacts` proxy mode) and Litestream uploads SQLite WAL frames; both need credentials.
3. **Keep api/lora-server services writing the same rune.db file** (via the same bind mount) — preserves existing `DATABASE_URL=sqlite:////data/rune.db` semantics; the mount path is what changes.
4. **Add unit test for URI default flip** — `test_training_common.py` already exists; one extra test asserts the new default.
5. **Document single-writer caveat** in `libs/model-training/README.md` — the registry has multiple call-sites (`swarm.py`, `rune_runner.py`, training-svc) but Litestream tolerates concurrent SQLite writers since it reads the WAL; the real concern is two replicators competing, not two writers.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `infra/docker-compose.yml` | modify | Add `litestream` + `litestream-restore` services, switch mlflow backend to sqlite + S3 artifact root, bind-mount `${HOME}/.rune` |
| `infra/litestream/litestream.yml` | create | Litestream replica config for `mlflow.db` + `rune.db` |
| `libs/model-training/src/model_training/training_common.py` | modify | Default tracking URI: `sqlite:///./mlflow.db` → `http://localhost:5000` |
| `scripts/optimization/run_training_hpo.py` | modify | Same default flip at lines 728, 990 |
| `scripts/run_hpo.sh` | modify | Export `MLFLOW_TRACKING_URI`, `RUNE_DATABASE_URL`; pre-flight curl `/health` |
| `libs/model-training/tests/test_training_common.py` | modify | Add test asserting new default URI |
| `libs/model-training/README.md` | modify | Document `docker compose up -d mlflow litestream` prereq + single-writer caveat |

---

## Task 1: Litestream config

**Files:**
- Create: `infra/litestream/litestream.yml`

- [ ] **Step 1: Create litestream.yml**

```yaml
# Litestream replica config for Rune persistence.
# Watches the bind-mounted mlflow.db (single MLflow server is the sole writer)
# and the bind-mounted rune.db (HPO trainer + api/lora-server share this file).
# 1-second sync interval keeps RPO near real-time; retention 168h keeps a week
# of WAL history for point-in-time restores.
dbs:
  - path: /data/mlflow/mlflow.db
    replicas:
      - type: s3
        bucket: elixirtrials-949678234935-eu-west-2-artifacts
        path: mlflow/db
        region: eu-west-2
        sync-interval: 1s
        retention: 168h
  - path: /data/rune/rune.db
    replicas:
      - type: s3
        bucket: elixirtrials-949678234935-eu-west-2-artifacts
        path: rune-registry/db
        region: eu-west-2
        sync-interval: 1s
        retention: 168h
```

- [ ] **Step 2: Verify YAML parses**

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('infra/litestream/litestream.yml'))"`
Expected: exits 0 silently.

- [ ] **Step 3: Commit**

```bash
git add infra/litestream/litestream.yml
git commit -m "infra(litestream): add replica config for mlflow.db + rune.db"
```

---

## Task 2: docker-compose rewrite

**Files:**
- Modify: `infra/docker-compose.yml` (full rewrite of the `mlflow` service + add `litestream` + `litestream-restore`; api/lora-server volume change)

- [ ] **Step 1: Replace mlflow service block**

Replace the existing block (lines 45–56 today):
```yaml
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlflow
    volumes:
      - mlflow_data:/mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
```

with:
```yaml
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    ports:
      - "5000:5000"
    environment:
      AWS_DEFAULT_REGION: eu-west-2
    command: >
      mlflow server
      --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////data/mlflow/mlflow.db
      --default-artifact-root s3://elixirtrials-949678234935-eu-west-2-artifacts/mlflow/artifacts/
      --serve-artifacts
    volumes:
      - mlflow_data:/data/mlflow
      - ${HOME}/.aws:/root/.aws:ro
    depends_on:
      litestream-restore:
        condition: service_completed_successfully
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
```

- [ ] **Step 2: Add litestream-restore + litestream services**

Add at the end of the `services:` block (before `volumes:`):

```yaml
  # One-shot restore: pulls latest replica from S3 before mlflow boots.
  # `service_completed_successfully` ensures the server starts only after
  # this exits 0. Idempotent — `-if-replica-exists` no-ops on first ever
  # boot when S3 prefix is empty.
  litestream-restore:
    image: litestream/litestream:0.3.13
    environment:
      AWS_DEFAULT_REGION: eu-west-2
    volumes:
      - mlflow_data:/data/mlflow
      - ${HOME}/.rune:/data/rune
      - ./litestream/litestream.yml:/etc/litestream.yml:ro
      - ${HOME}/.aws:/root/.aws:ro
    entrypoint: ["sh", "-c"]
    command:
      - |
        mkdir -p /data/mlflow /data/rune
        litestream restore -if-replica-exists -config /etc/litestream.yml /data/mlflow/mlflow.db
        litestream restore -if-replica-exists -config /etc/litestream.yml /data/rune/rune.db

  # Continuous WAL shipping. Watches both DBs; tolerates the trainer
  # process writing rune.db on the host because the bind mount exposes
  # the same inode.
  litestream:
    image: litestream/litestream:0.3.13
    restart: unless-stopped
    environment:
      AWS_DEFAULT_REGION: eu-west-2
    volumes:
      - mlflow_data:/data/mlflow
      - ${HOME}/.rune:/data/rune
      - ./litestream/litestream.yml:/etc/litestream.yml:ro
      - ${HOME}/.aws:/root/.aws:ro
    depends_on:
      litestream-restore:
        condition: service_completed_successfully
    command: replicate -config /etc/litestream.yml
```

- [ ] **Step 3: Update api + lora-server services to use the new rune.db path**

Change `api.environment.DATABASE_URL` from `sqlite:////data/rune.db` to `sqlite:////data/rune/rune.db` and `api.volumes` from `- rune_data:/data` to `- ${HOME}/.rune:/data/rune`.

Apply the identical volume swap to `lora-server.volumes` (`rune_data:/data` → `${HOME}/.rune:/data/rune`).

Remove `rune_data:` from the bottom-level `volumes:` block (no longer used).

- [ ] **Step 4: Verify compose config parses**

Run: `docker compose -f infra/docker-compose.yml config --quiet`
Expected: exits 0 with no output.

- [ ] **Step 5: Commit**

```bash
git add infra/docker-compose.yml
git commit -m "infra(compose): mlflow→sqlite+S3, litestream sidecar, bind ~/.rune"
```

---

## Task 3: Default tracking URI flip — training_common.py

**Files:**
- Modify: `libs/model-training/src/model_training/training_common.py:61`
- Test: `libs/model-training/tests/test_training_common.py`

- [ ] **Step 1: Add failing test**

Append to `libs/model-training/tests/test_training_common.py`:

```python
def test_setup_mlflow_default_uri_is_local_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither arg nor env override the URI, default is the in-pod server."""
    monkeypatch.delenv("RUNE_DISABLE_MLFLOW", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    captured: dict[str, str] = {}

    class _FakeMlflow:
        @staticmethod
        def active_run() -> None:
            return None

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            captured["uri"] = uri

        @staticmethod
        def set_experiment(name: str) -> None:
            captured["experiment"] = name

        @staticmethod
        def get_tracking_uri() -> str:
            return captured.get("uri", "")

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow)

    from model_training.training_common import setup_mlflow

    assert setup_mlflow("exp-x", None) is True
    assert captured["uri"] == "http://localhost:5000"
    assert captured["experiment"] == "exp-x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest libs/model-training/tests/test_training_common.py::test_setup_mlflow_default_uri_is_local_server -x -q`
Expected: FAIL with `assert 'sqlite:///./mlflow.db' == 'http://localhost:5000'`.

- [ ] **Step 3: Flip the default in training_common.py**

Change line 61 from:
```python
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
```
to:
```python
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
```

Update the docstring on lines 26–30:
```python
    Tracking URI precedence: explicit ``tracking_uri`` arg, then the
    ``MLFLOW_TRACKING_URI`` env var, then ``http://localhost:5000`` (the
    in-pod MLflow server started by ``infra/docker-compose.yml``). The
    sqlite/filesystem backends were dropped from the default path because
    they cannot be replicated to S3 by Litestream when multiple processes
    open the file concurrently.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest libs/model-training/tests/test_training_common.py -x -q`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Commit**

```bash
git add libs/model-training/src/model_training/training_common.py \
        libs/model-training/tests/test_training_common.py
git commit -m "feat(mlflow): default tracking URI → http://localhost:5000"
```

---

## Task 4: Default tracking URI flip — run_training_hpo.py

**Files:**
- Modify: `scripts/optimization/run_training_hpo.py:728` and `:990`

- [ ] **Step 1: Update the trial-level fallback**

Change line 728 from:
```python
        or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
```
to:
```python
        or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
```

- [ ] **Step 2: Update the study-summary fallback**

Change line 990 from:
```python
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
```
to:
```python
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
```

- [ ] **Step 3: Run lint + typecheck on the file**

Run: `uv run ruff check scripts/optimization/run_training_hpo.py && uv run mypy scripts/optimization/run_training_hpo.py`
Expected: 0 errors. (Mypy may emit pre-existing warnings; only fail on new ones.)

- [ ] **Step 4: Commit**

```bash
git add scripts/optimization/run_training_hpo.py
git commit -m "feat(hpo): default MLflow tracking URI → http://localhost:5000"
```

---

## Task 5: run_hpo.sh — env exports + pre-flight

**Files:**
- Modify: `scripts/run_hpo.sh`

- [ ] **Step 1: Add env exports + pre-flight after the existing prereq block**

Insert between the existing `[[ -f "$DATASET" ]] || ...` line (~line 53) and the `mkdir -p "${HOME}/.rune"` line (~line 57):

```bash
# ── persistence: route MLflow + AdapterRegistry through the docker stack ───
# HPO runs on the host but writes go to the in-pod MLflow server (which
# Litestream backs up to S3) and to the bind-mounted rune.db that Litestream
# also watches. If the user opted out via env, respect their override.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export RUNE_DATABASE_URL="${RUNE_DATABASE_URL:-sqlite:///${HOME}/.rune/rune.db}"

# Pre-flight only when the URI looks HTTP — sqlite:// fallbacks (used by
# users who explicitly want local-only) skip the curl check.
if [[ "$MLFLOW_TRACKING_URI" =~ ^https?:// ]]; then
    if ! curl -fsS --max-time 2 "${MLFLOW_TRACKING_URI%/}/health" >/dev/null; then
        echo "MLflow server not reachable at $MLFLOW_TRACKING_URI" >&2
        echo "Start the stack first:  docker compose -f infra/docker-compose.yml up -d mlflow litestream" >&2
        exit 1
    fi
fi
```

- [ ] **Step 2: Update the URI echo line**

Change the existing line:
```bash
echo "MLflow URI:     ${MLFLOW_TRACKING_URI:-sqlite:///./mlflow.db}"
```
to:
```bash
echo "MLflow URI:     ${MLFLOW_TRACKING_URI}"
```

- [ ] **Step 3: Shellcheck the file**

Run: `command -v shellcheck >/dev/null && shellcheck scripts/run_hpo.sh || echo "shellcheck unavailable; skipping"`
Expected: 0 errors, or "shellcheck unavailable; skipping".

- [ ] **Step 4: Dry-run of the script's prereq path**

Run: `MLFLOW_TRACKING_URI=http://127.0.0.1:9 bash scripts/run_hpo.sh --smoke --dataset /dev/null 2>&1 | grep -F "MLflow server not reachable" >/dev/null && echo OK`
Expected: prints `OK` (the curl fails fast, the script exits 1 with the expected message).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_hpo.sh
git commit -m "feat(hpo): export MLFLOW_TRACKING_URI + RUNE_DATABASE_URL, add MLflow pre-flight"
```

---

## Task 6: Operator docs

**Files:**
- Modify: `libs/model-training/README.md`

- [ ] **Step 1: Read current README to find the right insertion point**

Run: `grep -n "^#" libs/model-training/README.md | head -20`
Expected: a list of headings — pick the section before "Training" or after "Setup" / "Prerequisites" (whichever exists).

- [ ] **Step 2: Add a "Persistence" section**

Append the following at the end of the README (or under a "Setup" heading if one exists):

```markdown
## Persistence (MLflow + AdapterRegistry)

The HPO trainer writes runs to MLflow and adapter records to SQLite. Both
are replicated to S3 by a Litestream sidecar so a pod termination doesn't
lose state.

**Prereq before any HPO/training run:**

```bash
docker compose -f infra/docker-compose.yml up -d mlflow litestream
```

This boots the MLflow server (sqlite-backed at `/data/mlflow/mlflow.db`,
artifacts at `s3://elixirtrials-949678234935-eu-west-2-artifacts/mlflow/artifacts/`)
plus a Litestream container that ships both `mlflow.db` and the host's
`~/.rune/rune.db` to S3 every second.

**MLflow tracking URI default:** `http://localhost:5000`. Override with
`MLFLOW_TRACKING_URI=…` to talk to a different server (CI, remote pod).
The legacy `sqlite:///./mlflow.db` fallback was removed — sqlite backends
are not safe for concurrent writers and don't survive pod termination.

**Adapter registry DB default:** `sqlite:///${HOME}/.rune/rune.db`.
Override with `RUNE_DATABASE_URL=…`.

**Single-writer caveat:** Litestream replicates SQLite WAL frames; this
is safe even when several processes write the same DB (the trainer plus
the api/lora-server containers share `~/.rune/rune.db` via bind mount).
What is *not* safe is running two Litestream containers against the same
S3 prefix. The compose file enforces a single replicator.

**Cleanup:** `docker compose down` does not delete S3 data. Use
`docker volume rm` to drop the local `mlflow_data` volume; S3 replicas
remain available for subsequent restores.
```

- [ ] **Step 3: Commit**

```bash
git add libs/model-training/README.md
git commit -m "docs(model-training): document persistent MLflow + Litestream setup"
```

---

## Task 7: End-to-end verification (manual, post-merge)

Not automatable inside this session because it needs Docker + AWS creds + a GPU. The smoke checklist that owners should run after merging:

```bash
# 1. Boot stack
docker compose -f infra/docker-compose.yml up -d mlflow litestream

# 2. Run a smoke study (creates 2 trials + adapter records)
bash scripts/run_hpo.sh --smoke --dataset data/pairs/fastapi_fastapi.jsonl \
    --output-root /tmp/hpo_persist_test --experiment-name persist-survive

# 3. Confirm S3 replicas + artifacts exist
aws s3 ls s3://elixirtrials-949678234935-eu-west-2-artifacts/mlflow/db/
aws s3 ls s3://elixirtrials-949678234935-eu-west-2-artifacts/mlflow/artifacts/
aws s3 ls s3://elixirtrials-949678234935-eu-west-2-artifacts/rune-registry/db/

# 4. Wipe local volumes — simulate pod termination
docker compose -f infra/docker-compose.yml down
docker volume rm rune_mlflow_data 2>/dev/null || true

# 5. Boot again — restore should pull from S3
docker compose -f infra/docker-compose.yml up -d mlflow litestream

# 6. Confirm experiments + adapters survived
curl -s http://localhost:5000/api/2.0/mlflow/experiments/search \
    -H 'Content-Type: application/json' -d '{}' | jq -r '.experiments[].name'
# expect: includes "persist-survive"

uv run python -c "
from sqlalchemy import create_engine
from sqlmodel import Session, select
from adapter_registry.models import AdapterRecord
e = create_engine('sqlite:///' + str(__import__('pathlib').Path.home() / '.rune' / 'rune.db'))
with Session(e) as s:
    print(f\"{len(list(s.exec(select(AdapterRecord)).all()))} adapters\")
"
# expect: ≥2
```

Pass criterion: step 6 returns the smoke experiment + ≥2 adapters from a
freshly-restored local volume.

---

## Risks & rollback

- **Litestream lag (≤1s):** bounded RPO. Acceptable for dev.
- **Restore-on-empty races:** prevented by `service_completed_successfully` dependency.
- **Concurrent dev pods:** unsupported (would mean two Litestream containers writing the same S3 prefix). Pick one pod as the writer.
- **AWS creds outage:** Litestream buffers locally and retries; trainer keeps writing to local SQLite. Data loss only if the pod terminates *during* the outage.
- **Rollback:** revert the 6 commits above. Local SQLite + filesystem artifacts work as today. S3 data isn't touched by rollback — available for forensics or re-import.

## NOT in scope

- Migrating existing `./mlruns/` filesystem data — pre-fix, unrepresentative.
- Postgres backend — defer until concurrent pods are needed.
- SSE-KMS — bucket already has SSE-S3.
- Retention beyond 7 days — bump `retention:` in `litestream.yml` later if needed.
