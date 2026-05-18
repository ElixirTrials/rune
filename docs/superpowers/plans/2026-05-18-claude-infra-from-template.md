# Claude Infrastructure from Template — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap `.claude/` infrastructure (hooks, skills, commands, settings) from the ElixirTrials Template into rune-gpu, adapting for an ML training pipeline (not medtech/frontend).

**Architecture:** The Template at `/workspaces/Template` provides a manifest-based sync system. We'll do a selective first-time import — copying relevant files, adapting medtech/frontend references, relaxing test-suite restrictions to match rune's fast test suite, and wiring the sync script for future updates.

**Tech Stack:** Bash hooks, Python guard script, Claude Code settings.json, Makefile targets.

---

## File Structure

### New files to create:

| Path | Responsibility |
|------|---------------|
| `.claude/settings.json` | Hook wiring, model default, env vars |
| `.claude/hooks/guard.py` | PreToolUse guard (blocks mutating git, long-running ops, dangerous writes/reads) |
| `.claude/hooks/post-edit-lint.sh` | PostToolUse auto-lint (ruff for .py) |
| `.claude/hooks/session-start.sh` | SessionStart PRODUCT.md stub warning + CRG refresh |
| `.claude/skills-index.md` | Keyword→skill trigger table |
| `.claude/.template-manifest` | Manifest for future `make claude-sync` |
| `.claude/skills/plan-first/SKILL.md` | Multi-file change planning protocol |
| `.claude/skills/commit-prep/SKILL.md` | Conventional Commits message generator |
| `.claude/skills/pr-description/SKILL.md` | PR title+body from branch diff |
| `.claude/skills/debug-systematic/SKILL.md` | Reproduce-then-hypothesize debugging |
| `.claude/skills/senior-architect/SKILL.md` | CTO-level architecture review (Opus) |
| `.claude/skills/senior-devops/SKILL.md` | Infrastructure/deploy design (Opus) |
| `.claude/skills/risk-classifier/SKILL.md` | Feature blast-radius classification |
| `.claude/skills/long-run-fallback/SKILL.md` | Protocol for blocked long-running commands |
| `.claude/skills/codebase-onboarding/SKILL.md` | First-time repo orientation |
| `.claude/skills/alembic-safe/SKILL.md` | Safe migration generation |
| `.claude/skills/langgraph-debug/SKILL.md` | LangGraph agent debugging (adapted for rune-agent) |
| `.claude/skills/fastapi-test-fixture/SKILL.md` | FastAPI endpoint test recipe (adapted for rune services) |
| `.claude/skills/compliance-check/SKILL.md` | Compliance review (Opus) |
| `.claude/commands/architect.md` | `/architect` slash command |
| `.claude/commands/commit.md` | `/commit` slash command |
| `.claude/commands/compact-safe.md` | `/compact-safe` slash command |
| `.claude/commands/compliance.md` | `/compliance` slash command |
| `.claude/commands/context.md` | `/context` slash command |
| `.claude/commands/devops.md` | `/devops` slash command |
| `.claude/commands/errors.md` | `/errors` slash command |
| `.claude/commands/onboard.md` | `/onboard` slash command |
| `.claude/commands/opus.md` | `/opus` slash command |
| `.claude/commands/plan.md` | `/plan` slash command |
| `.claude/commands/pr.md` | `/pr` slash command |
| `.claude/commands/quiet.md` | `/quiet` slash command |
| `.claude/commands/risk.md` | `/risk` slash command |
| `.claude/commands/sonnet.md` | `/sonnet` slash command |
| `.claude/commands/sync-template.md` | `/sync-template` slash command |
| `.claude/commands/tail.md` | `/tail` slash command |
| `scripts/sync-claude-template.sh` | Template sync mechanism |
| `scripts/setup-git-hooks.sh` | Git hooks wiring |

### Files to modify:

| Path | Change |
|------|--------|
| `Makefile` | Add `claude-sync`, `claude-crg-enable/disable` targets |

### Files NOT touched (and why):

| Path | Reason |
|------|--------|
| `CLAUDE.md` | Already comprehensive and customized for rune-gpu |
| `.claudeignore` | Already customized |
| `PRODUCT.md` | Already exists; sync script preserves it |
| `.github/workflows/ci.yml` | Already customized for rune-gpu |
| `.githooks/post-commit` | Already exists |

### Files deliberately SKIPPED from Template (and why):

| Template path | Reason |
|------|--------|
| `.claude/skills/ux-pathway/` | No frontend/UI in rune-gpu |
| `.claude/skills/frontend-design/` | No React/Vite/Radix in rune-gpu |
| `.claude/skills/event-trace/` | No Pub/Sub emulator; rune uses EventBridge+SQS via AWS |
| `.claude/commands/ux.md` | No UX pathway skill to invoke |
| `Template CLAUDE.md` | Our CLAUDE.md is already adapted |
| `Template .claudeignore` | Our .claudeignore is already adapted |
| `scripts/check-all.sh` | We already have quality checks in CLAUDE.md and Makefile |
| `scripts/create-service.sh` | Already exists in rune-gpu |
| `scripts/bootstrap-claude.sh` | One-shot bootstrapper; we're doing a selective import |
| Various doc scripts | Already exist in rune-gpu |

---

## Adaptations from Template

### guard.py — 2 changes:
1. **Remove pytest full-suite block.** Rune's test suite runs in ~30s; CLAUDE.md explicitly documents `uv run pytest` as OK. The Template blocks bare `pytest` to force per-file runs in large suites — not needed here.
2. **Remove vitest/playwright patterns.** No JS/TS testing in rune-gpu.

### skills-index.md — 3 changes:
1. Remove frontend-design, ux-pathway, event-trace entries.
2. Rename "Stack-specific (ElixirTrials)" → "Stack-specific (Rune)".
3. Update langgraph-debug trigger to reference `rune-agent` instead of `agent-a/b-service`.

### risk-classifier SKILL.md — adapt tiers:
Replace medtech tiers (clinical-adjacent, clinical-direct, safety-critical) with ML-pipeline tiers (model-adjacent, training-direct, data-critical).

### langgraph-debug SKILL.md — update paths:
- `services/agent-{a,b}-service` → `services/rune-agent`

### fastapi-test-fixture SKILL.md — update paths:
- `services/api-service` → rune's services (`training-svc`, `api-service`)
- Remove PHI-specific test assertions

### settings.json — 2 changes:
1. Remove Vanta MCP placeholder (not relevant to ML research pipeline).
2. Keep CRG MCP notes (already set up via code-review-graph in this repo).

---

### Task 1: Create hooks infrastructure

**Files:**
- Create: `.claude/hooks/guard.py`
- Create: `.claude/hooks/post-edit-lint.sh`
- Create: `.claude/hooks/session-start.sh`

- [ ] **Step 1: Create `.claude/hooks/` directory**

```bash
mkdir -p .claude/hooks
```

- [ ] **Step 2: Create guard.py (adapted)**

Copy from `/workspaces/Template/.claude/hooks/guard.py` with these changes:
- Remove the pytest full-suite block pattern (lines 59-63 in template)
- Remove the vitest block pattern (lines 64-66)
- Remove the playwright block pattern (lines 67-70)
- Keep everything else (mutating git, rm -rf, force ops, destructive SQL, dev servers, builds/installs, migrations, compose, write blocks, read blocks, model-for-skill check)

The resulting `BASH_BLOCK` list should NOT contain:
```python
# REMOVE these three patterns:
(r"\bpytest\b(?!.*(::|\s[\w/.\-]+\.py))", "..."),
(r"\bvitest\b(?!\s+run\s+\S+\.[tj]sx?\b)(?!\s+--version)(?!\s+-h)", "..."),
(r"\bplaywright\s+test\b(?!.*--list)", "..."),
```

- [ ] **Step 3: Create post-edit-lint.sh (as-is from template)**

Copy from `/workspaces/Template/.claude/hooks/post-edit-lint.sh` verbatim. The biome fallback for JS/TS is a harmless no-op in rune-gpu (no TS files to lint).

- [ ] **Step 4: Create session-start.sh (as-is from template)**

Copy from `/workspaces/Template/.claude/hooks/session-start.sh` verbatim. Warns about PRODUCT.md stubs and refreshes CRG index.

- [ ] **Step 5: Make hooks executable**

```bash
chmod +x .claude/hooks/guard.py .claude/hooks/post-edit-lint.sh .claude/hooks/session-start.sh
```

---

### Task 2: Create settings.json

**Files:**
- Create: `.claude/settings.json`

- [ ] **Step 1: Create settings.json (adapted)**

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "model": "opus",
  "env": {
    "MAX_THINKING_TOKENS": "10000"
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": ".claude/hooks/guard.py bash" }
        ]
      },
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          { "type": "command", "command": ".claude/hooks/guard.py write" }
        ]
      },
      {
        "matcher": "Read",
        "hooks": [
          { "type": "command", "command": ".claude/hooks/guard.py read" }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          { "type": "command", "command": ".claude/hooks/post-edit-lint.sh" }
        ]
      }
    ],
    "SessionStart": [
      { "hooks": [ { "type": "command", "command": ".claude/hooks/session-start.sh" } ] }
    ]
  },
  "_mcpServers_NOTE_crg": "code-review-graph MCP. Per-repo install: `pipx install code-review-graph==2.3.2` then `code-review-graph build` once. Then move the example block into mcpServers below. Skills (pr-description, senior-architect, debug-systematic) will use it automatically.",
  "_mcpServers_example_crg": {
    "code-review-graph": {
      "command": "code-review-graph",
      "args": ["serve", "--repo", "."]
    }
  }
}
```

---

### Task 3: Create core workflow skills (plan-first, commit-prep, pr-description, debug-systematic)

**Files:**
- Create: `.claude/skills/plan-first/SKILL.md`
- Create: `.claude/skills/commit-prep/SKILL.md`
- Create: `.claude/skills/pr-description/SKILL.md`
- Create: `.claude/skills/debug-systematic/SKILL.md`

- [ ] **Step 1: Create plan-first skill**

Copy from `/workspaces/Template/.claude/skills/plan-first/SKILL.md` verbatim. No adaptation needed — the protocol is generic.

- [ ] **Step 2: Create commit-prep skill**

Copy from `/workspaces/Template/.claude/skills/commit-prep/SKILL.md` verbatim. Remove the PHI/medtech citation line from the rules:
```
- If the diff touches PHI handling, auth, audit logs, or migrations: add a `Refs:` line citing the relevant PRODUCT.md section or compliance control.
```
Replace with:
```
- If the diff touches model training, evaluation gates, or kill switches: add a `Refs:` line noting the affected safety mechanism.
```

- [ ] **Step 3: Create pr-description skill**

Copy from `/workspaces/Template/.claude/skills/pr-description/SKILL.md` with one change:
- In the body template, replace `## PRODUCT.md / regulatory` section with:
```markdown
## Impact
<Which pipeline phases, training components, or evaluation gates are affected, or "n/a — internal refactor".>
```
- Replace the PHI/auth/migrations rule with:
```
- If the PR touches model training, loss functions, evaluation gates, or kill switches, the Impact section is required, not optional.
```

- [ ] **Step 4: Create debug-systematic skill**

Copy from `/workspaces/Template/.claude/skills/debug-systematic/SKILL.md` verbatim. The debugging protocol is generic and applies perfectly to ML pipeline debugging.

---

### Task 4: Create architecture & review skills (senior-architect, senior-devops, risk-classifier, compliance-check)

**Files:**
- Create: `.claude/skills/senior-architect/SKILL.md`
- Create: `.claude/skills/senior-devops/SKILL.md`
- Create: `.claude/skills/risk-classifier/SKILL.md`
- Create: `.claude/skills/compliance-check/SKILL.md`

- [ ] **Step 1: Create senior-architect skill**

Copy from `/workspaces/Template/.claude/skills/senior-architect/SKILL.md` verbatim. The review framework is generic and valuable for rune's pipeline architecture.

- [ ] **Step 2: Create senior-devops skill**

Copy from `/workspaces/Template/.claude/skills/senior-devops/SKILL.md` verbatim. Rune uses the same AWS stack (ECS Fargate, RDS, S3, SageMaker).

- [ ] **Step 3: Create risk-classifier skill (adapted)**

Copy from `/workspaces/Template/.claude/skills/risk-classifier/SKILL.md` with adapted tiers:

```markdown
---
name: risk-classifier
description: Use early when the user describes a new feature, spec, or requirement. Classifies blast radius and names which downstream skills/reviews are mandatory.
---

# Risk Classifier

Output a tier and a list of required follow-ups. One paragraph max.

## Tiers

- **Trivial** — UI tweak, copy change, single-file refactor with tests. No downstream required.
- **Model-adjacent** — touches infra, data shape, config, or pipeline orchestration but not the training/inference path itself. Required: `code-review-self`.
- **Training-direct** — feature changes model training, loss functions, adapter merging, evaluation logic, or inference behavior. Required: `senior-architect`, targeted test coverage for the changed component.
- **Data-critical** — could corrupt adapters, lose training data, produce wrong model outputs, or break evaluation gates/kill switches. Required: all of training-direct + dual review + explicit rollback plan. Stop and ask the user to confirm scope before any code.

## Protocol

1. Ask for the smallest concrete description of the feature if not already given.
2. Read PRODUCT.md if it has content; check CLAUDE.md for relevant safety mechanisms (kill switches, evaluation gates).
3. Pick the highest-applicable tier (one tier, not "between"). State the reason in one line.
4. List required downstream skills + slash commands the user should invoke.
5. List the single biggest risk if we proceed without those steps.

## Output format
```
Tier: <name>
Why: <one line>
Required: <skill> (`/cmd`), <skill> (`/cmd`), ...
Biggest risk if skipped: <one line>
```
```

- [ ] **Step 4: Create compliance-check skill**

Copy from `/workspaces/Template/.claude/skills/compliance-check/SKILL.md` verbatim. Even though rune-gpu isn't medtech, the skill's framework for checking data handling, secrets, and audit trails is valuable. PRODUCT.md's regulatory surface will guide scope (likely N/A for most checks in this repo).

---

### Task 5: Create utility skills (long-run-fallback, codebase-onboarding, alembic-safe)

**Files:**
- Create: `.claude/skills/long-run-fallback/SKILL.md`
- Create: `.claude/skills/codebase-onboarding/SKILL.md`
- Create: `.claude/skills/alembic-safe/SKILL.md`

- [ ] **Step 1: Create long-run-fallback skill**

Copy from `/workspaces/Template/.claude/skills/long-run-fallback/SKILL.md` verbatim.

- [ ] **Step 2: Create codebase-onboarding skill**

Copy from `/workspaces/Template/.claude/skills/codebase-onboarding/SKILL.md` verbatim.

- [ ] **Step 3: Create alembic-safe skill**

Copy from `/workspaces/Template/.claude/skills/alembic-safe/SKILL.md` verbatim. Rune uses Postgres and may use Alembic migrations.

---

### Task 6: Create stack-specific skills (langgraph-debug, fastapi-test-fixture)

**Files:**
- Create: `.claude/skills/langgraph-debug/SKILL.md`
- Create: `.claude/skills/fastapi-test-fixture/SKILL.md`

- [ ] **Step 1: Create langgraph-debug skill (adapted)**

Copy from `/workspaces/Template/.claude/skills/langgraph-debug/SKILL.md` with path changes:
- Replace `services/agent-{a,b}-service` → `services/rune-agent`
- Replace `services/agent-{a,b}-service/agent/graph.py` → `services/rune-agent` (check actual graph.py location)
- Replace the escalation section:
  - "Two agents (a + b) diverging" → "Pipeline phases producing inconsistent results"
  - "Performance regression > 2x" → kept as-is

- [ ] **Step 2: Create fastapi-test-fixture skill (adapted)**

Copy from `/workspaces/Template/.claude/skills/fastapi-test-fixture/SKILL.md` with changes:
- Replace `services/api-service/tests/conftest.py` → note to check actual conftest in the target service (`services/training-svc/tests/conftest.py` or `services/api-service/tests/conftest.py`)
- Remove the PHI-specific testing rule:
  ```
  - For PHI-bearing endpoints: also assert that the response does **not** echo PHI fields the caller shouldn't see.
  ```
- Keep the rest (async client pattern, fixture reference, naming conventions)

---

### Task 7: Create slash commands

**Files:**
- Create: `.claude/commands/architect.md`
- Create: `.claude/commands/commit.md`
- Create: `.claude/commands/compact-safe.md`
- Create: `.claude/commands/compliance.md`
- Create: `.claude/commands/context.md`
- Create: `.claude/commands/devops.md`
- Create: `.claude/commands/errors.md`
- Create: `.claude/commands/onboard.md`
- Create: `.claude/commands/opus.md`
- Create: `.claude/commands/plan.md`
- Create: `.claude/commands/pr.md`
- Create: `.claude/commands/quiet.md`
- Create: `.claude/commands/risk.md`
- Create: `.claude/commands/sonnet.md`
- Create: `.claude/commands/sync-template.md`
- Create: `.claude/commands/tail.md`

- [ ] **Step 1: Create all 16 command files**

Copy each from `/workspaces/Template/.claude/commands/` verbatim. These are thin wrappers that invoke skills — no adaptation needed. The `/ux` command is intentionally omitted (no ux-pathway skill).

---

### Task 8: Create skills-index.md (adapted)

**Files:**
- Create: `.claude/skills-index.md`

- [ ] **Step 1: Create skills-index.md**

```markdown
# Skills Index — trigger table

Match user keywords → load just that skill. Don't preload.

## Workflow
- `plan / implement / build / refactor / add feature` (multi-file) → **plan-first**
- `commit / ready to commit / commit message` → **commit-prep**
- `pr / pull request / pr description` → **pr-description**
- `bug / broken / fails / error / debug / not working` → **debug-systematic**

## Architecture & review (Opus required)
- `design / architect / approach / how should I structure / tradeoff` → **senior-architect** (`/architect`)
- `dockerfile / terraform / infra / deploy / aws / ecs / fargate / rds / s3 / cicd / pipeline` → **senior-devops** (`/devops`)
- `compliance / data handling / secrets / audit / auth / encrypt` → **compliance-check** (`/compliance`)
- `new feature / spec / requirement (classify)` → **risk-classifier** (`/risk`)
- `onboard / new repo / orient / map this codebase` → **codebase-onboarding** (`/onboard`)

## Stack-specific (Rune)
- `agent / langgraph / checkpoint / state / node / graph not advancing` → **langgraph-debug**
- `migration / alembic / schema change / new column / drop column` → **alembic-safe**
- `endpoint test / api test / fastapi test / new test` → **fastapi-test-fixture**

## Quality (T2)
- `done / ready / self-review` → **code-review-self**
- `test / spec / write a test` → **test-writing**
- `refactor` → **refactor-safe**
- `install / add dependency / new package version` → **dependency-add**

## Monorepo (T2)
- `affected / changed / scope to changes` → **affected-only**
- `edit shared / change libs/shared` → **cross-package-change**
- `new package / new service / scaffold` → **new-package**

## Meta
- `audit context / token usage` → **context-audit**
- `compact / wrapping up` → **strategic-compact**
- `quick question / aside` → **quick-aside**
- *(any blocked long-running command)* → **long-run-fallback**
```

---

### Task 9: Create template-manifest and sync infrastructure

**Files:**
- Create: `.claude/.template-manifest`
- Create: `scripts/sync-claude-template.sh`
- Create: `scripts/setup-git-hooks.sh`
- Modify: `Makefile`

- [ ] **Step 1: Create .template-manifest (adapted)**

Same as template but without ux-pathway, frontend-design, event-trace skill entries. Remove CLAUDE.md and PRODUCT.md from the manifest (we don't want sync to overwrite our customized versions):

```
# Template manifest — files the template owns and will overwrite on `make claude-sync`.
# Anything in .claude/ NOT listed here is repo-local and will NEVER be touched by sync.

# Claude infra
.claude/skills-index.md

# Hooks (entire dir, no --delete: lets repos add custom hooks alongside)
.claude/hooks/

# Slash commands (entire dir, no --delete)
.claude/commands/

# Skills — explicit list.
.claude/skills/plan-first/SKILL.md
.claude/skills/commit-prep/SKILL.md
.claude/skills/pr-description/SKILL.md
.claude/skills/debug-systematic/SKILL.md
.claude/skills/senior-architect/SKILL.md
.claude/skills/senior-devops/SKILL.md
.claude/skills/compliance-check/SKILL.md
.claude/skills/risk-classifier/SKILL.md
.claude/skills/codebase-onboarding/SKILL.md
.claude/skills/long-run-fallback/SKILL.md
.claude/skills/langgraph-debug/SKILL.md
.claude/skills/alembic-safe/SKILL.md
.claude/skills/fastapi-test-fixture/SKILL.md

# code-review-graph integration
.githooks/post-commit
scripts/setup-git-hooks.sh
```

- [ ] **Step 2: Copy sync-claude-template.sh**

Copy from `/workspaces/Template/scripts/sync-claude-template.sh` verbatim. Make executable.

- [ ] **Step 3: Copy setup-git-hooks.sh**

Copy from `/workspaces/Template/scripts/setup-git-hooks.sh` verbatim. Make executable.

- [ ] **Step 4: Add claude-sync targets to Makefile**

Add after the existing `.PHONY` declaration and before the discovery helpers section:

```makefile
# ----- Claude template sync ------------------------------------------------
claude-sync:
	@./scripts/sync-claude-template.sh

claude-sync-init:
	@./scripts/sync-claude-template.sh --init

claude-sync-dry:
	@./scripts/sync-claude-template.sh --dry-run

# ----- code-review-graph (CRG) — OPTIONAL per-repo add-on ------------------
CRG_VERSION := 2.3.2

claude-crg-enable:
	@command -v pipx >/dev/null || { echo "Install pipx first: brew install pipx"; exit 1; }
	@command -v claude >/dev/null || { echo "Claude Code CLI not on PATH"; exit 1; }
	@echo "→ Installing code-review-graph==$(CRG_VERSION) (one-time per machine)…"
	@pipx install code-review-graph==$(CRG_VERSION) 2>/dev/null || pipx upgrade code-review-graph || true
	@echo "→ Building initial CRG index (one-time per repo, ~10s per 500 files, USER-RUN)…"
	@code-review-graph build
	@echo "→ Wiring post-commit auto-update…"
	@./scripts/setup-git-hooks.sh
	@echo "→ Registering as project-scoped MCP server…"
	@claude mcp add code-review-graph --scope project -- code-review-graph serve --repo . 2>&1 || \
		echo "  (already registered — that's fine)"
	@echo
	@echo "✓ CRG installed + indexed + registered."

claude-crg-disable:
	@claude mcp remove code-review-graph 2>/dev/null || echo "  (not registered)"
	@git config --unset core.hooksPath 2>/dev/null || true
	@echo "✓ CRG MCP unregistered, post-commit hook unwired."
```

Also update the `.PHONY` line and `help` target to include the new targets.

- [ ] **Step 5: Verify the full setup**

```bash
# Hooks exist and are executable
ls -la .claude/hooks/
# Settings.json is valid JSON
python3 -c "import json; json.load(open('.claude/settings.json'))"
# All skills have SKILL.md
find .claude/skills -name SKILL.md | sort
# All commands exist
ls .claude/commands/
# Manifest lists real files
cat .claude/.template-manifest
```

- [ ] **Step 6: Commit**

```bash
git add .claude/ scripts/sync-claude-template.sh scripts/setup-git-hooks.sh Makefile
git commit -m "feat(claude): bootstrap .claude/ infrastructure from template

- Hooks: guard.py (mutating git, dangerous ops, long-running blocks),
  post-edit-lint (ruff auto-fix), session-start (PRODUCT.md warnings)
- Skills: 13 skills (plan-first, commit-prep, pr-description,
  debug-systematic, senior-architect, senior-devops, risk-classifier,
  compliance-check, long-run-fallback, codebase-onboarding, alembic-safe,
  langgraph-debug, fastapi-test-fixture)
- Commands: 16 slash commands (/architect, /commit, /pr, /plan, etc.)
- Sync infra: make claude-sync for future template updates
- Adapted: removed medtech/frontend-specific items, relaxed pytest
  restriction (30s suite), updated paths for rune-agent"
```
