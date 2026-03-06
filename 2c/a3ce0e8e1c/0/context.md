# Session Context

## User Prompts

### Prompt 1

<objective>
Check project progress, summarize recent work and what's ahead, then intelligently route to the next action - either executing an existing plan or creating the next one.

Provides situational awareness before continuing work.
</objective>

<execution_context>
@/Users/noahdolevelixir/.claude-elixirtrials/get-shit-done/workflows/progress.md
</execution_context>

<process>
Execute the progress workflow from @/Users/noahdolevelixir/.claude-elixirtrials/get-shit-done/workflows/progress...

### Prompt 2

<objective>
Mark milestone {{version}} complete, archive to milestones/, and update ROADMAP.md and REQUIREMENTS.md.

Purpose: Create historical record of shipped version, archive milestone artifacts (roadmap + requirements), and prepare for next milestone.
Output: Milestone archived (roadmap + requirements), PROJECT.md evolved, git tagged.
</objective>

<execution_context>
**Load these files NOW (before proceeding):**

- @/Users/noahdolevelixir/.claude-elixirtrials/get-shit-done/workflows/com...

### Prompt 3

<objective>
Verify milestone achieved its definition of done. Check requirements coverage, cross-phase integration, and end-to-end flows.

**This command IS the orchestrator.** Reads existing VERIFICATION.md files (phases already verified during execute-phase), aggregates tech debt and deferred gaps, then spawns integration checker for cross-phase wiring.
</objective>

<execution_context>
@/Users/noahdolevelixir/.claude-elixirtrials/get-shit-done/workflows/audit-milestone.md
</execution_conte...

### Prompt 4

/gsd:plan-milestone-gaps

### Prompt 5

<objective>
Create all phases necessary to close gaps identified by `/gsd:audit-milestone`.

Reads MILESTONE-AUDIT.md, groups gaps into logical phases, creates phase entries in ROADMAP.md, and offers to plan each phase.

One command creates all fix phases — no manual `/gsd:add-phase` per gap.
</objective>

<execution_context>
@/Users/noahdolevelixir/.claude-elixirtrials/get-shit-done/workflows/plan-milestone-gaps.md
</execution_context>

<context>
**Audit results:**
Glob: .planning/v*-MILESTO...

