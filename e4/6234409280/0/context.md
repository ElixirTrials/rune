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

Let's do a check that formatting and linting (ruff), typing (mypy) and testing (pytest) all pass with no errors or warnings. Then let's locally merge this branch into it's parent branch feat/e2e_with_hypernetwork

### Prompt 3

Merge only when ruff, mypy and pytest fully pass. If they all do, merge.

### Prompt 4

[Request interrupted by user]

### Prompt 5

Commit only when ruff, mypy and pytest fully pass then merge.

