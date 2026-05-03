# Trajectory-Based Mining Plan

**Date:** 2026-05-03
**Author's note:** Supersedes the abortive "fetch full pre/post file bodies" mining change (reverted). The reframing came from the user's observation:

> *"If we mine whole-file bodies and train toward file reconstruction, we risk teaching the adapter to re-encode lots of static code context instead of the procedural change signal carried by diffs, feedback, and revisions."*

That's correct. The LoRA's job is **episodic memory of corrective trajectories**, not file-state reconstruction. The mining target should reflect that.

---

## What we're trying to achieve

The adapter sits on top of a frozen Qwen3.5-9B base. The base already knows how to write code that *looks like* the post-edit file — it does not need the adapter to re-encode static structure. The adapter's low-rank capacity should be spent on:

- *what changed* between revisions
- *why it changed* (review feedback, test failure, lint)
- *how the next correction should be shaped* given the trajectory so far

So the corpus should be a sequence of **(state_t, action_t, feedback_t) → action_{t+1}** episodes, where:

- `state_t` = the prior diff (the change in flight)
- `feedback_t` = reviewer comment, CI failure log, lint output, …
- `action_{t+1}` = the next corrective diff

That's what the hypernetwork compresses into LoRA weights.

---

## What's wrong with the current corpus

| Concern | Current behaviour | Why it hurts |
|---|---|---|
| Single-shot mode | Each pair is treated as `(prompt → response)` with no notion of "we just tried this, here's the feedback, now revise" | The adapter never sees the corrective dynamic — only the final answer |
| Feedback is opaque | Review comments interleave chronologically with commits in `steps[]` but aren't *attached* to the diff they critiqued | Training can't condition on "this diff was rejected because X" |
| Revision rounds aren't explicit | Multi-turn unrolling currently treats each revision as independent of the prior one | We lose the "diff → review → next-diff" chain that should drive memory |
| No CI / test signal | We capture only commit + review-comment data | Missing the strongest correction signal in real PR workflows |
| Quality filter is naive | Filters on `min_review_comments=1, min_commits=2` and approval-only | Doesn't preferentially select PRs with rich corrective structure |
| Whole-file fetch (the path I almost took) | Would have added pre/post bodies | Wastes adapter capacity on context the base already knows; doesn't capture the trajectory |

---

## Proposed shape of the new corpus

Each record is a **trajectory** with explicit corrective episodes. Schema sketch:

```jsonc
{
    "task_id": "pr_owner/repo_NNNN",
    "task_description": "PR title + body",
    "metadata": { "outcome": "merged", "language": "python", "n_rounds": 3, ... },

    // Ordered episodes. Each episode covers ONE corrective round:
    //   prior diff → feedback that came after it → next diff that addresses it.
    "episodes": [
        {
            "round": 0,
            // The change as it stood at the start of this round.
            // For round 0, this is empty (initial submission).
            "prior_diff": "",
            // The feedback that motivated this round's correction.
            // For round 0, this is the PR description / first review request.
            "feedback": {
                "kind": "task_description",
                "body": "Original PR description"
            },
            // The change submitted in response.
            "action_diff": "--- file ---\n@@ -1,3 +1,4 @@\n+import x\n …"
        },
        {
            "round": 1,
            "prior_diff": "<the round-0 action_diff>",
            "feedback": {
                "kind": "review_comment",
                "author": "reviewer1",
                "body": "this allocates inside the hot loop — pull it out",
                "anchor": { "file": "src/foo.py", "line": 42 }
            },
            "action_diff": "--- src/foo.py ---\n@@ -38,5 +38,8 @@\n …"
        },
        {
            "round": 2,
            "prior_diff": "<concatenation or last action_diff>",
            "feedback": {
                "kind": "ci_failure",
                "body": "FAILED tests/test_foo.py::test_bar - AssertionError: …",
                "test": "tests/test_foo.py::test_bar"
            },
            "action_diff": "…"
        }
    ]
}
```

Key shape rules:

1. **Diffs stay as unified-diff strings.** The adapter encodes procedural deltas; the base model handles file-state reconstruction. Don't store full file bodies.
2. **Feedback is a typed object** (`kind` ∈ `{task_description, review_comment, ci_failure, test_failure, lint, build_failure}`) with `body` plus optional anchor (file/line/test). Training prompts can format this however they want, but the structure is preserved at mining time.
3. **Episodes are ordered**, one per corrective round. A round = the prior change + the feedback that landed against it + the next change submitted.
4. **The training input for episode `t` is the TRAJECTORY PREFIX up to and including `feedback_t`.** The training target is `action_diff_t`. So a 3-round PR yields three training examples, each strictly more context than the last.

---

## Mining algorithm (what to actually implement)

Per-PR pseudocode:

```
def mine_trajectory(pr) -> Trajectory | None:
    commits = chronological list of commits in the PR
    review_comments = chronological list of inline comments
    issue_comments = chronological list of top-level discussion comments (filter: only those with corrective intent)
    ci_runs = list of CI/check-suite results, with status + (best-effort) failed-test extraction

    if not has_corrective_structure(commits, review_comments, ci_runs):
        return None  # skip — this PR is single-commit or has no review/CI signal

    rounds = []
    cumulative_diff = ""

    # Round 0: initial submission
    head_commit = commits[0]
    initial_diff = aggregate_patch_strings(head_commit.files)  # capped at file-content-max-bytes per patch
    rounds.append(Episode(
        round=0,
        prior_diff="",
        feedback=Feedback(kind="task_description", body=pr.title + pr.body),
        action_diff=initial_diff,
    ))
    cumulative_diff = initial_diff

    # Rounds 1..N: each interleaved (feedback, next-commit) pair
    for fb_event, next_commit in pair_feedback_with_next_commit(review_comments, ci_runs, commits[1:]):
        diff = aggregate_patch_strings(next_commit.files)
        rounds.append(Episode(
            round=len(rounds),
            prior_diff=cumulative_diff,
            feedback=Feedback(
                kind=fb_event.kind,
                body=fb_event.body,
                anchor=fb_event.anchor,
            ),
            action_diff=diff,
        ))
        cumulative_diff = diff   # OR concatenate; see "open question 2"

    return Trajectory(task_id=..., episodes=rounds, metadata=...)
```

Implementation notes:

- **Pairing feedback to commits.** A reviewer comment counts as the feedback for the *next* commit by the PR author after the comment timestamp. CI failures count as feedback for the next commit by anyone. Use timestamps and reviewer-vs-author bucketing — already partially in place in the chronological-interleaving code.
- **`aggregate_patch_strings`.** Concatenate `f["patch"]` from each modified file with `--- {filename} ---` separators. Same as today. Cap each patch at e.g. 2000 lines — patches that big aren't teaching corrective deltas, they're full rewrites.
- **`has_corrective_structure`.** A PR qualifies if at least one of: (a) ≥ 2 commits AND ≥ 1 review comment between commits, (b) ≥ 1 commit AND ≥ 1 CI failure resolved by a subsequent commit, (c) the PR description explicitly references a prior PR/issue ("addresses review from #N", "fixes failing test X").
- **CI fetch.** For each commit, GET `/repos/{repo}/commits/{sha}/check-runs` and `/check-suites`. Filter to non-success results; pull the failure body if available. This is the missing supervision signal.
- **Lint / build feedback.** Where CI runs include linter output (e.g. `flake8`, `eslint`, `clippy`) parse the failure summary into `Feedback(kind="lint")`.

---

## Quality filter changes

Replace the current `search_quality_prs` filter (`approved AND comments > 1 AND commits >= 2`) with one that prioritises corrective richness:

| Signal | Weight | Rationale |
|---|---|---|
| `review_comments_with_anchor >= 2` | + | The strongest correction signal — line-anchored critiques |
| `ci_failures_resolved > 0` | + | Trains the failure-recovery dynamic the LoRA is supposed to compress |
| `n_commits between 3 and 12` | + | Sweet spot — multi-round but not pathological |
| `outcome == merged` | + | Only PRs that actually landed (we know the final correction worked) |
| Author/reviewer disjoint | + | A self-reviewed PR has no real corrective structure |
| `n_files_changed <= 20` per commit | + | Mass-edit commits drown the signal |
| Bot author (any of: dependabot, renovate, github-actions[bot]) | exclude | Trivial mechanical edits |
| Labels: `documentation`, `chore`, `ci` | exclude | Already done; keep |

Score each candidate PR; select top-K by score. Concretely: aim for **5-10K trajectories** at this filter quality, with the median PR having 3-5 rounds.

---

## Storage / shape decisions

| Question | Decision |
|---|---|
| Keep diffs as the primary representation? | **Yes.** Procedural deltas are the right unit. |
| Mine full file bodies? | **No** — only as a tiny aux corpus (≤ 100 records) explicitly for "teach the model what a syntactically valid unified diff looks like" if we observe persistent diff-syntax errors. |
| Per-episode flat record vs. nested per-trajectory? | **Nested** (one row per PR) at mine-time; the data-prep stage unrolls into per-episode training rows. Lets future loss-shaping decisions (e.g. trajectory-level KL terms) operate on the structured record without re-mining. |
| JSONL or other? | JSONL, one trajectory per line, gz-compressed at upload time. |

---

## Code changes required (downstream of mining)

These are *follow-ups* — the user only asked for a mining plan now. Listing them so the implications of the new corpus shape are visible.

1. **`d2l_data.py`** — replace `pairs_to_chat_messages` with a `trajectory_to_chat_messages` that unrolls each `episode` into a multi-turn conversation:
   - System: the model role
   - User turn 0: task description
   - Assistant turn 0: `action_diff_0`
   - User turn 1: feedback_1 (with anchor)
   - Assistant turn 1: `action_diff_1`
   - …
2. **Diff-aware loss** — `_compute_hunk_ranges` becomes line-level on `prior_diff` vs `action_diff` (now both are per-round diffs against the same base, not interleaved diffs of the same file). Or, drop diff-aware loss and use plain SFT — re-evaluate after the new corpus is mined.
3. **Eval harness** — patch-quality evaluator should score against the *correct round's* ground-truth `action_diff`, given the trajectory prefix.
4. **HPO** — re-run on new corpus; the conservative-HP optima from the prior study were tuned under the old shape and won't transfer cleanly.

---

## Cost / wall-clock estimate

Per-PR API cost on the new mining:

| Call | Existing | New | Per-PR delta |
|---|---|---|---|
| `/pulls/{n}` | 1 | 1 | 0 |
| `/pulls/{n}/commits` (paginated) | 1-3 | 1-3 | 0 |
| `/pulls/{n}/comments` (review comments) | 1-2 | 1-2 | 0 |
| `/pulls/{n}/issues/{n}/comments` (top-level) | 0 | 1-2 | +1-2 |
| `/commits/{sha}` | N (commits) | N | 0 |
| `/commits/{sha}/check-runs` | 0 | N | +N |
| `/commits/{sha}/check-suites` | 0 | N | +N |

So ~2 + 2N extra API calls per PR. For a typical PR with 4 commits, that's ~10 extra calls. At GitHub's ~5000 calls/hour authenticated rate limit and ~10-20 calls per mine_pr_diff_chains today, going from ~15 calls/PR to ~25 means roughly 2× wall clock. **For a 5,000-PR mine: ~3-6 hours.** Manageable.

---

## What to revert before implementing this

- ✅ Already reverted: the `_fetch_file_pre_post` helper + `get_raw_url` client method + `--no-fetch-file-contents` CLI flag.
- The `pyproject.toml` and devcontainer changes for the Mamba fast-path kernels are **unrelated** to this — keep those.

---

## Open questions for the next implementation pass

1. **Cumulative vs. last-only `prior_diff`.** When round 2 sees feedback on round 1, does it see:
   - (a) Just the round-1 action diff, or
   - (b) The cumulative diff = round-0 + round-1 stacked?
   I lean (b) since reviewers in practice see the cumulative state; but the per-token cost is higher.

2. **CI failure body length.** Pytest failures can be 10 KB+ of stack trace. Truncate to first/last N lines? Use a summary extractor? For now: take first 2000 chars + a "tail" of the same length, with the middle elided.

3. **Should we fold the original `pair`-shape data through this pipeline?** I.e. emit BOTH a trajectory record AND the legacy single-step pairs, so existing training code keeps working during transition. Probably yes — gives us a clean A/B between old and new shape.

4. **What to do about issue-commit chains.** The current code also has `mine_issue_commit_chains`. That's a different flavour of trajectory (issue → commit-claiming-fix → outcome). Worth re-thinking under the same lens. Defer.

---

## Concrete next step

When the user is ready: a separate implementation PR that does, in this order:

1. Add `Episode` and `Trajectory` Pydantic models to `model_training/d2l_models.py` (or wherever `models.py` lives).
2. Refactor `mine_pr_diff_chains` into `mine_pr_trajectories`, returning the new shape.
3. Add `/check-runs` + `/check-suites` fetching to capture CI feedback.
4. Update quality filter per the table above.
5. Validate on a single repo (e.g. `babel/babel`) — verify the resulting JSONL has trajectories with at least 2 rounds and the feedback objects are populated.
6. Run the full mine on a CPU box (which is what the user will do).
7. Update `d2l_data.py` to consume the new shape (separate PR, after mining lands).

The current branch `fix/diff-loss-per-turn-alignment` should NOT carry these changes — it should land the eval/diagnostic/devcontainer changes and merge. New mining gets its own branch.
