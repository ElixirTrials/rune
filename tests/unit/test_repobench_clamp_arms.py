"""Pre-registered C1 arms of the keystone harness (a2_tail / filler / swap + stats).

The harness is a script under tools/ (not a package), so it is loaded from its
file path; its module-level imports are stdlib-only (GPU imports are deferred),
keeping this CPU-safe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from rune.bench.repobench import RepoBenchRow, render_episodic

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "_repobench_clamp_run.py"
_spec = importlib.util.spec_from_file_location("_repobench_clamp_run", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
clamp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clamp)


class _FakeModel:
    """Whitespace-token fake exposing the two methods the prompt helpers use."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def clamp_to_window(self, text: str, max_tokens: int) -> str:
        toks = text.split()
        if max_tokens <= 0:
            return ""
        if len(toks) <= max_tokens:
            return text
        return " ".join(toks[-max_tokens:])


class _MultiTokenModel(_FakeModel):
    """Long words cost 2 tokens — exercises the filler trim/pad logic."""

    def count_tokens(self, text: str) -> int:
        return sum(2 if len(w) > 6 else 1 for w in text.split())


# --- a2_tail prompt assembly ---


def test_tail_prompt_places_cond_before_cursor_within_budget() -> None:
    model = _FakeModel()
    prefix = " ".join(f"w{i}" for i in range(200))
    cond = "## Task\nmust use `Pooler` (from m.pool)"
    window = 60
    prompt, clamped = clamp._assemble_tail_prompt(model, prefix, cond, window)
    assert prompt.startswith("# Current file:\n")
    assert prompt.endswith(f"\n{cond}\n# Next line:")
    assert model.count_tokens(prompt) <= window
    # prefix is tail-clamped: the cursor-adjacent code survives
    assert clamped.split()[-1] == "w199"
    assert model.count_tokens(clamped) < model.count_tokens(prefix)


def test_tail_prompt_short_prefix_unclamped() -> None:
    model = _FakeModel()
    prefix = "x = 1"
    prompt, clamped = clamp._assemble_tail_prompt(model, prefix, "cond words", 100)
    assert clamped == prefix
    assert prompt == f"# Current file:\n{prefix}\ncond words\n# Next line:"


def test_tail_prompt_window_smaller_than_overhead_evicts_prefix() -> None:
    model = _FakeModel()
    prefix = " ".join(f"w{i}" for i in range(50))
    cond = "one two three four five six seven eight nine ten"
    prompt, clamped = clamp._assemble_tail_prompt(model, prefix, cond, 5)
    # the fixed scaffolding + conditioning exceed the window: prefix fully evicted
    assert clamped == ""
    assert prompt.endswith(f"\n{cond}\n# Next line:")


# --- a2_tail_filler neutrality + length matching ---


def test_filler_matches_token_count_exactly() -> None:
    model = _FakeModel()
    filler = clamp._neutral_filler(model, 30, {"Pooler"})
    assert model.count_tokens(filler) == 30


def test_filler_excludes_forbidden_words_case_insensitive() -> None:
    model = _FakeModel()
    forbidden = {"Meadow", "THE", "breeze", "Pooler"}
    filler = clamp._neutral_filler(model, 40, forbidden)
    words = {w.lower() for w in filler.split()}
    assert not words & {f.lower() for f in forbidden}


def test_filler_carries_no_code_like_tokens() -> None:
    model = _FakeModel()
    filler = clamp._neutral_filler(model, 25, set())
    assert filler
    assert all(w.isalpha() for w in filler.split())


def test_filler_length_match_with_multi_token_words() -> None:
    model = _MultiTokenModel()
    filler = clamp._neutral_filler(model, 25, set())
    assert model.count_tokens(filler) == 25


def test_filler_empty_on_zero_target() -> None:
    assert clamp._neutral_filler(_FakeModel(), 0, set()) == ""


# --- swap conditioning ---


def test_swap_replaces_all_occurrences() -> None:
    cond = (
        "## Task\nComplete the next line. It must use `Pooler` (from m.pool):\n\n"
        "class Pooler(nn.Module):"
    )
    swapped, n = clamp._swap_conditioning(cond, "Pooler", "Encoder")
    assert n == 2
    assert "Pooler" not in swapped
    assert swapped.count("Encoder") == 2


def test_swap_whole_token_only() -> None:
    swapped, n = clamp._swap_conditioning("tool tooltip tool_x atool", "tool", "kit")
    assert n == 1
    assert swapped == "kit tooltip tool_x atool"


def test_swap_inapplicable_when_gold_absent() -> None:
    cond = "## Task\nuse `Other`"
    swapped, n = clamp._swap_conditioning(cond, "Pooler", "Encoder")
    assert n == 0
    assert swapped == cond


def test_swap_inapplicable_when_gold_empty() -> None:
    assert clamp._swap_conditioning("anything", "", "X") == ("anything", 0)


def test_swap_on_rendered_episodic_conditioning() -> None:
    row = RepoBenchRow(
        task_id="t",
        cropped_code="x = 1\ny = 2",
        import_statement="",
        context=(
            {"identifier": "noise", "path": "x.py", "snippet": "X = 1"},
            {
                "identifier": "Pooler",
                "path": "m/pool.py",
                "snippet": "class Pooler(nn.Module):\n    def __init__(self, dim):\n        self.dim = dim",
            },
        ),
        gold_snippet_index=1,
        next_line="self.p = Pooler(dim)",
        level="8k",
        token_num=0,
        repo_name="r",
        file_path="f",
    )
    cond = render_episodic(row, "use", anchor_chars=0)
    swapped, n = clamp._swap_conditioning(cond, row.gold_identifier, "Encoder")
    assert n >= 1
    assert "Pooler" not in swapped
    assert "Encoder" in swapped
    # the rest of the conditioning surface is intact
    assert swapped.startswith("## Task")
    assert "## Review Feedback" in swapped


def test_pick_swap_identifier_next_distinct_cyclic() -> None:
    golds = ["alpha", "beta", "gamma"]
    assert clamp._pick_swap_identifier(0, golds) == "beta"
    assert clamp._pick_swap_identifier(2, golds) == "alpha"
    # skips empties and same-name duplicates
    assert clamp._pick_swap_identifier(0, ["a", "", "a", "b"]) == "b"


def test_pick_swap_identifier_none_when_no_distinct_donor() -> None:
    assert clamp._pick_swap_identifier(0, ["a", "a"]) is None
    assert clamp._pick_swap_identifier(0, ["a"]) is None


def test_pick_swap_identifier_skips_donor_containing_gold() -> None:
    # donor carrying the gold's surface form would prime the original gold
    golds = ["Config", "ConfigLoader", "load_config", "Encoder"]
    assert clamp._pick_swap_identifier(0, golds) == "Encoder"


def test_pick_swap_identifier_skips_donor_contained_in_gold() -> None:
    # donor extendable back into the gold (prefix/substring) is also inadmissible
    golds = ["ConfigLoader", "Config", "Loader", "Encoder"]
    assert clamp._pick_swap_identifier(0, golds) == "Encoder"


def test_pick_swap_identifier_containment_is_case_insensitive() -> None:
    assert clamp._pick_swap_identifier(0, ["config", "CONFIG_PATH", "Encoder"]) == "Encoder"


def test_pick_swap_identifier_none_when_all_donors_share_surface_form() -> None:
    assert clamp._pick_swap_identifier(0, ["Config", "ConfigLoader", "config_dir"]) is None


def test_pick_swap_identifier_none_on_empty_gold() -> None:
    assert clamp._pick_swap_identifier(0, ["", "Encoder"]) is None


# --- Wilson 95% score interval ---


def test_wilson_known_values() -> None:
    lo, hi = clamp._wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 1e-3
    assert abs(hi - 0.7634) < 1e-3
    lo, hi = clamp._wilson_ci(0, 10)
    assert lo == 0.0
    assert abs(hi - 0.2775) < 1e-3
    lo, hi = clamp._wilson_ci(10, 10)
    assert abs(lo - 0.7225) < 1e-3
    assert hi == 1.0
    lo, hi = clamp._wilson_ci(1, 10)
    assert abs(lo - 0.0179) < 1e-3
    assert abs(hi - 0.4042) < 1e-3


def test_wilson_empty_denominator_full_width() -> None:
    assert clamp._wilson_ci(0, 0) == (0.0, 1.0)


# --- McNemar pair extraction + metrics ---


def _trace(**arms: Any) -> dict[str, Any]:
    return {
        "arms": {
            k: (v if isinstance(v, dict) else {"recovered": v}) for k, v in arms.items()
        }
    }


def test_paired_discordants_excludes_null_missing_skipped() -> None:
    traces = [
        _trace(a=True, b=False),  # a-only
        _trace(a=False, b=True),  # b-only
        _trace(a=True, b=True),  # concordant
        _trace(a=None, b=True),  # null gold -> excluded
        _trace(a=True),  # arm missing -> excluded
        _trace(a=True, b={"skipped": "swap-inapplicable"}),  # skipped -> excluded
    ]
    assert clamp._paired_discordants(traces, "a", "b") == (1, 1, 3)


def _full_row(floor: bool, episodic: bool, swap: Any) -> dict[str, Any]:
    return _trace(
        floor=floor,
        a2_clamp=False,
        a2_full=False,
        episodic_use=episodic,
        dump_gf=False,
        a2_tail=False,
        a2_tail_filler=False,
        swap=swap,
    )


def test_metrics_new_arms_pairs_and_attributable_fraction() -> None:
    traces = [
        _full_row(False, True, True),
        _full_row(False, True, False),
        _full_row(False, True, False),
        _full_row(False, True, False),
    ]
    m = clamp._metrics(traces)
    # existing metric names/values preserved
    assert m["recovery_floor"] == 0.0
    assert m["recovery_episodic_use"] == 1.0
    assert (m["mcnemar_adapter_only"], m["mcnemar_floor_only"]) == (4, 0)
    assert m["mcnemar_p"] == clamp._two_sided_binom_p(4, 0)
    # new arms rated with Wilson bounds
    assert m["recovery_swap"] == 0.25
    assert 0.0 < m["recovery_swap_wilson_lo"] < 0.25 < m["recovery_swap_wilson_hi"] < 1.0
    # pre-registered pairs
    assert m["mcnemar_episodic_use_vs_a2_tail_n"] == 4
    assert m["mcnemar_episodic_use_vs_a2_tail_first_only"] == 4
    assert m["mcnemar_swap_vs_floor_first_only"] == 1
    assert m["mcnemar_swap_vs_episodic_use_second_only"] == 3
    assert m["mcnemar_a2_tail_vs_a2_tail_filler_p"] == 1.0
    # e=1.0, s=0.25, f=0.0 -> (e-s)/(e-f) = 0.75 on the common support
    assert m["attrib_n"] == 4
    assert abs(m["attributable_fraction"] - 0.75) < 1e-9
    assert m["swap_inapplicable"] == 0


def test_metrics_swap_skipped_rows_and_fraction_guard() -> None:
    traces = [
        _full_row(True, True, {"skipped": "swap-inapplicable: gold absent from conditioning"}),
    ]
    m = clamp._metrics(traces)
    assert m["swap_inapplicable"] == 1
    assert m["denom_swap"] == 0
    assert m["attrib_n"] == 0
    assert "attributable_fraction" not in m
    # unscored arm -> full-width Wilson interval
    assert (m["recovery_swap_wilson_lo"], m["recovery_swap_wilson_hi"]) == (0.0, 1.0)


def test_metrics_fraction_guard_when_episodic_equals_floor() -> None:
    traces = [_full_row(True, True, True), _full_row(False, False, False)]
    m = clamp._metrics(traces)
    assert m["attrib_n"] == 2
    assert m["attrib_rate_episodic"] == m["attrib_rate_floor"]
    assert "attributable_fraction" not in m


def test_fmt_metrics_renders_new_sections() -> None:
    traces = [
        _full_row(False, True, True),
        _full_row(False, True, False),
    ]
    text = clamp._fmt_metrics(clamp._metrics(traces))
    assert "wilson95" in text
    assert "a2_tail_filler" in text
    assert "McNemar swap vs floor" in text
    assert "attributable fraction" in text
    assert "swap-inapplicable rows" in text


def test_metric_names_mlflow_safe() -> None:
    m = clamp._metrics([_full_row(False, True, True)])
    assert all("@" not in k for k in m)


def test_metrics_a2_tail_skipped_rows_counted_and_excluded() -> None:
    over = {"skipped": "tail_overhead_tokens>768", "cond_tokens": 2020}
    traces = [
        _full_row(True, True, True),
        _trace(
            floor=False, a2_clamp=False, a2_full=False, episodic_use=True,
            dump_gf=False, a2_tail=dict(over), a2_tail_filler=dict(over), swap=False,
        ),
    ]
    m = clamp._metrics(traces)
    assert m["a2_tail_inapplicable"] == 1
    assert m["denom_a2_tail"] == 1
    assert m["denom_a2_tail_filler"] == 1
    # skipped row drops out of the paired McNemar for the tail pairs
    assert m["mcnemar_episodic_use_vs_a2_tail_n"] == 1
    assert m["mcnemar_a2_tail_vs_a2_tail_filler_n"] == 1
    assert "a2_tail-inapplicable rows: 1" in clamp._fmt_metrics(m)
