"""C4 capacity runner: stats, gate, and label plumbing on synthetic traces."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "_c4_capacity_run.py"
_spec = importlib.util.spec_from_file_location("_c4_capacity_run", _TOOL)
assert _spec is not None and _spec.loader is not None
cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap)


def _trace(i: int, arms: dict) -> dict:
    return {"task_id": f"t/{i}", "arms": {
        label: {"pred": "", "recovered": rec} for label, rec in arms.items()
    }}


def test_gate_go_when_one_mode_clears_margin_and_p() -> None:
    # 60 rows: floor recovers 9; adapter_b_k2 recovers 40 (superset). p ~ 2^-31.
    traces = [
        _trace(i, {
            "floor": i < 9,
            "adapter_a_k2": i < 12,
            "adapter_b_k2": i < 40,
        })
        for i in range(60)
    ]
    g = cap.stage1_gate(traces, margin=0.15)
    assert g["go"] is True
    assert g["adapter_b_k2"]["delta"] > 0.15
    assert g["adapter_b_k2"]["p"] < 0.05
    assert g["adapter_a_k2"]["passes"] is False  # delta 0.05 < margin


def test_gate_no_go_when_neither_mode_clears() -> None:
    traces = [
        _trace(i, {"floor": i < 9, "adapter_a_k2": i < 11, "adapter_b_k2": i < 10})
        for i in range(60)
    ]
    g = cap.stage1_gate(traces, margin=0.15)
    assert g["go"] is False


def test_gate_delta_computed_over_paired_support_only() -> None:
    # 2 rows lose the adapter arm (bundle exception) -> gate pairs on the 58.
    traces = [
        _trace(i, {"floor": i < 9, "adapter_a_k2": i < 12, "adapter_b_k2": i < 40})
        for i in range(60)
    ]
    for i in (58, 59):
        del traces[i]["arms"]["adapter_b_k2"]
    traces[58]["errors"] = ["k2: RuntimeError('boom')"]
    traces[59]["errors"] = ["k2: RuntimeError('boom')"]
    g = cap.stage1_gate(traces, margin=0.15)
    assert g["adapter_b_k2"]["n_pairs"] == 58
    assert g["adapter_b_k2"]["delta"] == (40 - 9) / 58
    assert g["adapter_a_k2"]["n_pairs"] == 60
    assert g["error_rows"] == 2


def test_metrics_report_per_arm_wilson_and_infeasible() -> None:
    traces = [
        _trace(0, {"floor": False, "tail_k8": False}),
        _trace(1, {"floor": True, "tail_k8": True}),
    ]
    traces[0]["arms"]["tail_k8"]["infeasible"] = True
    m = cap.capacity_metrics(traces)
    assert m["recovery_tail_k8"] == 0.5      # infeasible row scored as failure
    assert m["recovery_tail_k8_n"] == 1      # successes: C1 _metrics convention
    assert m["denom_tail_k8"] == 2
    assert m["infeasible_tail_k8"] == 1
    assert 0.0 <= m["recovery_tail_k8_wilson_lo"] < 0.5


def test_metrics_report_tail_recovery_excluding_infeasible() -> None:
    traces = [
        _trace(0, {"tail_k8": False}),
        _trace(1, {"tail_k8": True}),
        _trace(2, {"tail_k8": False}),
    ]
    traces[0]["arms"]["tail_k8"]["infeasible"] = True
    m = cap.capacity_metrics(traces)
    assert m["recovery_tail_k8"] == 1 / 3
    assert m["recovery_tail_k8_excl_infeasible"] == 0.5


def test_metrics_report_kstar_crossover_both_views() -> None:
    tail2 = [True, True, True, True]
    ad2 = [True, True, False, False]
    tail4 = [False, True, True, False]   # row 0 is an infeasible failure
    ad4 = [True, True, False, False]
    traces = [
        _trace(i, {
            "tail_k2": tail2[i], "adapter_b_k2": ad2[i],
            "tail_k4": tail4[i], "adapter_b_k4": ad4[i],
        })
        for i in range(4)
    ]
    traces[0]["arms"]["tail_k4"]["infeasible"] = True
    m = cap.capacity_metrics(traces)
    assert m["kstar_b"] == 4                   # 2/4 < 4/4 at K=2; 2/4 >= 2/4 at K=4
    assert m["kstar_b_excl_infeasible"] == -1  # dropping row 0: 1/3 < 2/3 at K=4
    assert "kstar_a" not in m                  # no mode-(a) arms in these traces


def test_bundle_sign_test_counts_bundle_means() -> None:
    # 4 bundles of 2: adapter better in 3 bundles, tied in 1 -> n_eff 3, pos 3.
    traces = []
    for b in range(4):
        for j in range(2):
            i = b * 2 + j
            adapter = b < 3  # bundles 0-2 recover both rows; bundle 3 none
            traces.append(_trace(i, {"floor": False, "adapter_b_k2": adapter}))
    pos, neg, n_eff = cap.bundle_sign_counts(traces, "adapter_b_k2", "floor", k=2)
    assert (pos, neg, n_eff) == (3, 0, 3)
    # pre-registered sensitivity: wired into the emitted metrics, not just here
    m = cap.capacity_metrics(traces)
    assert m["bundle_sign_adapter_b_k2_pos"] == 3
    assert m["bundle_sign_adapter_b_k2_neg"] == 0
    assert m["bundle_sign_adapter_b_k2_n_eff"] == 3


def test_compare_to_c1_disjoint_task_ids_is_not_exact(tmp_path: Path) -> None:
    # No task_id overlap -> 0 comparisons over 1 expected row: NOT bit-exact.
    ours = [_trace(0, {"floor": True, "adapter_k1": True})]
    c1 = [{"task_id": "other/99", "arms": {
        "floor": {"pred": "x"}, "episodic_use": {"pred": "x"},
    }}]
    c1_path = tmp_path / "c1_traces.json"
    c1_path.write_text(json.dumps(c1))
    out = cap.compare_to_c1(ours, c1_path)
    assert out["expected_rows"] == 1
    assert out["match_floor"] == "0/1"
    assert out["match_floor_compared"] == 0
    assert out["match_floor_exact"] is False
    assert out["match_adapter_k1"] == "0/1"
    assert out["match_adapter_k1_exact"] is False


def test_compare_to_c1_errored_row_breaks_exactness(tmp_path: Path) -> None:
    # An errored sanity row (empty arms) must not shrink S1-ANCHOR-1's 60/60.
    ours = [
        {"task_id": "t/0", "arms": {
            "floor": {"pred": "x"}, "adapter_k1": {"pred": "x"},
        }},
        {"task_id": "t/1", "arms": {}},
    ]
    c1 = [{"task_id": f"t/{i}", "arms": {
        "floor": {"pred": "x"}, "episodic_use": {"pred": "x"},
    }} for i in range(2)]
    c1_path = tmp_path / "c1_traces.json"
    c1_path.write_text(json.dumps(c1))
    out = cap.compare_to_c1(ours, c1_path)
    assert out["match_floor"] == "1/2"
    assert out["match_floor_compared"] == 1
    assert out["match_floor_exact"] is False
    assert out["match_adapter_k1"] == "1/2"
    assert out["match_adapter_k1_exact"] is False
