"""Smoke tests for the 6 root factory fixtures defined in conftest.py.

These tests verify that:
1. Each factory returns the correct type with correct defaults when called with no args.
2. Keyword overrides apply correctly while leaving other defaults unchanged.
"""


def test_make_adapter_record_defaults(make_adapter_record):
    obj = make_adapter_record()
    assert obj.id == "test-adapter-001"
    assert obj.task_type == "bug-fix"


def test_make_adapter_record_override(make_adapter_record):
    obj = make_adapter_record(task_type="code-gen")
    assert obj.task_type == "code-gen"
    assert obj.id == "test-adapter-001"  # other defaults unchanged


def test_make_adapter_ref_defaults(make_adapter_ref):
    obj = make_adapter_ref()
    assert obj.adapter_id == "test-adapter-001"


def test_make_coding_session_defaults(make_coding_session):
    obj = make_coding_session()
    assert obj.session_id == "test-session-001"
    assert obj.adapter_refs == []


def test_make_coding_session_override(make_coding_session):
    obj = make_coding_session(outcome="success")
    assert obj.outcome == "success"


def test_make_training_job_defaults(make_training_job):
    obj = make_training_job()
    assert obj.id == "test-job-001"
    assert obj.status == "pending"


def test_make_evolution_job_defaults(make_evolution_job):
    obj = make_evolution_job()
    assert obj.id == "test-evol-job-001"


def test_make_evol_metrics_defaults(make_evol_metrics):
    obj = make_evol_metrics()
    assert obj.pass_rate == 0.75
    assert obj.fitness_score == 0.8


def test_make_evol_metrics_override(make_evol_metrics):
    obj = make_evol_metrics(pass_rate=0.5)
    assert obj.pass_rate == 0.5
    assert obj.fitness_score == 0.8  # unchanged
