"""TDD tests for shared.rune_models — field types, defaults, and serialization."""

from shared.rune_models import AdapterRef, CodingSession, EvolMetrics, PipelinePhase

# --- AdapterRef tests ---


def test_adapter_ref_required_fields(make_adapter_ref) -> None:
    """AdapterRef has adapter_id and task_type as required fields."""
    ref = make_adapter_ref()
    assert ref.adapter_id == "test-adapter-001"
    assert ref.task_type == "bug-fix"


def test_adapter_ref_fitness_score_defaults_none(make_adapter_ref) -> None:
    """AdapterRef.fitness_score defaults to None."""
    ref = make_adapter_ref()
    assert ref.fitness_score is None


def test_adapter_ref_round_trip_serialization(make_adapter_ref) -> None:
    """AdapterRef serializes and deserializes correctly via model_dump."""
    original = make_adapter_ref(fitness_score=0.85)
    data = original.model_dump()
    restored = AdapterRef(**data)
    assert restored.adapter_id == original.adapter_id
    assert restored.task_type == original.task_type
    assert restored.fitness_score == original.fitness_score


# --- CodingSession tests ---


def test_coding_session_required_fields(make_coding_session) -> None:
    """CodingSession has session_id, task_description, and task_type as required."""
    session = make_coding_session()
    assert session.session_id == "test-session-001"
    assert session.task_description == "Fix the off-by-one error in list slicing"
    assert session.task_type == "bug-fix"


def test_coding_session_defaults(make_coding_session) -> None:
    """CodingSession has correct default values for optional fields."""
    session = make_coding_session()
    assert session.adapter_refs == []
    assert session.attempt_count == 0
    assert session.outcome is None


def test_coding_session_round_trip_serialization(make_coding_session) -> None:
    """CodingSession round-trips through model_dump and reconstruction."""
    original = make_coding_session(attempt_count=3, outcome="success")
    data = original.model_dump()
    restored = CodingSession(**data)
    assert restored.session_id == original.session_id
    assert restored.attempt_count == 3
    assert restored.outcome == "success"


# --- EvolMetrics tests ---


def test_evol_metrics_required_fields(make_evol_metrics) -> None:
    """EvolMetrics has adapter_id, pass_rate, and fitness_score as required."""
    metrics = make_evol_metrics()
    assert metrics.adapter_id == "test-adapter-001"
    assert metrics.pass_rate == 0.75
    assert metrics.fitness_score == 0.8


def test_evol_metrics_generalization_delta_defaults_none(make_evol_metrics) -> None:
    """EvolMetrics.generalization_delta defaults to None."""
    metrics = make_evol_metrics()
    assert metrics.generalization_delta is None


def test_evol_metrics_round_trip_serialization(make_evol_metrics) -> None:
    """EvolMetrics round-trips through model_dump and reconstruction."""
    original = make_evol_metrics(generalization_delta=0.05)
    data = original.model_dump()
    restored = EvolMetrics(**data)
    assert restored.adapter_id == original.adapter_id
    assert restored.pass_rate == original.pass_rate
    assert restored.fitness_score == original.fitness_score
    assert restored.generalization_delta == 0.05


# --- PipelinePhase tests ---


def test_pipeline_phase_has_all_five_canonical_values() -> None:
    """PipelinePhase enumerates all 5 Rune pipeline phases (incl. DIAGNOSE)."""
    values = {p.value for p in PipelinePhase}
    assert values == {"decompose", "plan", "code", "integrate", "diagnose"}


def test_pipeline_phase_diagnose_round_trips_via_string() -> None:
    """PipelinePhase is a str-Enum — DIAGNOSE round-trips through its string value."""
    assert PipelinePhase("diagnose") is PipelinePhase.DIAGNOSE
    assert PipelinePhase.DIAGNOSE.value == "diagnose"
