from rune.engine.graph import render_training_format_trajectory


def test_inference_trajectory_uses_training_headers() -> None:
    txt = render_training_format_trajectory(
        task="implement find_tuples",
        current_code="def f(): pass",
        feedback="use all() not any()",
    )
    assert "## Task" in txt and "## Current Code" in txt and "## Review Feedback" in txt
