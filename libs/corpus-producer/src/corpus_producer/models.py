"""Core data models for the phase corpus producer.

PhaseArtifact is the unit of data flowing from a single pipeline run
through filtering, binning, manifest emission, and training invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseArtifact:
    """One phase-boundary output from a single Rune pipeline run.

    Attributes:
        phase: Pipeline phase name: "decompose" | "plan" | "code" | "integrate"
            | "diagnose".
        benchmark: Benchmark identifier (e.g. "humaneval").
        problem_id: Benchmark-specific problem identifier.
        pipeline_run_id: UUID of the pipeline run that produced this artifact.
        input_text: The prompt / trajectory input fed to the model for this phase.
        output_text: The model output for this phase.
        pass_at_1: True if the integrated final code passed the benchmark test suite.
            None if not yet evaluated.
        rationalized: True if this artifact was produced by STaR rationalization
            (ground-truth test hints), not a live pipeline run.
        metadata: Optional extra key/value pairs (e.g. subtask name, layer index).
    """

    phase: str
    benchmark: str
    problem_id: str
    pipeline_run_id: str
    input_text: str
    output_text: str
    pass_at_1: bool | None = None
    rationalized: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def bin_key(self) -> str:
        """Return the training bin key for this artifact.

        Returns:
            "diagnose_pooled" for diagnose phase; "<phase>_<benchmark>" otherwise.
        """
        if self.phase == "diagnose":
            return "diagnose_pooled"
        return f"{self.phase}_{self.benchmark}"

    def to_manifest_record(self) -> dict[str, object]:
        """Serialize to a JSONL training manifest record.

        Schema is drop-in compatible with ``model_training.d2l_data`` pair
        records consumed by ``trainer.train_and_register``:
        - ``task_id``: ``"<benchmark>/<problem_id>/<phase>"``
        - ``activation_text``: model input for this phase
        - ``teacher_text``: activation + model output (what the trainer supervises)
        - ``metadata``: provenance fields

        Returns:
            Dict suitable for ``json.dumps`` and JSONL emission.
        """
        task_id = f"{self.benchmark}/{self.problem_id}/{self.phase}"
        activation = self.input_text
        teacher = f"{activation}\n\n{self.output_text}".strip()
        return {
            "task_id": task_id,
            "activation_text": activation,
            "teacher_text": teacher,
            "metadata": {
                "phase": self.phase,
                "benchmark": self.benchmark,
                "problem_id": self.problem_id,
                "pipeline_run_id": self.pipeline_run_id,
                "pass_at_1": self.pass_at_1,
                "rationalized": self.rationalized,
                **self.metadata,
            },
        }
