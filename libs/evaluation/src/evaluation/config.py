"""Configuration dataclasses and YAML loader for the benchmark runner.

Defines the full configuration hierarchy (``ModelConfig``, ``DatasetConfig``,
``BenchmarkRunConfig``) and provides ``load_config()`` to hydrate them from a
``.yaml`` file.  Path constants and the dataset registry live here too.

Example::

    from evaluation.config import load_config
    cfg = load_config("configs/qwen3_5_olym_easy.yaml")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Path constants ─────────────────────────────────────────────────────────────

_DATA_ROOT = Path(__file__).parent / "data"
_CONFIGS_DIR = Path(__file__).parent / "configs"
# Shared Jinja2 templates — libs/shared/src/shared/templates/
_TEMPLATES_DIR = (
    Path(__file__).resolve().parents[4]
    / "libs"
    / "shared"
    / "src"
    / "shared"
    / "templates"
)

# ── Dataset registry ───────────────────────────────────────────────────────────

# Maps dataset alias → path relative to _DATA_ROOT.
# Extend as datasets are downloaded via evaluation.download_data.
DATASET_REGISTRY: dict[str, str] = {
    # ── Evaluation sets ───────────────────────────────────────────────────────
    "daft_math": "daft_math/train",
    "olym_math_easy": "olym_math/en-easy/test",
    "olym_math_hard": "olym_math/en-hard/test",
    "gsm8k_test": "gsm8k/main/test",
    "aime_2025": "aime_2025/test",
    # ── Training sets ─────────────────────────────────────────────────────────
    "numina_tir": "numina_tir/train",
    "numina_cot": "numina_cot/train",
    "deepmath": "deepmath/train",
    # competition_math — 7 subjects × 2 splits; each subject is its own Arrow dir.
    # Use the per-subject keys below, or load via data_path in your config yaml.
    "competition_math_algebra_train": "competition_math/algebra/train",
    "competition_math_algebra_test": "competition_math/algebra/test",
    "competition_math_counting_train": "competition_math/counting_and_probability/train",
    "competition_math_counting_test": "competition_math/counting_and_probability/test",
    "competition_math_geometry_train": "competition_math/geometry/train",
    "competition_math_geometry_test": "competition_math/geometry/test",
    "competition_math_intalgebra_train": "competition_math/intermediate_algebra/train",
    "competition_math_intalgebra_test": "competition_math/intermediate_algebra/test",
    "competition_math_numtheory_train": "competition_math/number_theory/train",
    "competition_math_numtheory_test": "competition_math/number_theory/test",
    "competition_math_prealgebra_train": "competition_math/prealgebra/train",
    "competition_math_prealgebra_test": "competition_math/prealgebra/test",
    "competition_math_precalculus_train": "competition_math/precalculus/train",
    "competition_math_precalculus_test": "competition_math/precalculus/test",
}

# ── Config dataclasses ─────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Model identity, backend selection, and generation hyper-parameters.

    Attributes:
        model_id: HuggingFace model ID or absolute local path.
        provider: Backend selector.  ``"vllm"`` (default) with no ``base_url``
            loads the model in-process via ``vllm.LLM``.  Any other value, or
            ``"vllm"`` *with* a ``base_url``, delegates to the corresponding
            ``InferenceProvider`` from ``libs/inference/``.
        base_url: Server base URL for HTTP-backed providers.
        tensor_parallel_size: GPU count for tensor parallelism (in-process vLLM).
        max_model_len: Maximum KV-cache sequence length (in-process vLLM).
        dtype: Weight dtype forwarded to the backend.
        enable_prefix_caching: Enable prefix KV-cache sharing (in-process vLLM).
        language_model_only: Skip vision encoder init for hybrid-architecture
            checkpoints that ship without vision weights.
        temperature: Sampling temperature.  When ``n_samples > 1`` the runner
            enforces a minimum of 0.4 to ensure diversity across samples.
        max_new_tokens: Maximum output tokens per generation call.
        enable_thinking: Enable Qwen3-style ``<think>…</think>`` thinking mode.
        batch_size: Problems packed into a single ``llm.generate()`` call.
        n_samples: Number of independent samples (trajectories) per problem.
            When > 1, majority vote over extracted ``\\boxed{}`` answers is
            applied and all per-sample results are saved.
            For ``use_tools=False``: vLLM uses ``SamplingParams(n=N)``; HTTP
            providers issue N concurrent requests.
            For ``use_tools=True``: N fully independent agentic trajectories
            are run per problem, each with its own ``MathSandbox`` kernel, all
            batched together in the same generate calls so GPU utilisation is
            unchanged.
        use_tools: When ``True`` (default) the runner uses the full
            generate → tool-call → execute agentic loop backed by
            ``MathSandbox``.  Disable for simple single-pass inference.
        max_iterations: Maximum agentic-loop steps per problem.
        sandbox_timeout: Code-execution subprocess wall-clock timeout (seconds).
        repetition_penalty: Token-level repetition penalty forwarded to vLLM's
            ``SamplingParams``.  Values > 1.0 discourage the model from
            repeating the same token sequences.  A value of 1.05–1.1 is
            effective at preventing the degenerate ``\\boxed{N}`` repetition
            loop without noticeably affecting answer quality.  Default 1.0
            (no penalty, preserving the original distribution).
    """

    model_id: str
    provider: str = "vllm"
    base_url: str | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    dtype: str = "float16"
    enable_prefix_caching: bool = True
    language_model_only: bool = True
    temperature: float = 0.6
    max_new_tokens: int = 32768
    enable_thinking: bool = False
    batch_size: int = 8
    n_samples: int = 1
    use_tools: bool = True
    max_iterations: int = 15
    sandbox_timeout: float = 60.0
    repetition_penalty: float = 1.0


@dataclass
class DatasetConfig:
    """Configuration for a single benchmark dataset.

    Attributes:
        name: Dataset alias resolved via ``DATASET_REGISTRY``, or a freeform
            label when ``data_path`` is set explicitly.
        prompt_key: Column / dict key holding the question text.
        answer_key: Column / dict key holding the ground-truth answer.
        id_key: Optional column providing a stable per-problem identifier.
            Falls back to ``"{name}_{row_index}"`` when absent.
        split: HuggingFace dataset split (informational; the Arrow path already
            bakes in the split from the registry).
        n_samples: Maximum number of problems to evaluate.  ``None`` = all.
        data_path: Absolute or relative path that overrides the registry lookup.
        scorer: Scoring strategy name.  One of:

            * ``"boxed_extract"`` — extract ``\\boxed{}`` then match numerically
              or as a string (default; good for math).
            * ``"exact_match"`` — case-insensitive exact string comparison.
            * ``"numeric"`` — numeric equality with near-integer tolerance.
        system_prompt: Inline system prompt text.  Overrides the run-level
            ``BenchmarkRunConfig.system_prompt`` when set.  Ignored when
            ``system_prompt_template`` is also set (template wins).
        system_prompt_template: Jinja2 template filename (e.g.
            ``"math_prompt_v2.j2"``) resolved from the shared templates dir or
            a custom ``templates_dir``.  When set, the rendered output is used
            as the system prompt for this dataset, overriding any inline value.
        template_vars: Variables passed to the Jinja2 template.  Merged on top
            of the run-level ``BenchmarkRunConfig.template_vars``.
        filter_integer_answers: When ``True``, rows whose ground-truth answer
            cannot be parsed as a non-negative integer inside ``integer_range``
            are skipped during loading.  Useful for datasets like OlymMATH that
            mix LaTeX-expression answers with plain integer answers.
        integer_range: Inclusive ``[lo, hi]`` range used by
            ``filter_integer_answers``.  Defaults to ``[0, 99999]``.
    """

    name: str
    prompt_key: str
    answer_key: str
    id_key: str | None = None
    split: str = "test"
    n_samples: int | None = 9       # default 9 — easy to change for full runs
    data_path: str | None = None
    scorer: str = "boxed_extract"
    system_prompt: str | None = None
    system_prompt_template: str | None = None
    template_vars: dict[str, Any] = field(default_factory=dict)
    filter_integer_answers: bool = False
    integer_range: tuple[int, int] = (0, 99999)


@dataclass
class RetrieverConfig:
    """Configuration for the ``MathContextRetriever`` used in few-shot prompting.

    When ``use_retriever=True``, the runner embeds every query problem and
    fetches the most similar training examples from the index.  Those examples
    are formatted and injected as the ``retrieved_examples`` variable in the
    Jinja2 system-prompt template (``math_prompt.j2``).

    Attributes:
        use_retriever: Master on/off switch.  Set ``false`` to disable retrieval
            without removing the section from the YAML.
        embedding_model: Sentence-transformers model ID used to encode problems.
            Defaults to ``"BAAI/bge-base-en-v1.5"`` (110 M params, 768-dim).
        top_k: Total retrieved examples injected per problem.
        tir_top_k: Guaranteed reserved slots from ``numina_tir`` (the only
            tool-integrated-reasoning dataset).  Remaining ``top_k - tir_top_k``
            slots come from all other indexed sources by similarity.
        max_solution_chars: Soft truncation limit for solution text in the
            formatted few-shot block.
        similarity_threshold: Minimum cosine similarity to include a result
            (``0.0`` disables the threshold).
        datasets: Per-dataset sampling configuration.  Each key is one of
            ``"numina_tir"``, ``"numina_cot"``, ``"competition_math"``,
            ``"deepmath"``.  Each value is a dict accepted by
            ``shared.math_retriever.DatasetConfig`` (keys: ``n``, ``sources``,
            ``levels``, ``subjects``, ``split``).
        dedup_test_path: Path to the HuggingFace Arrow test dataset whose
            ``"problem"`` column is checked for train/test overlap.  Problems
            found in the test set are excluded from the index.  ``null`` →
            auto-detected from the benchmark dataset names.
        index_dir: Root directory for persisted index files.  ``null`` →
            ``~/.rune/math_index``.
    """

    use_retriever: bool = True
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 5
    tir_top_k: int = 2
    max_solution_chars: int = 800
    similarity_threshold: float = 0.0
    datasets: dict[str, Any] = field(
        default_factory=lambda: {
            "numina_tir": {"n": 10000},
            "numina_cot": {"n": 8000, "sources": ["olympiads", "amc_aime"]},
            "competition_math": {"n": 4000, "levels": ["Level 4", "Level 5"]},
        }
    )
    dedup_test_path: str | None = None
    index_dir: str | None = None


@dataclass
class BenchmarkRunConfig:
    """Top-level configuration for a complete benchmark run.

    Attributes:
        model: Model identity and generation settings.
        datasets: Ordered list of datasets to evaluate.
        output_dir: Root directory for all run artefacts.
        run_id: Unique run identifier.  Auto-generated when not supplied.
        system_prompt: Default inline system prompt.  Used when no template is set.
        system_prompt_template: Jinja2 template filename resolved from the
            shared templates dir (e.g. ``"math_prompt_v2.j2"``).  Per-dataset
            templates override this.
        template_vars: Default template variables (merged under per-dataset
            overrides).  E.g. ``{"answer_range": "0 to 99999"}``.
        templates_dir: Override the directory searched for Jinja2 templates.
            Defaults to ``libs/shared/src/shared/templates/``.
        max_retries: Times to retry a failed batch before recording errors and
            moving to the next batch.
        retriever: Optional few-shot context retrieval configuration.  When set
            and ``use_retriever=True``, a ``MathContextRetriever`` index is
            built at run start and each problem's system prompt is augmented
            with the *k* most similar training examples.
    """

    model: ModelConfig
    datasets: list[DatasetConfig]
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    run_id: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
    )
    system_prompt: str | None = None
    system_prompt_template: str | None = None
    template_vars: dict[str, Any] = field(default_factory=dict)
    templates_dir: str | None = None
    max_retries: int = 2
    retriever: RetrieverConfig | None = None


# ── YAML config loader ─────────────────────────────────────────────────────────


def load_config(path: Path | str) -> BenchmarkRunConfig:
    """Load a ``BenchmarkRunConfig`` from a YAML file.

    The YAML structure mirrors the dataclass hierarchy exactly.  Unknown keys
    are silently ignored so that comments and future fields don't break older
    runners.  See ``configs/`` for annotated examples.

    Args:
        path: Path to a ``.yaml`` config file.

    Returns:
        Fully populated ``BenchmarkRunConfig``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required field (``model.model_id``) is missing.

    Example::

        cfg = load_config("configs/qwen3_5_olym_easy.yaml")
        BenchmarkRunner(cfg).run()
    """
    import yaml  # deferred — pyyaml optional dep

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    # ── model ──
    model_raw = data.get("model", {})
    _known_model = {f.name for f in ModelConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    model_cfg = ModelConfig(**{k: v for k, v in model_raw.items() if k in _known_model})

    # ── datasets ──
    _known_ds = {f.name for f in DatasetConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    datasets = []
    for ds in data.get("datasets", []):
        ds_data = {k: v for k, v in ds.items() if k in _known_ds}
        # YAML parses [0, 99999] as a list; coerce to tuple for the dataclass.
        if "integer_range" in ds_data and isinstance(ds_data["integer_range"], list):
            ds_data["integer_range"] = tuple(ds_data["integer_range"])
        datasets.append(DatasetConfig(**ds_data))

    # ── retriever (optional) ──
    retriever_cfg: RetrieverConfig | None = None
    retriever_raw = data.get("retriever")
    if retriever_raw and isinstance(retriever_raw, dict):
        _known_ret = {f.name for f in RetrieverConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        retriever_cfg = RetrieverConfig(
            **{k: v for k, v in retriever_raw.items() if k in _known_ret}
        )

    # ── top-level ──
    return BenchmarkRunConfig(
        model=model_cfg,
        datasets=datasets,
        output_dir=Path(data.get("output_dir", "benchmark_results")),
        run_id=data.get(
            "run_id",
            BenchmarkRunConfig.__dataclass_fields__["run_id"].default_factory(),  # type: ignore[misc]
        ),
        system_prompt=data.get("system_prompt"),
        system_prompt_template=data.get("system_prompt_template"),
        template_vars=dict(data.get("template_vars") or {}),
        templates_dir=data.get("templates_dir"),
        max_retries=int(data.get("max_retries", 2)),
        retriever=retriever_cfg,
    )
