"""Math context retrieval engine for few-shot prompting.

Indexes math QA pairs from training datasets and retrieves the most similar
examples at query time to inject as few-shot context into ``math_prompt.j2``.

The index is built once and persisted on disk.  Subsequent calls with the same
configuration load in milliseconds without re-embedding.

Typical usage::

    from pathlib import Path
    from shared.math_retriever import (
        DatasetConfig,
        MathContextRetriever,
        MathRetrievalConfig,
    )

    config = MathRetrievalConfig(
        datasets={
            "numina_tir": DatasetConfig(n=5000),
            "numina_cot": DatasetConfig(n=3000, sources=["olympiads"]),
            "competition_math": DatasetConfig(n=2000, levels=["Level 4", "Level 5"]),
        },
        top_k=3,
        dedup_test_path=Path(
            "libs/evaluation/src/evaluation/data/olym_math/en-hard/test"
        ),
    )

    retriever = MathContextRetriever(config)
    retriever.build_index()           # idempotent — loads from disk if already built

    examples = retriever.query("Find all primes p such that p^2 + 2 is also prime.")
    context  = retriever.format_context(examples)
    # Pass context as extra_context= to math_prompt.j2
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATA_ROOT = (
    _REPO_ROOT / "libs" / "evaluation" / "src" / "evaluation" / "data"
)

_COMPETITION_MATH_SUBJECTS: list[str] = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Sampling and filter configuration for a single dataset.

    Attributes:
        n: Maximum number of examples to sample.  Sampling is deterministic
            (first *n* rows after optional filtering).  ``None`` = keep all
            rows (no limit applied).
        sources: ``numina_cot`` only — restrict to these ``source`` values,
            e.g. ``["olympiads", "amc_aime"]``.  ``None`` = all sources.
        levels: ``competition_math`` only — restrict to these difficulty
            strings, e.g. ``["Level 4", "Level 5"]``.  ``None`` = all levels.
        subjects: ``competition_math`` only — restrict to these subject
            directory names (e.g. ``["algebra", "number_theory"]``).
            ``None`` = all 7 subjects.
        split: Dataset split to load (``"train"`` or ``"test"``).  Ignored for
            ``competition_math``, which exposes per-subject/split directories.
    """

    n: int | None = 1000
    sources: list[str] | None = None
    levels: list[str] | None = None
    subjects: list[str] | None = None
    split: str = "train"


@dataclass
class MathRetrievalConfig:
    """Full configuration for building and querying a math context index.

    Attributes:
        datasets: Mapping of dataset name to its :class:`DatasetConfig`.
            Supported keys: ``"numina_tir"``, ``"numina_cot"``,
            ``"competition_math"``, ``"deepmath"``.
        embedding_model: Sentence-transformers model ID or local path used to
            encode problem texts.  Defaults to ``"BAAI/bge-base-en-v1.5"``
            (110 M params, 768-dim, good technical-text coverage).
        batch_size: Number of texts to embed per forward pass.
        index_dir: Parent directory for persisted index files.  A sub-folder
            named after the config hash is created inside so multiple configs
            can coexist without collision.
        top_k: Total number of examples to return per :meth:`~MathContextRetriever.query`
            call.
        tir_top_k: Guaranteed slots reserved for ``numina_tir`` (tool-integrated
            reasoning) examples within each result.  The remaining
            ``top_k - tir_top_k`` slots are filled from all other indexed
            datasets by similarity.  Set to ``0`` to disable the reservation
            and rank everything together.  Clamped to ``top_k`` at query time.
        similarity_threshold: Minimum cosine similarity for an example to be
            included in results.  ``0.0`` means no threshold.
        max_solution_chars: Soft truncation limit applied to solution text when
            :meth:`~MathContextRetriever.format_context` assembles the prompt
            snippet.  Keeps prompts from ballooning.
        dedup_test_path: Optional path to a HuggingFace Arrow dataset whose
            ``"problem"`` column is used to detect train/test overlap.  Any
            training example whose normalised problem text exactly matches a
            test problem is logged and excluded from the index.
        data_root: Root directory that contains the Arrow dataset sub-folders.
    """

    datasets: dict[str, DatasetConfig] = field(
        default_factory=lambda: {
            "numina_tir": DatasetConfig(n=5000),
            "numina_cot": DatasetConfig(n=5000, sources=["olympiads", "amc_aime"]),
            "competition_math": DatasetConfig(n=3000, levels=["Level 4", "Level 5"]),
        }
    )
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 256
    index_dir: Path = field(default_factory=lambda: Path.home() / ".rune" / "math_index")
    top_k: int = 5
    tir_top_k: int = 1
    similarity_threshold: float = 0.0
    max_solution_chars: int = 800
    dedup_test_path: Path | None = None
    data_root: Path = field(default_factory=lambda: _DEFAULT_DATA_ROOT)

    def config_hash(self) -> str:
        """Return a 10-character MD5 hex digest representing this configuration.

        The hash covers dataset names, per-dataset counts and filters, and the
        embedding model.  Changing any of those values produces a different
        hash — and therefore a different index sub-directory — so no stale
        index is ever silently re-used.

        Returns:
            10-character lowercase hex string.
        """
        payload = json.dumps(
            {
                "datasets": {
                    k: {
                        "n": v.n,
                        "sources": sorted(v.sources) if v.sources else None,
                        "levels": sorted(v.levels) if v.levels else None,
                        "subjects": sorted(v.subjects) if v.subjects else None,
                        "split": v.split,
                    }
                    for k, v in sorted(self.datasets.items())
                },
                "embedding_model": self.embedding_model,
            },
            sort_keys=True,
        )
        return hashlib.md5(payload.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# MathExample — a single retrieved result
# ---------------------------------------------------------------------------


@dataclass
class MathExample:
    """A single retrieved math QA pair with its similarity score.

    Attributes:
        problem: The problem statement (LaTeX-formatted text).
        solution: Full solution text from the source dataset.
        source: Dataset or category label (e.g. ``"numina_tir"``,
            ``"olympiads"``, ``"competition_math"``).
        metadata: Extra dataset-specific fields such as ``level``, ``type``,
            or ``source_tag``.
        similarity: Cosine similarity to the query problem (range 0–1 for
            normalised vectors; 1.0 = identical embedding).
    """

    problem: str
    solution: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0


# ---------------------------------------------------------------------------
# MathContextRetriever
# ---------------------------------------------------------------------------


class MathContextRetriever:
    """Builds and queries a vector index of math QA pairs for few-shot prompting.

    The index is stored as two files inside ``config.index_dir/<hash>/``:

    * ``embeddings.npy`` — L2-normalised float32 array of shape ``[N, D]``.
    * ``records.jsonl``  — one JSON object per line (problem, solution, source,
      metadata).
    * ``meta.json``      — human-readable build metadata.

    Cosine similarity is computed via a dot-product on the normalised vectors,
    so no external ANN library is required.

    Args:
        config: :class:`MathRetrievalConfig` controlling datasets, model, and
            storage paths.  Defaults to the built-in defaults when omitted.

    Example::

        retriever = MathContextRetriever(config)
        retriever.build_index()       # idempotent; loads if already built
        examples = retriever.query("Find all integers n where n^3 + 1 is prime.")
        context  = retriever.format_context(examples)
        # inject context into math_prompt.j2 via extra_context=context
    """

    def __init__(self, config: MathRetrievalConfig | None = None) -> None:
        self._cfg = config or MathRetrievalConfig()
        self._embeddings: np.ndarray | None = None  # [N, D] float32, L2-normalised
        self._records: list[dict[str, Any]] = []
        self._model: Any = None  # lazy-loaded SentenceTransformer

        # Boolean masks built after every index load; used by query() to
        # enforce the tir_top_k reservation.
        self._tir_mask: np.ndarray | None = None    # True where source == "numina_tir"
        self._other_mask: np.ndarray | None = None  # True everywhere else

        self._index_path = Path(self._cfg.index_dir) / self._cfg.config_hash()
        self._embeddings_file = self._index_path / "embeddings.npy"
        self._records_file = self._index_path / "records.jsonl"
        self._meta_file = self._index_path / "meta.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, force_rebuild: bool = False) -> int:
        """Build the vector index from configured datasets, or load from disk.

        If an index with a matching config hash already exists on disk, it is
        loaded without re-embedding.  Pass ``force_rebuild=True`` to discard
        the existing index and start fresh.

        Args:
            force_rebuild: Ignore existing on-disk index and rebuild from
                scratch.

        Returns:
            Number of examples in the (newly built or loaded) index.
        """
        if not force_rebuild and self._index_exists():
            logger.info("Loading existing index from %s", self._index_path)
            self._load_index()
            self._build_source_masks()
            n = len(self._records)
            logger.info("Loaded %d indexed examples", n)
            return n

        logger.info("Building math context index …")
        t0 = time.monotonic()

        records = self._collect_records()
        logger.info("Collected %d records (after dedup)", len(records))

        if not records:
            logger.warning("No records collected — index will be empty")
            self._records = []
            self._embeddings = np.zeros((0, 1), dtype=np.float32)
            return 0

        texts = [r["problem"] for r in records]
        logger.info(
            "Embedding %d texts with %s …", len(texts), self._cfg.embedding_model
        )
        embeddings = self._embed(texts)  # [N, D]

        self._records = records
        self._embeddings = embeddings
        self._build_source_masks()
        self._save_index()

        elapsed = time.monotonic() - t0
        logger.info("Index built in %.1fs — %d examples", elapsed, len(records))
        return len(records)

    def query(
        self,
        problem: str,
        top_k: int | None = None,
        tir_top_k: int | None = None,
    ) -> list[MathExample]:
        """Retrieve the most similar training examples for *problem*.

        Results are drawn from two pools:

        * **TIR pool** — ``numina_tir`` examples only.  Exactly *tir_top_k*
          slots are reserved for this pool (or ``config.tir_top_k`` when the
          argument is omitted).  This guarantees at least one tool-integrated-
          reasoning example regardless of what the general similarity ranking
          would produce.
        * **Other pool** — all remaining indexed sources.  Fills the
          ``top_k - tir_top_k`` remaining slots by similarity.

        When ``tir_top_k == 0`` (or ``numina_tir`` was not indexed), the
        reservation is disabled and all *top_k* slots come from the full index
        ranked by similarity alone.

        Args:
            problem: The query math problem text.
            top_k: Total examples to return.  Falls back to ``config.top_k``
                when ``None``.
            tir_top_k: Reserved slots for ``numina_tir``.  Falls back to
                ``config.tir_top_k`` when ``None``.

        Returns:
            List of :class:`MathExample` sorted by descending cosine
            similarity.  TIR examples appear interleaved in their natural
            rank order, not pinned to the front.

        Raises:
            RuntimeError: If :meth:`build_index` has not been called yet.
        """
        if self._embeddings is None or not self._records:
            raise RuntimeError("Index is empty. Call build_index() first.")

        k = top_k if top_k is not None else self._cfg.top_k
        tir_k = tir_top_k if tir_top_k is not None else self._cfg.tir_top_k
        tir_k = min(tir_k, k)
        other_k = k - tir_k

        query_vec = self._embed([problem])  # [1, D]
        sims: np.ndarray = (query_vec @ self._embeddings.T).squeeze(0)  # [N]

        tir_examples = self._top_k_from_mask(sims, self._tir_mask, tir_k)
        other_examples = self._top_k_from_mask(sims, self._other_mask, other_k)

        combined = tir_examples + other_examples
        combined.sort(key=lambda x: x.similarity, reverse=True)
        return combined

    def _top_k_from_mask(
        self,
        sims: np.ndarray,
        mask: np.ndarray | None,
        k: int,
    ) -> list[MathExample]:
        """Return up to *k* best examples from the subset selected by *mask*.

        Args:
            sims: Full similarity vector ``[N]`` for the current query.
            mask: Boolean array ``[N]`` selecting the desired subset, or
                ``None`` to select the full index.
            k: Maximum examples to return.

        Returns:
            Up to *k* :class:`MathExample` objects above
            ``config.similarity_threshold``, sorted by descending similarity.
        """
        if k <= 0:
            return []

        if mask is None or not mask.any():
            return []

        masked_sims = np.where(mask, sims, -2.0)  # push excluded entries below any valid sim
        top_indices: np.ndarray = np.argsort(masked_sims)[::-1]

        threshold = self._cfg.similarity_threshold
        results: list[MathExample] = []
        for idx in top_indices:
            if len(results) >= k:
                break
            if not mask[int(idx)]:
                break  # past all valid entries
            sim = float(sims[idx])
            if sim < threshold:
                break
            rec = self._records[int(idx)]
            results.append(
                MathExample(
                    problem=rec["problem"],
                    solution=rec["solution"],
                    source=rec["source"],
                    metadata=rec.get("metadata", {}),
                    similarity=sim,
                )
            )
        return results

    def format_context(
        self,
        examples: list[MathExample],
        max_solution_chars: int | None = None,
    ) -> str:
        """Format retrieved examples as a string for ``extra_context`` in math_prompt.j2.

        Pass the returned string as ``extra_context=`` when rendering
        ``math_prompt.j2``::

            from jinja2 import Environment, FileSystemLoader
            context_str = retriever.format_context(examples)
            system_prompt = env.get_template("math_prompt.j2").render(
                answer_range="0 to 999",
                extra_context=context_str,
            )

        Args:
            examples: List of :class:`MathExample` from :meth:`query`.
            max_solution_chars: Override the config truncation limit for this
                call.

        Returns:
            Multi-line formatted string, or ``""`` when *examples* is empty.
        """
        if not examples:
            return ""

        limit = (
            max_solution_chars
            if max_solution_chars is not None
            else self._cfg.max_solution_chars
        )
        parts: list[str] = [
            f"The following {len(examples)} solved example(s) illustrate "
            "potentially relevant techniques. Study the reasoning approach — "
            "do not copy answers directly.\n",
        ]
        for i, ex in enumerate(examples, 1):
            meta_parts: list[str] = [ex.source]
            if ex.metadata.get("level"):
                meta_parts.append(str(ex.metadata["level"]))
            if ex.metadata.get("type"):
                meta_parts.append(str(ex.metadata["type"]))
            meta_str = " · ".join(meta_parts)

            solution = ex.solution
            if len(solution) > limit:
                solution = solution[:limit] + "\n…[truncated]"

            parts.append(
                f"--- Example {i} ({meta_str}, similarity={ex.similarity:.3f}) ---\n"
                f"Problem: {ex.problem}\n\n"
                f"Solution:\n{solution}\n"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Source masks
    # ------------------------------------------------------------------

    def _build_source_masks(self) -> None:
        """Precompute boolean masks separating TIR from non-TIR records.

        Called once after every index build or load so that :meth:`query`
        can split similarity scores into two pools without iterating the
        full records list on every call.
        """
        self._tir_mask = np.array(
            [r["source"] == "numina_tir" for r in self._records],
            dtype=bool,
        )
        self._other_mask = ~self._tir_mask
        n_tir = int(self._tir_mask.sum())
        logger.debug(
            "Source masks built: %d TIR, %d other",
            n_tir,
            len(self._records) - n_tir,
        )

    # ------------------------------------------------------------------
    # Index existence / persistence
    # ------------------------------------------------------------------

    def _index_exists(self) -> bool:
        return (
            self._embeddings_file.exists()
            and self._records_file.exists()
            and self._meta_file.exists()
        )

    def _save_index(self) -> None:
        self._index_path.mkdir(parents=True, exist_ok=True)
        assert self._embeddings is not None
        np.save(str(self._embeddings_file), self._embeddings)
        with open(self._records_file, "w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta: dict[str, Any] = {
            "embedding_model": self._cfg.embedding_model,
            "config_hash": self._cfg.config_hash(),
            "n": len(self._records),
            "dim": int(self._embeddings.shape[1]) if self._embeddings.ndim > 1 else 0,
            "datasets": {k: v.n for k, v in self._cfg.datasets.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self._meta_file, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _load_index(self) -> None:
        self._embeddings = np.load(str(self._embeddings_file))
        with open(self._records_file, encoding="utf-8") as fh:
            self._records = [
                json.loads(line) for line in fh if line.strip()
            ]

    # ------------------------------------------------------------------
    # Dataset loading / deduplication
    # ------------------------------------------------------------------

    def _collect_records(self) -> list[dict[str, Any]]:
        """Load all configured datasets, apply filters, and deduplicate."""
        exclude: set[str] = self._load_test_problems()

        all_records: list[dict[str, Any]] = []
        for ds_name, ds_cfg in self._cfg.datasets.items():
            loader = _DATASET_LOADERS.get(ds_name)
            if loader is None:
                logger.warning("Unknown dataset %r — skipping", ds_name)
                continue

            records = loader(ds_cfg, Path(self._cfg.data_root))
            before = len(records)

            if exclude:
                records = [
                    r for r in records
                    if _normalise(r["problem"]) not in exclude
                ]
                n_removed = before - len(records)
                if n_removed:
                    logger.info(
                        "Removed %d duplicate(s) from %s matching the test set",
                        n_removed,
                        ds_name,
                    )

            logger.info("Loaded %d records from %s", len(records), ds_name)
            all_records.extend(records)

        return all_records

    def _load_test_problems(self) -> set[str]:
        """Return normalised problem strings from the configured dedup test set.

        Returns:
            Set of normalised problem strings, or an empty set when no
            ``dedup_test_path`` is configured or the path does not exist.
        """
        if self._cfg.dedup_test_path is None:
            return set()

        path = Path(self._cfg.dedup_test_path)
        if not path.exists():
            logger.warning(
                "dedup_test_path %s does not exist — skipping dedup", path
            )
            return set()

        from datasets import load_from_disk  # noqa: PLC0415

        ds = load_from_disk(str(path))
        problems = {_normalise(str(row["problem"])) for row in ds}
        logger.info(
            "Loaded %d test problems for dedup from %s", len(problems), path
        )
        return problems

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info(
                "Loading embedding model %s …", self._cfg.embedding_model
            )
            self._model = SentenceTransformer(self._cfg.embedding_model)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* and return an L2-normalised float32 array of shape [N, D].

        Args:
            texts: List of strings to encode.

        Returns:
            ``numpy.ndarray`` of shape ``(len(texts), embedding_dim)``,
            dtype ``float32``, L2-normalised so cosine similarity reduces to
            a dot product.
        """
        model = self._get_model()
        vecs: np.ndarray = model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Text normalisation (for exact-match dedup)
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Strip and collapse internal whitespace for robust exact-match comparison.

    Args:
        text: Raw problem string.

    Returns:
        Whitespace-normalised string.
    """
    return " ".join(text.strip().split())


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
# Each loader returns list[dict] with keys: problem, solution, source, metadata.
# Heavy deps (datasets, sentence_transformers) are imported inside function
# bodies following the INFRA-05 deferred-import pattern so this module stays
# importable in CPU-only or minimal environments.


def _cap(n: int | None, total: int) -> int:
    """Return the number of rows to take, respecting an optional cap.

    Args:
        n: Maximum rows requested.  ``None`` = no limit.
        total: Total available rows in the dataset.

    Returns:
        ``total`` when *n* is ``None``; otherwise ``min(n, total)``.
    """
    return total if n is None else min(n, total)


def _load_numina_tir(cfg: DatasetConfig, data_root: Path) -> list[dict[str, Any]]:
    """Load examples from the NuminaMath-TIR dataset.

    Args:
        cfg: Sampling configuration.
        data_root: Root directory containing the Arrow dataset sub-folders.

    Returns:
        List of record dicts.
    """
    from datasets import load_from_disk  # noqa: PLC0415

    path = data_root / "numina_tir" / cfg.split
    if not path.exists():
        logger.warning("numina_tir path not found: %s", path)
        return []

    ds = load_from_disk(str(path))
    n = _cap(cfg.n, len(ds))
    records: list[dict[str, Any]] = []
    for row in ds.select(range(n)):
        records.append(
            {
                "problem": str(row["problem"]),
                "solution": str(row["solution"]),
                "source": "numina_tir",
                "metadata": {},
            }
        )
    return records


def _load_numina_cot(cfg: DatasetConfig, data_root: Path) -> list[dict[str, Any]]:
    """Load examples from the NuminaMath-CoT dataset.

    Supports filtering by ``source`` tag (e.g. ``"olympiads"``, ``"amc_aime"``).

    Args:
        cfg: Sampling configuration.
        data_root: Root directory containing the Arrow dataset sub-folders.

    Returns:
        List of record dicts.
    """
    from datasets import load_from_disk  # noqa: PLC0415

    path = data_root / "numina_cot" / cfg.split
    if not path.exists():
        logger.warning("numina_cot path not found: %s", path)
        return []

    ds = load_from_disk(str(path))
    if cfg.sources:
        source_set = set(cfg.sources)
        ds = ds.filter(lambda row: row["source"] in source_set)

    n = _cap(cfg.n, len(ds))
    records: list[dict[str, Any]] = []
    for row in ds.select(range(n)):
        records.append(
            {
                "problem": str(row["problem"]),
                "solution": str(row["solution"]),
                "source": str(row.get("source", "numina_cot")),
                "metadata": {"source_tag": str(row.get("source", ""))},
            }
        )
    return records


def _load_competition_math(
    cfg: DatasetConfig, data_root: Path
) -> list[dict[str, Any]]:
    """Load examples from the Hendrycks MATH (competition_math) dataset.

    Concatenates all configured subject directories, then optionally filters
    by difficulty level.

    Args:
        cfg: Sampling configuration.  ``cfg.subjects`` selects subject
            directories; ``cfg.levels`` filters by difficulty string.
        data_root: Root directory containing the Arrow dataset sub-folders.

    Returns:
        List of record dicts.
    """
    from datasets import concatenate_datasets, load_from_disk  # noqa: PLC0415

    subjects = cfg.subjects or _COMPETITION_MATH_SUBJECTS
    subject_datasets = []
    for subject in subjects:
        path = data_root / "competition_math" / subject / cfg.split
        if not path.exists():
            logger.warning("competition_math path not found: %s", path)
            continue
        subject_datasets.append(load_from_disk(str(path)))

    if not subject_datasets:
        return []

    ds = concatenate_datasets(subject_datasets)
    if cfg.levels:
        level_set = set(cfg.levels)
        ds = ds.filter(lambda row: row["level"] in level_set)

    n = _cap(cfg.n, len(ds))
    records: list[dict[str, Any]] = []
    for row in ds.select(range(n)):
        records.append(
            {
                "problem": str(row["problem"]),
                "solution": str(row["solution"]),
                "source": "competition_math",
                "metadata": {
                    "level": str(row.get("level", "")),
                    "type": str(row.get("type", "")),
                },
            }
        )
    return records


def _load_deepmath(cfg: DatasetConfig, data_root: Path) -> list[dict[str, Any]]:
    """Load examples from the DeepMath-103K dataset.

    The ``prompt`` field is a list of message dicts; the user turn
    (``role == "user"``) is extracted as the problem text.

    Args:
        cfg: Sampling configuration.
        data_root: Root directory containing the Arrow dataset sub-folders.

    Returns:
        List of record dicts.
    """
    from datasets import load_from_disk  # noqa: PLC0415

    path = data_root / "deepmath" / cfg.split
    if not path.exists():
        logger.warning("deepmath path not found: %s", path)
        return []

    ds = load_from_disk(str(path))
    n = _cap(cfg.n, len(ds))
    records: list[dict[str, Any]] = []
    for row in ds.select(range(n)):
        prompt_field = row["prompt"]
        if isinstance(prompt_field, list) and prompt_field:
            problem = str(prompt_field[0].get("content", ""))
        else:
            problem = str(prompt_field)

        records.append(
            {
                "problem": problem,
                "solution": str(row.get("solution", "")),
                "source": "deepmath",
                "metadata": {},
            }
        )
    return records


_DATASET_LOADERS: dict[
    str,
    Any,
] = {
    "numina_tir": _load_numina_tir,
    "numina_cot": _load_numina_cot,
    "competition_math": _load_competition_math,
    "deepmath": _load_deepmath,
}
