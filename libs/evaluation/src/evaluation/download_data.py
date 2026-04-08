"""Download benchmark datasets from HuggingFace and save them locally.
Some datasets are completely open access, others require you to request access. 
Set your HF_TOKEN environment variable, try download a dataset, 
if it's gated follow the link in error message to accept terms.

Usage:
    uv run src/evaluation/download_data.py --datasets numina_tir swe_bench_lite
    uv run src/evaluation/download_data.py --datasets all
    uv run src/evaluation/download_data.py --hf-id "some/custom-dataset" --split test --name my_dataset

Known dataset aliases
---------------------
Coding:
  swe_bench             SWE-bench/SWE-bench                (test)
  swe_bench_lite        SWE-bench/SWE-bench_Lite            (test)
  swe_bench_verified    SWE-bench/SWE-bench_Verified        (test)
  swe_evo               Fsoft-AIC/SWE-EVO                   (test)
  swe_perf              SWE-Perf/SWE-Perf                   (test)

Math — tiered difficulty:
  gsm8k                 openai/gsm8k                        (train + test)
  competition_math      EleutherAI/hendrycks_math            (train + test, Level 1–5, 7 subjects)
  omni_math             KbsdJames/Omni-MATH                 (test, 10+ difficulty levels)
  harp                  HARP-benchmark/HARP                 (test, 6 difficulty levels)
  olym_math             RUC-AIBOX/OlymMATH                  (test, en-easy + en-hard)

Math — olympiad / ultra-hard:
  numina_tir            AI-MO/NuminaMath-TIR                (train)
  numina_cot            AI-MO/NuminaMath-CoT                (train, streaming)
  numina_15             AI-MO/NuminaMath-1.5                (train, streaming)
  olympiad_bench        Hothan/OlympiadBench                (test)
  openbmb_olympiad      math-ai/olympiadbench               (test, multimodal + bilingual)
  daft_math             metr-evals/daft-math                (test, ultra-hard)
  live_math_bench       opencompass/LiveMathBench           (test, EN configs: CNMO/CCEE/AMC/WLPMC/hard/v202505)
  deepmath              trl-lib/DeepMath-103K               (train)
  aime_2025             MathArena/aime_2025                 (test)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Registry of well-known dataset aliases
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # --- Coding ---
    "swe_bench": {
        "hf_id": "SWE-bench/SWE-bench",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "SWE-bench (2294 Python issues)",
    },
    "swe_bench_lite": {
        "hf_id": "SWE-bench/SWE-bench_Lite",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "SWE-bench Lite (300 curated Python issues)",
    },
    "swe_bench_verified": {
        "hf_id": "SWE-bench/SWE-bench_Verified",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "SWE-bench Verified (500 human-validated issues)",
    },
    "swe_evo": {
        "hf_id": "Fsoft-AIC/SWE-EVO",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "SWE-EVO (48 long-horizon software evolution tasks)",
    },
    "swe_perf": {
        "hf_id": "SWE-Perf/SWE-Perf",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "SWE-Perf (140 performance optimization tasks)",
    },
    # --- Math (tiered difficulty) ---
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "splits": ["train", "test"],
        "config": "main",
        "streaming": False,
        "description": "GSM8K grade-school math (8.5K problems)",
    },
    "competition_math": {
        "hf_id": "EleutherAI/hendrycks_math",
        "splits": ["train", "test"],
        "configs": [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
        "streaming": False,
        "description": "MATH (12,500 problems: 7,500 train / 5,000 test, Level 1–5, 7 subjects) — EleutherAI mirror",
    },
    "omni_math": {
        "hf_id": "KbsdJames/Omni-MATH",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "Omni-MATH (4,428 olympiad problems, 33+ sub-domains, 10+ difficulty levels)",
    },
    "harp": {
        "hf_id": "HARP-benchmark/HARP",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "HARP (5,409 US competition problems: AJHSME/AMC/AIME/USAMO, 6 difficulty levels)",
    },
    "olym_math": {
        "hf_id": "RUC-AIBOX/OlymMATH",
        "splits": ["test"],
        "configs": ["en-easy", "en-hard"],
        "streaming": False,
        "description": "OlymMATH (200 verified problems, AIME-easy / ultra-hard tiers, EN)",
    },
    # --- Math (olympiad / ultra-hard) ---
    "numina_tir": {
        "hf_id": "AI-MO/NuminaMath-TIR",
        "splits": ["train"],
        "config": None,
        "streaming": False,
        "description": "NuminaMath-TIR (70K olympiad problems, tool-integrated reasoning)",
    },
    "numina_cot": {
        "hf_id": "AI-MO/NuminaMath-CoT",
        "splits": ["train"],
        "config": None,
        "streaming": True,
        "description": "NuminaMath-CoT (860K olympiad problems, CoT solutions) — streamed",
    },
    "numina_15": {
        "hf_id": "AI-MO/NuminaMath-1.5",
        "splits": ["train"],
        "config": None,
        "streaming": True,
        "description": "NuminaMath-1.5 (900K competition math) — streamed",
    },
    "olympiad_bench": {
        "hf_id": "Hothan/OlympiadBench",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "OlympiadBench (8,476 olympiad math & physics problems)",
    },
    "openbmb_olympiad": {
        "hf_id": "math-ai/olympiadbench",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "OlympiadBench (8,476 math+physics, multimodal, EN+ZH — math-ai/olympiadbench)",
    },
    "daft_math": {
        "hf_id": "metr-evals/daft-math",
        "splits": ["train"],
        "config": None,
        "streaming": False,
        "description": "DAFT-Math (199 ultra-hard problems; 50%+ failure rate for SOTA models, IMO shortlist)",
    },
    "live_math_bench": {
        "hf_id": "opencompass/LiveMathBench",
        "splits": ["test"],
        "configs": [
            "v202412_CNMO_en",
            "v202412_CCEE_en",
            "v202412_AMC_en",
            "v202412_WLPMC_en",
            "v202412_hard_en",
            "v202505_all_en",
            "v202505_hard_en",
        ],
        "streaming": False,
        "description": "LiveMathBench (contamination-resistant: CNMO, AMC, CCEE, WLPMC — EN only)",
    },
    "deepmath": {
        "hf_id": "trl-lib/DeepMath-103K",
        "splits": ["train"],
        "config": None,
        "streaming": False,
        "description": "DeepMath-103K (TRL curated, used in GRPO examples)",
    },
    "aime_2025": {
        "hf_id": "MathArena/aime_2025",
        "splits": ["test"],
        "config": None,
        "streaming": False,
        "description": "AIME 2025 (fresh, likely uncontaminated)",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _ensure_datasets() -> None:
    """Deferred import guard — keeps module importable without datasets installed."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'datasets' package not found.\n"
            "Install it with:  uv add datasets\n"
            "or:               pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)


def _save_dataset(
    ds: Any,
    out_dir: Path,
    split: str,
    streaming: bool,
) -> None:
    """Persist a dataset split to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / split

    if streaming:
        # Materialise the iterable dataset into a regular Dataset then save.
        from datasets import Dataset

        print(f"  Materialising streamed split '{split}' (this may take a while)…")
        ds_materialised = Dataset.from_generator(lambda: ds)
        ds_materialised.save_to_disk(str(split_dir))
    else:
        ds.save_to_disk(str(split_dir))

    print(f"  Saved → {split_dir}")


def download(
    alias: str,
    *,
    out_root: Path,
    hf_id: str | None = None,
    config: str | None = None,
    splits: list[str] | None = None,
    streaming: bool | None = None,
) -> None:
    """Download one dataset (by alias or ad-hoc HF ID) and save to *out_root/alias/*.

    Args:
        alias: Short name used as the local directory name.
        out_root: Root directory under which ``alias/`` will be created.
        hf_id: HuggingFace dataset ID, e.g. ``"openai/gsm8k"``.  When *alias*
            is a known registry key this defaults to the registered value.
        config: HuggingFace dataset config name (subset), e.g. ``"main"``.
        splits: List of splits to download, e.g. ``["train", "test"]``.
        streaming: If ``True`` the dataset is loaded in streaming mode and
            materialised before saving.  Defaults to the registry value.
    """
    _ensure_datasets()
    from datasets import load_dataset  # type: ignore[import-untyped]

    reg = DATASET_REGISTRY.get(alias, {})

    hf_id = hf_id or reg.get("hf_id")
    if not hf_id:
        print(
            f"ERROR: '{alias}' is not in the registry and no --hf-id was supplied.",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved_config: str | None = config if config is not None else reg.get("config")
    resolved_configs: list[str] | None = reg.get("configs") if resolved_config is None else None
    resolved_splits: list[str] = splits or reg.get("splits") or ["train"]
    resolved_streaming: bool = streaming if streaming is not None else reg.get("streaming", False)

    description = reg.get("description", hf_id)
    out_dir = out_root / alias

    print(f"\n{'─'*60}")
    print(f"Dataset : {description}")
    print(f"HF ID   : {hf_id}")
    if resolved_configs:
        print(f"Configs : {resolved_configs}")
    print(f"Splits  : {resolved_splits}")
    print(f"Out dir : {out_dir}")
    print(f"{'─'*60}")

    configs_to_use: list[str | None] = resolved_configs if resolved_configs else [resolved_config]
    for cfg in configs_to_use:
        cfg_out_dir = out_dir / cfg if cfg else out_dir
        for split in resolved_splits:
            label = f"{cfg}/{split}" if cfg else split
            print(f"  Downloading '{label}'…")
            load_kwargs: dict[str, Any] = {"streaming": resolved_streaming}
            if cfg:
                load_kwargs["name"] = cfg

            ds = load_dataset(hf_id, split=split, **load_kwargs)
            _save_dataset(ds, cfg_out_dir, split, resolved_streaming)

    print(f"Done: {alias}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help=(
            "One or more dataset aliases to download (e.g. numina_tir swe_bench_lite). "
            "Pass 'all' to download every registered dataset. "
            "See the module docstring for the full list of aliases."
        ),
    )
    parser.add_argument(
        "--hf-id",
        metavar="HF_ID",
        help=(
            "HuggingFace dataset ID for a one-off download not in the registry "
            "(e.g. 'some-org/some-dataset'). Requires exactly one value in --datasets "
            "to use as the local directory name."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="CONFIG",
        help="HuggingFace dataset config / subset name (overrides registry default).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        metavar="SPLIT",
        help="Splits to download (overrides registry default, e.g. train test).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=None,
        help="Force streaming mode (useful for very large datasets).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        metavar="DIR",
        help=f"Root directory to save datasets (default: {_DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all registered dataset aliases and exit.",
    )

    return parser


def _print_registry() -> None:
    print("\nRegistered datasets:\n")
    max_alias = max(len(k) for k in DATASET_REGISTRY)
    for alias, info in DATASET_REGISTRY.items():
        print(f"  {alias:<{max_alias}}  {info['description']}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_registry()
        return

    if not args.datasets:
        parser.print_help()
        sys.exit(0)

    targets: list[str] = (
        list(DATASET_REGISTRY.keys()) if args.datasets == ["all"] else args.datasets
    )

    if args.hf_id and len(targets) != 1:
        print(
            "ERROR: --hf-id requires exactly one alias in --datasets to use as the "
            "local directory name.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving datasets to: {out_root.resolve()}")

    for alias in targets:
        download(
            alias,
            out_root=out_root,
            hf_id=args.hf_id if args.hf_id else None,
            config=args.config,
            splits=args.splits,
            streaming=args.streaming if args.streaming else None,
        )

    print("All downloads complete.")


if __name__ == "__main__":
    main()
