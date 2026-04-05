# Rune: Parametric Episodic Memory for Local Coding Agents

<div class="author-block" markdown="1">
**Noah**
Rune Project
2026
</div>

!!! warning "Research Status: Pre-Validation"
    Rune's infrastructure is built and tested (five-phase pipeline, benchmark
    evaluation framework, 433+ tests passing). GPU training runs and adapter
    evaluations have not been conducted. All performance claims are qualified
    with a claim tier (validated/expected/proposed).

## Article Sections

- [Abstract](abstract.md)
- [Background](background.md)
- [Methods](methods.md)
- [Results](results.md)
- [Discussion](discussion.md)
- [References](references.md)

## What This Article Covers

This article presents the theoretical foundation and algorithmic design of Rune's
parametric episodic memory approach. It surveys composable weight-space memory
strategies, specifies the Doc-to-LoRA trajectory extension, five-phase pipeline
(decompose, plan, code, integrate, diagnose/repair), and Evolution Operator,
proposes an experimental design for empirical validation using a tiered benchmark
framework (HumanEval+, MBPP+, BigCodeBench), and discusses expected contributions
and limitations honestly in terms of claim tier.
