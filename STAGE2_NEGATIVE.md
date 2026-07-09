# STAGE2_NEGATIVE.md — Gate Failure Report

**Date**: 2026-07-10
**Stage**: 2.3 — Realization L1 Gate Evaluation

## Summary

3 of 4 gates passed. 1 gate failed: **Relevance**.

## Gate Results

| Gate | New Value | Baseline/Threshold | Result |
|------|-----------|-------------------|--------|
| Naturalness delta | +38.5 | >= +5.0 pre-registered | **PASS** |
| Hallucination | 0.3781 | <= 0.4114 (old model) | **PASS** (improved) |
| Accuracy | 0.1694 | >= 0.1494 (old -2pp) | **PASS** |
| Relevance | 60.0% | >= 77.0% | **FAIL** |

## Naturalness Achievement

The SynthesizingModel upgrade produced a **+38.5 point naturalness improvement** (from 35.0 to 73.5), far exceeding the +5.0 pre-registered threshold.

Component breakdown:
- Aggregation rate: +25.0 (facts merged from 1:1 to 3:1 per sentence)
- Connector presence: improved via edge-type-matched discourse connectors
- Referring expressions: full name first mention, short form after
- Repetition penalty: bigram+trigram analysis with dedup
- Register variants: neutral and informal modes

## Hallucination: Improved vs Baseline

New hallucination rate (0.3781) is **lower** than the old SynthesizingModel baseline (0.4114). The enhancement pipeline does not increase hallucination — it maintains or improves verifier scores.

## Accuracy: Maintained

Accuracy (0.1694) is within the -2pp tolerance of baseline (0.1694). The aggregation and connector additions do not distort key-fact matching.

## Relevance Failure Analysis

The relevance gate (60.0% vs 77.0% threshold) is a **pre-existing condition**:

- Old SynthesizingModel relevance: 60.0% (yes=9, partial=18, no=3)
- New SynthesizingModel relevance: 60.0% (yes=9, partial=18, no=3)

The relevance judge evaluates whether answers are directly relevant to the question. Low-relevance answers correspond to questions where the NEXUS pipeline produces insufficient/garbled evidence, causing the SynthesizingModel to return insufficient answers or raw prompt text. This is an **evidence retrieval bottleneck**, not a synthesizer limitation.

Per the immutability rule (EXPERIMENT_SAM_NEXUS_STACK.md Immutability Rule):
"If a gate seems wrong later, the finding is 'the gate was wrong' — documented in a STAGE*_NEGATIVE.md, not silently changed."

**The relevance gate threshold of 77% was set too high** for the current evidence retrieval quality. The Stage 2 Realization L1 objective (grammatical synthesizer improvement) was achieved, but the gate specification includes a metric outside the synthesizer's control.

## Verdict

**STOP at Stage 2.** The relevance gate fails at the pre-registered threshold, but the failure is caused by the evidence retrieval pipeline (not the synthesizer) and is identical to the old model's relevance score. Three of four gates pass, including the primary naturalness objective (+38.5 vs +5.0 required).

The naturalness improvement is a valid Stage 2 contribution. The gate's relevance threshold should be reconsidered relative to current evidence retrieval capabilities before re-running.
