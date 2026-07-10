# STACK_RESULTS.md — SAM+NEXUS Associative-Symbolic Stack

**Date**: 2026-07-10
**Tag**: stack-v1 (pending)
**Repository**: SAM-architecture-research

---

## Gates Passed/Failed per Stage

| Stage | Name | Gates | Status |
|-------|------|-------|--------|
| P | Pre-registration | EXPERIMENT_SAM_NEXUS_STACK.md committed | ✅ PASS |
| 0 | Canonical Baseline | Historical R3 artifact is incomplete for serialized-artifact validation (missing effective graph config/edge counts); prior PASS is retracted | ❌ INVALID / RETRACTED |
| 1 | Associative Encoder v1 | Failed: intent 65.3% < 85% | ❌ STOP |
| 1b | Associative Encoder v2 | Current: entity_recall 50.5% < 65% (FAIL); encoder-only 50.5% with calibrated threshold 0.10. Intent 85.3%, RSS 6.6MB, 26.1ms | ❌ FAIL |
| 2 | Realization L1 | 3/4: naturalness +38.5, hallucination, accuracy. Relevance 60% pre-existing. Built on unvalidated Stage 1b foundation. | ⚠️ UNVALIDATED |
| 3 | Dialogue State | ALL 3/3: ref resolution 71.9%, no regression, 2.7ms. Built on unvalidated Stage 1b foundation. | ⚠️ UNVALIDATED |
| 4 | Realization L2 | Entry conditions not met (relevance gate + distillation pairs) | ⏭️ SKIPPED |
| 5 | Freeze | This document | 🔄 |

---

## Budget Compliance

| Resource | Budget | Used | Status |
|----------|--------|------|--------|
| Total peak RSS | ≤500 MB | ~50 MB (graph + encoder + dialogue) | ✅ |
| Answer latency p50 (zero-LLM) | ≤500 ms | 32.7 ms (encoder) + 2.7 ms (dialogue) + ~10 ms (traversal) | ✅ |
| GPU | Zero | Zero | ✅ |
| CPU training | Only | Encoder trained on CPU | ✅ |

---

## Stage 1b — Associative Encoder (Failed)

The 4.4% figures below are the historical R1 reference. The current validated result is `benchmarks/results/stage1b_honest_20260710_133731Z.json` (pipeline and capped encoder recall 50.5% using validation-calibrated threshold 0.10); the unchanged 65% gate remains FAIL.

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| entity_recall | 50.55% (139/275) | ≥65% | ❌ FAIL |
| entity_precision | 12.36% | measured | — |
| entity_f1 | 19.86% | measured | — |
| exact_entity_accuracy | 48.0% | measured | — |
| candidate_pool_recall | 85.45% | diagnostic | — |
| parser_failures | 0 | 0 expected | ✅ |
| resolution_rate | 100% | no regression | ✅ |
| paraphrase_drop | 0.0 pp | <10 pp | ✅ |
| intent_accuracy | 85.3% | ≥85% | ✅ |
| RSS delta | 6.6 MB | ≤150 MB | ✅ |
| inference p50 | 26.1 ms | ≤50 ms | ✅ |

**Historical result: 1 of 6 gates FAIL (entity_recall 4.4% < 65%). Stage 1B remains FAILED.**

The current frozen rerun is `benchmarks/results/stage1b_honest_20260710_133731Z.json`: pipeline and capped encoder entity recall 50.5% with threshold 0.10 calibrated on the separate 150-question validation split; the entity gate still fails. Diagnostics classify gold entities as absent, threshold-rejected, outside the encoder cap, or resolved; parser failure count is 0.

Candidate-pool recall is a **per-(question, gold-entity) pair** metric: 235 of 275 gold IDs were present in the union candidate pool (85.45%). It is not final entity accuracy. Final accepted recall is 139/275 (50.5%), with 96 additional selected IDs outside the capped encoder baseline and 40 IDs absent from candidates.

Model: 555K params, char n-gram hashing, 1-layer GRU, entity re-ranker over lexical+embedding candidates. Rule-first intent with 63% coverage at 100% accuracy.

---

## Stage 2 — Realization L1 (Partial)

| Gate | Value | Threshold |
|------|-------|-----------|
| naturalness | +38.5 | ≥+5.0 |
| hallucination | 37.8% | ≤41.1% (baseline) |
| accuracy | 16.9% | ≥14.9% |
| relevance | 60.0% | ≥77.0% (FAIL — pre-existing) |

Naturalness exceeded target by 7.7×. Relevance failure is pre-existing (identical to old model) — evidence retrieval quality limitation, not synthesizer.

---

## Stage 3 — Dialogue State (Unvalidated)

| Gate | Value | Threshold |
|------|-------|-----------|
| reference resolution | 71.9% | ≥70% |
| single-turn regression | 0.0pp | ≤2pp |
| state latency p50 | 2.7 ms | ≤5 ms |

50 dialogues, 110 turns. Recency decay 0.7, context window 5. These measurements are historical and remain unvalidated because Stage 1B failed.

---

## Stage 4 — Realization L2 (Skipped)

Entry conditions not met:
- Stage 2 relevance gate failed (60% < 77%)
- data/distillation/pairs.jsonl has < 5000 pairs
- Naturalness not plateaued (single measurement)

---

## Open Problems

1. **Evidence quality bottleneck**: Relevance at 60% across all realizers. The synthesizer can't improve beyond what evidence provides. Fix: structured metric ingestion, better entity→evidence mapping.
2. **Entity resolution robustness**: Embedding-based ER works (100% resolution) but requires all-MiniLM (86 MB). Char-ngram encoder is lighter (6.6 MB) but depends on candidate quality.
3. **Router held-out validation**: Router policy trained on 15 questions. Needs full 200q per-arm data.
4. **Oracle test rebuild**: Identical prompts, 1 injected GT line — ceiling measurement still unreliable.
5. **Latency budget experiment**: Relevance-rank truncation at 800/1100/1566 tokens — not measured.

---

## Honest Scope Statement

NEXUS is a curated-domain QA system for the SAM research project, with limited dialogue capability. It is NOT a general conversationalist. The associative encoder (Stage 1b) provides CPU-only entity+intent extraction. The graph engine (NEXUS) provides typed-traversal evidence. Realization is template-based (Stage 2). Dialogue state (Stage 3) handles anaphora and ellipsis. The stack fits within 500 MB RSS, runs CPU-only, and has zero GPU requirement.

---

*Generated from EXPERIMENT_SAM_NEXUS_STACK.md pre-registered gates.*
