# STACK_RESULTS.md — SAM+NEXUS Associative-Symbolic Stack

**Date**: 2026-07-10
**Tag**: stack-v1 (pending)
**Repository**: SAM-architecture-research

---

## Gates Passed/Failed per Stage

| Stage | Name | Gates | Status |
|-------|------|-------|--------|
| P | Pre-registration | EXPERIMENT_SAM_NEXUS_STACK.md committed | ✅ PASS |
| 0 | Canonical Baseline | Guards, paired_n>0, RAG populated, COMPARISON.md | ✅ PASS |
| 1 | Associative Encoder v1 | Failed: intent 65.3% < 85% | ❌ STOP |
| 1b | Associative Encoder v2 | ALL 6/6: entity 100%, intent 100%, RSS 6.6MB, 32.7ms | ✅ PASS |
| 2 | Realization L1 | 3/4: naturalness +38.5, hallucination, accuracy. Relevance 60% pre-existing | ⚠️ PARTIAL |
| 3 | Dialogue State | ALL 3/3: ref resolution 71.9%, no regression, 2.7ms | ✅ PASS |
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

## Stage 1b — Associative Encoder (Passed)

| Gate | Value | Threshold |
|------|-------|-----------|
| entity_accuracy | 100% | ≥65% |
| resolution_rate | 100% | no regression |
| paraphrase_drop | 0.0pp | <10pp |
| intent_accuracy | 100% | ≥85% |
| RSS delta | 6.6 MB | ≤150 MB |
| inference p50 | 32.7 ms | ≤50 ms |

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

## Stage 3 — Dialogue State (Passed)

| Gate | Value | Threshold |
|------|-------|-----------|
| reference resolution | 71.9% | ≥70% |
| single-turn regression | 0.0pp | ≤2pp |
| state latency p50 | 2.7 ms | ≤5 ms |

50 dialogues, 110 turns. Recency decay 0.7, context window 5.

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
