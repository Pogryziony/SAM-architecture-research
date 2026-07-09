# EXPERIMENT: SAM+NEXUS Associative-Symbolic Stack

**Pre-registered**: 2026-07-09
**Status**: Stage P — thresholds frozen before implementation
**Repository**: SAM-architecture-research (branch: main)

---

## Layer Architecture

```
stack/                        # associative-symbolic layers (new)
  encoder/                    #   Stage 1: SAM-as-encoder (model, training, loader)
  normalization/              #   Stage 1: PL/EN lemmatization
  dialogue/                   #   Stage 3: dialogue state (activation subgraph)
  realization/                #   Stage 2/4: grammatical synthesizer + realizer
nexus/                        # unchanged — pure graph engine, zero weights
sam-lm/                       # unchanged — archive (Stage 1 imports design from here)
models/                       # trained artifacts (gitignore weights, commit configs+hashes)
data/distillation/            # distillation pairs (Stage 2+)
benchmarks/                   # shared harness — extended, never duplicated
```

**Dependency direction**: `stack → nexus` only, never reverse. NEXUS must not import anything from `stack/`. Integration via existing extension points only (NEXUSConfig flags, candidate interface in parser, ModelInterface for realization). CI test: `nexus/` imports and passes tests without PyTorch in the environment.

---

## Hard Resource Budget

This is the project's thesis — not a preference:

- **Total peak RSS**: <= 500 MB (full stack, all layers active)
- **Answer latency p50**: <= 500 ms (zero-LLM path)
- **Zero GPU**: CPU-only training for any learned component
- **CPU training only**: no GPU code paths

A layer that breaks the budget is falsified in that form — record it and stop that layer's track.

---

## Layer List

| # | Layer | Function | Stage |
|---|-------|----------|-------|
| 1 | Normalization | PL/EN lemmatization for entity spotting | 1 |
| 2 | Associative Understanding | Small encoder: utterance → (entities, intent, category) | 1 |
| 3 | Dialogue State | Activation subgraph with recency decay | 3 |
| 4 | NEXUS Knowledge+Reasoning | Graph traversal, evidence building (unchanged) | (existing) |
| 5 | Realization L1 | Grammatical synthesizer (template-based) | 2 |
| 6 | Realization L2 | Small CPU realizer (structure-to-text) | 4 |

---

## Immutable Gate Thresholds

Thresholds are frozen once committed. A failed gate STOPS the program at that stage. 
Do not proceed past a failed gate. Do not tune thresholds after seeing results.

### Stage 0 — Baseline

| Gate | Threshold |
|------|-----------|
| Guard unit tests pass | all |
| paired_n > 0 | > 0 |
| RAG arm populated | rows > 0 |
| COMPARISON.md regenerated | no hand-typed numbers |

Accuracy values are recorded, not gated — whatever they are, they become the frozen baseline.

### Stage 1 — Associative Encoder

Measured on untouched test split, encoder path enabled:

| Gate | Threshold |
|------|-----------|
| entity_accuracy | >= 65% (baseline: 40%) |
| resolution_rate | >= current lexical path (no regression) |
| paraphrase_30: entity_accuracy drop vs original | < 10pp |
| intent accuracy | >= 85% |
| encoder RSS delta | <= 150 MB |
| inference p50 | <= 50 ms/question (CPU) |

IF FAILED: write STAGE1_NEGATIVE.md with per-head metrics, 20 worst cases, and failure hypothesis. STOP THE PROGRAM.

### Stage 2 — Realization L1

| Gate | Threshold |
|------|-----------|
| naturalness score increase vs current SynthesizingModel | > pre-registered margin |
| relevance | >= 77% |
| hallucination | <= current synth level |
| accuracy | no worse than -2pp vs current synth |

Pre-registered naturalness margin: **+5 points** (composite score 0-100).

### Stage 3 — Dialogue State

| Gate | Threshold |
|------|-----------|
| reference resolution accuracy | >= 70% (dialogue set) |
| single-turn benchmark unchanged | no regression (spot-check 30q) |
| state latency p50 | <= 5 ms |

### Stage 4 — Realization L2

| Gate | Threshold |
|------|-----------|
| naturalness | > L1 by >= 3 points |
| hallucination | <= L1 |
| accuracy | within 2pp of L1 |
| realizer RSS | <= 300 MB (int8) |
| generation | >= 20 tok/s CPU |
| full-stack budget | <= 500 MB total |

ENTRY CONDITIONS (all required, else skip):
- Stage 2 gate passed
- Naturalness score plateaued (two consecutive tuning attempts < +2 points)
- data/distillation/pairs.jsonl >= 5000 verifier-passed pairs

### Stage 5 — Freeze

| Gate | Threshold |
|------|-----------|
| Full test suite | passes |
| Every stage artifact in INDEX.md | yes |
| No metric in COMPARISON.md hand-typed | yes |

---

## Immutability Rule

Thresholds above are immutable once this file is committed. If a gate seems wrong later, the finding is "the gate was wrong" — documented in a STAGE*_NEGATIVE.md, not silently changed. A stopped program is a successful experiment.

---

## Naturalness Metric Definition (Stage 2)

Composite score 0-100, computed by `benchmarks/naturalness_eval.py`:

| Component | Weight | Description |
|-----------|--------|-------------|
| Aggregation rate | 25 | Facts merged per sentence (higher = fewer robot-like one-fact-per-sentence) |
| Connector presence | 20 | Edge-type-matched discourse connectors (caused_by→"because"/"ponieważ") |
| Referring expressions | 20 | Full name first mention, short form after |
| Repetition penalty | 20 | Lower score for repeated identical phrases |
| Mean sentence length | 15 | In natural band (10-25 words), not too short or long |

---

## Dialogue Set Specification (Stage 3)

50 dialogues, 2-5 turns each, covering:
- Anaphora: "why did IT fail"
- Ellipsis: "and at 3 hops?"
- Topic continuation
- Topic switch
- Clarification requests

Each turn annotated with:
- gt_entities: expected entity IDs
- resolution_source: "context" | "global"

---

## Stage 1B — Associative Encoder v2 (revision after Stage 1 gate failure)

**Date**: 2026-07-10
**Rationale**: Stage 1 stopped on intent_accuracy (65.3% < 85%). The threshold was not wrong — the implementation was. Same immutable thresholds, different architecture.

### Four Architectural Changes

1. **Intent: rule-first, model for remainder**. The QA dataset uses templated patterns ("Compare...", "Why did...", "What if...", "What was the goal of..."). A rule-based prefix/pattern classifier handles the majority; the encoder model classifies only cases without a pattern match. This costs zero weights and is in the spirit of the architecture (symbolic where symbolic suffices).

2. **Representation: char n-grams + sequence**. Char tri/penta-gram hashing (fastText-style) captures subword patterns for OOV terms like "verifier", "BCE", "PKM", "InfoNCE" — which caused 8 of 20 worst-case failures. A light sequential component (1-layer GRU or 2-layer mini-transformer, CPU-trainable) adds positional information the EmbeddingBag lacked.

3. **Entity head: re-ranker, not classifier**. Candidate entities come from the existing lexical/alias index (top-20 per question). The encoder only scores them — converting an open-set 1500-class problem into a 20-candidate ranking problem, appropriate for the dataset size. This aligns with the original chain-set retriever design (scoring candidates, not open-set classification).

4. **Class weights / focal loss** on intent head — mechanical fix for the factual_lookup majority-class dominance (93.8%→22.4% imbalance).

### Gates (same as Stage 1)

| Gate | Threshold |
|------|-----------|
| entity_accuracy | >= 65% |
| resolution_rate | >= current lexical path (no regression) |
| paraphrase_30: entity_accuracy drop | < 10 pp |
| intent_accuracy | >= 85% |
| encoder RSS delta | <= 150 MB |
| inference p50 | <= 50 ms/question (CPU) |

### Polish Robustness Note

Stage 1 showed a -16.7pp drop on the Polish paraphrase subset vs -8.0pp on English. Ensure the lemmatizer from Stage 1 normalize step is connected to the encoder input path and handles Polish inflection.

### Verdict Rule

If Stage 1b also fails: the negative is about the hypothesis, not the implementation, and STOP is final. If it passes: proceed to Stage 2.

---
