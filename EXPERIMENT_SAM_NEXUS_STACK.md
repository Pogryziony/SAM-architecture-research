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
