# EXPERIMENT: Rule Engine V2 — Frozen Split Opened

**Pre-registered**: 2026-07-21  
**Status**: ACTIVE — frozen eval opened  
**Repository**: SAM-architecture-research  
**Preregistration ID**: `rule-engine-v2`  
**Rules source**: `benchmarks/qa-dataset/rule_corpus_v1.json` (development identity `rule-engine-v1`)  
**Frozen corpus**: `benchmarks/qa-dataset/rule_corpus_v1_frozen.json`  
**Eval**: `python benchmarks/eval_rule_engine.py --mode frozen`

---

## Purpose

Open the Stage 4 frozen rule evaluation after growing the development corpus
beyond the toy engine. The frozen graph/gold are held out; rules must not be
retuned against the frozen artifact after this preregistration lands.

## Frozen corpus identity

| Field | Value |
|---|---|
| `corpus_id` | `rule_corpus_v1_frozen` |
| `frozen_file_sha256` | `58c36ca889b9e0d44ac476dac92de30ad7ed60715ddcaf2e24e0a6c480d8f03b` |
| Rules | Shared with development `rule_corpus_v1` (≥12 Horn rules) |
| Frozen graph | Disjoint node IDs from development (`P`…`Cause`) |
| Gold inferred facts | Embedded `frozen_gold` (closure under shared rules) |

## Preregistered frozen gates

| Metric | Threshold |
|---|---:|
| Precision | ≥ 0.90 |
| Recall | ≥ 0.90 |
| F1 | ≥ 0.90 |
| File SHA-256 match | exact |
| Every inferred fact has `rule_id` + non-empty `premises` | 100% |

## Relationship to V1

`EXPERIMENT_RULE_ENGINE_V1.md` remains the development prereg. V2 only opens
frozen evaluation; development growth continues under V1 thresholds with the
raised minimum rule count (≥12).

## Non-goals

- Retuning rules after inspecting frozen failures
- Unbounded recursion / full Datalog
- Treating development F1 as a frozen PASS
