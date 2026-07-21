# EXPERIMENT: Rule Engine V1 — Bounded Horn Corpus + Preregistered F1

**Pre-registered**: 2026-07-21  
**Status**: ACTIVE — development corpus; frozen eval opened under `rule-engine-v2`  
**Repository**: SAM-architecture-research  
**Preregistration ID**: `rule-engine-v1`  
**Corpus**: `benchmarks/qa-dataset/rule_corpus_v1.json`  
**Eval**: `python benchmarks/eval_rule_engine.py --corpus ... --mode development`

---

## Purpose

Grow the Stage 4 rule engine beyond a single transitive toy rule. Lock F1
thresholds and corpus identity **before** any frozen evaluation. The
development set may be used to verify the harness; the frozen split must not
be opened until this preregistration is committed.

## Corpus identity

| Field | Value |
|---|---|
| `corpus_id` | `rule_corpus_v1` |
| `rule_count` | ≥12 (versioned Horn rules; grown beyond toy engine) |
| Relations used | `depends_on`, `derived_from`, `replaces`, `validates` |
| Development graph | Embedded in corpus JSON (`development_graph`) |
| Gold inferred facts | Embedded in corpus JSON (`development_gold`) |

## Preregistered development gates

| Metric | Threshold |
|---|---:|
| Precision | ≥ 0.90 |
| Recall | ≥ 0.90 |
| F1 | ≥ 0.90 |
| Every inferred fact has `rule_id` + non-empty `premises` | 100% |

## Frozen evaluation

- Frozen split opened under **`EXPERIMENT_RULE_ENGINE_V2.md`**
  (`preregistration_id=rule-engine-v2`) with published file SHA-256.
- Do not retune rules against the frozen artifact after V2 lands.

## Non-goals

- Full Datalog / recursion without bounds
- Production KG rule mining
- Treating development F1 as a frozen PASS
