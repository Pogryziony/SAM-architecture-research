# EXPERIMENT: Contradiction Policy V1

**Pre-registered**: 2026-07-21  
**Status**: ACTIVE — development F1 / calibration campaign opened  
**Preregistration ID**: `contradiction-policy-v1`  
**Gold**: `benchmarks/qa-dataset/contradiction_gold_v1.jsonl`  
**Eval**: `python benchmarks/eval_contradiction_policy.py`

## Policy classes

| Class | Meaning |
|---|---|
| `contradiction` | Explicit `contradicts` edge among proof/evidence nodes |
| `supersession` | `replaces` edge indicates newer fact supersedes older |
| `validity_mismatch` | Bi-temporal validity windows do not overlap for a claim |
| `source_disagreement` | Distinct sources assert incompatible relations |

## Gate

Any **unresolved** conflict ⇒ readiness recommendation must not be
unconditional `answer` (at most `conditional_answer` or `abstain`).

## Development campaign thresholds

| Metric | Threshold |
|---|---:|
| Conflict-class macro F1 | ≥ 0.90 |
| Policy recommendation accuracy | ≥ 0.90 |
| Unconditional answer leaks on unresolved conflicts | 0 |
| Calibration | report Brier + ECE on readiness scores (diagnostic) |

Frozen contradiction F1 is opened under **`EXPERIMENT_CONTRADICTION_POLICY_V2.md`**
(`contradiction-policy-v2`) with a published LF-normalized gold SHA-256.
Development gold may still evolve under V1 thresholds; do not retune against
the frozen file.
