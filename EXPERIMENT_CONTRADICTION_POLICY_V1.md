# EXPERIMENT: Contradiction Policy V1

**Pre-registered**: 2026-07-21  
**Status**: ACTIVE — partial Stage 5 scaffold  
**Preregistration ID**: `contradiction-policy-v1`

## Policy classes

| Class | Meaning |
|---|---|
| `contradiction` | Explicit `contradicts` edge among proof/evidence nodes |
| `supersession` | `replaces` edge indicates newer fact supersedes older |
| `validity_mismatch` | Bi-temporal validity windows do not overlap for a claim |
| `source_disagreement` | Distinct sources assert incompatible relations |

## Gate (partial)

Any **unresolved** conflict ⇒ readiness recommendation must not be
unconditional `answer` (at most `conditional_answer` or `abstain`).

## Frozen contradiction F1

Sealed until a future prereg publishes gold conflict labels and thresholds.
Development unit tests only in this version.
