# EXPERIMENT: Contradiction Policy V2 — Frozen Gold Opened

**Pre-registered**: 2026-07-21  
**Status**: ACTIVE — frozen gold opened  
**Preregistration ID**: `contradiction-policy-v2`  
**Frozen gold**: `benchmarks/qa-dataset/contradiction_gold_v1_frozen.jsonl`  
**Eval**: `python benchmarks/eval_contradiction_policy.py --mode frozen`

---

## Purpose

Seal the Stage 5 development contradiction campaign behind a published frozen
file hash. Development gold under V1 may continue to grow; frozen eval must
not be retuned after this preregistration lands.

## Frozen corpus identity

| Field | Value |
|---|---|
| `gold_id` | `contradiction_gold_v1_frozen` |
| `frozen_file_sha256` | `2cee684a620402fa58bbbd7006edbeb36fd65c813475fba9ea6f795e86cce3d5` (LF-normalized) |
| Record count | 12 |
| Classes | contradiction, supersession, validity_mismatch, source_disagreement, none |

## Preregistered frozen gates

| Metric | Threshold |
|---|---:|
| Conflict-class macro F1 | ≥ 0.90 |
| Policy recommendation accuracy | ≥ 0.90 |
| Unconditional answer leaks on unresolved conflicts | 0 |
| File SHA-256 match | exact (LF-normalized) |

## Relationship to V1

`EXPERIMENT_CONTRADICTION_POLICY_V1.md` remains the development campaign.
V2 only opens frozen evaluation against the sealed gold file.
