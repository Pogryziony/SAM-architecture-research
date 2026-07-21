# NEXUS architecture validation verdict

**Date:** 2026-07-21  
**Artifact:** `benchmarks/results/architecture_validation_20260721T241500Z.json`  
**Paired evidence:** `oracle_vs_predicted_union_l1_acceptance_full_20260721T241500Z.json`  
**Decision:** **VALIDATED**

## Question

Is NEXUS a good architecture that can perform well under honest paired
evaluation (without AnswerPlan weight training)?

## Checks (preregistered)

| Check | Threshold | Observed | Pass |
|---|---:|---:|:---:|
| Entry recall | ≥ 0.90 | 0.9786 | yes |
| Path recall | ≥ 0.90 | 0.9697 | yes |
| Proof valid rate | ≥ 0.90 | 0.9634 | yes |
| Predicted fact | ≥ 0.70 | 0.7159 | yes |
| Beats RAG fact | NEXUS > RAG | 0.7159 > 0.0482 | yes |
| Beats LLM-only fact | NEXUS > LLM | 0.7159 > 0.0000 | yes |
| AnswerPlan binding | oracle≥0.50 and lag≥0.15 | lag≈0.016 | no (sealed) |

## Arms

- **NEXUS:** union lexical∪ER3 + `l1_acceptance` + SynthesizingModel
- **RAG:** lexical chunk retrieval + SynthesizingModel
- **LLM-only:** EvidenceBlindModel (no evidence pack)

## Training decision

- **Rung A (lexical ER hygiene):** done — entry ≥ 0.95 without ER3 retrain.
- **Rung B (ER3 refresh):** not authorized — remaining misses were alias gaps,
  not new-canonical-node coverage after hygiene.
- **Rung C (AnswerPlan):** sealed — predicted does not lag oracle by ≥0.15.
- **Rung D (full realizer train):** not needed for the architecture verdict.

## Summary

On frozen `oracle_v1` (191q), NEXUS L1 saturates graph/ER metrics, clears the
surface fact gate, and substantially outperforms RAG and evidence-blind
baselines. The architecture claim stands on structured retrieval +
deterministic L1 realization — not on learned AnswerPlan weights.
