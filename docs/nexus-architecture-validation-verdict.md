# NEXUS architecture validation verdict

**Date:** 2026-07-21 (campaign) / 2026-07-22 (claim-scope clarification)  
**Artifact:** `benchmarks/results/architecture_validation_20260721T241500Z.json`  
**Paired evidence:** `oracle_vs_predicted_union_l1_acceptance_full_20260721T241500Z.json`  
**Decision (historical artifact string):** `VALIDATED`  
**Decision (current interpretation / new runner):** `VALIDATED_INTERNAL`  
**Claim scope:** **internal repository validation contract only**

> Historical status note: this document previously used unqualified “VALIDATED”
> language. The underlying arms are deterministic placeholders, not real LLMs
> or modern RAG. See [`CURRENT_STATE.md`](CURRENT_STATE.md).

## Question

Does NEXUS pass a preregistered **internal** paired campaign on frozen
`oracle_v1` without AnswerPlan weight training?

## Checks (preregistered)

| Check | Threshold | Observed | Pass |
|---|---:|---:|:---:|
| Entry recall | ≥ 0.90 | 0.9786 | yes |
| Path recall | ≥ 0.90 | 0.9697 | yes |
| Proof valid rate | ≥ 0.90 | 0.9634 | yes |
| Predicted fact | ≥ 0.70 | 0.7159 | yes |
| Beats placeholder RAG fact | NEXUS > placeholder RAG | 0.7159 > 0.0482 | yes |
| Beats evidence-blind placeholder | NEXUS > placeholder | 0.7159 > 0.0000 | yes |
| AnswerPlan binding | oracle≥0.50 and lag≥0.15 | lag≈0.016 | no (sealed) |

## Arms

- **NEXUS:** union lexical∪ER3 + `l1_acceptance` + `SynthesizingModel` (deterministic)
- **RAG placeholder:** lexical chunk retrieval + `SynthesizingModel` (**not** modern RAG)
- **“LLM-only” placeholder:** `EvidenceBlindModel` (**not** a real closed-book LLM)

## Training decision

- **Rung A (lexical ER hygiene):** done — entry ≥ 0.95 without ER3 retrain.
- **Rung B (ER3 refresh):** not authorized — remaining misses were alias gaps,
  not new-canonical-node coverage after hygiene.
- **Rung C (AnswerPlan):** sealed — predicted does not lag oracle by ≥0.15.
- **Rung D (full realizer train):** not needed for the internal architecture verdict.

## Summary

On frozen `oracle_v1` (191q), NEXUS L1 saturates graph/ER metrics, clears the
surface fact gate, and substantially outperforms **deterministic placeholder**
baselines. This supports an **internal** architecture contract for structured
retrieval + deterministic L1 realization. It does **not** authorize claims that
NEXUS outperforms real, version-pinned LLM or competitive RAG systems.
