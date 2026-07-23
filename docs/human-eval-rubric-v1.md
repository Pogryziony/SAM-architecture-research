# Human adjudication rubric v1

**Schema:** `nexus-human-eval-rubric-v1`  
**Use:** Blinded free-form adjudication when deterministic scorers do not apply.  
**Rule:** An LLM judge may be diagnostic only — never sole source of truth.

## Protocol

1. Anonymize system identity (`system_A` / `system_B`, shuffled per item).
2. Two independent raters; third rater breaks ties.
3. Publish inter-annotator agreement (Cohen's κ or Krippendorff's α).
4. Version this file; do not silently edit after a sealed campaign starts.

## Decision per question

| Field | Values |
|-------|--------|
| `answer_correct` | yes / no / abstain_correct |
| `material_claims_supported` | yes / no / n_a |
| `citations_entail` | yes / no / n_a |
| `temporal_ok` | yes / no / n_a |
| `unsupported_material_claim` | yes / no |
| `grounded_correct` | derived: all required yes (or correct abstain) |

## Edge cases

- Empty answer on answerable question → not grounded.
- Correct abstain on `should_abstain` → grounded.
- Extra harmless flourish with all material claims supported → may still be grounded.
- Any unsupported material claim → not grounded.
