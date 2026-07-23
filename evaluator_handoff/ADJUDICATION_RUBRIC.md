# Adjudication rubric

Score each dimension in {0, 0.5, 1} unless noted:

1. **Conclusion correctness** — final answer matches gold conclusion.
2. **Material-claim support** — key claims supported by permitted evidence.
3. **Citation entailment** — cited evidence actually entails claims.
4. **Completeness** — required facets present.
5. **Temporal correctness** — as-of / validity / known-time correct.
6. **Unsupported claims** — 1 = none; 0 = material hallucinations.
7. **Abstention appropriate** — for unanswerable/insufficient cases.

Process: two independent annotators; track disagreements; third adjudicator or defined resolution; report agreement (e.g. Cohen's κ).

LLM judges may be diagnostic only — never sole ground truth.
