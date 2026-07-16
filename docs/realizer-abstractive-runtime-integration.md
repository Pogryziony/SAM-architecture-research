# NEXUS comparison-plan Realizer — runtime integration report

**Scope:** connect the accepted `abstractive_v1_plan_v3` pilot checkpoint to
the real NEXUS answer pipeline. This change does not launch or continue
training.

## Outcome

The checkpoint is no longer benchmark-only. The opt-in
`ProductionNEXUSConfig.comparison_plan()` configuration routes comparison
questions through `answer_question()` to the new fail-closed runtime in
`nexus/realizer/comparison_plan.py`. The recommended
`ProductionNEXUSConfig.grounded()` profile combines this path with Pointer/Copy
for factual lookups in one immutable pipeline configuration.

The integration preserves the architectural boundary established by the
pilot: NEXUS decides what is true; the neural Realizer only follows a verified
plan; immutable evidence is inserted after neural inference.

## Runtime flow

1. The existing parser identifies the question intent as `comparison` and
   resolves graph entry nodes.
2. Traversal or zero-hop evidence construction produces the evidence pack.
3. The comparison runtime accepts exactly two distinct, supported evidence
   facts. It does not attempt free-form information extraction from unknown
   prose.
4. Source, subject and value slots are recovered from the registered config or
   table evidence forms.
5. NEXUS compares the two normalized immutable values and creates a verified
   `SAME` or `DIFFERENT` plan.
6. The loader verifies the accepted manifest status, training configuration
   hash, weights hash and configured expected checkpoint SHA-256 before loading
   the model on CPU.
7. Constrained candidate scoring permits only the complete labels `SAME` and
   `DIFFERENT`.
8. A model label that differs from the symbolic plan is rejected.
9. The runtime materializes the exact sources, subjects and values into the
   registered answer template and sends the answer through the existing
   verifier and reasoning audit.

## Bound artifact

| Field | Registered value |
|---|---|
| Backend | `abstractive_plan_v3` |
| Combined grounded profile | `grounded_v1` |
| Model directory | `models/realizer/abstractive_v1_plan_v3` |
| Training config | `training/nexus_realizer_abstractive_v1.json` |
| Parameters | 959,747 |
| Weights SHA-256 | `bfa5855a57fba8db34e896d77848942733c5570049c927d4310646bea444e152` |
| Pilot validation | 356/356 exact |
| Full training launched | No |

## Fail-closed matrix

| Condition | Runtime result |
|---|---|
| Exactly two supported facts and matching model plan | Materialized comparison |
| Missing or more than two unresolvable facts | Insufficient evidence |
| Two facts from the same source | Insufficient evidence |
| Unknown evidence prose | Insufficient evidence |
| Missing PyTorch | Insufficient evidence |
| Missing model/config/manifest | Insufficient evidence |
| Config, manifest or weights hash mismatch | Insufficient evidence |
| Unsupported model label | Insufficient evidence |
| Model label contradicts symbolic plan | Insufficient evidence |
| Non-comparison question | Existing configured NEXUS path; this backend is not invoked |

No failure in the comparison backend falls through to unconstrained LLM text
generation for a comparison question.

## Verification

`tests/test_abstractive_realizer_runtime.py` covers:

- symbolic same/different plans;
- config and table evidence forms;
- unknown and ambiguous evidence;
- a deliberately contradictory model result;
- missing checkpoint behavior;
- immutable production configuration identity;
- the complete `answer_question()` zero-hop flow without a synthesis model;
- actual checkpoint inference on a registered validation record;
- actual checkpoint inference through `answer_question()` on a balanced
  validation sample containing both relation classes.

The normal Python 3.11/3.12 jobs exercise the dependency-free behavior. The
CPU-PyTorch CI job explicitly includes this test module so the neural tests run
with the committed checkpoint instead of being silently skipped.

## Activation

```python
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.reasoning.answer import answer_question

config = ProductionNEXUSConfig.grounded()
result = answer_question(question, graph, config=config)
print(result["answer"])
print(result["realization"])
```

PyTorch is an optional runtime dependency and is required only when this neural
backend is selected. Install the `train` extra for local inference.

## Honest limitations

- The backend supports the two registered config/table evidence forms, not
  arbitrary prose.
- It supports exactly two evidence units and one comparison operator.
- The emitted sentence is currently an English registered template.
- The neural output language contains only two control labels. The 356 unique
  complete answers come from evidence materialization, not open-ended neural
  generation.
- This integration does not establish quality for aggregation, explanation,
  causality, arbitrary multi-hop synthesis or Polish surface realization.
- A separate unseen real-question benchmark is still required before calling
  the broader Realizer production-ready.

## Next decision

Do not run a longer training job on the unchanged comparison dataset: answer
quality saturated after epoch 1. The next justified experiment is a new,
train-only controlled surface-realization dataset with multiple templates and
Polish/English output contracts, followed by another bounded 1→3 epoch pilot.
That experiment must preserve exact evidence slots and must not weaken the
fail-closed runtime introduced here.
