# Current Status

What has been confirmed, what hasn't, and what the risks are.

**Updated: 2026-06-18**

## Confirmed (strong evidence from controlled experiments)

These results are backed by experiments using the same synthetic dense dataset
and the same model architecture. They are internally consistent and reproducible.

| Result | Evidence | Confidence |
|--------|----------|------------|
| **Oracle memory works** — SAM core can use memory for reasoning | 0.6: oracle_memory 99.87% | High |
| **Controlled noise tolerance** — SAM handles up to +8 random distractors with >91% accuracy | 0.13A: +8 dist = 91.6% overall | High |
| **One distractor does not collapse SAM** — 99.82% with +1 | 0.13A: vs 100% clean | High |
| **3-hop degrades first under noise** — from 100% at +0 to 79.3% at +8 to 39% at +16 | 0.13A | High |
| **Chain retrieval is solved** — all_required@32 = 100% with chain-set BCE | 0.11 | High |
| **Dual encoder retrieval works for dense data** — Rec@8 99.0% | 0.5 | High |
| **SAM core matches dense baseline at equal params** — both 68.74% without memory | 0.6 | High |
| **Product-key memory addressing works** — used successfully in all experiments | All 0.5-0.13A | High |
| **Critical padding bug is fixed** — `-1` slots no longer contaminate memory | Post-0.12 | High |

## Not yet confirmed (hypotheses or experiments pending)

These are NOT yet supported by experiments. They may be true or false — we
don't know yet.

| Hypothesis | Status |
|-----------|--------|
| Realistic non-oracle retrieved memory beats core-only after padding fix | **0.13B pending** |
| Realistic hard distractors are as tolerable as random distractors | **0.13B pending** |
| Learned selector path works post-fix | **Not yet rerun** |
| Reducing topK (4, 8, 16 instead of 32, 64) helps accuracy | **Configs prepared, not yet run** |
| Chain-set retriever external text query is well-wired post-padding-fix | **Not yet validated** |
| SAM is more CPU/RAM efficient than dense baselines | **Not measured** |
| SAM scales to larger models or datasets | **Not tested** |
| The current architecture beats a well-trained small dense baseline with RAG | **Not compared** |
| Latent memory aggregation (averaging) is better than slot-wise reader | **Not compared** |
| Product-key memory scales to millions of slots | **Not tested** |
| Noise tolerance holds for realistic, semantically misleading distractors | **0.13B will test** |

## Known risks and concerns

### 1. Synthetic-only data risk

All experiments use the `data/synthetic_dense` dataset — template-generated
questions about made-up entities. The templates are designed to exercise
reasoning chains, but:

- Real-world QA has much more complex language
- Real distractors may be more deceptive
- Templates may have unintentional patterns the model can exploit
- No out-of-distribution evaluation exists

### 2. Tiny scale risk

- 16M parameters, 1,650 slots, 853 vocabulary tokens
- Results may not transfer to larger, more realistic model sizes
- The core model memorizes the token templates (core_only = 68.74% on a task
  with 42K possible answers — well above random)

### 3. Post-bug revalidation needed

The padding bug (slot `-1` clamped to slot 0) may have invalidated:
- Experiment 0.11 non-oracle baselines
- Experiment 0.12 selector training
- Any result where padding masks interacted with the memory path

These should be rerun on the fixed codebase.

### 4. Realistic distractor quality risk

0.13A showed that random distractors are tolerable. But realistic distractors
from the retriever may be qualitatively different:
- They score highly because they are semantically related
- They may activate competing "fact chains"
- The selector's distractors with 50% precision might be misleading, not random

### 5. Aggregation architecture risk

All memory values are flattened (averaged) into one vector. With more slots
and more distractors, this averaging may lose information. A slot-wise reader
that can attend to individual slots differently might be needed. The current
aggregation may hit a ceiling at some noise level.

### 6. Gate training-dynamics risk

The gate learns to suppress memory when it's noisy during early training.
There is no mechanism yet to force the gate to re-open later in training when
memory quality improves. This could create a one-way ratchet: noisy early
batches → gate shuts → good later batches can't reopen it.

### 7. No formal validation

The project is in an early exploration phase. No peer review, no external
reproduction, no standardized benchmarks. All results are preliminary.

## Where the project stands today

```
SAM Architecture Research
│
├── Core implementation ......... COMPLETE (v0.1)
├── Product-key memory .......... WORKING
├── Dual encoder retriever ...... WORKING (99% Rec@8 on dense data)
├── Chain-set retriever ......... WORKING (all_required@32 = 100%)
├── Oracle memory validation .... DONE (99.87%)
├── Controlled noise tolerance .. DONE (0.13A — positive signal)
│
├── Learned selector ............ PARTIAL (96.6% recall, 50% precision, not yet usable)
├── Realistic distractor replay . IN PROGRESS (0.13B)
├── Full non-oracle pipeline .... NOT YET VALIDATED
├── Efficiency metrics .......... NOT MEASURED
├── Scale testing ............... NOT STARTED
├── Real-world data ............. NOT STARTED
│
└── Formal validation ........... NOT STARTED
```

---

*Last updated: 2026-06-18*
