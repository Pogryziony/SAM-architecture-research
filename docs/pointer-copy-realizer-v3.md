# NEXUS Pointer/Copy Realizer v3

**Decision (2026-07-16): accepted for extractive factual QA. The neural v2
checkpoint remains rejected.**

## Why the previous checkpoint failed

The v2 pilot learned output shape: it reached end-of-sequence reliably, stopped
producing empty answers and avoided mode collapse. It did not learn the binding
between a question and the exact value, identifier or path in its evidence.
Consequently, all raw neural answers failed grounding even though token F1 and
similarity improved. Increasing model size or lowering the grounding threshold
would not fix that verified failure mode.

The dataset audit revealed a more fundamental issue: this is not currently a
generation task. All 5,693 train records and all 1,434 validation records have
an answer that exactly equals one complete structured evidence candidate. The
target is always available without reconstructing it byte by byte.

## Chosen architecture

Pointer/Copy v3 performs four bounded operations:

1. Extract deduplicated candidates from `node_facts`, `snippets`, path nodes and
   facts.
2. Score each candidate from its evidence kind, confidence, overlap with the
   question and overlap between the question and its source identifier.
3. Select using a stable candidate identifier as the final tie-break. Candidate
   list position and the answer label are never features.
4. Copy the selected text verbatim. If no candidate exists, its score is too
   low, or the top two candidates are too close, return `Insufficient evidence`
   and a machine-readable rejection reason.

This preserves filenames, configuration keys, integers, floating-point values,
percentages, quotes and Unicode. It also makes the answer traceable through
`selected_candidate_id`, `selected_candidate_kind`, `evidence_source`,
`selection_score` and `selection_margin`.

## Why the selector is deterministic, not trained

The current evidence ranking and the lexical Pointer/Copy baseline both achieve
100% top-1/exact accuracy on the registered validation split. In addition, the
answer is the first ranked evidence candidate in every current record. A learned
selector would add checkpoint, reproducibility and overfitting risk without
demonstrated benefit. Position-shuffled evaluation is mandatory to ensure the
implementation does not merely reuse the first-item shortcut.

No neural weights are needed for the accepted v3 path. The existing v1/v2
checkpoints remain historical diagnostics and are not silently promoted.

## Runtime integration

`nexus/realizer/pointer_copy.py` contains the selector. It is connected to
`nexus.reasoning.answer.answer_question` for `factual_lookup` intent only. Enable
it explicitly:

```python
from nexus.pipeline.config import ProductionNEXUSConfig

config = ProductionNEXUSConfig.pointer_copy()
```

The default backend remains `synth` to preserve the semantics and hashes of
previously registered Stage 2 runs. Comparison, multi-evidence synthesis and
other non-extractive intents are not rerouted to Pointer/Copy.

## Reproducible evaluation

```bash
python benchmarks/evaluate_pointer_copy_v3.py \
  --config training/pointer_copy_realizer_v3.json \
  --manifest data/distillation/realizer_v1/manifest.json \
  --output /tmp/pointer_copy_v3.json
```

The evaluator validates the dataset manifest, records dataset and split hashes,
the effective selector configuration and source tree, writes an exact SHA-256
sidecar and refuses to overwrite existing evidence. Labels are used only after
realization for target classification and scoring. Realization reads only the
question and evidence pack.

Acceptance gates are:

| Metric | Gate |
|---|---:|
| Candidate availability | at least 99% |
| Exact match | at least 98% |
| Mean token F1 | at least 99% |
| Unsupported numbers | 0% |
| Unsupported identifiers | 0% |
| Hallucination | at most 1% |
| Unique outputs | at least 80% |
| Exact-match drop after deterministic candidate permutation | at most 1 pp |
| Median selection latency | at most 5 ms |
| Synthetic adversarial checks | all pass |

The adversarial set includes a correct answer outside the first position, wrong
number, wrong file, wrong configuration key, missing evidence and conflicting
evidence. Missing and conflicting evidence must fail closed.

### Registered result

The registered 1,434-record validation run is
`benchmarks/results/realizer/pointer_copy_v3_20260716.json`. It reports 100%
candidate availability, exact match, token F1, uniqueness and shuffled-order
exact match; unsupported-number, unsupported-identifier, wrong-candidate and
hallucination rates are all 0%. Median selection latency is 0.037 ms and all six
adversarial checks pass. Its canonical SHA-256 is
`046b53747fb2e722f4ed6cbd56b392df1920360a87a347d5e5de2c5caef1deab`.

## Grounding diagnostics

`nexus.realizer.grounded.grounding_diagnostics` reports both a continuous
support score and the exact reason a strict grounding decision failed. It
distinguishes unreadable output, unsupported numbers, unsupported identifiers
and excessive unsupported tokens. This avoids the old diagnostic problem where
every failed answer appeared only as `grounding=0`.

## Scope and limitations

The accepted claim is deliberately narrow: NEXUS can reliably realize factual
answers already represented as complete evidence candidates. It does not prove
that NEXUS can compose a comparison, explanation or answer spanning multiple
facts. The current dataset is strongly position-biased even though the
evaluation neutralizes this bias by permuting candidates.

Before another neural training run, acquire new train-only records that are
genuinely unique and genuinely abstractive. They should require controlled
composition from multiple evidence items, remain source-family-disjoint from
validation and have explicit support annotations. Do not duplicate current
records, paraphrase labels into artificial variants or lower grounding gates.

## Related files

- `nexus/realizer/pointer_copy.py` — accepted selector and diagnostics payload.
- `nexus/realizer/grounded.py` — candidate extraction and grounding diagnostics.
- `nexus/reasoning/answer.py` — production runtime connection.
- `training/pointer_copy_realizer_v3.json` — frozen selection policy.
- `benchmarks/evaluate_pointer_copy_v3.py` — registered full evaluation.
- `tests/test_pointer_copy_realizer.py` — exact-copy, invariance, fail-closed and
  runtime integration tests.
- `benchmarks/train_nexus_realizer_v2.py` — historical neural trainer; scheduler
  horizon and checkpoint metadata are corrected, but it is not the accepted v3
  runtime.
