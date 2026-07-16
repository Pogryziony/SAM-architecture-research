# NEXUS Realizer AnswerPlan v1

**Data status:** `READY_FOR_BOUNDED_PILOT`

**Overall status:** `FULL_TRAINING_BLOCKED`
**Test:** sealed

## Why this stage was necessary

Corpus v2 originally carried a semantic operator and gold evidence, but not a
fully resolved proposition. Feeding that form directly to a neural model would
quietly make the Realizer infer the answer from text. That is the job of NEXUS
graph traversal, path scoring and verification—not of the language layer.

`AnswerPlan v1` fixes the boundary. Upstream logic supplies the canonical
answer, immutable values and exact provenance. The Realizer can improve wording
but cannot choose evidence, change facts or replace a symbolic decision. This
matches the project flow:

`query → graph reasoning → verified evidence → AnswerPlan → Realizer → verifier`

## Contract

`nexus/realizer/answer_plan.py` compiles and validates:

- language, operator and final `answer`/`abstain` decision;
- the canonical resolved answer and aliases;
- immutable values that must survive generation;
- one resolved claim bound to every required evidence identifier;
- source title, locator and evidence hash for provenance;
- the invariant `reasoning_owner = nexus_upstream`;
- the invariant `realizer_may_change_facts = false`;
- a content-derived stable plan identifier.

Any changed answer, hash, evidence ID, locator or ownership flag invalidates the
plan. Raw evidence paragraphs remain in corpus v2; the compact neural
serialization carries the resolved fact and provenance references. This avoids
turning the Realizer into a second, hidden reasoner.

## Prepared data

Run:

```bash
python benchmarks/prepare_realizer_answer_plan_v1.py \
  --corpus-root /external/NEXUS-realizer-corpus-v2 \
  --output /external/NEXUS-answer-plan-v1 \
  --max-pieces 4096
```

The current verified run keeps 144,746 train records, 3,660 native validation
records and a new 2,408-record document-disjoint holdout. The holdout moves 344
whole document groups out of train and contains 500 abstentions. It creates no
additional examples and performs no translation or paraphrase.
The 12,126 test records were not opened. Their registered source hash is copied
into a seal record for a single final evaluation after checkpoint selection.

External artifacts are intentionally not committed. Their identities are:

| Artifact | SHA-256 |
|---|---|
| Prepared manifest | `a4c5045b4b416cd6c7299d629faab16ed3c5dff7603e1d9ef3ccb9d50a517a2e` |
| Tokenizer | `32bd240c30054a147521a72246da20bb535e53e83f8a9467cf9d5a453162fb9d` |
| Data readiness | `2ca4b28ba9bbf1a0e8dc190854888a3906feda727bcdb681ab396683059b950f` |
| Full-training readiness | `10663240140424febb463622f005c0e366c9b769ff63e9abe49e3874004d8e75` |

## Tokenizer and length audit

`TrainOnlySubwordTokenizer` learns the 4,096 most frequent lexical and
whitespace pieces from serialized train inputs and train targets only. Unknown
pieces fall back losslessly to UTF-8 bytes. This is deliberately described as
a frequency subword tokenizer, not as BPE (Byte Pair Encoding) or SentencePiece.

It solves the immediate correctness issue: unlike the old byte tokenizer it
does not truncate every input to 256 bytes and it has no unknown-token path.
All prepared train and validation input/target strings round-tripped with zero
failures. The tokenizer never sees the moved holdout while fitting.

| Tokens | Train p99 | Train max | Validation p99 | Validation max | Budget |
|---|---:|---:|---:|---:|---:|
| Input | 412 | 1,202 | 396 | 1,559 | 2,048 |
| Target | 116 | 992 | 46 | 1,366 | 2,048 |

Required plan fields may never be truncated. A later production tokenizer may
replace this implementation with a pinned BPE/Unigram model, but it must retain
train-only fitting, lossless Polish support and the same leakage tests.

## Deterministic validation baselines

| Baseline | Exact match | Token F1 | Immutable values | Unsupported numbers |
|---|---:|---:|---:|---:|
| AnswerPlan copy | 85.58% | 89.08% | 100% | 0% |
| Registered PL/EN template | 0% | 72.74% | 100% | 0% |
| First evidence-title pointer | 5.75% | 10.08% | 16.32% | 2.21% |

Copy is a safety baseline, not a target for stylistic improvement. A neural
checkpoint is useful only if it retains the same factual safety while improving
natural wording on tasks whose target is more than a short canonical answer.

## Pilot protocol

The registered configuration is
`training/realizer_answer_plan_v1.json`. The permitted sequence is:

1. overfit smoke on 64 records; no promotable weights;
2. one epoch on 2,048 stratified records;
3. one epoch on at most 17,000 unique neural-eligible records;
4. only after both pass, request separate authorization for at most three full
   epochs with one-epoch early-stopping patience.

Every stage must report PL/EN and operator slices, immutable preservation,
unsupported numbers, exact match, token F1, EOS, empty output and repetition.
Generation must use immutable-value constraints and fail closed to AnswerPlan
copy when the verifier rejects neural output.

## Data correction and pilot result

The PoQuAD builder now preserves the extractive `text` as a fact alias and the
human `generative_answer` as the target. Previously it discarded the fact alias,
which made the target appear verbatim in AnswerPlan. The corrected train split
contains 27,255 genuine surface-transform records, without adding a single row.

The abstention coverage blocker is resolved, but neural generation is not. A
one-epoch 2,048-record Transformer pilot reached validation loss 2.70 and
teacher-forced token accuracy 41.5%, yet generated `exact_match=0`, token F1
29.3% and EOS 52.1%. Pointer-generator, fact-only copy masks, byte-granularity
tokens and beam decoding were tested; none passed the overfit generation gate.

`check_answer_plan_full_training_readiness.py` therefore reports
`FULL_TRAINING_BLOCKED` for three checks: overfit generation, small generation
and the intentionally unlaunched representative pilot. The final immutable
contract preserves numbers and identifiers exactly while permitting validated
linguistic inflection; earlier pilot checkpoints are diagnostics and are not
identity-compatible with this final artifact. Full training and frozen test
evaluation remain prohibited. Rejected pilot checkpoints stay external.

## Next technical work

The next implementation should use a non-autoregressive copy/edit transducer or
a constrained edit-script target, not another unconstrained sequence decoder.
Abstention stays deterministic because upstream NEXUS has already made that
decision. The new architecture must first pass 64-record overfit and the 2,048
pilot before the representative stage can run.
