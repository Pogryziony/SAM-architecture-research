# NEXUS Realizer — bounded multi-evidence pilot

**Status:** `PILOT_CHECKPOINT_ACCEPTED` and `READY_FOR_FULL_TRAINING`. The
bounded 1→3 epoch pilot is complete, the full 356-record evaluation passes and
the accepted checkpoint is committed. Full training was not launched.

## Objective

Pointer/Copy v3 already solves factual answers that equal one evidence
candidate. The next neural experiment tests a different capability: combining
two independently sourced facts into one comparison while preserving every
source, subject and value exactly.

This is not a continuation of the rejected byte-level v2 checkpoint. It uses a
new dataset contract, a new split and a slot-preserving output contract.

## Honest data scope

The builder derives new task examples from the immutable train-only atomic
claim archive. It does not duplicate existing records or create paraphrases.
Each new record consumes two different atomic claims exactly once and produces
one comparison target that is not equal to either evidence candidate.

The source facts already exist in the train-only corpus, so this dataset adds a
new composition task rather than new external domain knowledge. All 44 source
families used by the consumed Realizer v1 validation split are quarantined from
the new dataset. Normalized questions and answers are checked against all v1
records and must have zero overlap.

Supported task families are:

- comparison of the same configuration key across two configuration files;
- comparison of the same table field across two independently authored source
  documents.

Source families are paired into disjoint components before splitting. An atomic
claim cannot appear in two compositions, and a source family cannot cross the
train/validation boundary.

## Correct reasoning/realization boundary

The first pilot exposed an architectural error. A free byte-level decoder was
asked both to compare values and to reproduce six control placeholders. Its
teacher-forced validation loss reached 0.0043 while free generation achieved
0% correct slots and 0% correct relations. A shorter relation-only target fixed
the structure but reached only 72% balanced relation accuracy (52% for
`DIFFERENT`). Neither failed checkpoint was promoted.

Comparison is deterministic symbolic reasoning and does not belong in the
Realizer. NEXUS now computes and verifies the relation from the two immutable
values. The neural input contains this verified answer plan as `SAME` or
`DIFFERENT`; a plan contradicting its values fails closed before inference.
The model is evaluated on adherence to the plan, not credited with discovering
the relation.

Constrained decoding selects only an allowed relation label. The runtime then
uses the fixed slot-preserving template:

```text
[SOURCE_1] reports [VALUE_1] for [SUBJECT_1], while [SOURCE_2] reports
[VALUE_2] for [SUBJECT_2]; the values are different.
```

Only after relation-plan selection does the runtime replace placeholders with
immutable sources, subjects and values. The model never regenerates paths,
keys or numbers. This is the intended NEXUS split: symbolic components decide
what is true; the constrained Realizer controls how the verified result is
expressed.

## Why BPE is not introduced in this pilot

Byte Pair Encoding (BPE) remains a possible later experiment, but the verified
v2 failure was evidence binding, not insufficient vocabulary size. Slots remove
the need to regenerate arbitrary identifiers. Keeping the existing tokenizer
for the first controlled comparison changes one major variable at a time. BPE
should be evaluated only if the slot model fails despite correct data and
serialization.

## Preparation gates

Before training, all of the following must pass:

| Gate | Requirement |
|---|---:|
| Unique multi-evidence records | at least 1,000 |
| Train records | at least 750 |
| Validation share | 15–25% |
| Consumed v1 validation families | all excluded |
| Old normalized question/answer overlap | 0 |
| Atomic claim reuse | 0 |
| Single-candidate targets | 0 |
| Largest relation class | at most 80% |
| Input binding coverage | 100% |
| CPU model parameters | at most 50 million |
| Initial loss | finite and at most 10 |
| Overfit smoke | at least 15% loss reduction |
| Weights written by preparation | none |

## Registered preparation result

The immutable dataset contains 1,642 compositions: 1,286 train and 356
validation records (21.6809%). It consumes 3,284 atomic claims exactly once,
quarantines 44 source families from the consumed v1 validation split and has
zero normalized question or answer overlap with all v1 records. It contains
1,383 configuration comparisons and 259 table comparisons.

The current readiness artifact reports `READY_FOR_BOUNDED_PILOT` with no
blocking checks, including independent verification of every symbolic relation
plan. The CPU model has 959,747 parameters. Preparation writes no weights.

Registered identities:

- source commit: `0d1aeae63712eacf5d3da014799a0303d8d0e61d`;
- source tree: `13373da1f034b9bc2ab44f5703f7dfa9f5c628f1`;
- dataset SHA-256: `7aaa4ee9566da98d67bf07f8a773b47e7dfb479a85b64d488e97446d2ef9b5c1`;
- config SHA-256: `92f5610fd900e927a43a7d40fc331d6c6425e87fbaaa61e2162e1766055a8ee4`;
- readiness canonical SHA-256: `c0eaaddaa96ef05b02d5bca2aaa5d60dfcbb5f59e7ad7f1c0a1e538ea46a3a50`.

Two independent readiness evaluations of the same immutable inputs produced
the same canonical hash. Timing remains diagnostic and is excluded from the
canonical identity.

## Pilot schedule and stop conditions

Training was limited to one epoch first and three epochs maximum. Epoch 1 was
allowed to continue only after all conditions passed on a relation-balanced
validation subset:

- slot placeholder exact rate 100%;
- overall relation-plan adherence at least 80%;
- per-class relation-plan adherence at least 70%;
- materialized exact match at least 80%.

Promotion requires at least 98% exact slot preservation, 95% overall and
per-class relation-plan adherence, 95% materialized exact match and at most 1%
hallucination. Falling loss cannot override a failed text-level gate.

## Registered pilot result

The final bounded pilot completed three CPU epochs in 64.9 seconds. Training
loss decreased from 0.1371 at epoch 1 to 0.00077 at epoch 3; validation loss
decreased from 0.000351 to 0.000129. Quality was already saturated after epoch
1 and remained unchanged:

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Full validation |
|---|---:|---:|---:|---:|
| Materialized exact match | 100% | 100% | 100% | 100% (356/356) |
| Relation-plan adherence | 100% | 100% | 100% | 100% |
| Minimum per-class adherence | 100% | 100% | 100% | 100% |
| Slot preservation | 100% | 100% | 100% | 100% |
| Hallucination | 0% | 0% | 0% | 0% |

The accepted epoch-3 weights have SHA-256
`bfa5855a57fba8db34e896d77848942733c5570049c927d4310646bea444e152`.
The full evaluation canonical SHA-256 is
`6a9d5e5756ebbdedd57432295de56196b003daa9e64336febc42ad15ac8ef6a2`.
The final readiness artifact says `READY_FOR_FULL_TRAINING`, has no blocking
checks and canonical SHA-256
`4fc860a48aa992d5daa22cf53a174bd423a5dc73c480bef21ec078b16d315139`.
Its `full_training_launched` field is `false`.

## Reproduction commands

```bash
python benchmarks/build_abstractive_realizer_dataset.py

python benchmarks/prepare_abstractive_realizer_run.py \
  --manifest data/distillation/realizer_abstractive_v1/manifest.json \
  --config training/nexus_realizer_abstractive_v1.json \
  --output benchmarks/results/realizer/abstractive_v1_readiness.json
```

The historical one-epoch command was:

```bash
python benchmarks/train_nexus_realizer_v2.py \
  --mode pilot \
  --epochs 1 \
  --gen-val-samples 50 \
  --checkpoint-epochs 1 \
  --manifest data/distillation/realizer_abstractive_v1/manifest.json \
  --config training/nexus_realizer_abstractive_v1.json \
  --output-dir models/realizer/abstractive_v1_pilot
```

Full checkpoint evaluation is reproduced with:

```bash
python benchmarks/evaluate_abstractive_realizer_checkpoint.py \
  --manifest data/distillation/realizer_abstractive_v1/manifest.json \
  --config training/nexus_realizer_abstractive_v1.json \
  --weights models/realizer/abstractive_v1_plan_v3/model.pt \
  --output /tmp/abstractive-v1-evaluation.json
```

No additional training is justified by the pilot metrics: epochs 2 and 3
reduced loss without improving answer quality. A full run remains technically
authorized but must be an explicit future decision; it was not launched here.

## Runtime integration

The accepted checkpoint is now available as the opt-in
`abstractive_plan_v3` backend. `ProductionNEXUSConfig.comparison_plan()` binds
the exact model directory, training configuration and expected checkpoint
SHA-256 into the immutable pipeline identity. `answer_question()` invokes this
backend only for questions parsed as comparisons.

For a complete grounded profile, `ProductionNEXUSConfig.grounded()` routes
factual lookups to Pointer/Copy and comparisons to this checkpoint while
leaving other intents on their existing path.

The runtime accepts exactly two supported registered evidence facts, derives
`SAME` or `DIFFERENT` symbolically, verifies the checkpoint and configuration
hashes, asks the model to follow that immutable plan and materializes sources,
subjects and values outside neural weights. Missing PyTorch, missing or changed
artifacts, ambiguous evidence, unknown evidence syntax and a neural label that
contradicts the plan all return `Insufficient evidence to answer.` without
falling through to unconstrained generation.

See `docs/realizer-abstractive-runtime-integration.md` for the integration
report and remaining scope limits.
