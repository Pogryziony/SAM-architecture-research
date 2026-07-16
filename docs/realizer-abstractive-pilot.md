# NEXUS Realizer — bounded multi-evidence pilot

**Status:** `READY_FOR_BOUNDED_PILOT`. The committed readiness artifact passes
all 15 blocking checks. No training was launched and no checkpoint was
promoted by this preparation run.

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

## Slot-preserving target

The neural target contains placeholders for exact evidence bindings:

```text
[SOURCE_1] reports [VALUE_1] for [SUBJECT_1], while [SOURCE_2] reports
[VALUE_2] for [SUBJECT_2]; the values are different.
```

The model must reproduce all six placeholders exactly once and decide whether
the values are `the same` or `different`. Only after generation does the
runtime replace placeholders with immutable evidence values. This prevents a
neural decoder from inventing or misspelling paths, keys and numbers.

The input serializer contains both evidence bindings and the question. It does
not contain the training target or the final materialized answer.

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

The readiness artifact reports `READY_FOR_BOUNDED_PILOT` with no blocking
checks. The CPU model has 1,058,051 parameters. The no-write preflight loss is
6.179254 with finite gradients; the 30-step overfit smoke reduces loss from
6.192694 to 1.360204 (78.04%).

Registered identities:

- source commit: `cd3824bf6a4f7f158eec58dc552706accdc2a3a8`;
- source tree: `45e31734f384688c2a7e49bf8b4cfd575f2c9c5e`;
- dataset SHA-256: `7aaa4ee9566da98d67bf07f8a773b47e7dfb479a85b64d488e97446d2ef9b5c1`;
- readiness canonical SHA-256: `e854a80695aede2fa703898923f47aa8f54353393ff8088143ccabdb0dce3361`.

Two independent readiness evaluations of the same immutable inputs produced
the same canonical hash. Timing remains diagnostic and is excluded from the
canonical identity.

## Pilot schedule and stop conditions

Training is limited to one epoch first and three epochs maximum. Epoch 1 may
continue only when all three conditions pass on raw neural output:

- slot placeholder exact rate at least 80%;
- relation accuracy at least 70%;
- materialized exact match at least 40%.

Promotion requires at least 98% exact slot preservation, 95% relation accuracy,
95% materialized exact match and at most 1% hallucination. Falling loss cannot
override a failed text-level gate. A failed epoch-1 gate stops the run and marks
the checkpoint rejected.

## Reproduction commands

```bash
python benchmarks/build_abstractive_realizer_dataset.py

python benchmarks/prepare_abstractive_realizer_run.py \
  --manifest data/distillation/realizer_abstractive_v1/manifest.json \
  --config training/nexus_realizer_abstractive_v1.json \
  --output benchmarks/results/realizer/abstractive_v1_readiness.json
```

After a committed readiness result passes, the next action is exactly one epoch:

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

That command is documented for the next run; this preparation change does not
execute full training or promote a checkpoint.
