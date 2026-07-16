# NEXUS Realizer v2 — quality recovery

**Status (2026-07-16): grounded runtime passes; neural checkpoint remains
unselected until a new short pilot passes text-level gates.**

## What failed

The historical v1 checkpoints are not deployable. Epoch 1 produced one unique
answer for 100 validation questions and epoch 3 produced four. Falling
teacher-forced loss and byte-level repetition metrics therefore did not prove
answer quality.

## Verified root causes

The follow-up diagnosis found two stronger causes than the earlier tokenizer
and capacity hypotheses:

1. The legacy tied embedding/output head was badly scaled. The untrained model
   started at cross-entropy about `191.8`, while a uniform 259-token model is
   expected near `ln(259) = 5.56`. Its initial logits had standard deviation
   about `16.7` and absolute maximum above `217`.
2. The current distillation task is extractive. On all 7,127 accepted records,
   the target answer is already present as an answer-bearing graph fact. Asking
   an autoregressive language model to recreate the same bytes adds failure
   modes without adding information.

BPE (Byte Pair Encoding), a larger network and scheduled sampling remain valid
experiments for future abstractive data, but they were not established as the
primary cause of this failure and are not prerequisites for the extractive
contract now present in the repository.

## Implemented recovery

### Grounded answer path

`nexus/realizer/grounded.py` ranks structured evidence without reading the
evaluation label. It accepts a neural answer only when the answer is readable,
contains no unsupported number or material token claim, and passes the
grounding threshold. Otherwise it copies the highest-ranked complete evidence
fact. With no usable evidence it fails closed with `Insufficient evidence to
answer.`

This is the deliberate move away from an LLM-first architecture: NEXUS already
stores the answer as provenance-bearing graph evidence, so deterministic
realization is safer, faster and auditable. A language model is an optional
surface-form improver, not the source of truth and not a mandatory dependency.

### Stable neural path

`stable_transformer_v2` keeps historical v1 checkpoints loadable but uses:

- a final layer normalization;
- separate input embedding and output projection weights;
- controlled `d_model^-0.5` initialization;
- an initial-loss fail-closed gate;
- compact evidence-first serialization;
- text-level exact match, token F1, similarity, grounding and uniqueness;
- checkpoint selection based only on raw neural output quality, never on the
  deterministic fallback;
- early stopping on mode collapse, repetition, validation regression and
  sampled memory-budget violation.

The v2 configuration uses 910k–1.1M parameters (depending on position budget),
batch size 2, three epochs, patience 1 and complete 704-byte source/target
budgets. Per-epoch diagnostic generation is capped at 128 tokens to keep the
CPU gate bounded; complete grounded answers are not truncated. It deliberately
avoids another 50-epoch run.

## Current evidence

| Check | Result |
|---|---:|
| Full validation records | 1,434 |
| Grounded exact match | 100% |
| Grounded token F1 | 100% |
| Grounded hallucination rate | 0% |
| Unique final answers | 1,434 / 1,434 |
| Median grounded latency | about 0.04 ms |
| Stable v2 initial loss | about 6.0 |
| Stable v2 overfit smoke | 6.03 → 1.61 in 50 steps |
| Weights written by safety checks | none |

Labels are used only after realization to score the validation output. They are
not available to the ranking or answer path.

## Promotion rule

The grounded runtime may pass independently. A neural checkpoint may be
promoted only after a fresh 1–3 epoch pilot passes raw-neural text metrics on an
untouched validation split. The fallback metrics are reported separately and
must never hide a collapsed checkpoint. Historical v1 checkpoints remain
rejected.
