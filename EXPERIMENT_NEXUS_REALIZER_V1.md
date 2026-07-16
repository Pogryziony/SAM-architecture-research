# EXPERIMENT: NEXUS Realizer v1 — CPU-First Evidence-to-Answer Model

**Pre-registered**: 2026-07-11
**Status**: V1 and neural v2 checkpoints rejected; Pointer/Copy v3 accepted for
extractive factual QA. The separate constrained comparison-plan pilot
checkpoint is accepted after three bounded CPU epochs and full validation.
Full training is ready but was not launched.
**Repository**: SAM-architecture-research

---

## Summary

Train a CPU-first, small-parameter evidence-to-answer model that produces
grounded natural-language answers from structured NEXUS evidence packs.
The model must NOT store the knowledge graph in its weights — it acts as
a language interface that reads structured evidence and produces answers.

## Entry Conditions

1. Stage 2 relevance >= 77%
2. Stage 2 naturalness >= 5pt improvement over baseline
3. Stage 2 hallucination no worse than baseline
4. Realizer v1 manifest >= 5000 verifier-passed unique pairs under the
   `nexus-realizer-v1` contract
5. No pairs derived from validation or holdout labels; zero entity-family
   overlap between train and validation

The historical `data/distillation/pairs.jsonl` file is not eligible: its
records predate structured evidence, proof audit, split identity, and source
hash requirements.

## Architecture

```
Input:  question + structured evidence pack (JSON)
Output: grounded natural-language answer
```

### Model

- Base: Small transformer or T5-style encoder-decoder (≤ 50M params)
- Tokenizer: deterministic UTF-8 byte/character tokenizer (259 symbols)
- Input serialization: `[QUESTION] <text> [EVIDENCE] <json> [ANSWER]`
- Maximum evidence length: 1024 tokens
- CPU-first training with deterministic seed handling

### Training Protocol

- Loss: cross-entropy on answer tokens
- Optimizer: AdamW with cosine annealing
- Learning rate: 1e-4 (to be validated on learning curves)
- Seed: 20260711
- Pilot order: 1, then 3, then at most 5 epochs with generation-aware checks
- Extended presets: 8 or 12 epochs only after a separate decision based on registered pilot metrics
- Batch size: 16
- Train/validation split: 80/20 by entity-family to reduce leakage

### Baselines

1. **Template realizer**: Rule-based answer construction from evidence
2. **SynthesizingModel**: Current template-based synthesizer
3. **Trivial evidence-copy baseline**: Copies evidence text verbatim

### Selection Rule

- Winner = argmax(relevance), tie-break by naturalness, then accuracy
- Validation-only model selection
- Never use the frozen split for selection

## Immutable Gates

| Gate | Threshold |
|------|-----------|
| Relevance | >= 77% |
| Naturalness improvement | >= 5 points over template baseline |
| Hallucination rate | <= registered baseline |
| Accuracy | >= baseline - 2pp |
| CPU peak RSS | <= 500 MB |
| Inference p50 | <= 500 ms |
| Weight size | <= 100 MB |

## Failure Conditions

1. Training fails to converge (loss plateau > 20 epochs)
2. Validation relevance < 77% after full training
3. Any model output shown to be memorizing graph facts
4. Checkpoint corruption or hash mismatch
5. Hardware exhaustion (RSS > 500 MB during training)

## Outputs

Per example:
- Question
- Evidence pack
- Predicted answer
- Ground truth
- Relevance
- Naturalness
- Accuracy
- Hallucination

## Artifact Policy

- Weights: may be committed under a configured `models/` subdirectory; SHA-256 and byte size are mandatory in the manifest.
- Config: committed
- Tokenizer: committed (text file, not binary)
- Training log: committed
- Validation artifact: committed
- Per-example predictions: committed

## Pre-training implementation

- `training/nexus_realizer_v1.json` freezes the 2.78M-parameter CPU model and
  optimizer configuration.
- `benchmarks/run_nexus_oracle.py` evaluates the 181-case oracle contract.
- `benchmarks/build_distillation_dataset.py` creates hash-verified,
  entity-family-disjoint data.
- `benchmarks/acquire_realizer_train_data.py` extracts unique atomic targets
  from an explicit train-only repository corpus; it does not paraphrase the
  existing questions or read validation/test labels.
- `benchmarks/check_realizer_readiness.py` aggregates model/data gates into one
  `READY_FOR_TRAINING` or `BLOCKED` artifact.
- `benchmarks/check_phase4_readiness.py` combines Phase 0–4 evidence into the
  final `GO_FOR_REALIZER_TRAINING` decision.
- `benchmarks/train_nexus_realizer.py` provides no-write preflight,
  overfit-smoke, and guarded training modes. Repository weights are accepted
  only below the configured `models/realizer/` root.

Current verified results and the launch procedure are documented in
`docs/nexus-realizer-pretraining-status.md`.

## Separate comparison pilot readiness

The next neural experiment does not reuse the extractive v1 task. The
`nexus-realizer-abstractive-v1` dataset composes two independent train-only
claims into one comparison answer. Its 1,642 records use 3,284 atomic claims
without reuse, quarantine all 44 consumed validation source families and have
zero normalized question/answer overlap with v1. Source families are split as
disjoint components, yielding 1,286 train and 356 validation records.

The initial output contract incorrectly asked the neural Realizer to perform
comparison reasoning and reproduce six byte-level placeholders. It failed the
epoch-1 quality gates despite very low teacher-forced loss. NEXUS now computes
and verifies the relation symbolically, then passes a `SAME`/`DIFFERENT` answer
plan to the neural Realizer. Constrained decoding preserves the plan; exact
source paths, subjects and values are materialized from immutable evidence.
This tests instruction adherence without crediting the model for reasoning
performed by the graph-first system.

The accepted 959,747-parameter checkpoint completes the bounded 1→3 epoch
schedule. All three epochs score 100% on their balanced generation checks. The
full 356-record validation reports 100% exact materialization, 100% adherence
for both relation classes, 100% slots, 0% hallucination and fail-closed handling
of a contradictory plan. Weights SHA-256:
`bfa5855a57fba8db34e896d77848942733c5570049c927d4310646bea444e152`.

`benchmarks/results/realizer/abstractive_v1_plan_v3_full_readiness.json`
reports `READY_FOR_FULL_TRAINING`, canonical SHA-256
`4fc860a48aa992d5daa22cf53a174bd423a5dc73c480bef21ec078b16d315139`
and `full_training_launched: false`. Quality did not improve after epoch 1, so
additional epochs require a separate justification rather than falling loss.

## First-run outcome and protocol amendment

The first external CPU run completed 50 epochs and reduced training loss from
191.815 to 1.867, with best validation loss 1.778. It nevertheless regressed on
all registered answer-quality measures. Decoder diagnostics showed that the
byte-level greedy autoregressive path entered repetition loops. Repetition
penalty `1.2` and no-repeat trigram blocking removed the observed loops.

This does not retroactively make the trained checkpoint acceptable. The
corrected Phase 0–4 contract now passes: 7,127 unique train-only pairs, valid
oracle and Stage 0, deterministic registered Stage 2, passing Stage 3, model
readiness, CPU preflight and overfit smoke. The next run must use the corrected
generation-aware trainer and proceed through 1, 3 and at most 5 epochs. Every
run records effective preset values. Training stops on non-finite loss,
validation regression, repetition recurrence or worsening registered
answer-quality metrics.

## Pilot run outcome: `REALIZER_PILOT_FAIL`

**Run ID**: `run_20260716T100428Z`
**Date**: 2026-07-16
**Commit**: `e1575bbb2c18496141d71969c13d9b5e586cd789` (PR #22 merge)
**Tree SHA**: `7b1920947a155404c58bf98701fdd4f8d54c696e`

### Training summary

| Metric | Value |
|--------|-------|
| Epochs completed | 4 (stopped early: gen quality regression) |
| Configured epochs | 5 |
| Total elapsed | 54.5 minutes |
| Initial loss | 191.815 |
| Final train loss | 3.539 |
| Best validation loss | 2.591 (epoch 4) |
| Model parameters | 2,770,752 |
| Peak RSS | ~6,860 MB (exceeded 500 MB budget) |

### Epoch metrics

| Epoch | Train Loss | Val Loss | Gen Coh | Rep3 | EOS | Time |
|-------|-----------|----------|---------|------|-----|------|
| 1 | 47.612 | 4.481 | 100% | 0.00 | 100% | 13.9 min |
| 2 | 4.546 | 2.970 | 100% | 0.00 | 100% | 13.4 min |
| 3 | 3.810 | 2.678 | 100% | 0.00 | 100% | 13.4 min |
| 4 | 3.539 | 2.591 | 100% | 0.00 | 100% | 13.6 min |

### Checkpoint evaluation (100 validation samples)

| Epoch | SHA-256 | Unique outputs | Uniqueness |
|-------|---------|---------------|------------|
| 1 | `d27885f1...` | 1/100 | 1.0% |
| 3 | `de3bcb60...` | 4/100 | 4.0% |

### Gate results

| Gate | Status | Value | Threshold |
|------|--------|-------|-----------|
| Relevance | **FAIL** | 0.0% | >= 77% |
| Accuracy | **FAIL** | 0.0% | >= baseline - 2pp |
| Naturalness | **FAIL** | 0.0 | >= 5pt improvement |
| Hallucination | **FAIL** | 100% | <= baseline |
| Coherence (bytes) | PASS | 100% | — |
| EOS | PASS | 100% | — |
| Repetition (bytes) | PASS | 0.00 | — |

### Root cause: Mode collapse

Both checkpoints suffer from catastrophic mode collapse:
- **Epoch 1**: ALL 100 samples produce identical output: `ooo senamtri___plll,,, cccfffIII`
- **Epoch 3**: Only 4 unique outputs across 100 samples, all variations of `In sam/coretigl, ...`

The byte-level coherence/EOS/repetition metrics misleadingly show perfect scores
because they operate on byte-level token sequences that happen to be short and
non-repeating at the byte level, while the text-level output is completely
meaningless.

### Key findings

1. **Loss is misleading**: Train loss dropped from 191.8 to 3.5, but generation quality did not improve. Loss alone cannot determine model quality.
2. **Generation-aware checks caught the issue**: The gen_patience safety stop triggered at epoch 4, preventing wasted computation.
3. **Memory budget exceeded**: Peak RSS of ~6.9 GB far exceeds the 500 MB budget. The Python process with PyTorch, the model, and dataset all contribute.
4. **Byte-level tokenizer limitations**: The byte-level tokenizer (259 symbols) prevents the model from learning meaningful word/subword patterns, contributing to mode collapse.
5. **Model capacity likely insufficient**: 2.78M parameters may be inadequate for learning evidence-to-answer mapping across 7,127 training pairs with diverse question types.

### Recommended next steps

1. Switch from byte-level to BPE/subword tokenizer (e.g., 1000-8000 tokens)
2. Increase model capacity: d_model ≥ 256, more encoder/decoder layers
3. Add diversity-promoting training: unlikelihood loss, scheduled sampling
4. Add beam search or nucleus sampling (top-p) for diverse generation
5. Implement streaming/batched inference to control memory
6. Consider curriculum: train on short factual answers first, then longer ones
7. Add runtime text-level diversity metrics (not just byte-level rep_3gram)

### Decision

**No checkpoint is acceptable.** Both checkpoint epochs (1 and 3) fail all
registered answer-quality gates. Training artifacts are preserved for
diagnostics. This does NOT authorize a longer training run — the identified
architectural issues must be addressed first.

Full report: `benchmarks/results/realizer/run_20260716T100428Z/pilot_report.json`

## Corrected diagnosis and v2 amendment

The section above is the immutable historical v1 result. Subsequent direct
diagnostics showed that the strongest verified causes were not model size
alone: the tied embedding/output design began at loss about `191.8` instead of
the uniform expectation `ln(259) = 5.56`, and all accepted targets are already
present in the structured evidence.

The amended implementation adds `stable_transformer_v2` and a fail-closed
grounded Realizer. On all 1,434 validation records, grounded realization reaches
100% exact match, 100% token F1, 0% hallucination and 1,434 unique outputs,
without using labels during answer generation. Stable v2 starts near loss 6.0
and reduces it to about 1.61 in the 50-step no-write overfit smoke.

These results validate the runtime fallback and training mechanics, not a new
neural checkpoint. Only raw neural output metrics may promote a checkpoint; the
grounded fallback is reported separately. See
`docs/realizer-v2-quality-recovery.md`.

## Budget Compliance

| Resource | Budget |
|----------|--------|
| CPU training | Required |
| GPU | Not required |
| Peak RSS | <= 500 MB |
| Inference p50 | <= 500 ms |
| Model size | <= 100 MB |
| Training time | <= 24 hours on single CPU |
