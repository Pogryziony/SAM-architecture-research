# EXPERIMENT: NEXUS Realizer v1 — CPU-First Evidence-to-Answer Model

**Pre-registered**: 2026-07-11
**Status**: PREREGISTERED — training blocked until Stage 4 entry conditions are met.
**Repository**: SAM-architecture-research

---

## Summary

Train a CPU-first, small-parameter evidence-to-answer model that produces
grounded natural-language answers from structured NEXUS evidence packs.
The model must NOT store the knowledge graph in its weights — it acts as
a language interface that reads structured evidence and produces answers.

## Entry Conditions

1. Stage 2 relevance >= 77% (currently 60% — BLOCKED)
2. Stage 2 naturalness >= 5pt improvement over baseline
3. Stage 2 hallucination no worse than baseline
4. `data/distillation/pairs.jsonl` >= 5000 verifier-passed unique pairs
5. No pairs derived from validation or holdout labels

## Architecture

```
Input:  question + structured evidence pack (JSON)
Output: grounded natural-language answer
```

### Model

- Base: Small transformer or T5-style encoder-decoder (≤ 50M params)
- Tokenizer: character n-gram or BPE
- Input serialization: `[QUESTION] <text> [EVIDENCE] <json> [ANSWER]`
- Maximum evidence length: 1024 tokens
- CPU-first training with deterministic seed handling

### Training Protocol

- Loss: cross-entropy on answer tokens
- Optimizer: AdamW with cosine annealing
- Learning rate: 1e-4 (to be validated on learning curves)
- Seed: 20260711
- Epoch limit: 50 with early stopping (patience 10 on validation loss)
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

- Weights: NEVER committed to git. SHA-256 recorded in manifest.
- Config: committed
- Tokenizer: committed (text file, not binary)
- Training log: committed
- Validation artifact: committed
- Per-example predictions: committed

## Budget Compliance

| Resource | Budget |
|----------|--------|
| CPU training | Required |
| GPU | Not required |
| Peak RSS | <= 500 MB |
| Inference p50 | <= 500 ms |
| Model size | <= 100 MB |
| Training time | <= 24 hours on single CPU |
