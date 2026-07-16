# NEXUS Realizer v1 — pre-training runbook

Training is fail-closed. A run may start only when the readiness artifact says
`READY_FOR_TRAINING`; old `data/distillation/pairs.jsonl` records do not satisfy
the v1 contract because they lack structured evidence, proof audit, source
split, and reproducibility hashes.

## Required order

1. Build the oracle benchmark from the registered validation and relation-gold
   sources with `benchmarks/run_nexus_oracle.py`.
2. Acquire atomic train-only source claims with
   `benchmarks/acquire_realizer_train_data.py`. It creates one target per
   source property/table cell/claim/API contract and rejects semantic,
   question, and answer duplicates.
3. Build at least 5,000 verifier-passed pairs with
   `benchmarks/build_distillation_dataset.py --acquisition-manifest ...`.
   Use `--archived-acquisition` only to reproduce the committed, hash-verified
   acquisition snapshot after source documents have legitimately changed.
   The builder creates source-family-disjoint train/validation files.
4. Run Stage 0, registered Stage 2 for seeds 0/1/42, and the complete Stage 3.
5. Run `benchmarks/check_realizer_readiness.py` and resolve every blocking gate.
6. Install `pip install -e '.[train]'`, then run
   `benchmarks/train_nexus_realizer.py --mode preflight` and
   `--mode overfit-smoke`.
7. Run `benchmarks/check_phase4_readiness.py`. Start training only when it says
   `GO_FOR_REALIZER_TRAINING`.
8. Execute 1, then 3, then at most 5 epochs. `models/realizer/` is the only
   configured in-repository checkpoint root; an external output directory is
   also valid. Every checkpoint SHA-256 is recorded in its run manifest.

The historical configuration is `training/nexus_realizer_v1.json`. New runs use
`training/nexus_realizer_v2.json`: three epochs, patience 1, batch size 2,
stable initialization and compact evidence-first input. No command overwrites an existing
dataset, evaluation artifact, report or training directory.

## Pilot outcome: `REALIZER_PILOT_FAIL`

**Run**: `run_20260716T100428Z` (2026-07-16)
**Result**: Mode collapse at all checkpoint epochs. No acceptable checkpoint.
**Artifacts**: `models/realizer/run_20260716T100428Z/`
**Report**: `benchmarks/results/realizer/run_20260716T100428Z/pilot_report.json`

Key findings:
- Loss decreased from 191.8 to 3.5, but generation quality did not improve
- Epoch 1: 1 unique output across 100 validation samples
- Epoch 3: 4 unique outputs across 100 validation samples
- Byte-level coherence/EOS/repetition metrics are misleading — they pass while text-level quality fails
- Memory budget (500 MB) exceeded: peak RSS ~6.9 GB
- The later diagnosis superseded the initial tokenizer/capacity hypothesis: pathological v1 initialization and an extractive objective were the verified primary causes

## V2 recovery status

- `nexus/realizer/grounded.py` returns complete supported evidence and rejects
  unreadable, numerically invented or materially unsupported neural answers.
- `stable_transformer_v2` starts near the theoretical uniform loss instead of
  about 191 and passes a 50-step overfit smoke without writing weights.
- `benchmarks/evaluate_grounded_realizer.py` evaluates labels only after answer
  generation and writes a hash sidecar.
- Full validation result: 1,434/1,434 exact, 0% hallucination and 1,434 unique
  outputs. This validates the grounded runtime, not a neural checkpoint.
- The next run is a single 1–3 epoch neural pilot. Never extend it merely
  because loss falls; promote only raw-neural text quality.

See `docs/realizer-v2-quality-recovery.md` for the design and promotion gates.
