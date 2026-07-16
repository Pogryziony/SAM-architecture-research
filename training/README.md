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

The committed configuration is `training/nexus_realizer_v1.json`; its default
is 5 epochs with early-stopping patience 3. No command overwrites an existing
dataset, evaluation artifact, report or training directory.
