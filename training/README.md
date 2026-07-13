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
   The builder verifies every source hash and creates source-family-disjoint
   train/validation files.
4. Produce a Stage 2 artifact containing the four preregistered baseline deltas:
   relevance, naturalness, hallucination, and accuracy.
5. Run `benchmarks/check_realizer_readiness.py` and resolve every blocking gate.
6. Install `pip install -e '.[train]'`, then run
   `benchmarks/train_nexus_realizer.py --mode preflight` and
   `--mode overfit-smoke`.
7. Start `--mode train` only with the valid readiness artifact. The output
   directory must be outside this repository; its weight SHA-256 is recorded
   in an external manifest.

The committed configuration is `training/nexus_realizer_v1.json`. No command
overwrites an existing dataset, evaluation artifact, or training directory.
