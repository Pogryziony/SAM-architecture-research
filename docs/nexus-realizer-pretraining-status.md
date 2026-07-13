# NEXUS Realizer v1 — pre-training status

**Decision: READY TO LAUNCH TRAINING in the PyTorch training environment.**

Training has not started. The repository-side data, quality, leakage, oracle,
budget, and artifact-policy gates pass. The final launch command must still run
the readiness check in the same environment that has the `train` extra; the
command remains fail-closed unless its exact status is `READY_FOR_TRAINING`.

## Verified result (2026-07-13)

| Gate | Observed | Required | Status |
|---|---:|---:|---|
| Unique acquired train-only targets | 8,282 | >= 5,000 candidates | PASS |
| Verifier/audit-passed pairs | 7,127 | >= 5,000 | PASS |
| Validation share after source-family grouping | 20.12% | 15–25% | PASS |
| Semantic-target overlap between splits | 0 | 0 | PASS |
| Source-file overlap between splits | 0 | 0 | PASS |
| Stage 2 relevance | 78.33% | >= 77% | PASS |
| Stage 2 naturalness improvement | +22.40 | >= +5 | PASS |
| Stage 2 hallucination delta | -0.0512 | <= 0 | PASS |
| Stage 2 accuracy delta | +0.1534 | >= -0.02 | PASS |
| Oracle cases | 181 | >= 150 | PASS |
| Oracle proof validity | 96.13% | >= 95% | PASS |
| Oracle gold-path recall | 85.71% | >= 80% | PASS |
| Oracle provenance coverage | 96.13% | >= 90% | PASS |
| Top-three evidence retention | 100% | 100% | PASS |
| Estimated parameters | 2,779,200 | <= 50M | PASS |
| CPU PyTorch forward/backward preflight | passed in GitHub Actions | pass | PASS |

The 8,282 candidates come from 199 source families and contain one record per
source property, Markdown table cell, authored claim, or public API contract.
They are not paraphrases of the old question set. Acquisition rejects repeated
semantic targets, normalized questions, and normalized answers. It excludes
evaluation/result code, generated result directories, and all validation,
test, and holdout labels.

The dataset builder accepted 7,127 records and rejected 1,155 fail-closed,
primarily because the target verifier did not support the authored wording.
Accepted records contain only the atomic fact being trained and its one-hop
`claim -> source` proof; other claims from the same source document are not
included as neighbor context. The generated clean dataset is about 26 MB
(5,693 train and 1,434 validation records).

## Reproduction and launch order

```bash
python benchmarks/acquire_realizer_train_data.py \
  --output data/realizer_train/source_claims_v1

python benchmarks/build_distillation_dataset.py \
  --acquisition-manifest data/realizer_train/source_claims_v1/manifest.json \
  --output-dir data/distillation/realizer_v1 \
  --min-pairs 5000

python benchmarks/run_nexus_oracle.py \
  --output /tmp/nexus-oracle-realizer-v1.json

python benchmarks/run_stage2_stage3.py \
  --stage 2 --limit 30 --output-dir /tmp/nexus-stage2-realizer-v1

pip install -e '.[train]'

python benchmarks/check_realizer_readiness.py \
  --dataset-manifest data/distillation/realizer_v1/manifest.json \
  --oracle-artifact /tmp/nexus-oracle-realizer-v1.json \
  --stage2-artifact /tmp/nexus-stage2-realizer-v1/STAGE2_FILE.json \
  --output /tmp/nexus-realizer-v1-readiness.json
```

Start `benchmarks/train_nexus_realizer.py --mode train` only after the last
artifact says `READY_FOR_TRAINING`. Model weights must remain outside the
repository; their SHA-256 is recorded in the external training manifest.
