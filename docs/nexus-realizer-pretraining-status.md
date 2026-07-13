# NEXUS Realizer v1 — pre-training status

**Decision: BLOCKED. Do not start training yet.**

The repository now contains the complete fail-closed path to the first model:
an oracle benchmark, safe dataset builder, compact evidence serialization,
training/readiness configuration, CPU Transformer, preflight, overfit smoke,
training loop, artifact hashing, and weight-output protection.

## Diagnostic result (2026-07-13)

The diagnostics below were generated from the repository's allowed train and
validation inputs. They are engineering preflight results, not published model
results.

| Gate | Observed | Required | Status |
|---|---:|---:|---|
| Safe distillation pairs | 85 | >= 5,000 | BLOCKED |
| Validation share after entity grouping | 47.06% | 15–25% | BLOCKED |
| Stage 2 relevance | 63.33% | >= 77% | BLOCKED |
| Stage 2 naturalness improvement | +11.29 | >= +5 | PASS |
| Stage 2 hallucination delta | -0.1266 | <= 0 | PASS |
| Stage 2 accuracy delta | +0.0334 | >= -0.02 | PASS |
| Oracle cases | 181 | >= 150 | PASS |
| Oracle proof validity | 96.13% | >= 95% | PASS |
| Oracle gold-path recall | 85.71% | >= 80% | PASS |
| Oracle provenance coverage | 96.13% | >= 90% | PASS |
| Top-three evidence retention | 100% | 100% | PASS |
| Estimated parameters | 2,779,200 | <= 50M | PASS |
| PyTorch training runtime | absent | installed | BLOCKED |

The 375 eligible train questions produced 85 accepted records. Rejections were
driven mainly by target-verifier failure (231 occurrences) and audit abstention
(111 occurrences); a record can have more than one rejection reason. The
manifest correctly counts 290 rejected records.

## Required next actions

1. Add genuinely new train-only questions/evidence-answer pairs until the
   builder accepts at least 5,000. Do not multiply the current 375 questions
   with superficial templates and do not use validation/test labels.
2. Increase the number of disconnected entity families so an 80/20 grouped
   split is possible without entity leakage.
3. Improve evidence selection/realization until the registered 30-question
   Stage 2 relevance gate reaches 77%; keep the already-passing deltas intact.
4. Install the `train` extra in the training environment and run preflight and
   the eight-example overfit smoke. Neither mode writes weights.
5. Regenerate the readiness artifact. Start training only if its exact status
   is `READY_FOR_TRAINING`.

See `training/README.md` for command order. Generated datasets and model
weights are ignored or rejected inside the repository; final weight SHA-256 is
recorded in the external training manifest.
