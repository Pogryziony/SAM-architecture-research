# Stage 1D — HONEST PASS

Date: 2026-07-10

Stage 1D evaluated a parser handoff cap selected from validation only. The cap search began at 30 and expanded to `[100, 150, 200]` after the 30-cap validation result remained below the immutable gate. The selected cap was **200** with threshold **0.20**, selected from `stack/encoder/data/val.jsonl` (150 questions); the frozen test split was not read during calibration.

## Validation calibration

Artifact: `benchmarks/results/entity_threshold_calibration_stage1d_capupper_20260710.json`

| Cap | Selected validation recall | p50 (ms) | Parser success |
|---:|---:|---:|---:|
| 100 | 63.74% | 30.18 | 100.0% |
| 150 | recorded in artifact | recorded in artifact | 100.0% |
| 200 | 69.23% | 30.32 | 100.0% |

Selection rule: maximum validation recall for the parser cap; within the selected cap, maximum validation F1, then recall, then lowest threshold. No test IDs or test answers were used for selection.

## Frozen evaluation

Artifact: `benchmarks/results/stage1b_honest_20260710_162457Z.json`

The artifact was written, read back from disk, and passed `validate_stage1b_artifact`. Its metadata and configuration agree on `max_entry_nodes=200`, `entity_threshold=0.20`, the validation calibration artifact, and the frozen split identifier.

| Metric | Result | Gate |
|---|---:|---:|
| Entity recall | **65.45% (180/275)** | >= 65% PASS |
| Resolution rate | 100.0% | >= baseline PASS |
| Paraphrase drop | 0.0 pp | < 10 pp PASS |
| Intent accuracy | 85.3% | >= 85% PASS |
| RSS delta | 6.4 MB | <= 150 MB PASS |
| Inference p50 | 34.9 ms | <= 50 ms PASS |

Decision: **HONEST PASS**. The 65% gate and all other checks remained unchanged and enabled.

## Targeted implementation change

The calibration and evaluation CLIs now propagate an explicit parser handoff cap into `NEXUSConfig.max_entry_nodes`. The serialized artifact validator checks that the selected cap is present and agrees with the runtime configuration. Targeted tests cover cap propagation and metadata/configuration mismatch rejection.

Historical artifacts were preserved; the final artifact is additive and does not replace Stage 1B/1C result files.
