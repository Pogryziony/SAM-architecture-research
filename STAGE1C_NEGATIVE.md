# Stage 1C — HONEST FAIL

Date: 2026-07-10

Stage 1C added deterministic graph-only candidate expansion from aliases, key findings, and descriptions, plus 1,560 provenance-bearing generated alias/key-finding/relation training pairs. Calibration used only `stack/encoder/data/val.jsonl` (150 questions), selected threshold `0.10`, and did not read or modify the frozen test split during calibration.

## Frozen evaluation

Artifact: `benchmarks/results/stage1b_honest_20260710_152608Z.json`

| Metric | Result | Gate |
|---|---:|---:|
| Entity recall | 50.55% (139/275) | >= 65% **FAIL** |
| Candidate-pool recall | 85.82% | diagnostic |
| Resolution rate | 100.0% | >= 100% PASS |
| Intent accuracy | 85.3% | >= 85% PASS |
| RSS delta | 6.8 MB | <= 150 MB PASS |
| Inference p50 | 40.3 ms | <= 50 ms PASS |

The serialized artifact was read back and passed `validate_stage1b_artifact`. All 225 frozen question IDs matched the expected frozen split. The immutable gate was not changed or disabled.

## Exact blocker

Candidate generation improved only from the Stage 1B diagnostic (~82.97%) to 85.82%, leaving 39 gold IDs absent from the candidate pool. A further 97 gold IDs were selected by the reranker but fell outside the existing five-entry parser cap. The resulting recall remains exactly 139/275, so candidate expansion alone did not change accepted recall; the reranker/parser-cap path is now the dominant blocker after candidate coverage.

## Concrete next proposal

Run a separately preregistered Stage 1D experiment that trains a graph-only pairwise reranker on the generated pairs and adds a validation-calibrated, provenance-preserving top-k policy for the parser handoff. First measure this policy on `val.jsonl`; do not tune against `test.jsonl`. Keep `max_entry_nodes` and all gate thresholds unchanged, and stop if validation shows a resolution, latency, or precision regression.
