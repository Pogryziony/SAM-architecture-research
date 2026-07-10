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

## Phase C4 final endpoint (2026-07-10)

Artifact: `benchmarks/results/stage1c_final_20260710T180311Z.json`.
The artifact was read back and validated with `validate_final_artifact`; all 225 frozen IDs were present and unique, and the recorded evaluation commit was `98169e8a23bb63d78a03c5f0dfa41eda6e01badf`.

| Metric | Test result | Gate |
|---|---:|---:|
| Winner recall@1 | 9.45% | diagnostic |
| Winner recall@5 | 37.82% | diagnostic |
| Winner recall@10 | 53.45% | >=65% **FAIL** |
| Winner precision@10 | 6.56% | reported |
| Trivial baseline recall@10 | 35.64% | reported |
| Intent accuracy | 90.67% | >=85% PASS |
| Paraphrase drop at K=10 | -14.71 pp | absolute <10 pp **FAIL** |
| Resolution rate | 100.0% | no regression PASS |
| Inference p50 | 20.8 ms | <=50 ms PASS |
| RSS | 47.3 MB | <=150 MB PASS |

The validation control passed mechanically: baseline validation recall@10 was 35.71%, while the frozen selected winner validation recall@10 was 71.32%, a gap of 35.61 percentage points (at least 15 pp). The control also reports the baseline test metrics in the final artifact. Because the primary and paraphrase gates failed, the mechanical decision is **HONEST FAIL**. The feature-logistic winner is not integrated or enabled; integration readiness is documented by the tested in-path K=10 ranker, but protocol requires lexical-path closure with the encoder disabled.

Exact non-frozen test commands run:
- `python -m pytest -q tests/test_stage1c_final.py tests/test_c2_c3.py tests/test_trivial_baseline.py` — 9 passed.
- `python -m py_compile benchmarks/stage1c_final.py tests/test_stage1c_final.py` — passed.
- `python -m benchmarks.stage1c_final` — the sole command that read `stack/encoder/data/test.jsonl`.
- Read-back validation command used `validate_final_artifact` and returned `validation_errors= []`.

`eval_gates.py` and full pytest were not run because they can read the frozen test split. Historical artifacts remain unchanged.
