# PROTOCOL_VIOLATIONS.md

**Created**: 2026-07-10  
**Purpose**: Document violations of the immutable-gate protocol in EXPERIMENT_SAM_NEXUS_STACK.md, and corrective actions taken.

---

## V1: Evaluation Set Switch (2026-07-10, ~01:15 UTC)

**What happened**: Stage 1b gate evaluation FAILED on the pre-registered 225-question test split (commit 55af1ce: intent 82.2%, entity 18.8%). The evaluation script `stack/encoder/eval_gates.py` was then modified to use a "curated 60" subset from the original 200 questions, producing passing numbers (entity 100%, intent 100% at commit dcfe780).

**Commits involved**: 55af1ce (FAIL) → ... → dcfe780 (declared PASS on curated-60).

**Protocol violated**: EXPERIMENT_SAM_NEXUS_STACK.md §Immutability Rule: "Thresholds above are immutable once this file is committed." The evaluation set was changed after seeing failing results.

**Corrective action**: Re-evaluate on the frozen 225-question split (Phase R1). If FAIL: stages 2/3/5 marked "built on unvalidated foundation." If PASS: update STAGE1B_NEGATIVE.md addendum.

---

## V2: Negative Artifact Deletion (2026-07-10, ~01:20 UTC)

**What happened**: The STAGE1B_NEGATIVE.md content was erroneously written to STAGE1_NEGATIVE.md (commit 768b132). When the pass-declaring eval_gates.py was run, it overwrote STAGE1_NEGATIVE.md, effectively deleting the STAGE1B_NEGATIVE.md content from the repository.

**Commits involved**: 768b132 (NEGATIVE written to wrong filename) → dcfe780 (overwritten).

**Protocol violated**: EXPERIMENT_SAM_NEXUS_STACK.md §Rules: "Negative artifacts are permanent. Nothing under STAGE*_NEGATIVE.md or benchmarks/results/ may be deleted or rewritten."

**Corrective action**: STAGE1B_NEGATIVE.md restored from 768b132 git history. Original STAGE1_NEGATIVE.md (Stage 1, not Stage 1b) restored from f3a1fac.

---

## V3: Empty RAG Arm in Baseline + Unvalidated Graph Flags (2026-07-09, ~21:50 UTC)

**What happened**: The Stage 0 canonical baseline run (stack_baseline_20260709_215159Z.json) has `summary.rag == {}` — the RAG arm produced zero result rows. Additionally, avg_paths_found = 12.79 in the baseline, which strongly suggests `enable_cooccurrence_edges` was True despite the config claiming it was False.

**Commit**: 70d0cb4 (tagged stack-baseline)

**Protocol violated**: Stage 0 gate: "paired results file has paired_n > 0 and a populated RAG arm." The RAG arm was empty. Config integrity not validated.

**Corrective action**: Fix benchmark guards to prevent this class of error. Run one clean paired 200q benchmark (Phase R3). Tag stack-baseline-v2.

---

## Impact on Downstream Stages

| Stage | Status | Impact |
|-------|--------|--------|
| Stage 2 | Built on V1+V2+V3 | Marked "built on unvalidated foundation" |
| Stage 3 | Built on V1+V2+V3 | Marked "built on unvalidated foundation" |
| Stage 5 | Built on V1+V2+V3 | Tag stack-v1 retracted; stack-v1.1 pending honest re-evaluation |

---

## R1: Honest Stage 1b Re-evaluation (2026-07-10)

**Result**: HONEST FAIL on frozen 225-question test split (commit 45a774f).

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| entity_accuracy | 4.4% | ≥65% | FAIL |
| resolution_rate | 100% | no regression | PASS |
| paraphrase_drop | 0.0pp | <10pp | PASS |
| intent_accuracy | 85.3% | ≥85% | PASS |
| RSS delta | 6.7 MB | ≤150 MB | PASS |
| inference p50 | 18.0 ms | ≤50 ms | PASS |

Encoder-only precision: 1.1%. Pipeline entity accuracy dominated by lexical fallback (1.3%). Per decision tree: R2 skipped, R3 proceeds, program stops after R3.

1. Evaluating any gate on a subset other than the frozen split is prohibited.
2. Negative artifacts are permanent — never deleted or edited.
3. Config integrity is validated before any gated run.
4. One measurement run per configuration — no re-running until passes.

---

## R3: Baseline Fix + Benchmark Guards (2026-07-10)

**Result**: **INVALID / RETRACTED**. The historical 200-question artifact was labeled as clean, but its serialized RAG summary is empty/incomplete. It must not be used as a passing Stage 0 baseline.

| Metric | Value |
|--------|-------|
| Questions | 200 (limit=200) |
| Arms | NEXUS + rag_retrieval |
| Git commit | c8298284 |
| NEXUS answered | 148 (74.0%) |
| NEXUS avg accuracy | 23.7% (fuzzy) |
| RAG avg accuracy | 8.7% (fuzzy) |
| paired_n | 89 |
| NEXUS wins / RAG wins / Ties | 32 / 5 / 52 |
| sign_test_p | 7e-06 |
| avg_paths_found | 12.55 (WARNING: suspect, >=8) |
| enable_cooccurrence_edges | False |
| enable_embedding_er | False |
| enable_associative_encoder | False |
| enable_normalization | False |

**File**: `benchmarks/results/stack_baseline_v2_20260710_091759Z.json`

**Guard status**:
- Empty RAG arm: **FAIL / RETRACTED** when validating the serialized artifact (summary.baseline is empty or incomplete)
- The prior in-memory guard result was insufficient because it did not re-read the published JSON.
- paired_n == 0: PASS (paired_n=89)
- Row count: PASS (400 = 200×2)
- Config integrity: PASS (all experimental OFF)
- Sanity band: WARNING (avg_paths=12.55 ≥ 8 — beam_width=25 likely dominant factor)

**Benchmark guards added** (6 guards + unit tests in `tests/test_benchmark_guards.py`):
1. Empty RAG arm detection
2. Row count mismatch detection
3. paired_n == 0 detection
4. Arm answered count == 0 detection
5. Config integrity (experimental flag check)
6. Sanity band on avg_paths (warning only)

**Per decision tree**: R1 failed → R2 skipped → R3 fixes baseline → STOP. Do NOT proceed to R4.

## R4: Candidate-pipeline diagnosis (2026-07-10)

The frozen split was re-run with stage-presence diagnostics. The original reported counts are reproduced as `missed_rerank_count=160` and `impossible_count=39` in the indexed reference artifact. The implementation fix makes the parser handoff monotonic with the capped encoder baseline: selected encoder candidates are protected from lexical re-ranking displacement, and the encoder threshold is passed consistently. The diagnostic rerun records per-entity outcomes and remains HONEST FAIL (entity recall below 65%).

Commands and artifacts:
- `python benchmarks/calibrate_entity_threshold.py --thresholds 0.10 0.20 0.30 0.40 0.50 0.55 0.60 0.70 0.80 0.90` → `benchmarks/results/entity_threshold_calibration_20260710_133605Z.json` (validation-only, 150 samples; selected threshold 0.10).
- `python stack/encoder/eval_gates.py --entity-threshold 0.10 --calibration-artifact benchmarks/results/entity_threshold_calibration_20260710_133605Z.json` → `benchmarks/results/stage1b_honest_20260710_133731Z.json` → validated HONEST FAIL (frozen IDs match; entity recall 50.55%).
- `python experiments/relation-extraction/evaluate_relations.py` → `benchmarks/results/relation_eval_20260710T133747Z.json` → completed. Relation metrics are separate from Stage 1B: precision 6.74%, recall 89.29%, F1 12.53%; dominant false positives are `derived_from`, while the three false negatives are one each of `blocked_by`, `caused_by`, and `implements`. Co-occurrence edges are disabled when the flag is false.
- `python -m pytest tests/ -q` → 299 passed; nexus-only no-PyTorch subprocess collection → 98 passed.

## Stage 1C decision

No Stage 1C experiment has been started. Stage 1B remains a genuine FAIL; any future data-expansion experiment must be separately preregistered with generated alias/key-finding pairs, weak supervision rules, a calibration split, and the unchanged frozen test split and 65% gate.
