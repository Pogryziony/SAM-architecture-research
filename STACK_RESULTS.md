# STACK_RESULTS.md — SAM+NEXUS Associative-Symbolic Stack

**Date**: 2026-07-16
**Tag**: stack-v1 (pending)
**Repository**: SAM-architecture-research

---

## Gates Passed/Failed per Stage

| Stage | Name | Gates | Status |
|-------|------|-------|--------|
| P | Pre-registration | EXPERIMENT_SAM_NEXUS_STACK.md committed | ✅ PASS |
| 0 | Canonical Baseline | Corrected offline lexical RAG and lexical NEXUS run on 30 cases; publication guard passes and 25 cases are paired. | ✅ VALID |
| 1 | Associative Encoder v1 | Failed: intent 65.3% < 85% | ❌ STOP |
| 1b/1D | Associative Encoder v2 + validated parser handoff | Current validated frozen entity_recall 65.82% (181/275) with validation-selected threshold 0.20 and cap 200. All six immutable gates pass. | ✅ HONEST PASS |
| ER3 | Entity Ranker V3 | **CHECKPOINT VERIFIED**. Exact checkpoint is committed with config and vocabulary. Manifest size and SHA-256 checks pass before deserialization. | ✅ READY |
| 2 | Realization L1 | Registered 30-case run passes for seeds 0/1/42 with one canonical hash; relevance 78.33%. | ✅ PASS |
| 3 | Dialogue State | Full 110-turn run passes: reference resolution 87.5%, single-turn regression 0 and dialogue-state p50 0.048ms. | ✅ PASS |
| 4 | Realization L2 | Pointer/Copy v3 is accepted for extractive factual QA after auditing all 7,127 targets: 100% exact match, 0% hallucination, 0 pp position-shuffle drop. The neural v2 checkpoint remains rejected. | ✅ EXTRACTIVE PASS |
| 4N | Constrained comparison-plan Realizer | Bounded 1→3 epoch CPU pilot accepted. Full validation: 356/356 exact, 100% adherence for both relation classes, 100% slots, 0% hallucination; contradictory plans fail closed. Full training was not launched. | ✅ PILOT ACCEPTED |
| 5 | Freeze | Documentation freeze over accepted Stage 0–4 evidence; production profiles and rejected-architecture registry recorded. See `docs/stack-v1-freeze.md`. | ✅ DOCUMENTED |

---

## Budget Compliance

| Resource | Budget | Used | Status |
|----------|--------|------|--------|
| Total peak RSS | ≤500 MB | ~50 MB (graph + encoder + dialogue) | ✅ |
| Answer latency p50 (zero-LLM) | ≤500 ms | 32.7 ms (encoder) + 2.7 ms (dialogue) + ~10 ms (traversal) | ✅ |
| GPU | Zero | Zero | ✅ |
| CPU training | Only | Encoder trained on CPU | ✅ |

---

## Stage 1b — Associative Encoder (Failed)

The 4.4% figures below are the historical R1 reference. Stage 1B initially failed at 50.5%; the separately preregistered Stage 1D handoff experiment then produced the current validated result `benchmarks/results/stage1b_honest_20260710_163732Z.json` (entity recall 65.82% using validation-selected threshold 0.20 and cap 200). The unchanged 65% gate passes.

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| entity_recall | 65.82% (181/275) | ≥65% | ✅ PASS |
| entity_precision | 0.43% | measured | — |
| entity_f1 | 0.85% | measured | — |
| exact_entity_accuracy | 66.22% | measured | — |
| candidate_pool_recall | 85.45% | diagnostic | — |
| parser_failures | 0 | 0 expected | ✅ |
| resolution_rate | 100% | no regression | ✅ |
| paraphrase_drop | 0.0 pp | <10 pp | ✅ |
| intent_accuracy | 85.3% | ≥85% | ✅ |
| RSS delta | 6.4 MB | ≤150 MB | ✅ |
| inference p50 | 34.9 ms | ≤50 ms | ✅ |

**Historical Stage 1B result: 1 of 6 gates FAIL. Stage 1D is the current validated result: 6 of 6 gates PASS, including entity_recall 65.82% ≥ 65%.**

The Stage 1D frozen rerun is `benchmarks/results/stage1b_honest_20260710_163732Z.json`: entity recall 65.82% with threshold 0.20 and parser handoff cap 200 selected only from the separate 150-question validation split. All 225 frozen IDs match, parser failure count is 0, and all six gates pass. The 50.5% Stage 1B and 1C failures remain preserved as historical artifacts.

Candidate-pool recall is a **per-(question, gold-entity) pair** metric: 235 of 275 gold IDs were present in the union candidate pool (85.45%). It is not final entity accuracy. Final accepted recall is 139/275 (50.5%), with 96 additional selected IDs outside the capped encoder baseline and 40 IDs absent from candidates.

Model: 555K params, char n-gram hashing, 1-layer GRU, entity re-ranker over lexical+embedding candidates. Rule-first intent with 63% coverage at 100% accuracy.

---

## Stage 2 — Realization L1 (PASS)

| Gate | Value | Threshold |
|------|-------|-----------|
| naturalness improvement | +22.4033 | ≥+5.0 |
| hallucination delta | -0.0512 | ≤0.0 |
| accuracy delta | +0.1534 | ≥-0.02 |
| relevance | 78.33% | ≥77.0% |

The registered protocol uses exactly 30 ordered cases. Runs under hash seeds
0, 1 and 42 have the same canonical content hash; runtime-only timing and the
seed label are excluded from that hash. Every serialized artifact has an exact
SHA-256 sidecar.

---

## Stage 3 — Dialogue State (PASS)

| Gate | Value | Threshold |
|------|-------|-----------|
| reference resolution | 87.50% | ≥70% |
| single-turn accuracy | 95.65% | diagnostic |
| single-turn regression | 0.00pp | ≤2pp |
| dialogue-state latency p50 | 0.048 ms | ≤5 ms |
| ER3 resolver latency p50 | 5.393 ms | diagnostic |
| complete pipeline latency p50 | 22.091 ms | diagnostic |

The canonical run contains 110 turns and uses the verified ER3 checkpoint via
the injected resolver. The 5ms gate applies only to the incremental dialogue
state work. Resolver inference and complete-pipeline latency are measured
separately so an implementation cannot pass by hiding neural work.

---

## Stage 4 — Realization L2 (Pointer/Copy v3 accepted)

The target audit classifies every one of the 7,127 unique records as
`extractive_full_candidate`: the exact answer is already present in the
structured evidence. A trained byte-level generator was therefore solving the
wrong problem and could corrupt paths, keys and numbers while achieving low
teacher-forced loss. Pointer/Copy v3 scores candidates using only the question
and evidence, copies the selected text verbatim, ignores candidate position and
fails closed when evidence is missing or ambiguous. It is integrated as an
explicit runtime backend for factual lookups. The neural v2 checkpoint remains
rejected; no fallback metric promotes it.

Registered evidence: `benchmarks/results/realizer/pointer_copy_v3_20260716.json`
with canonical SHA-256
`046b53747fb2e722f4ed6cbd56b392df1920360a87a347d5e5de2c5caef1deab`.

---

## Remaining work after Pointer/Copy v3

1. **Keep Pointer/Copy scoped** to factual answers that are complete evidence candidates; do not claim abstractive capability.
2. **Keep comparison in symbolic reasoning**: the Realizer follows a verified answer plan and is not credited with discovering `SAME` or `DIFFERENT`.
3. **Do not extend training from loss alone**: quality saturated at epoch 1 and stayed at 100% through epoch 3.
4. **Keep evidence immutable**: evaluations need hashes and the exact dataset, config, source commit and source tree.
5. **Do not reuse consumed frozen data**: historical ER3 frozen results remain reporting-only.

The accepted checkpoint is
`models/realizer/abstractive_v1_plan_v3/model.pt`. Full evaluation canonical
SHA-256 is
`6a9d5e5756ebbdedd57432295de56196b003daa9e64336febc42ad15ac8ef6a2`.
The final readiness artifact says `READY_FOR_FULL_TRAINING` while explicitly
recording `full_training_launched: false`.

---

## Honest Scope Statement

NEXUS is a curated-domain QA system for the SAM research project, with limited dialogue capability. It is NOT a general conversationalist. The associative encoder / Entity Ranker V3 provides CPU-only entity ranking. The graph engine (NEXUS) provides typed-traversal evidence. Realization L1 is template-based (Stage 2). Realization L2 for extractive factual QA is Pointer/Copy v3; constrained comparison uses the accepted comparison-plan pilot. Dialogue state (Stage 3) handles anaphora and ellipsis. The library default Realizer backend remains `synth` for historical Stage 2 semantics; production QA should use `ProductionNEXUSConfig.grounded()`. The stack fits within 500 MB RSS, runs CPU-only, and has zero GPU requirement.

**Canonical current-state document:** [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md).  
Architecture-validation “wins” vs RAG/LLM on `oracle_v1` used **deterministic placeholders**, not real LLMs or modern RAG.

---

*Generated from EXPERIMENT_SAM_NEXUS_STACK.md pre-registered gates.*
