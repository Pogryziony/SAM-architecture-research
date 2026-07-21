# Stack-v1 Freeze Attestation

**Status:** `STACK_V1_FREEZE_DOCUMENTED`  
**Date:** 2026-07-21  
**Basis commit at freeze documentation:** recorded by the implementing branch tip

## Purpose

Close Stage 5 of `EXPERIMENT_SAM_NEXUS_STACK.md` as a **documentation and
identity freeze** over already-accepted Stage 0–4 evidence. This file does not
re-run registered gates or invent new metrics.

## Authoritative status chain

1. [`README.md`](../README.md) — project status table  
2. [`STACK_RESULTS.md`](../STACK_RESULTS.md) — stage gate verdicts  
3. Hashed artifacts under `benchmarks/results/` (immutable)  
4. This attestation — freeze interpretation and production profile guidance

Historical 200-question Phase 4 FAIL numbers in
[`ANALYSIS_AND_ROADMAP.md`](../ANALYSIS_AND_ROADMAP.md) (2026-07-09) remain
**historical evidence**. They are not the Stage 0–4 stack-v1 verdict.

## Accepted language layer (production profiles)

| Profile | Factory | Scope |
|---------|---------|-------|
| Library default | `NEXUSConfig.realizer_backend = "synth"` | Backward-compatible registered Stage 2 semantics |
| Extractive factual | `ProductionNEXUSConfig.pointer_copy()` | Pointer/Copy v3 |
| Comparison | `ProductionNEXUSConfig.comparison_plan()` | Hash-verified comparison-plan pilot |
| Recommended production QA | `ProductionNEXUSConfig.grounded()` | Pointer/Copy + comparison-plan |

Architecture allow/deny list: [`training/REJECTED_ARCHITECTURES.json`](../training/REJECTED_ARCHITECTURES.json).

## Freeze inventory (do not rewrite)

| Gate | Verdict | Primary evidence |
|------|---------|------------------|
| Stage 0 | VALID | `benchmarks/results/phase4/stage0.json` |
| Stage 1D | HONEST PASS | `STAGE1D_RESULT.md`, `stage1b_honest_20260710_163732Z.json` |
| ER3 | CHECKPOINT VERIFIED | `entity_ranker_v3_selection_20260711T081545Z.json` |
| Stage 2 | PASS | `phase4/stage2_seed{0,1,42}/` |
| Stage 3 | PASS | `phase4/stage3/stage3_20260716T010219Z.json` |
| Pointer/Copy v3 | ACCEPTED | `realizer/pointer_copy_v3_20260716.json` |
| Comparison-plan pilot | PILOT ACCEPTED | `models/realizer/abstractive_v1_plan_v3/` |
| Phase 4 readiness | GO_FOR_REALIZER_TRAINING | `phase4/phase4_readiness.json` |

## Explicit non-claims

- Pointer/Copy v3 is **not** abstractive realization.
- Comparison-plan full training was **not** launched (`full_training_launched: false`).
- AnswerPlan autoregressive pilots remain **blocked**; copy/edit transducer pilots were not part of this freeze.
- ER3 frozen-split aggregate remains **reporting-only** (do not re-evaluate).

## Related history

Merged topic branches used to produce this freeze (and earlier NEXUS stages) are
catalogued with tip SHAs in [`docs/branch-history.md`](branch-history.md).

## Hard budgets (unchanged)

- Peak RSS ≤ 500 MB  
- Zero-LLM answer p50 ≤ 500 ms  
- CPU-only; no GPU requirement  
- Dependency direction: `stack → nexus` only
