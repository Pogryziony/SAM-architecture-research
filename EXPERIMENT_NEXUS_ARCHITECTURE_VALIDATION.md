# EXPERIMENT: NEXUS architecture validation (next training gate)

**Status:** COMPLETE — decision **VALIDATED** (2026-07-21).  
**Verdict:** `docs/nexus-architecture-validation-verdict.md`  
**Campaign artifact:** `benchmarks/results/architecture_validation_20260721T241500Z.json`  
**Goal:** Answer whether NEXUS is a *good* architecture that can *perform
well* under honest paired evaluation — not whether AnswerPlan weights can
paper over surface gaps.

## Decision question

NEXUS is validated as an architecture iff a preregistered paired campaign
shows, on the frozen oracle contract (`oracle_v1` or a successor freeze):

1. **Graph/ER saturated** — entry ≥ 0.90, path ≥ 0.90, proof ≥ 0.90
2. **Surface competence** — predicted fact ≥ 0.70 on the numeric+key-fact
   scorer (or a preregistered qualitative token-F1 secondary)
3. **Realization gap** — if oracle fact ≥ 0.50 and predicted lag ≥ 0.15,
   then (and only then) authorize bounded AnswerPlan copy/edit pilots
4. **Baselines** — NEXUS L1 beats classic RAG and LLM-only on the same
   freeze for fact + abstain calibration (Stage 4 comparison arm)

## Observed (2026-07-21)

| Metric | Value |
|---|---:|
| Entry | 0.9786 |
| Path | 0.9697 |
| Proof | 0.9634 |
| Predicted fact | 0.7159 |
| Oracle fact | 0.7319 |
| RAG fact | 0.0482 |
| LLM-only fact | 0.0000 |
| AnswerPlan lag | ≈0.016 (sealed) |

## What is already established (do not retrain for this)

- L1 zero-LLM path: path render, node-fact, qualitative dual-compare,
  metric compare, dependency chains, PIT abstain families
- Bi-temporal stamps + family valid-window / retract / active cases
- AnswerPlan sealed while predicted does not lag a strong oracle
- Question-grounded aliases for remaining entry zeros (verifier /
  distractor / phase-transition prompts)

## Training ladder (only escalate when the prior rung fails)

| Rung | Train? | Purpose | Status |
|---|---|---|---|
| A. Lexical/union ER hygiene | No weights | Exact-ID + grounded spotting + aliases | **PASS** (entry ≥ 0.95) |
| B. ER3 refresh (bounded) | **Small** ranker finetune | New canonical vocab only | **NOT authorized** |
| C. AnswerPlan overfit+2048 | **Only if** lag ≥ 0.15 | Realization gap | **Sealed** |
| D. Full AnswerPlan / realizer | **Ultimately only** after C | Production verbalizer | **Not needed** for verdict |

## Recommended next run sequence

1. ~~Publish current L1 `union` + `l1_acceptance` paired artifact; record AnswerPlan status.~~ **Done** (`…T241500Z`).
2. ~~If entry < 0.95 and misses are new canonical nodes → rung B.~~ **N/A** — entry 0.9786 via aliases.
3. ~~Run Stage 4 NEXUS vs RAG vs LLM-only on the same freeze.~~ **Done**
   (`python benchmarks/run_architecture_validation.py --output …`).
4. ~~Write one-page verdict.~~ **Done** (`docs/nexus-architecture-validation-verdict.md`).

## Explicit non-goals

- Do not train AnswerPlan to lift qualitative prose that curated dual-facts
  already cover.
- Do not consume the frozen test split for ER3 calibration.
- Do not declare “NEXUS works” from oracle-only arms.
