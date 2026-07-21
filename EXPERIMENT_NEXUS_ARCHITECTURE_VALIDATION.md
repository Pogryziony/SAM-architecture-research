# EXPERIMENT: NEXUS architecture validation (next training gate)

**Status:** PLAN ONLY — do not start full training until the binding
conditions below are met.  
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

## What is already established (do not retrain for this)

- L1 zero-LLM path: path render, node-fact, qualitative dual-compare,
  metric compare, dependency chains, PIT abstain families
- Bi-temporal stamps + family valid-window / retract / active cases
- AnswerPlan sealed while predicted does not lag a strong oracle

## Training ladder (only escalate when the prior rung fails)

| Rung | Train? | Purpose | Stop if |
|---|---|---|---|
| A. Lexical/union ER hygiene | No weights | Exact-ID + grounded spotting; family nodes stay `Concept_*`/`Decision_*`/`Exp_*` | Entry ≥ 0.95 on freeze |
| B. ER3 refresh (bounded) | **Small** ranker finetune on validation paraphrases + new canonical nodes | Recover entry for new graph vocabulary | Entry ≥ 0.95; no frozen-test peek |
| C. AnswerPlan overfit+2048 | **Only if** oracle fact ≥ 0.50 and predicted lag ≥ 0.15 | Close realization gap, not ER | Pilot gates pass; else abort |
| D. Full AnswerPlan / realizer | **Ultimately only** after C | Production verbalizer | Architecture claim already decided at A–B + baseline |

## Recommended next run sequence (no full train yet)

1. Publish current L1 `union` + `l1_acceptance` paired artifact; record
   AnswerPlan status (expect still sealed).
2. If entry < 0.95 **and** misses are mostly new canonical nodes → authorize
   **rung B** only (ER3 refresh), not AnswerPlan.
3. Run Stage 4 NEXUS vs RAG vs LLM-only on the same freeze
   (`benchmarks/run_benchmark.py` comparison arm / documented RAG baseline).
4. Write a one-page verdict: architecture validated / conditional / rejected
   against the thresholds above.

## Explicit non-goals

- Do not train AnswerPlan to lift qualitative prose that curated dual-facts
  already cover.
- Do not consume the frozen test split for ER3 calibration.
- Do not declare “NEXUS works” from oracle-only arms.
