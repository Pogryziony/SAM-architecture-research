# Sealed evaluation protocol

## Roles

- **System owner** — develops NEXUS/baselines; must not hold sealed gold.
- **Evaluator** — freezes corpus, holds hidden questions/gold, adjudicates, publishes.

## Steps

1. **Preregister** using `PREREGISTRATION_TEMPLATE.md` (metrics, power, configs, budget).
2. **Freeze corpus** — hash all source files; record URLs/commits; no post-disclosure edits.
3. **Freeze ingestion** — graph snapshot identity recorded before question disclosure.
4. **Disclose questions** — question text + IDs only; no gold answers/entities.
5. **Execute systems** — pinned configs; emit `nexus-eval-result-v1`.
6. **Validate artifacts** — schema + one terminal outcome per question.
7. **Adjudicate** — automated routes first; blinded human packet for remainder.
8. **Statistics** — controlled and system-level families separately; refuse placeholders/NOT_RUN/pending adjudication.
9. **Release** — hash-identified package per `FINAL_RELEASE_PROCEDURE.md`.

## Hard rules

- No graph/alias/prompt edits after sealed disclosure.
- No TLS disablement for model downloads.
- Every question remains in the primary-metric denominator.
- Failures/timeouts/abstentions stay visible.
