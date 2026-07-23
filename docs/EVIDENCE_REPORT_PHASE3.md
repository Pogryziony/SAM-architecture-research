# Evidence Report — Phase 3

**Date (UTC):** 2026-07-22  
**Analyzed HEAD:** `518103188e797d1aded310ff56134045264f6968`  
**Working tree:** Phase 1 + Phase 2 + Phase 3 (uncommitted)  
**Commits created this phase:** none

## Executive summary

Phase 3 produced the first trustworthy **full-SAM / `oracle_v1`** performance
package, fail-closed safe-profile behavior, adjudication routing with a blinded
human packet, local BM25 retrieval-only evaluation, and a sealed-evaluator
handoff package. Real LLM and generated modern-RAG arms remain `NOT_RUN` (no
API key / spending authorization). Sealed external execution remains `BLOCKED`.
Kuzu remains experimental by ADR.

**NO SUPERIORITY VERDICT — insufficient real comparable evidence.**

---

## Infrastructure implemented

| Item | Location |
|------|----------|
| `allow_synth_fallback` (safe profiles default `false`) | `nexus/utils/config.py`, `nexus/pipeline/config.py` |
| Fail-closed cascade / abstain | `nexus/reasoning/answer.py`, `nexus/pipeline/runner.py` |
| Fallback audit in eval export | `nexus/evaluation/export.py` |
| Full-oracle performance harness | `nexus/evaluation/performance.py`, `benchmarks/run_performance_grounded.py` |
| Adjudication routes + blinded packet | `nexus/evaluation/adjudication.py`, `benchmarks/run_adjudication_routes.py` |
| Pending-adjudication refusal in paired stats | `nexus/evaluation/compare.py` |
| Local BM25 retrieval-only | `nexus/baselines/retrieval.py`, `benchmarks/run_bm25_retrieval.py` |
| Sealed evaluator handoff | `evaluator_handoff/` |
| Kuzu scope ADR | `docs/adr/ADR-0001-kuzu-persistence-phase3.md` |
| HF SSL diagnosis | `docs/HF_SSL_DIAGNOSIS_PHASE3.md` |
| CI gates | `.github/workflows/eval-gates.yml` |

---

## Tests run

| Suite | Result |
|-------|--------|
| Phase 3 focused (fail-closed, adjudication, BM25, handoff, config identity, Phase 2 export/perf) | Pass (after fixes) |
| HF semantic entity-candidate (with `HF_HUB_OFFLINE=1` + certifi) | Pass |
| HF semantic entity-candidate (default env) | Fail / environment SSL |
| Full base suite | See completion report |
| Kuzu | `NOT_RUN` (module missing; not required by ADR) |

---

## Internal NEXUS runs

### Schema-v1 eval — full `oracle_v1` grounded

- Artifact: `benchmarks/results/eval_oracle_v1_grounded_phase3.json`
- Status: `VALID`, 191 questions, one terminal outcome each
- Profile: `ProductionNEXUSConfig.grounded()` (`allow_synth_fallback=false`)

### Performance — full SAM graph + `oracle_v1` (separate from mini)

| Artifact | Profile | Nodes/Edges | Warm p50/p95/p99 (ms) | Peak RSS (MB) | Gates |
|----------|---------|-------------|------------------------|---------------|-------|
| `performance_grounded_oracle_v1_phase3.json` | grounded (lexical ER) | 2301 / 596 | 36.06 / 63.35 / 79.84 | 55.8 | PASS / PASS |
| `performance_lexical_oracle_v1_phase3.json` | lexical (`allow_synth_fallback=true`) | 2301 / 596 | 40.13 / 68.35 / 81.43 | 56.0 | PASS / PASS |
| `performance_grounded_er3_oracle_v1_phase3.json` | grounded+ER3 (`entity_ranker_v3_20260711T081545Z`) | 2317 / 596 | 41.8 / 69.5 / 114.1 | 278.3 | PASS / PASS |

Cold latency (grounded lexical): 86.7 ms. Setup graph construction ≈1.7 s.
ER3 checkpoint load ≈1.6 s. Mini-domain Phase 2 results are **not** used as
full-graph evidence.

Outcome counts are diagnostic (success/abstention during perf sampling), not
a correctness superiority claim.

---

## Real LLM runs

`NOT_RUN` — `NEXUS_LLM_API_KEY` absent; no spending authorization.

Artifacts: `eval_oracle_closed_book_not_run_phase3.json` (and related arm smokes).

---

## Retrieval-only RAG runs

| Arm | Status | Artifact |
|-----|--------|----------|
| BM25 retrieval-only (mini) | `OK_RETRIEVAL_ONLY` | `bm25_retrieval_mini_phase3.json` |
| BM25 retrieval-only (oracle sample, 50q) | `OK_RETRIEVAL_ONLY` | `bm25_retrieval_oracle_v1_sample_phase3.json` |
| Dense / hybrid / rerank answer generation | `NOT_RUN` | stub artifacts |

Do **not** compare retrieval-only scores to NEXUS answer correctness.

---

## Complete RAG runs

`NOT_RUN` — no pinned answer-model credentials / budget.

---

## Pending adjudication

| Item | Value |
|------|-------|
| Routes artifact | `adjudication_routes_oracle_v1_phase3b.json` |
| Automatically scorable | 120 |
| Human-dependent | 71 |
| Scores | `adjudication_scores_oracle_v1_phase3.json` (`PENDING` for human) |
| Blinded packet | `adjudication_packet_oracle_v1_phase3b.json` |
| Annotators | 0 (agreement `NOT_RUN`) |
| Primary denominator | 191 |
| Superiority eligible | **false** |

---

## Controlled vs system-level

Infrastructure labels `comparison_mode` and refuses mixed-mode paired stats.
No valid real controlled or system-level superiority package was produced.

---

## Unsupported claims (still)

- Real-LLM superiority
- Modern-RAG superiority
- Sealed external generalization
- Authoritative Kuzu persistence
- That mini-domain latency equals full-SAM (they are reported separately; both measured)

Full-system ≤500 ms / ≤500 MB: **measured PASS** on this Windows host for the
three full-oracle campaigns above; still not a sealed external claim.

---

## Security and cost

- External providers contacted: Hugging Face (diagnosis / offline cache warm only)
- Models downloaded: none new beyond existing local MiniLM cache warm
- Credentials used: none for LLM APIs
- API cost: $0
- TLS: verification **not** disabled; offline+certifi documented
