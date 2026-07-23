# NEXUS current state (canonical source of truth)

**Document role:** canonical current-state attestation for this repository.  
**Supersedes for status claims:** informal “validated architecture” wording in older docs.  
**Analyzed HEAD:** `518103188e797d1aded310ff56134045264f6968`  
**Working tree:** Phases 1–4 (uncommitted at authorship)  
**Date (UTC):** 2026-07-23  
**Evidence:** [`EVIDENCE_REPORT_PHASE1.md`](EVIDENCE_REPORT_PHASE1.md) … [`PHASE4`](EVIDENCE_REPORT_PHASE4.md)

> If this file and another document disagree on *current* status, **this file wins**.

---

## Active architecture

| Item | Value |
|------|-------|
| Active architecture | **NEXUS** |
| Recommended production profile | `ProductionNEXUSConfig.grounded()` with `allow_synth_fallback=false` |
| Local LLM for Phase 4 comparisons | Ollama `qwen3.6:latest` (digest `07d35212…`, 36B MoE Q4_K_M) |
| Kuzu | Experimental; deferred (ADR-0001) |

---

## Supported claims

- NEXUS active; SAM Classic archived; safe profiles fail closed.
- Internal L1 beat deterministic placeholders (`VALIDATED_INTERNAL`).
- Schema-v1 terminals; regenerable aggregates; denominators exposed.
- Paired stats refuse placeholders, `NOT_RUN`, pending adjudication, and mixed comparison families.
- Mini and full-SAM internal grounded performance passed ≤500 ms / ≤500 MB on the tested host (Phase 3).
- BM25 retrieval-only and BM25/dense/hybrid/rerank **answer** RAG with local Qwen executed on `oracle_v1`.
- Local Qwen 3.6 closed-book and long-context executed on all 191 questions.
- Evaluation-only NEXUS-graph-evidence+Qwen arm exists and does not alter production `grounded()`.
- Sealed-evaluator handoff package exists; sealed run **BLOCKED**.
- Exploratory auto-subset paired stats available; **not** a full primary superiority verdict.

`SynthesizingModel` / `EvidenceBlindModel` are **not** LLMs.

---

## Unsupported claims

- General LLM superiority  
- General modern-RAG superiority  
- Sealed external generalization  
- Completed human adjudication (71q pending)  
- Authoritative Kuzu  

**NO FULL SUPERIORITY VERDICT — human adjudication incomplete.**

---

## Phase 4 highlights

| Arm | Status | Artifact |
|-----|--------|----------|
| Qwen closed-book | OK | `phase4_qwen_closed_book_oracle_v1.json` |
| Qwen long-context | OK | `phase4_qwen_long_context_oracle_v1.json` |
| BM25/dense/hybrid/rerank RAG+Qwen | OK | `phase4_*_rag_qwen_oracle_v1.json` |
| NEXUS graph-evidence+Qwen | OK | `phase4_nexus_graph_evidence_qwen_oracle_v1.json` |
| Human adjudication | PENDING | `phase4_adjudication_export/` |
| Sealed external | BLOCKED | `evaluator_handoff/` |

---

## Next acceptance gates

1. Import dual human annotator responses; compute agreement; resolve disagreements.  
2. Bind complete primary metrics; publish family-separated paired stats.  
3. Independent evaluator + sealed external corpus.  
4. Revisit Kuzu only if product scope requires persistence.

---

## Where to look

| Need | Location |
|------|----------|
| This file | `docs/CURRENT_STATE.md` |
| Phase 4 evidence | `docs/EVIDENCE_REPORT_PHASE4.md` |
| Preregistration | `benchmarks/phase4_preregistration_v1.json` |
| Local Qwen runner | `benchmarks/run_phase4_arms.py` |
| Backlog | `docs/REMAINING_WORK_BACKLOG.md` |
