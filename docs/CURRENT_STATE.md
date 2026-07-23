# NEXUS current state (canonical source of truth)

**Document role:** canonical current-state attestation for this repository.  
**Supersedes for status claims:** informal “validated architecture” wording in older docs.  
**Analyzed HEAD:** `93cd009a8d6ccb91f48a436395f2d62006f31470`  
**Generated (UTC):** 2026-07-23 via `benchmarks/sync_current_state.py`  
**Evidence manifest:** `91dfb4fcf288ae544d824f91fbbdd4e08b6d8565fe3c906adea9312ff392e7b1`

> If this file and another document disagree on *current* status, **this file wins**.
> Regenerate with: `python benchmarks/sync_current_state.py`

---

## Active architecture

| Item | Value |
|------|-------|
| Active architecture | **NEXUS** |
| Recommended production profile | `ProductionNEXUSConfig.grounded()` with `allow_synth_fallback=false` |
| Local LLM for Phase 4 comparisons | Ollama `qwen3.6:latest` (full digest enforced) |
| Kuzu | Experimental; deferred (ADR-0001) |
| Dataset identity | full canonical record hash (`nexus-canonical-dataset-v1`) |
| Retrieval relevance | entity/fact→chunk map (`188/191` nonzero) |

---

## Supported claims

- NEXUS active; SAM Classic archived; safe profiles fail closed.
- Internal L1 beat deterministic placeholders (`VALIDATED_INTERNAL`).
- Schema-v1 terminals; regenerable aggregates; denominators exposed.
- Paired stats refuse placeholders, `NOT_RUN`, pending adjudication, and mixed comparison families.
- Exploratory `proxy_key_fact_correct` must not be quoted as primary `grounded_correct`.
- Evidence-bearing dual adjudication packets are required; empty evidence export is refused.
- Sealed-evaluator handoff package exists; sealed run **BLOCKED** until independent evaluator completes it.
- Metadata-only dataset rebinding is **invalid**.

`SynthesizingModel` / `EvidenceBlindModel` are **not** LLMs.

---

## Unsupported claims

- General LLM superiority  
- General modern-RAG superiority  
- Sealed external generalization  
- Completed dual human adjudication (responses still required)  
- Authoritative Kuzu  

**NO FULL SUPERIORITY VERDICT — human adjudication incomplete.**

---

## Artifact status (auto)

| Arm | Status | Artifact |
|-----|--------|----------|
| Qwen closed-book | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `phase4_qwen_closed_book_oracle_v1.json` |
| Qwen long-context | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `phase4_qwen_long_context_oracle_v1.json` |
| BM25 RAG+Qwen | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `phase4_bm25_rag_qwen_oracle_v1.json` |
| Dense RAG+Qwen | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `phase4_dense_rag_qwen_oracle_v1.json` |
| Hybrid RAG+Qwen | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `phase4_hybrid_rag_qwen_oracle_v1.json` |
| Hybrid+rerank RAG+Qwen | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `phase4_hybrid_rerank_rag_qwen_oracle_v1.json` |
| NEXUS graph-evidence+Qwen | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `phase4_nexus_graph_evidence_qwen_oracle_v1.json` |
| NEXUS grounded (evidence repair) | VALID @ `93cd009a8d6c` ds=`ca96877de869` | `eval_oracle_v1_grounded_evidence_repair.json` |
| Human adjudication | PENDING | `phase4_adjudication_export/` |
| Sealed external | BLOCKED | `evaluator_handoff/` |

NEXUS grounded source_commit in latest repair artifact: `93cd009a8d6ccb91f48a436395f2d62006f31470`

---

## Next acceptance gates

1. Import dual human annotator responses; compute κ; resolve disagreements.  
2. Bind complete primary metrics; publish family-wide Holm-corrected paired stats.  
3. Dense arms record degraded HF-cache snapshot identity until pinned revision c9745ed1 is available offline.  
4. Independent evaluator + sealed external corpus.  
5. Revisit Kuzu only if product scope requires persistence.

---

## Where to look

| Need | Location |
|------|----------|
| This file | `docs/CURRENT_STATE.md` |
| Evidence manifest | `benchmarks/results/evidence_manifest_v1.json` |
| Artifact governance | `docs/ARTIFACT_GOVERNANCE.md` |
| Phase reports | `docs/EVIDENCE_REPORT_PHASE*.md` |
