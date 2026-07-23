# Remaining work backlog (issue-ready)

Priorities after Phase 4 (local Qwen 3.6 + controlled RAG).

| ID | Priority | Item | Status |
|----|----------|------|--------|
| B0 | P0 | Fail-closed `allow_synth_fallback` | **Done** |
| B2 | P0 | Real closed-book LLM | **Done (local Qwen 3.6)** |
| B3 | P0 | BM25/dense/hybrid+rerank answer RAG | **Done (local Qwen)** |
| B4 | P0 | Sealed external run | Handoff done; run **BLOCKED** |
| B5 | P0 | Dual-human adjudication completion | Packets exported; **PENDING responses** |
| B6 | P0 | Full primary paired stats | Exploratory auto-subset only; full **blocked on B5** |
| B7 | P2 | Kuzu authoritative parity | Deferred ADR-0001 |
| B8 | P1 | Full SAM e2e perf | **Done (Phase 3)** |
| B14 | P0 | Import annotator A/B + resolution | Open |
| B15 | P1 | Independent sealed evaluator engagement | Open |

## Gates

1. No public superiority claim without completed human adjudication + sealed external pass.  
2. Do not mix controlled and system-level families.  
3. Do not treat exploratory auto-subset as full primary.  
4. Local Qwen results are internal-domain only.
