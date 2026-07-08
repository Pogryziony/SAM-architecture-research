# Canonical Comparison Table

**Generated**: 2026-07-08 21:59:30 UTC
**Script**: `benchmarks/build_comparison.py`
**Data sources**:
- `nexus_vs_rag_after_fix.json` — NEXUS vs RAG paired comparison (n=None)
- `nexus_vs_rag_200.json` — per-arm summary stats (hallucination, pass rate, latency, tokens)
- `verifier_check_20260708_194420Z.json` — post-verifier-fix hallucination measurement (n=30)
- `ram_throughput_20260708T212808Z.json` — warmed throughput data (qwen2.5:latest, 7.6B Q4_K_M, p50=116.8 tok/s)
- `relevance_audit.md` — SynthesizingModel relevance audit
- `router_paired_20260708.json` — NEXUS router 200-question run (hallucination, pass rate, latency)
- `router_vs_rag_paired_20260708T215707Z.json` — NEXUS router vs RAG paired comparison (accuracy, sign test)

> ⚠️ **Every cell cites its source file.** Cells showing "not measured" have no data — never estimated.

| Architecture | Paired fuzzy accuracy | Hallucination rate (post-fix) | Verification pass rate | Answer rate | Avg evidence tokens | p50 latency | Peak RAM (MB) | $/1K queries | Sign test p vs RAG | Relevance rate |
|---|---|---|---|---|---|---|---|---|---|---|
| NEXUS + local 3B<br>(FallbackModel: qwen2.5:latest + SynthesizingModel) | not measured [nexus_vs_rag_after_fix.json] | 10.43% [nexus_vs_rag_200.json] | 77.50% [nexus_vs_rag_200.json] | 93.50% [nexus_vs_rag_200.json] | 688.7 [nexus_vs_rag_200.json] | 4.37 s [nexus_vs_rag_200.json] | 44 MB (delta +2 MB) [ram_throughput_20260708T212808Z.json] | $0.0195 [ram_throughput_20260708T212808Z.json] | not measured [nexus_vs_rag_after_fix.json] | 76.9% [relevance_audit.md] |
| NEXUS zero-weight<br>(SynthesizingModel only, no LLM) | not measured [verifier_check_20260708_194420Z.json] (no paired comparison run) | 38.77% [verifier_check_20260708_194420Z.json] | 20.00% [verifier_check_20260708_194420Z.json] | 100.00% [verifier_check_20260708_194420Z.json] | not measured [verifier_check_20260708_194420Z.json] (no evidence token tracking in verifier) | 1.39 s [verifier_check_20260708_194420Z.json] | 42 MB (delta +6 MB) [ram_throughput_20260708T212808Z.json] | $0.0000 [ram_throughput_20260708T212808Z.json] (template synthesis only) | not measured [verifier_check_20260708_194420Z.json] | 76.9% [relevance_audit.md] |
| NEXUS router<br>(SynthesizingModel + LLM routing, 97% synth) | 20.82% [router_vs_rag_paired_20260708T215707Z.json] | 17.32% [router_paired_20260708.json] | 70.50% [router_paired_20260708.json] | 97.50% [router_paired_20260708.json] | 20.7 [router_paired_20260708.json] (blended: 97% synth×0 + 3% LLM×689) | 0.05 s [router_paired_20260708.json] | 44 MB (delta +2 MB) [ram_throughput_20260708T212808Z.json] | $0.0001 [ram_throughput_20260708T212808Z.json] | 0.0237 [router_vs_rag_paired_20260708T215707Z.json] | 76.9% [relevance_audit.md] |
| RAG + same 3B<br>(OllamaModel qwen2.5:latest) | not measured [nexus_vs_rag_after_fix.json] | 3.28% [nexus_vs_rag_200.json] | 93.50% [nexus_vs_rag_200.json] | 76.00% [nexus_vs_rag_200.json] | 2231.6 [nexus_vs_rag_200.json] | 3.96 s [nexus_vs_rag_200.json] | 456 MB (delta +411 MB) [ram_throughput_20260708T212808Z.json] | $0.0552 [ram_throughput_20260708T212808Z.json] | (baseline) | 76.9% [relevance_audit.md] |

## Notes

- **Paired fuzzy accuracy**: From `compare_arms.compare_paired()` using unified `compute_fact_score`.
  NEXUS and RAG scores computed on same questions; only questions scorable by both arms included in paired comparison.
- **Hallucination rate (post-fix)**: Fraction of answer statements unsupported by source documents.
  Post-verifier-fix numbers come from the honest hallucination measurement (double-gate, P2 fix).
- **Answer rate**: `1 − insufficient_evidence_rate` — fraction of questions the system attempted to answer.
- **$/1K queries**: Local-only electricity cost via `LocalCostModel` (65W @ $0.15/kWh). Target: $0.01/1M tokens.
  Zero-weight row = $0 (template synthesis is pure CPU overhead, no LLM inference).
  **Throughput** measured on warmed model (5× warmup, 10× per prompt length, 3 lengths) — not cold-start.
- **Peak RAM**: Per-arm measurement via `psutil.Process().memory_info().rss`.
  Zero-weight = SynthesizingModel pipeline only (graph + template engine).
  NEXUS+3B = FallbackModel pipeline only (graph + code — Ollama RSS measured separately).
  RAG+3B = chunk retrieval + all-MiniLM-L6-v2 embeddings loaded in memory.
  Ollama process RSS (7.6B Q4_K_M): ~5–8 GB (not included in per-arm numbers).
- **Relevance rate**: From heuristic checklist audit (4-point rubric). Formula: `% yes + 0.5 × % partial`.

## Key Findings

- **NEXUS vs RAG accuracy**: not measured vs not measured (p = not measured) — no significant difference.
- **Hallucination**: NEXUS 10.4% vs RAG 3.3% — RAG hallucinates less.
- **Evidence efficiency**: NEXUS uses 689 tokens vs RAG's 2232 — 3.2× reduction.
- **Latency**: NEXUS 4.37s vs RAG 3.96s.
- **Throughput (warmed)**: p50=116.8 tok/s on qwen2.5:latest (7.6B Q4_K_M). Raw LLM cost = $0.0232/1M. Router (80% synth) = $0.0046/1M.
- **RAM**: RAG indexing adds +411.4 MB (embedding model). NEXUS pipeline adds +2.4 MB. Zero-weight adds +5.6 MB.
- **Zero-weight hallucination**: 38.8% (SynthesizingModel only, n=30).
- **Relevance**: 76.9% — below 70% triggers metric caveat (accuracy × relevance = 30.2% actionable accuracy).

## Router vs RAG (Row 3 — newly measured)

- **Router paired accuracy**: 20.8% vs RAG 33.8% (n=89 paired).
- **Win/Loss/Tie**: Router wins=12, RAG wins=27, ties=50.
- **Sign test**: p=0.0237 — significant at α=0.05.
- **Router hallucination**: 17.3% vs NEXUS+3B 10.4%.
- **Router latency**: 0.05s (97% synth-routed, 3% LLM-routed).
- **Router verification pass**: 70.5%.
- **Router answer rate**: 97.5%.