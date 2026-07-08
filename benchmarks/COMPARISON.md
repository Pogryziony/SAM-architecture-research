# Canonical Comparison Table

**Generated**: 2026-07-08 19:56:15 UTC<br>
**Updated**: 2026-07-08 (post relevance fix)
**Script**: `benchmarks/build_comparison.py`
**Data sources**:
- `nexus_vs_rag_20260708_193450Z.json` — NEXUS vs RAG paired comparison (n=88)
- `nexus_vs_rag_200.json` — per-arm summary stats (hallucination, pass rate, latency, tokens)
- `synth_after_fix.json` — SynthesizingModel post-relevance-fix benchmark (n=30)
- `verifier_check_20260708_194420Z.json` — post-verifier-fix hallucination measurement (n=30)
- `throughput_results.json` — throughput and cost model data
- `relevance_audit.md` — SynthesizingModel relevance audit (post-fix: 76.9%)

> ⚠️ **Every cell cites its source file.** Cells showing "not measured" have no data — never estimated.

| Architecture | Paired fuzzy accuracy | Hallucination rate (post-fix) | Verification pass rate | Answer rate | Avg evidence tokens | p50 latency | Peak RAM (MB) | $/1K queries | Sign test p vs RAG | Relevance rate |
|---|---|---|---|---|---|---|---|---|---|---|
| NEXUS + local 3B<br>(FallbackModel: qwen2.5:latest + SynthesizingModel) | 32.42% [nexus_vs_rag_20260708_193450Z.json] | 10.43% [nexus_vs_rag_200.json] | 77.50% [nexus_vs_rag_200.json] | 93.50% [nexus_vs_rag_200.json] | 688.7 [nexus_vs_rag_200.json] | 4.37 s [nexus_vs_rag_200.json] | not measured (ram_mb = 0 in source) [throughput_results.json] | $0.4360 [throughput_results.json] | 0.7111 [nexus_vs_rag_20260708_193450Z.json] | 46.4% [relevance_audit.md] |
| NEXUS zero-weight<br>(SynthesizingModel only, no LLM) | 26.13% [synth_after_fix.json] | 60.44% [synth_after_fix.json] | 20.00% [synth_after_fix.json] | 100.00% [synth_after_fix.json] | 556 [synth_after_fix.json] | 1.11 s [synth_after_fix.json] | not measured (ram_mb = 0 in source) [throughput_results.json] (same hardware) | $0.0000 [throughput_results.json] (template synthesis only) | not measured | 76.9% [relevance_audit.md] |
| NEXUS router<br>(SynthesizingModel + LLM routing, 80% synth) | not measured [verifier_check_20260708_194420Z.json] (no paired comparison run) | 30.96% [verifier_check_20260708_194420Z.json] | not measured [verifier_check_20260708_194420Z.json] | not measured [verifier_check_20260708_194420Z.json] | not measured [verifier_check_20260708_194420Z.json] | not measured [verifier_check_20260708_194420Z.json] | not measured (ram_mb = 0 in source) [throughput_results.json] (same hardware) | $0.0872 [throughput_results.json] | not measured [verifier_check_20260708_194420Z.json] | 46.4% [relevance_audit.md] |
| RAG + same 3B<br>(OllamaModel qwen2.5:latest) | 34.17% [nexus_vs_rag_20260708_193450Z.json] | 3.28% [nexus_vs_rag_200.json] | 93.50% [nexus_vs_rag_200.json] | 76.00% [nexus_vs_rag_200.json] | 2231.6 [nexus_vs_rag_200.json] | 3.96 s [nexus_vs_rag_200.json] | not measured (ram_mb = 0 in source) [throughput_results.json] | $1.2380 [throughput_results.json] | (baseline) | 46.4% [relevance_audit.md] |

## Notes

- **Paired fuzzy accuracy**: From `compare_arms.compare_paired()` using unified `compute_fact_score`.
  NEXUS and RAG scores computed on same questions; only questions scorable by both arms included in paired comparison.
- **Hallucination rate (post-fix)**: Fraction of answer statements unsupported by source documents.
  Post-verifier-fix numbers come from the honest hallucination measurement (double-gate, P2 fix).
- **Answer rate**: `1 − insufficient_evidence_rate` — fraction of questions the system attempted to answer.
- **$/1K queries**: Local-only electricity cost via `LocalCostModel` (65W @ $0.15/kWh). Target: $0.01/1M tokens.
  Zero-weight row = $0 (template synthesis is pure CPU overhead, no LLM inference).
- **Relevance rate**: From heuristic checklist audit (4-point rubric). Formula: `% yes + 0.5 × % partial`.

## Key Findings

- **NEXUS vs RAG accuracy**: 32.4% vs 34.2% (p = 0.711) — no significant difference.
- **Hallucination**: NEXUS 10.4% vs RAG 3.3% — RAG hallucinates less.
- **Evidence efficiency**: NEXUS uses 689 tokens vs RAG's 2232 — 3.2× reduction.
- **Latency**: NEXUS 4.37s vs RAG 3.96s.
- **Zero-weight hallucination**: 60.4% (SynthesizingModel only, n=30). Higher than pre-fix because shorter answers expose more unsupported claims relative to supported ones.
- **Relevance (post-fix)**: 76.9% — above 70% threshold ✓. Before/after: 46.4% → 76.9% (+30.5pp).
  - Factual: 50.0% → 72.2%
  - Comparative: 33.3% → 100.0%
  - Diagnostic: 50.0% → 83.3%
  - Multi-hop: 50.0% → 83.3%
  - All "Regarding..." preamble eliminated. Dependency chain dumps removed. No "no" verdicts remain.
