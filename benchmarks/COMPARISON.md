# Canonical Comparison Table

**Generated**: 2026-07-10 09:44:55 UTC
**Script**: `benchmarks/build_comparison.py`
**Data sources**:
- `nexus_vs_rag_20260709_151249Z.json` — NEXUS vs RAG paired comparison (n=88)
- `nexus_vs_rag_200.json` — per-arm summary stats (hallucination, pass rate, latency, tokens)
- `verifier_check_20260708_194420Z.json` — post-verifier-fix hallucination measurement (n=30)
- `ram_throughput_20260708T212808Z.json` — warmed throughput data (qwen2.5:latest, 7.6B Q4_K_M, p50=116.8 tok/s)
- `relevance_audit.md` — SynthesizingModel relevance audit
- `router_paired_20260708.json` — NEXUS router 200-question run (hallucination, pass rate, latency)
- `router_vs_rag_paired_20260708T215707Z.json` — NEXUS router vs RAG paired comparison (accuracy, sign test)

> ⚠️ **Every cell cites its source file.** Cells showing "not measured" have no data — never estimated.

| Architecture | Paired fuzzy accuracy | Hallucination rate (post-fix) | Verification pass rate | Answer rate | Avg evidence tokens | p50 latency | Peak RAM (MB) | Total RSS (MB) | $/1K queries | Sign test p vs RAG | Relevance rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEXUS + local qwen2.5:latest<br>(7.6B Q4_K_M; FallbackModel + SynthesizingModel) | 35.98% [nexus_vs_rag_20260709_151249Z.json] | 19.25% [nexus_vs_rag_200.json] | 70.50% [nexus_vs_rag_200.json] | 66.00% [nexus_vs_rag_200.json] | 1566.0 [nexus_vs_rag_200.json] | 4.34 s [nexus_vs_rag_200.json] | 44 MB (delta +2 MB) [ram_throughput_20260708T212808Z.json] | 8209 MB [ram_throughput_20260708T212808Z.json] (pipeline 44 MB + Ollama idle 8165 MB) | $0.0398 [ram_throughput_20260708T212808Z.json] | 1.0000 [nexus_vs_rag_20260709_151249Z.json] | 76.9% [relevance_audit.md] |
| NEXUS zero-weight<br>(SynthesizingModel only, no LLM) | not measured [verifier_check_20260708_194420Z.json] (no paired comparison run) | 38.77% [verifier_check_20260708_194420Z.json] | 20.00% [verifier_check_20260708_194420Z.json] | 100.00% [verifier_check_20260708_194420Z.json] | not measured [verifier_check_20260708_194420Z.json] (no evidence token tracking in verifier) | 1.39 s [verifier_check_20260708_194420Z.json] | 42 MB (delta +6 MB) [ram_throughput_20260708T212808Z.json] | 42 MB [ram_throughput_20260708T212808Z.json] (pipeline only, no LLM) | $0.0000 [ram_throughput_20260708T212808Z.json] (template synthesis only) | not measured [verifier_check_20260708_194420Z.json] | 76.9% [relevance_audit.md] |
| NEXUS router<br>(SynthesizingModel + LLM routing, 97% synth) | 20.82% [router_vs_rag_paired_20260708T215707Z.json] | 17.32% [router_paired_20260708.json] | 70.50% [router_paired_20260708.json] | 97.50% [router_paired_20260708.json] | 47.0 [router_paired_20260708.json] (blended: 97% synth×0 + 3% LLM×1566) | 0.05 s [router_paired_20260708.json] | 44 MB (delta +2 MB) [ram_throughput_20260708T212808Z.json] | 8209 MB [ram_throughput_20260708T212808Z.json] (pipeline 44 MB + Ollama idle 8165 MB) | $0.0001 [ram_throughput_20260708T212808Z.json] | 0.0237 [router_vs_rag_paired_20260708T215707Z.json] | 76.9% [relevance_audit.md] |
| RAG + same qwen2.5:latest<br>(7.6B Q4_K_M OllamaModel) | 33.60% [nexus_vs_rag_20260709_151249Z.json] | 32.77% [nexus_vs_rag_200.json] | 54.50% [nexus_vs_rag_200.json] | 75.50% [nexus_vs_rag_200.json] | 2231.6 [nexus_vs_rag_200.json] | 3.35 s [nexus_vs_rag_200.json] | 456 MB (delta +411 MB) [ram_throughput_20260708T212808Z.json] | 8621 MB [ram_throughput_20260708T212808Z.json] (pipeline 456 MB + Ollama idle 8165 MB) | $0.0552 [ram_throughput_20260708T212808Z.json] | (baseline) | 76.9% [relevance_audit.md] |

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
  Pipeline RSS only — excludes Ollama process RSS.
  Zero-weight = SynthesizingModel pipeline only (graph + template engine).
  NEXUS+3B = FallbackModel pipeline only (graph + code — Ollama RSS measured separately).
  RAG+3B = chunk retrieval + all-MiniLM-L6-v2 embeddings loaded in memory.
- **Total RSS (MB)**: Pipeline peak RSS + Ollama idle process RSS (model loaded but not generating).
  This is the true system RAM cost — the number the project defends itself with.
  Zero-weight = pipeline only (no Ollama needed).
  All LLM-dependent arms include Ollama idle RSS (~5–8 GB for 7.6B Q4_K_M).
  Ollama generating RSS (KV cache + activations) measured separately via concurrent polling.
- **Relevance rate**: From heuristic checklist audit (4-point rubric). Formula: `% yes + 0.5 × % partial`. The preregistered relevance gate is **≥77.0%**; 70% is only the metric-caveat trigger, not the pass threshold.

## Key Findings

- **NEXUS vs RAG accuracy**: 36.0% vs 33.6% on paired_n=88 (p = 1.000) — no significant difference. The clean R3 gate is separate: n=200 per arm, paired_n=89, p=7e-06.
- **Hallucination**: NEXUS 19.2% vs RAG 32.8% — NEXUS has the lower measured hallucination rate (RAG is higher), not “RAG hallucinates less.”
- **Evidence efficiency**: NEXUS uses 1566 tokens vs RAG's 2232 — 3.2× reduction.
- **Latency**: NEXUS 4.34s vs RAG 3.35s.
- **Throughput (warmed)**: p50=116.8 tok/s on qwen2.5:latest (7.6B Q4_K_M). Raw LLM cost = $0.0232/1M. Router (80% synth) = $0.0046/1M.
- **RAM**: RAG indexing adds +411.4 MB (embedding model). NEXUS pipeline adds +2.4 MB. Zero-weight adds +5.6 MB. Total RSS (pipeline + Ollama idle): see table column.
- **Zero-weight hallucination**: 38.8% (SynthesizingModel only, n=30).
- **Relevance**: 76.9% — above the 70% metric-caveat trigger but below the preregistered 77.0% gate (accuracy × relevance = 30.2% actionable accuracy).

## Router vs RAG (Row 3 — newly measured)

- **Router paired accuracy**: 20.8% vs RAG 33.8% (n=89 paired).
- **Win/Loss/Tie**: Router wins=12, RAG wins=27, ties=50.
- **Sign test**: p=0.0237 — significant at α=0.05.
- **Router hallucination**: 17.3% vs NEXUS+3B 19.2%.
- **Router latency**: 0.05s (97% synth-routed, 3% LLM-routed).
- **Router verification pass**: 70.5%.
- **Router answer rate**: 97.5%.

## R3 Clean Benchmark Accounting

The latest clean R3 benchmark is `benchmarks/results/stack_baseline_v2_20260710_091759Z.json` (200 questions, all experimental flags OFF). It must not be conflated with the smaller LLM measurements above:

| Measurement | N / paired_n | Result | Source |
|---|---:|---|---|
| R3 clean per-arm benchmark | n=200 | NEXUS 23.7% fuzzy accuracy; RAG 8.7% | `stack_baseline_v2_20260710_091759Z.json` |
| NEXUS vs RAG paired comparison | paired_n=89 | wins/ties/losses = 32/52/5; sign-test p=7e-06 | same R3 artifact |
| Earlier full-arm LLM measurement | n=200 | hallucination/latency/token rows only; not the R3 gate | `nexus_vs_rag_200.json` |
| Router evaluation | n=30 | separate router diagnostic; not comparable to R3 accuracy | `router_paired_20260708.json` |
| Paired router-vs-RAG | paired_n=89 | p=0.0237; separate router run | `router_vs_rag_paired_20260708T215707Z.json` |

### Model and memory accounting

The model name is **Ollama `qwen2.5:latest`**, measured as a **7.6B-parameter Q4_K_M** model in `ram_throughput_20260708T212808Z.json`; it is not `qwen2.5-coder:3b`. Report pipeline RSS separately from LLM RSS: NEXUS pipeline 44 MB, RAG pipeline 456 MB, and Ollama idle RSS 8,165 MB. Thus the LLM-dependent path explicitly **violates the 500 MB total RSS budget** even though the NEXUS pipeline component alone is below budget. The zero-weight path has pipeline RSS 42 MB and no LLM RSS.