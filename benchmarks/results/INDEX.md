# Benchmarks Results Index

| Date (UTC) | File | Command | Summary |
|---|---|---|---|
| 2026-07-08 19:34:50 UTC | `nexus_vs_rag_20260708_193450Z.json` | `python benchmarks/regenerate_comparison.py` | N=88, NEXUS=32.42%, RAG=34.17%, W/L/T=13/16/59, p=0.711071 |
| 2026-07-08 19:44:20 UTC | `verifier_check_20260708_194420Z.json` | `python benchmarks/router_benchmark.py --limit 30 --output ...` | N=30, Synth halluc=38.77%, LLM halluc=23.14% (post verifier fix: catches fabrications) |
| 2026-07-08 21:48 UTC | `relevance_audit.md` | `python benchmarks/relevance_judge.py` | N=14 stratified, relevance rate=46.4% (<70% → metric caveat triggered), per-type: factual=50%, comparative=33.3%, diagnostic=50%, multi-hop=50% |
| 2026-07-08 19:55 UTC | `COMPARISON.md` | `python benchmarks/build_comparison.py` | Phase 5 canonical comparison table — 4 architecture rows × 11 columns, all cells cite source files, zero manually-typed numbers |
