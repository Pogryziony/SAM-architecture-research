# Benchmarks Results Index

*Auto-generated 2026-07-08 21:59 UTC from 9 result files. Updated 2026-07-09 19:35 UTC for Phase 4.*

| File | Date (UTC) | Size | Key Metrics | Command to Reproduce |
|---|---|---|---|---|
| `phase4_paired_20260709_183954Z.json` | 2026-07-09 18:39 UTC | — | N=200, NEXUS=24.17%, RAG=9.25%, W/L/T=33/1/166, p=0.0000, answer_rate=74.5%, ER=51.5%, hall=15.63%, cascade L0=51/L1=115/L3=34 | `python benchmarks/run_benchmark.py --limit 200 --arm-rag rag_retrieval --output benchmarks/results/phase4_paired_20260709_183954Z.json` |

| File | Date (UTC) | Size | Key Metrics | Command to Reproduce |
|---|---|---|---|---|
| `nexus_vs_rag_20260708_193450Z.json` | 2026-07-08 19:34 UTC | 815 KB | N=88, NEXUS=32.4%, RAG=34.2%, W/L/T=13/16/59, p=0.7111, questions=200 | `python benchmarks/regenerate_comparison.py benchmarks\nexus_vs_rag_200.json` |
| `nexus_vs_rag_after_fix.json` | 2026-07-08 20:20 UTC | 94 KB | questions=30, nodes=1870, edges=563, model=FallbackModel(OllamaModel(qwen2.5-coder:3b) + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `ram_throughput_20260708T212808Z.json` | 2026-07-08 21:28 UTC | 9 KB | - | see source |
| `relevance_sample.json` | 2026-07-08 20:18 UTC | 7 KB | questions=26 | see source |
| `router_paired_20260708.json` | 2026-07-08 21:40 UTC | 416 KB | questions=200, nodes=1866, edges=2105, model=OllamaModel(qwen2.5-coder:3b), limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `router_vs_rag_paired_20260708T215707Z.json` | 2026-07-08 21:57 UTC | 1 KB | N=89, RAG=33.8%, W/L/T=?/27/50, p=0.0237 | see source |
| `synth_after_fix.json` | 2026-07-08 20:22 UTC | 92 KB | questions=30, nodes=1870, edges=562, model=FallbackModel(SynthesizingModel + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `verifier_calibrated_20260708_232220.json` | 2026-07-08 21:22 UTC | 10 KB | results=30 | `python benchmarks/verifier_check.py` |
| `verifier_check_20260708_194420Z.json` | 2026-07-08 19:47 UTC | 90 KB | questions=30, nodes=1855, edges=563, model=OllamaModel(qwen2.5-coder:3b), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| 2026-07-09 15:12:49 UTC | `nexus_vs_rag_20260709_151249Z.json` | `python benchmarks/regenerate_comparison.py` | N=88, NEXUS=35.98%, RAG=33.60%, W/L/T=15/15/58, p=1.0 |
