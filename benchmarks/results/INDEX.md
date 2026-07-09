# Benchmarks Results Index

*Auto-generated 2026-07-09 22:15 UTC from 26 result files.*

| File | Date (UTC) | Size | Key Metrics | Command to Reproduce |
|---|---|---|---|---|
| `confidence_router_2026-07-09.json` | 2026-07-08 22:27 UTC | 65 KB | questions=30, nodes=1866, edges=2104, model=OllamaModel(qwen2.5-coder:3b), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `extended_evidence_20260708T222811Z.json` | 2026-07-08 22:28 UTC | 93 KB | questions=30, nodes=1866, edges=2105, model=FallbackModel(OllamaModel(qwen2.5-coder:3b) + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `extended_evidence_test.json` | 2026-07-08 22:27 UTC | 93 KB | questions=30, nodes=1866, edges=2105, model=FallbackModel(OllamaModel(qwen2.5-coder:3b) + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `final_arch_20260709_194800Z.json` | 2026-07-09 20:13 UTC | 2 MB | questions=200, nodes=1866, edges=87652, limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results/TIMESTAMPED_FILE.json` |
| `nexus_vs_rag_20260708_193450Z.json` | 2026-07-08 19:34 UTC | 815 KB | N=88, NEXUS=32.4%, RAG=34.2%, W/L/T=13/16/59, p=0.7111, questions=200 | `python benchmarks/regenerate_comparison.py benchmarks\nexus_vs_rag_200.json` |
| `nexus_vs_rag_20260709_151249Z.json` | 2026-07-09 15:12 UTC | 788 KB | N=88, NEXUS=36.0%, RAG=33.6%, W/L/T=15/15/58, p=1, questions=200 | `python benchmarks/regenerate_comparison.py benchmarks\results\nexus_vs_rag_paired_20260709_144656Z.json` |
| `nexus_vs_rag_after_fix.json` | 2026-07-08 20:20 UTC | 94 KB | questions=30, nodes=1870, edges=563, model=FallbackModel(OllamaModel(qwen2.5-coder:3b) + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `nexus_vs_rag_final_20260709_161758Z.json` | 2026-07-09 16:31 UTC | 931 KB | questions=200, nodes=1866, edges=2106, model=FallbackModel(OllamaModel(qwen2.5:latest) + SynthesizingModel), limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results/TIMESTAMPED_FILE.json` |
| `nexus_vs_rag_full_20260709_124135Z.json` | 2026-07-09 12:53 UTC | 1 MB | questions=200, nodes=1866, edges=2105, model=FallbackModel(OllamaModel(qwen2.5:latest) + SynthesizingModel), limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results/TIMESTAMPED_FILE.json` |
| `nexus_vs_rag_full_20260709_125347Z.json` | 2026-07-09 13:05 UTC | 1 MB | questions=200, nodes=1866, edges=2107, model=FallbackModel(OllamaModel(qwen2.5:latest) + SynthesizingModel), limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results/TIMESTAMPED_FILE.json` |
| `nexus_vs_rag_paired_20260709_144656Z.json` | 2026-07-09 15:12 UTC | 788 KB | questions=200, nodes=1866, edges=2107, limit=200 | `python benchmarks/run_benchmark.py` |
| `oracle_evidence_test.json` | 2026-07-09 13:37 UTC | 44 KB | questions=30 | see source |
| `phase4_paired_20260709_183954Z.json` | 2026-07-09 19:03 UTC | 2 MB | questions=200, nodes=1866, edges=2106, limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results/TIMESTAMPED_FILE.json` |
| `post_edit_2026-07-09T0029Z.json` | 2026-07-08 22:39 UTC | 100 KB | questions=30, nodes=1866, edges=2106, model=FallbackModel(OllamaModel(qwen2.5-coder:3b) + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `post_edit_v3.json` | 2026-07-08 22:39 UTC | 100 KB | questions=30, nodes=1866, edges=2106, model=FallbackModel(OllamaModel(qwen2.5-coder:3b) + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `ram_throughput_20260708T212808Z.json` | 2026-07-08 21:28 UTC | 9 KB | - | see source |
| `relevance_sample.json` | 2026-07-08 20:18 UTC | 7 KB | questions=26 | see source |
| `resolution_after_20260709_182327Z.json` | 2026-07-09 18:24 UTC | 97 KB | - | see source |
| `resolution_before_20260709_182132Z.json` | 2026-07-09 18:21 UTC | 97 KB | - | see source |
| `router_paired_20260708.json` | 2026-07-08 21:40 UTC | 416 KB | questions=200, nodes=1866, edges=2105, model=OllamaModel(qwen2.5-coder:3b), limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `router_vs_rag_paired_20260708T215707Z.json` | 2026-07-08 21:57 UTC | 1 KB | N=89, RAG=33.8%, W/L/T=?/27/50, p=0.0237 | see source |
| `stack_baseline_20260709_215159Z.json` | 2026-07-09 22:15 UTC | 2 MB | questions=200, nodes=1866, edges=2105, limit=200 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results/TIMESTAMPED_FILE.json` |
| `synth_after_fix.json` | 2026-07-08 20:22 UTC | 92 KB | questions=30, nodes=1870, edges=562, model=FallbackModel(SynthesizingModel + SynthesizingModel), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
| `tier3_diagnostic_20260709T183718Z.json` | 2026-07-09 18:37 UTC | 36 KB | questions=34 | see source |
| `verifier_calibrated_20260708_232220.json` | 2026-07-08 21:22 UTC | 10 KB | results=30 | `python benchmarks/verifier_check.py` |
| `verifier_check_20260708_194420Z.json` | 2026-07-08 19:47 UTC | 90 KB | questions=30, nodes=1855, edges=563, model=OllamaModel(qwen2.5-coder:3b), limit=30 | `python benchmarks/run_benchmark.py --limit 30 --output benchmarks/results.json` |
