# Evidence report — Phase 2 (executable evaluation foundation)

**Extends:** [`EVIDENCE_REPORT_PHASE1.md`](EVIDENCE_REPORT_PHASE1.md)  
**Canonical status:** [`CURRENT_STATE.md`](CURRENT_STATE.md)  
**Starting HEAD:** `518103188e797d1aded310ff56134045264f6968` (unchanged; Phase 1+2 uncommitted)  
**Date (UTC):** 2026-07-22  

## Executive summary

Phase 2 wired `NEXUSRunner` to `nexus-eval-result-v1`, made aggregates regenerable from per-question rows, connected paired statistics to real artifacts (fail-closed on placeholders/`NOT_RUN`), implemented baseline adapters with honest `NOT_RUN`, measured end-to-end `grounded()` performance on the **mini** domain pack, and added ablation/robustness + CI eval gates.

No real LLM/RAG API runs. No sealed external generalization. No claim that NEXUS outperforms modern RAG or version-pinned LLMs.

## Repository state

| Item | Value |
|------|-------|
| Starting / final HEAD | `518103188e797d1aded310ff56134045264f6968` |
| Working tree | Phase 1 preserved + Phase 2 additions (uncommitted) |
| Frozen `oracle_v1` / model weights | Unmodified |

## Implemented infrastructure

| Workstream | Deliverable |
|------------|-------------|
| Runner→schema | `NEXUSRunner.run_eval()`, `nexus/evaluation/export.py`, zero-hop failure fix |
| Aggregates | `nexus/evaluation/aggregate.py` (`regenerate_aggregates`) |
| Paired stats | `nexus/evaluation/compare.py` |
| Baselines | `nexus/baselines/adapters.py` (same registry) |
| Comparison modes | `comparison_mode` = `system_level` \| `controlled` on artifacts |
| Performance | `nexus/evaluation/performance.py`, `benchmarks/run_performance_grounded.py` |
| Ablations/robustness | `nexus/evaluation/ablations.py` |
| CLI runners | `benchmarks/run_eval_v1.py` |
| CI | `.github/workflows/eval-gates.yml` |

## Tests

| Suite | Result |
|-------|--------|
| Phase 2 focused eval/baseline/perf tests | **21 passed** |
| Full `tests/` excluding `test_entity_candidate.py` (HF SSL) | **741 passed**, 1 skipped |
| `test_entity_candidate.py` | **Skipped intentionally** — environment SSL to HuggingFace (do not disable TLS) |
| Kuzu | Package not installed → prior skip behavior |
| Architecture boundary | Covered in full suite |

## Evaluation runs (by arm)

| Arm | Status | Notes |
|-----|--------|-------|
| `nexus` / `grounded` on mini domain | **OK** | `eval_mini_grounded_phase2.json` |
| `closed_book_llm` | **NOT_RUN** | Missing `NEXUS_LLM_API_KEY` / `NEXUS_LLM_MODEL` |
| `long_context_llm`, BM25/dense/hybrid/rerank/graph RAG | **NOT_RUN** | Same credentials / optional deps |
| Placeholder synthesizing / evidence-blind | **NOT_RUN** | Explicit placeholders; not LLM/modern RAG |
| Sealed multi-domain external | **BLOCKED** | No independent evaluator / frozen external corpora |

## Performance (measured)

**Profile:** `ProductionNEXUSConfig.grounded()` via `NEXUSRunner._run_single`  
**Domain:** `mini` domain pack (3 questions; not SAM `oracle_v1` graph)  
**Artifact:** `benchmarks/results/performance_grounded_mini_phase2.json`

| Metric | Value |
|--------|-------|
| Cold latency | first sample in artifact |
| Warm p50 | **1.275 ms** |
| Peak RSS | **26.039 MB** |
| Latency ≤500 ms gate | **PASS** (mini) |
| RSS ≤500 MB gate | **PASS** (mini) |
| Lexical profile warm p50 | 1.833 ms (`performance_lexical_mini_phase2.json`) |

**Not claimed:** end-to-end budget on the full SAM curated graph / `oracle_v1` mixture.

## Statistical results

Paired helpers connected; superiority verdicts **refused** for `NOT_RUN` and placeholder arms. Deterministic fixture test only (no real LLM paired campaign).

## Supported claims (after Phase 2)

- NEXUS active; SAM Classic archived  
- `grounded()` recommended production profile; runner requires explicit config  
- Internal L1 beat deterministic placeholders (Phase 1 historical artifact)  
- Config identity v2 covers behavior-changing fields  
- Every evaluated question can emit one schema-valid terminal outcome  
- Aggregates regenerate from per-question rows with explicit denominators  
- Fair baseline adapters emit honest `NOT_RUN` without credentials  
- Exact `grounded()` profile measured e2e on **mini** domain (latency/RSS gates PASS there)

## Unsupported claims (still)

- Outperforms real version-pinned LLMs  
- Outperforms modern hybrid/reranked RAG  
- Sealed multi-domain external generalization  
- ≤500 MB / ≤500 ms on the full recommended SAM/`oracle_v1` grounded system  
- Kuzu authoritative production backend  

## Artifacts created

- `benchmarks/results/eval_mini_grounded_phase2.json`  
- `benchmarks/results/eval_mini_closed_book_not_run_phase2.json`  
- `benchmarks/results/performance_grounded_mini_phase2.json`  
- `benchmarks/results/performance_lexical_mini_phase2.json`  
- Docs: this file; updates to `CURRENT_STATE.md`, backlog, INDEX  

## Blockers

| Class | Detail |
|-------|--------|
| Missing credentials | `NEXUS_LLM_*` for real LLM/RAG |
| Environment/network | HuggingFace SSL in `test_entity_candidate` |
| Missing independent evaluator | Sealed external protocol |
| Missing external corpora | Sealed multi-domain |
| Scope | Full-graph e2e perf not yet run |

## Next phase (Phase 3) — dependency order

1. Real closed-book + BM25/dense/hybrid+rerank under controlled mode (creds + budget).  
2. E2E performance on SAM benchmark graph + `oracle_v1` mixture.  
3. Wire primary grounded-correct adjudication for all question types.  
4. Sealed external run only with separate evaluator.  
5. Harden `answer.py` cascade (`allow_synth_fallback`).  
6. Kuzu parity if persistence is required.
