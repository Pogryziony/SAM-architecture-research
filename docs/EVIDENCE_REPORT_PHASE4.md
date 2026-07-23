# Evidence Report — Phase 4

**Date (UTC):** 2026-07-22/23  
**Analyzed HEAD:** `518103188e797d1aded310ff56134045264f6968`  
**Working tree:** Phases 1–4 uncommitted  
**Commits created this phase:** none

## Executive summary

Phase 4 identified and executed **local Ollama `qwen3.6:latest`** (36B MoE, Q4_K_M) across all planned answer-generating arms on internal `oracle_v1` (191q). Controlled RAG arms and an evaluation-only NEXUS-graph+Qwen arm completed. Dual human adjudication packets were exported; **zero human responses imported**.

Exploratory auto-scorable-subset paired statistics exist and are labeled exploratory only.

**NO FULL SUPERIORITY VERDICT — human adjudication incomplete.**

---

## Local Qwen identity

| Field | Value |
|-------|-------|
| Runtime | Ollama |
| Runtime version | 0.32.1 |
| Model | `qwen3.6:latest` |
| Digest | `07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522` |
| Architecture | `qwen35moe` |
| Parameters | 36.0B |
| Quantization | Q4_K_M |
| Context length | 262144 |
| Think | `false` (required for usable answers) |
| Decoding | temperature 0, top_p 1, top_k 1, seed 0, num_predict 256 |
| Health | `phase4_qwen_health.json` — OK (~6 tok/s after load; TTFT via prompt_eval) |
| Device | Ollama-managed (host GPU/CPU as configured by Ollama) |

No remote commercial LLM. No API key. Candidates also present but **not** used: `qwen2.5:latest`, `qwen2.5-coder:3b`.

---

## Infrastructure

| Item | Path |
|------|------|
| Identity + adapter | `nexus/baselines/local_qwen.py` |
| Phase 4 arms | `nexus/baselines/phase4_arms.py` |
| RAG corpus freeze | `nexus/baselines/rag_corpus.py` |
| Adjudication I/O | `nexus/evaluation/adjudication_io.py` |
| Runner | `benchmarks/run_phase4_arms.py` |
| Preregistration | `benchmarks/phase4_preregistration_v1.json` |
| Corpus | `benchmarks/results/phase4_rag_corpus_v1.json` (20 files, 249 chunks) |

Production `grounded()` remains `allow_synth_fallback=false` (hash `1182c840a0855408`).

---

## Arm answer results (full 191; key-fact proxy)

| Arm | Family | Answered / Abstained | Grounded-correct proxy | Warm-ish p50 latency |
|-----|--------|----------------------|------------------------|----------------------|
| NEXUS grounded (Phase 3) | system | 138 / 53 | **28/191 (0.147)** | 35.3 ms |
| Qwen closed-book | system | 3 / 188 | 7/191 (0.037) | 1.93 s |
| Qwen long-context | system | 63 / 128 | 22/191 (0.115) | 35.3 s |
| BM25 RAG+Qwen | controlled | 87 / 104 | 18/191 (0.094) | 1.94 s |
| Dense RAG+Qwen | controlled | 72 / 119 | 20/191 (0.105) | 1.59 s |
| Hybrid RAG+Qwen | controlled | 90 / 101 | 21/191 (0.110) | 1.74 s |
| Hybrid+rerank RAG+Qwen | controlled | 99 / 92 | 25/191 (0.131) | 4.89 s |
| NEXUS graph-evidence+Qwen | controlled | 71 / 120 | **40/191 (0.209)** | 1.50 s |

API cost: **$0**. Local compute is not free (see latencies).

Dense embedding: `sentence-transformers/all-MiniLM-L6-v2` (offline cache).  
Reranker: **Qwen listwise LLM reranker** (explicitly not a cross-encoder).

---

## Exploratory statistics (auto-scorable subset n=120)

Full primary superiority remains ineligible.

| Pair | Family | Verdict | Mean diff | 95% CI | McNemar p |
|------|--------|---------|-----------|--------|-----------|
| NEXUS grounded vs Qwen closed-book | system | LEFT_BETTER | +0.15 | [0.083, 0.217] | 4e-5 |
| Qwen closed-book vs long-context | system | RIGHT_BETTER (long) | −0.05 | [−0.092, −0.017] | 0.031 |
| NEXUS graph+Qwen vs hybrid RAG+Qwen | controlled | LEFT_BETTER | +0.20 | [0.108, 0.292] | 1.2e-4 |
| BM25 vs hybrid+rerank | controlled | INCONCLUSIVE | −0.025 | [−0.058, 0.0] | 0.25 |

Permitted wording example (exploratory, internal only):

> On the internal `oracle_v1` contract, NEXUS `grounded()` achieved higher exploratory auto-subset grounded-correct proxy than local `qwen3.6:latest` closed-book under preregistered Phase-4 conditions (n=120). This is not a full primary verdict and not an external generalization claim.

---

## Adjudication

| Item | Value |
|------|-------|
| Auto-scorable | 120 |
| Human-dependent | 71 |
| Dual packets | `benchmarks/results/phase4_adjudication_export/` |
| Annotators completed | **0** |
| Agreement | NOT_RUN |
| Full primary eligible | **false** |

---

## Sealed external / Kuzu

- Sealed run: **BLOCKED**  
- Kuzu: deferred (ADR-0001)

---

## Unsupported claims

- General LLM superiority  
- General modern-RAG superiority  
- Sealed external generalization  
- Completed human adjudication  
- Authoritative Kuzu  

---

## Security

- Local Ollama contacted only  
- HF offline preferred for MiniLM  
- TLS not disabled  
- No remote LLM credentials  
- Monetary API cost: $0  
