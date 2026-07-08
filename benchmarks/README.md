# NEXUS Benchmarks

Benchmarks for evaluating the NEXUS graph-first reasoning architecture.

## Benchmark tracks

| Track | What it measures | Directory |
|-------|-----------------|-----------|
| QA dataset | Domain-specific questions with ground-truth answers | `qa-dataset/` |
| Graph eval | Graph-specific metrics: path accuracy, entity resolution, relation quality | `graph-eval/` |
| RAG baseline | Classic RAG implementation for fair comparison | `rag-baseline/` |

## Core comparison experiment

Run the same QA dataset through:
1. **NEXUS** — graph traversal → evidence → small LLM
2. **Classic RAG** — embeddings → top-K chunks → same LLM
3. **Hybrid** — graph + RAG → combined evidence → LLM
4. **LLM-only** — same LLM, closed-book (no external knowledge)

Compare on: accuracy, hallucination rate, context size, latency, source traceability.
