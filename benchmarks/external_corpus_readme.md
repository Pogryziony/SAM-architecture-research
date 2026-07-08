# NEXUS External Corpus Generalization Test

## Purpose

Test whether the NEXUS rule-based entity/relation extraction pipeline generalizes
to a corpus we did NOT author by hand. The benchmark ingests **Azanto** documentation
(Polish service marketplace platform), builds a knowledge graph with zero hand-curated
aliases, and runs QA against it.

## Corpus

- **Source:** `C:\Users\Pogry\Projects\azanto\docs\` (24 markdown files)
- **Domain:** Polish local services marketplace (classifieds)
- **Content:** PRD, architecture, domain model, deployment, CI/CD, roadmap, payment integration, runbook
- **Language:** Mostly Polish with some English technical terms

## Graph Statistics

| Metric | Value |
|--------|-------|
| Files ingested | 23 .md files |
| Total nodes | 1,173 |
| Total edges | 84 |
| Concept nodes | 853 (72.7%) |
| Entity nodes | 265 (22.6%) |
| Technology nodes | 23 (2.0%) |
| Edge types | 84 `depends_on` (timeline/arrow patterns) |

## Benchmark Results (15 questions)

**Key result: NEXUS does NOT generalize to unfamiliar domains without curated aliases.**

| Metric | NEXUS | Baseline |
|--------|-------|----------|
| Answer rate | 0.0% | 100% |
| Fuzzy accuracy | 0.00% | 0.00% |
| Exact accuracy | 0.00% | 0.00% |
| Entity resolution rate | 100.0% | N/A |
| Avg paths found | 0.0 | N/A |
| Avg hallucination | 0.00% | N/A |
| Avg latency | 0.19s | 2.64s |

### Root Causes

1. **Entity resolution resolves wrong entities:** 100% resolution rate, but it
   matches 5 random concept nodes per question (e.g., "Azanto Post MVP
   Implementation Plan", "Check Sundream Connectivity") — none of which are
   actually relevant to the question.

2. **No meaningful edges:** Only 84 `depends_on` edges from timeline arrow
   patterns. The rich semantic relationships (e.g., "PostgreSQL IS_USED_BY Azanto",
   "JWT IMPLEMENTS auth") are not captured by the generic extractor.

3. **Path count = 0:** Even when entities are resolved, the graph has no paths
   connecting them, so `traverse_with_intent` finds nothing. Every question
   returns "Insufficient evidence."

4. **Concept noise:** 853 concept nodes from bold text and headers, mostly
   irrelevant technical procedure names. No domain-specific vocabulary
   extraction.

### Comparison to sam-lm Benchmark

| Metric | sam-lm (curated) | Azanto (generic) |
|--------|-----------------|------------------|
| Graph nodes | ~250 | 1,173 |
| Graph edges | ~116 | 84 |
| Entity resolution rate | 96-100% | 100% |
| **Accuracy** | **~70%** | **0%** |
| Answer rate | ~90% | 0% |
| Paths per question | ~3-5 | 0 |

## Generalization Assessment

**FAIL (<25% threshold).** NEXUS achieves 0% accuracy on the external Azanto
corpus with generic, rule-based extraction. The architecture does NOT generalize
to unfamiliar domains without hand-curated aliases and domain vocabulary.

This validates the concern that extraction quality is **the primary bottleneck**
for generalization — precisely the "Phase 5 problem" identified in the roadmap.

## How to Point at a Different Corpus

```bash
# Ingest any directory
python -m nexus.ingestion.ingest_generic \
    --dir /path/to/your/corpus \
    --patterns "**/*.md" "**/*.txt"

# Run the external benchmark
python benchmarks/run_external_benchmark.py \
    --corpus /path/to/your/corpus \
    --qa benchmarks/external_qa/your_questions.jsonl \
    --limit 15 \
    --verbose
```

## QA Format

QA questions use JSONL format:
```json
{"id": "az001", "question": "...", "answer": "...", "question_type": "factual", "entities": ["..."], "difficulty": "easy", "hops": 1}
```

## Files

- `benchmarks/external_qa/azanto_questions.jsonl` — 15 factual questions about Azanto
- `nexus/ingestion/ingest_generic.py` — corpus-agnostic ingestion pipeline
- `benchmarks/run_external_benchmark.py` — benchmark runner for external corpora
