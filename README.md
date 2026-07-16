# SAM-architecture-research

**Architecture pivot: SAM → NEXUS** (2026-07-08)

*Experimental research project. Not production code. Not a validated architecture.*

---

## Active: NEXUS — Non-Parametric Execution and Understanding System

NEXUS is a **graph-first reasoning architecture**. It stores knowledge as an explicit graph
of entities, relations, and sources — not as dense weights or document chunks.

Instead of: `documents → embeddings → top-K chunks → general-purpose LLM`

NEXUS goes: `entities → relations → graph paths → evidence → constrained realizer`

**The core bet:** reasoning should start from graph traversal, not text generation.
The graph remains the source of domain knowledge. Learned components are small,
CPU-oriented selectors or realizers; they do not replace graph traversal with a
general-purpose LLM reasoning path.

| Property | RAG | NEXUS |
|----------|-----|-------|
| Knowledge | Text chunks | Graph: entities + typed relations + sources |
| Retrieval | Cosine similarity | Graph traversal + path scoring |
| Multi-hop | LLM infers chains from text | Graph paths ARE the chains |
| Hallucination | LLM may invent connections | Verifier checks answer against evidence |
| CPU-first | No (embedding model needs GPU) | Yes (traversal is CPU, small LLM runs on CPU) |

### Project status

| Area | Status |
|------|--------|
| Architecture design | ✅ Complete — see [docs/](docs/) |
| Graph data model | ✅ Defined — node types, edge types, confidence scoring |
| Reasoning pipeline | ✅ Designed — entity extraction → traversal → evidence → verify |
| Graph store | ✅ Implemented — `InMemoryGraphStore` with 1,866 nodes |
| Associative encoder (Stage 1D) | ✅ PASS — frozen-split entity_recall 65.82% (181/275) with validation-selected parser handoff cap 200; all immutable Stage 1 gates passed. |
| Entity Ranker V3 | ✅ **CHECKPOINT VERIFIED** — the 2 July model bundle includes config, vocabulary and the exact 3.49 MB checkpoint; file size and SHA-256 are checked before loading. Historical frozen claims remain consumed and are not reused for model selection. |
| Canonical baseline (Stage 0) | ✅ **VALID** — dependency-free lexical RAG and lexical NEXUS both answer the registered 30 cases; 25 cases form the paired comparison. |
| Realization L1 (Stage 2) | ✅ **PASS** — registered 30-case protocol passes for `PYTHONHASHSEED=0,1,42`: relevance 78.33%, accuracy delta +15.34pp, naturalness +22.40 and hallucination delta -5.12pp. All seeds have the same canonical content hash. |
| Dialogue state (Stage 3) | ✅ **PASS** — full 110-turn protocol: reference resolution 87.5%, zero single-turn regression and dialogue-state p50 0.048ms. Resolver and complete-pipeline latency remain separate diagnostics. |
| Realization L2 (Stage 4) | ✅ **GO FOR TRAINING** — 7,127 unique train-only pairs, oracle, readiness, CPU preflight and 50-step no-write overfit smoke pass. The default pilot is capped at 5 epochs with patience 3. |
| End-to-end QA | ✅ Phase 0–4 training-readiness contract implemented; full Realizer training has deliberately not started yet. |

### Quick start (NEXUS)

```bash
# Explore the graph data model
python -c "from nexus.graph import Node, Edge, EDGE_TYPE_WEIGHTS; print(EDGE_TYPE_WEIGHTS)"

# Create and traverse a knowledge graph
python -c "
from nexus.graph import Node, Edge
from nexus.graph.store import InMemoryGraphStore

g = InMemoryGraphStore()
g.add_node(Node(id='DHM', type='Entity'))
g.add_node(Node(id='MigrationTest', type='TestCase', properties={'status': 'failing'}))
g.add_node(Node(id='Bug_Visibility', type='Bug'))
g.add_edge(Edge(type='validates', source='MigrationTest', target='DHM', confidence=1.0))
g.add_edge(Edge(type='blocked_by', source='MigrationTest', target='Bug_Visibility', confidence=0.9))

paths = g.traverse(['MigrationTest'], max_depth=3)
for p in paths:
    print(p)
```

### Documentation (NEXUS)

- [Analysis & Roadmap](ANALYSIS_AND_ROADMAP.md) — full architecture, roadmap, research questions
- [Graph Memory Model](docs/graph-memory.md) — data model, node/edge types, construction pipeline
- [Graph Reasoning](docs/graph-reasoning.md) — traversal, path scoring, evidence building, verification
- [Auditability & Reasoning Roadmap](docs/nexus-auditability-roadmap.md) — proof traces, provenance, oracle evaluation, and staged acceptance gates
- [Realizer v1 Training Status](docs/nexus-realizer-pretraining-status.md) — verified Phase 0–4 gates, evidence and safe pilot procedure
- [Pilot Integrity and Next Run](docs/nexus-pilot-integrity.md) — preset wiring, artifact identity, resolver contracts and launch rules
- [RAG vs NEXUS](docs/rag-vs-graph-nexus.md) — detailed comparison, when to use which

### Repository structure

```
nexus/                  ← NEXUS implementation (ACTIVE)
  graph/                → Graph store, traversal, scoring
  ingestion/            → Entity/relation extraction pipelines
  query/                → Question parsing, entity disambiguation
  reasoning/            → Evidence building, prompt templates, verifier
  realizer/             → Byte tokenizer and CPU Transformer factory

experiments/            ← NEXUS experiments (ACTIVE)
  entity-extraction/
  relation-extraction/
  graph-traversal/
  path-ranking/

benchmarks/             ← NEXUS benchmarks (ACTIVE)
  qa-dataset/
  graph-eval/
  rag-baseline/

docs/                   ← NEXUS documentation (ACTIVE)
  graph-memory.md
  graph-reasoning.md
  rag-vs-graph-nexus.md

sam-lm/                 ← Original SAM experiments (ARCHIVED — reference only)
```

---

## Archived: SAM — Sparse Associative Memory (deprecated)

The original SAM architecture proved that a small reasoning core CAN use external
memory for multi-hop reasoning (100% oracle accuracy) and that chain-set retrieval
can find complete fact chains (100% all_required@32). These findings informed the NEXUS design.

SAM code, experiments, and documentation are preserved in `sam-lm/` for reference.
See [sam-lm/README.md](sam-lm/README.md).

**Key SAM findings that transferred to NEXUS:**
- Oracle memory = 100% on multi-hop → reasoning core CAN use external structured knowledge
- Chain-set retrieval = 100% all_required@32 → complete-set retrieval works; graph is the generalization
- Selector bottleneck (50% precision) → distinguishing relevant from misleading facts requires graph structure, not flat MLPs
- SAM tolerates +8 random distractors → architecture is noise-tolerant; the problem is semantic, not quantitative

---

*Last updated: 2026-07-16 (Phase 0–4 readiness passes; the next authorized action is a short, generation-aware Realizer pilot.)*
