# SAM-architecture-research

**Architecture pivot: SAM → NEXUS** (2026-07-08)

*Experimental research project. Not production code. Not a validated architecture.*

---

## Active: NEXUS — Non-Parametric Execution and Understanding System

NEXUS is a **graph-first reasoning architecture**. It stores knowledge as an explicit graph
of entities, relations, and sources — not as dense weights or document chunks.

Instead of: `documents → embeddings → top-K chunks → LLM`

NEXUS goes: `entities → relations → graph paths → evidence → small reasoning model`

**The core bet:** reasoning should start from graph traversal, not text generation.
The LLM is a language interface and lightweight reasoner; the domain intelligence
comes from the graph.

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
| Entity Ranker V3 (preregistered) | 🔄 Preregistered — `EXPERIMENT_ENTITY_RANKER_V3.md`; 10 critical defects documented, implementation pending. |
| Realization L1 (Stage 2) | ⚠️ Unvalidated — built on failed Stage 1b foundation |
| Dialogue state (Stage 3) | ⚠️ Unvalidated — built on failed Stage 1b foundation |
| Realization L2 (Stage 4) | ⏭️ Skipped — entry conditions not met |
| End-to-end QA | ❌ Not validated — stack halted at Stage 1b |

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
- [RAG vs NEXUS](docs/rag-vs-graph-nexus.md) — detailed comparison, when to use which

### Repository structure

```
nexus/                  ← NEXUS implementation (ACTIVE)
  graph/                → Graph store, traversal, scoring
  ingestion/            → Entity/relation extraction pipelines
  query/                → Question parsing, entity disambiguation
  reasoning/            → Evidence building, prompt templates, verifier

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

*Last updated: 2026-07-10 (validated Stage 1D frozen-split pass; Entity Ranker V3 preregistered at `EXPERIMENT_ENTITY_RANKER_V3.md`; current reference: `benchmarks/results/stage1b_honest_20260710_163732Z.json`; historical references remain preserved in the results index)*
