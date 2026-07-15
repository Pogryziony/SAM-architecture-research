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
| Entity Ranker V3 | ⚠️ **VALIDATION PASS / EXTERNAL CHECKPOINT REQUIRED** — latest retrain reports canonical recall@10=72.53% and +39.0pp over the trivial baseline. Config and vocabulary are tracked; weights remain external and must match the manifest SHA-256. Historical frozen claims remain non-repeatable because that split is consumed. |
| Realization L1 (Stage 2) | ⚠️ **RERUN REQUIRED** — the earlier registered 30-case baseline reported 78.33% relevance, but the committed July 15 artifacts are only 5-case smoke runs. The runner now distinguishes registered and smoke protocols and writes a real file-hash sidecar. |
| Dialogue state (Stage 3) | ❌ **FAIL** — latest 110-turn run: reference resolution 15.62% and resolver p50 12.166ms. Stage 3 now uses the injected resolver path; it must be rerun with the verified external ER3 checkpoint. |
| Realization L2 (Stage 4) | ⚠️ **PILOT BLOCKED** — 7,127 unique train-only pairs exist and the first 50-epoch CPU run completed, but post-training answer metrics regressed. Decoder repetition was diagnosed and mitigated; a new 1→3→5 epoch pilot is required after all gates pass. |
| End-to-end QA | ❌ Not validated — Stage 0, registered Stage 2 and Stage 3 evidence must be regenerated under the corrected contracts. |

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
- [Realizer v1 Training Status](docs/nexus-realizer-pretraining-status.md) — first-run outcome, current blockers, gates, and safe pilot procedure
- [Pilot Integrity and Next Run](docs/nexus-pilot-integrity.md) — corrected preset wiring, resolver contracts, blockers, and safe pilot order
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

*Last updated: 2026-07-16 (training and decoder diagnosis recorded; corrected pilot remains fail-closed until Stage 0/2/3 are regenerated and pass.)*
