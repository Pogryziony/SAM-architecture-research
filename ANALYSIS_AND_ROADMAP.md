# NEXUS Architecture Research — Analysis & Roadmap to First Model

**Date:** 2026-07-08 | **Architecture pivot:** SAM → NEXUS (Non-Parametric Execution and Understanding System)

---

## Part 0: The Pivot — From Associative Memory to Graph-First Reasoning

### 0.1 Why SAM Classic Is Being Deprecated

The original SAM (Sparse Associative Memory) architecture proved three things:

1. **A small core CAN use external memory** — 100% accuracy with oracle memory on multi-hop QA ✓
2. **Retrieval CAN be solved for complete fact chains** — chain-set BCE achieves 100% all_required@32 ✓
3. **But realistic end-to-end retrieval FAILS** — the selector bottleneck (50% precision) means SAM never beats the core_only baseline ✗

The fundamental problem is architectural: SAM treats memory as **flat slots in a vector space** — the retriever maps questions to slot embeddings via similarity. This works for controlled synthetic data but breaks on realistic distractor quality because:

- Similarity in embedding space ≠ factual relevance
- Multi-hop chains require understanding *how* facts relate, not just *that* they co-occur
- The selector is asked to solve a fundamentally graph-structured problem with a flat MLP

### 0.2 The NEXUS Vision

**NEXUS = Non-Parametric Execution and Understanding System**

The core insight: **reasoning should not start from text generation. It should start from graph traversal.**

| | Old SAM | NEXUS |
|---|---------|-------|
| Knowledge representation | Flat vector slots (PKM) | Graph: entities, relations, paths, sources |
| Retrieval | Embedding similarity (dual encoder, chain-set BCE) | Graph traversal + path scoring |
| Memory structure | Key-value vectors in RAM | Fact nodes + relation edges + source evidence |
| Reasoning | Small transformer with gated memory injection | Graph traversal → evidence pack → small reasoning model |
| Primary compute | Dense transformer forward pass | CPU graph traversal + lightweight LLM |
| Debuggability | Black-box slot embeddings | Traversible, inspectable graph paths |
| Knowledge update | Retrain slot embeddings | Add/remove nodes and edges |

### 0.3 What the Old Experiments Gave Us (Transferable Knowledge)

| SAM Classic Finding | NEXUS Takeaway |
|--------------------|----------------|
| Oracle memory = 100% on multi-hop | A reasoning core CAN use external structured knowledge. Graph paths are the natural next step. |
| Chain-set retrieval achieves 100% all_required@32 | Retrieving complete sets of related facts works. Graph traversal is the generalization of this. |
| Selector recall 96.6%, precision 50% | Distinguishing relevant from misleading facts is the hard problem. Graph structure provides the missing signal. |
| SAM tolerates +8 random distractors but struggles with semantic hard negatives | Random noise is easy; semantically-plausible-but-wrong information is hard. Graph relationships disambiguate. |
| Product-key memory works at small scale | The idea of sparse, selective knowledge access is valid. Graph edges are the natural addressing mechanism. |

### 0.4 Research Arc (SAM Classic → NEXUS Pivot)

```
Phase 1: Pipeline Setup (Exp 0–0.5)
  → Infrastructure working, retrieval 6.9% → 99.0% Rec@8

Phase 2: Core Validation (Exp 0.6–0.9)
  → Oracle memory = 99.87% — architecture CAN work
  → Realistic retrieval = core_only — it DOESN'T yet work

Phase 3: Retrieval Revolution (Exp 0.10–0.11)
  → Chain-set BCE: all_required@32 = 100% — retrieval solved for synthetic data
  → But SAM still = core_only — retrieval ≠ understanding

Phase 4: Selection & Noise (Exp 0.12–0.13A)
  → Selector: 96.6% recall, 50% precision — flat MLP can't solve graph problems
  → SAM tolerates +8 random distractors (91.6%) — architecture is noise-tolerant
  → Gate NOT the bottleneck — selection quality IS

═══════════ PIVOT POINT ═══════════
Phase 5: NEXUS (Architecture Redesign)
  → Knowledge as explicit graph, not latent vectors
  → Reasoning as traversal, not similarity search
  → LLM as language interface, not knowledge store
```

---

## Part 1: NEXUS Architecture

### 1.1 The Full Pipeline

```
Input question
     ↓
Entity & Intent Extraction   ← Lightweight NLP / small LLM
     ↓
Graph Lookup                 ← Find entry nodes by entity name/type
     ↓
Graph Traversal              ← Walk edges: depends_on, caused_by, validates, etc.
     ↓
Path Scoring                 ← Score paths by relevance, recency, source trust
     ↓
Evidence Building            ← Collect fact nodes, relations, sources along best paths
     ↓
Small Reasoning Model        ← Gets structured evidence (not raw chunks), generates answer
     ↓
Verifier                     ← Checks answer against evidence, flags unsupported claims
     ↓
Answer
```

### 1.2 Core Innovation: Reasoning Starts From Graph Traversal, Not Text Generation

```
RAG:                                      NEXUS:
                                          ┌──────────┐
┌──────────┐                              │  Entity   │
│ Document │                              │   DHM     │
│  Chunks  │                              └────┬─────┘
└────┬─────┘                              ┌────▼─────┐
     │                                    │  Entity   │
     ▼                                    │ WorkOrder │
┌──────────┐                              └────┬─────┘
│ Similarity│                              ┌────▼─────┐
│  Search   │                              │  Entity   │
└────┬─────┘                              │MigrationTest│
     │                                    └────┬─────┘
     ▼                                    ┌────▼─────┐
┌──────────┐                              │  Entity   │
│   LLM    │                              │ DataHub   │
│ (all text)│                             └────┬─────┘
└──────────┘                              ┌────▼─────┐
                                          │  Bug      │
                                          │ #KNOWN-42 │
                                          └──────────┘
                                             ↑
                                        Evidence path,
                                        not just a chunk
```

The LLM doesn't "invent connections" — it receives **already-discovered graph paths**.

### 1.3 Data Model

**Node types:**

| Type | What it represents | Example |
|------|-------------------|---------|
| Entity | Domain object | DHM, WorkOrder, DataHub |
| Concept | Abstract idea | Migration, Visibility, Status Change |
| Document | Source text | `doc.md`, `architecture.md` |
| CodeFile | Source code file | `migration_service.py` |
| Function | Specific function | `validate_migration()` |
| TestCase | A test | `MigrationDataTest` |
| Bug | Known issue | `Case not visible in Elements` |
| Decision | Design choice | "Use async queue for migration" |
| Requirement | Spec requirement | "Must support partial migration" |
| Experiment | Research experiment | "Experiment 0.11 — chain-set retrieval" |
| Metric | Measured value | "all_required@32 = 100%" |

**Edge types (relation semantics):**

| Relation | Meaning | Example |
|----------|---------|---------|
| depends_on | A requires B to function | MigrationTest → depends_on → DataHub |
| caused_by | A was triggered by B | Bug → caused_by → StatusChange |
| validates | A confirms B works | MigrationTest → validates → DHM_migration |
| contradicts | A conflicts with B | TestResult → contradicts → Requirement |
| implements | A realizes B in code | Function → implements → Decision |
| mentioned_in | A appears in document B | Entity → mentioned_in → Document |
| derived_from | A was conceptually derived from B | Decision → derived_from → Concept |
| sub_experiment | A is a structural child of B | Run → sub_experiment → Experiment |
| related_to | General association | Entity → related_to → Concept |
| replaces | A supersedes B | NewFunction → replaces → OldFunction |
| blocked_by | A cannot proceed until B | MigrationTest → blocked_by → Bug |

### 1.4 Example Knowledge Record

```json
{
  "node": "MigrationDataTest",
  "type": "TestCase",
  "properties": {
    "status": "failing",
    "last_run": "2026-06-15",
    "failure_rate": 1.0
  },
  "edges": [
    { "type": "validates",     "target": "DHM_migration",         "confidence": 1.0 },
    { "type": "depends_on",    "target": "DataHub_visibility",    "confidence": 0.95 },
    { "type": "blocked_by",    "target": "BUG_Elements_visibility", "confidence": 0.9 },
    { "type": "mentioned_in",  "target": "migration_spec.md",     "confidence": 1.0 }
  ],
  "sources": ["doc.md:45-67", "issue-123", "test-results.json:12"]
}
```

### 1.5 Concrete Reasoning Example

**Question:** "Why does the DHM migration test not pass after changing WO status?"

**RAG approach:**
1. Embed question → find similar chunks → "migration test", "WO status", "DHM"
2. Retrieve top-K semantically similar text fragments
3. LLM reads fragments, tries to connect them

**NEXUS approach:**
1. Extract entities: `DHM`, `WorkOrder`, `MigrationDataTest`, `status change`
2. Graph lookup: find `MigrationDataTest` node
3. Traverse edges:
   - `MigrationDataTest` → `validates` → `DHM_migration`
   - `MigrationDataTest` → `depends_on` → `DataHub_visibility`
   - `MigrationDataTest` → `blocked_by` → `BUG_Elements_visibility`
   - `BUG_Elements_visibility` → `caused_by` → `WO_status_change`
4. Build evidence path:
   ```
   WO_status_change → caused_by → BUG_Elements_visibility
                                  → blocked_by → MigrationDataTest
                                  → validates → DHM_migration
   ```
5. Reasoning model receives this exact path, generates answer referencing the chain

### 1.6 Target Compute Model

| Resource | What runs on it |
|----------|----------------|
| **CPU** | Graph traversal, path scoring, entity resolution |
| **RAM** | Graph store (nodes, edges, properties, source references) |
| **CPU/RAM** | Small reasoning model (< 1B params) — receives evidence packs |
| **Disk (mmap)** | Source documents, code, logs — referenced by nodes, not loaded into RAM |

The key efficiency thesis: **graph traversal is O(path_length × branching_factor), not O(total_knowledge)**. The reasoning model receives a small, curated evidence pack (~1-2KB of structured facts), not raw documents.

---

## Part 2: Comparison to LLMs and RAG

### 2.1 Why NEXUS Is Fundamentally Different From RAG

| Property | RAG | NEXUS |
|----------|-----|-------|
| Search mechanism | Semantic similarity (embedding cosine) | Graph structure (edge traversal) |
| What is found | Similar text chunks | Related entities via explicit relations |
| Noise source | Semantically similar but factually wrong chunks | Graph edges can be wrong/missing, but noise is structural not semantic |
| Reasoning | LLM infers connections from raw text | Graph provides explicit connections; LLM only verbalizes |
| Multi-hop | LLM must track chains implicitly in text | Graph paths ARE the chains — explicit, traversible |
| Debuggability | "Why did the model say that?" — opaque | "The answer comes from path: A → B → C, and here are the sources" |
| Knowledge update | Re-index documents | Add/remove nodes and edges |
| Hallucination surface | Large — LLM generates from noisy context | Small — LLM only verbalizes graph evidence |

### 2.2 Why NEXUS Is Fundamentally Different From Dense LLMs

| Property | Dense LLM (GPT, Claude, Llama) | NEXUS |
|----------|-------------------------------|-------|
| Knowledge location | Compressed into weight matrices | Explicit graph with source traceability |
| Knowledge capacity | Limited by parameter count (expensive to scale) | Limited by RAM (cheap to scale) |
| Knowledge freshness | Frozen at training cutoff | Updated by adding nodes/edges, no retraining |
| Reasoning mechanism | Implicit in transformer computations | Explicit graph traversal + lightweight model |
| Per-token compute | Scales with parameter count (all weights used) | Constant (small reasoning model) + graph traversal cost |
| Explainability | "The model predicted this" | "The graph shows path X → Y → Z, confirmed by source S" |
| Factual precision | Hallucinations are a known failure mode | Verifier checks answer against graph evidence |

### 2.3 The Key Research Hypothesis

> **Reasoning does not need to start from text generation. Reasoning can start from graph traversal. The language model's role shifts from "knowing everything" to "articulating what the graph already discovered."**

This means:
- The LLM is NOT the knowledge store — the graph is
- The LLM is NOT the reasoning engine — the traversal is
- The LLM IS the language interface — translating structured evidence into natural language
- The LLM IS the lightweight reasoner — filling gaps where graph evidence is incomplete

### 2.4 Target Metrics (New)

Moving from retrieval-centric metrics to graph-centric metrics:

| Metric | What it measures | Current (SAM) | Target (NEXUS) |
|--------|-----------------|---------------|----------------|
| **Entity resolution accuracy** | Correctly identified entities in query | N/A | >90% |
| **Relation extraction accuracy** | Correctly typed relations between entities | **100% F1** (2026-07-21; structural `sub_experiment` excluded) | >85% |
| **Path accuracy** | Retrieved paths contain the correct reasoning chain | N/A | >80% |
| **Path relevance@K** | Top-K paths are actually useful for answering | N/A (all_required@32 = 100% analogue) | >90% |
| **Evidence precision** | Evidence pack contains only relevant facts | 50% (selector precision analogue) | >85% |
| **Hallucination rate** | Claims in answer not supported by graph evidence | Not measured | <10% |
| **Traversal latency** | Time to traverse graph and build evidence pack | N/A | <500ms (CPU) |
| **RAM usage** | Graph store memory footprint | N/A (1,650 slots × 128-dim) | <8GB for 1M nodes |
| **Graph update cost** | Time/resources to add a new fact | Retrain slot embeddings | O(1) node/edge insertion |
| **Answer accuracy** | Final answer correctness | 68.74% (core_only) → 100% (oracle) | >90% on domain-specific QA |

---

## Part 3: Roadmap to First NEXUS Model

### 3.1 Phase 1: Graph Infrastructure & Ingestion (Weeks 1-4)

**Goal:** Build the graph construction pipeline from existing project artifacts.

#### Step 1.1: Graph Store Implementation
- Choose graph backend: **NetworkX** (prototyping) → **Neo4j** or ** KuzuDB** (production)
- Implement in-memory graph with node/edge types from §1.3
- Implement basic traversal: BFS/DFS with edge type filtering
- Implement path scoring: edge confidence × recency × source trust

#### Step 1.2: Entity & Relation Extraction
- **From structured sources** (issue tracker, test results, experiment metrics):
  - Parse JSON/YAML → auto-create typed nodes with properties
  - Infer edges from explicit references (issue mentions test, test validates requirement)
- **From semi-structured sources** (markdown docs, configs):
  - Regex-based entity extraction from doc headers, code references
  - Template-based relation extraction ("X depends on Y", "X validates Z")
- **From unstructured sources** (natural language in issues, docs):
  - Small LLM (e.g., Phi-3, Llama-3.2-3B) for entity extraction + relation extraction
  - Prompt: "Extract entities and their relationships from this text. Output as JSON."

#### Step 1.3: Graph Construction Pipeline
```
Documents, Issues, Tests, Code, Experiments
     ↓
Entity Extraction (rule-based + LLM)
     ↓
Relation Extraction (rule-based + LLM)
     ↓
Deduplication (fuzzy name matching)
     ↓
Normalization (entity name canonicalization)
     ↓
Graph Merge (add nodes, add edges, deduplicate)
     ↓
Source Annotation (attach evidence pointers)
```

#### Step 1.4: Initial Population
Populate the graph with existing project artifacts:
- All experiment reports → Experiment nodes
- All metrics from experiments → Metric nodes linked to experiments
- All issues/PRs → Bug/Decision nodes
- Code structure → CodeFile/Function nodes
- Documentation → Document nodes
- Cross-reference everything with `mentioned_in`, `derived_from`, `depends_on` edges

**Phase 1 Deliverables:**
- [ ] Graph store with typed nodes, typed edges, property support
- [ ] Entity extraction pipeline (rule-based + LLM)
- [ ] Relation extraction pipeline (rule-based + LLM)
- [ ] Graph populated with all existing project knowledge
- [ ] Basic traversal and path scoring working

### 3.2 Phase 2: Query Understanding & Traversal (Weeks 5-8)

**Goal:** Given a natural language question, find relevant graph paths.

#### Step 2.1: Query → Entity Mapping
- Extract entities from question text
- Fuzzy match against graph node names
- Disambiguate: "DHM" → `Entity:DHM`, not `Concept:DHM` or `TestCase:DHM`
- Handle multi-entity queries (multiple entry points into graph)

#### Step 2.2: Graph Traversal Engine
- From entry nodes, traverse edges up to depth D (configurable, default 4)
- Edge type weighting: `caused_by` > `depends_on` > `validates` > `related_to` > `mentioned_in`
- Beam search: at each hop, keep top-B paths (configurable, default 5)
- Path scoring: composite score = Σ(edge_weight × edge_confidence) / path_length

#### Step 2.3: Path Ranking
- Score paths by: relevance to query entities, structural coherence, source recency
- De-duplicate semantically equivalent paths (A→B→C and A→C via B are same)
- Select top-K paths (K=3-5) for evidence building

#### Step 2.4: Evidence Building
- From selected paths, collect: entities, relations, properties, sources
- Format as structured JSON evidence pack (~1-2KB)
- Include source references for every fact in the evidence

**Phase 2 Deliverables:**
- [ ] Entity extraction from natural language queries
- [ ] Fuzzy entity disambiguation against graph
- [ ] Multi-path traversal with scoring
- [ ] Evidence pack construction
- [ ] End-to-end: question → evidence pack

### 3.3 Phase 3: Reasoning Model & Verifier (Weeks 9-12)

**Goal:** Build the small reasoning model that converts evidence packs into answers.

#### Step 3.1: Evidence → Prompt Template
Design a structured prompt that presents the evidence pack to the reasoning model:
```
You are a reasoning assistant. Below is structured evidence from a knowledge graph.

QUESTION: {question}

EVIDENCE PATH:
  Entity: {entity1} ({type1})
    → {relation1} → Entity: {entity2} ({type2}) [source: {source1}]
    → {relation2} → Entity: {entity3} ({type3}) [source: {source2}]

FACTS:
  - {entity1} depends on {entity2} because {explanation}
  - {entity3} is blocked by {entity4}

SOURCES:
  - {source1}: {excerpt}
  - {source2}: {excerpt}

Based ONLY on the evidence above, answer the question. If the evidence is
insufficient, say so. Do not invent facts not present in the evidence.
```

#### Step 3.2: Reasoning Model Selection
- Candidates: Phi-3-mini (3.8B), Llama-3.2-3B, Qwen-2.5-3B
- Run on CPU (quantized INT8/INT4)
- Fine-tune on (evidence_pack, question, answer) triples
- Target: <2s inference time on CPU

#### Step 3.3: Verifier
- After reasoning model generates answer, verifier checks:
  1. Are all claimed entities present in the evidence pack?
  2. Are all claimed relations present in the evidence graph?
  3. Does the answer contradict any evidence fact?
- If verification fails → flag as potential hallucination, return "Insufficient evidence"
- Verifier is rule-based (not another LLM) for reliability and speed

#### Step 3.4: Training Data Generation
- Use the existing graph to auto-generate (evidence_pack, question, answer) triples
- Template-based question generation from graph paths
- Negative examples: evidence packs with deliberately wrong/missing facts

**Phase 3 Deliverables:**
- [ ] Evidence-to-prompt template
- [ ] Fine-tuned reasoning model (< 4B params, CPU-capable)
- [ ] Rule-based verifier
- [ ] Training dataset: 10K+ (evidence, question, answer) triples
- [ ] End-to-end: question → graph traversal → evidence → answer → verification

### 3.4 Phase 4: Benchmarking & Comparison (Weeks 13-16)

**Goal:** Quantitatively compare NEXUS against RAG and dense LLM baselines.

#### Step 4.1: QA Dataset Construction
- Curate 200-500 domain-specific questions about the project
- Include: factual, multi-hop, diagnostic, comparative questions
- Ground-truth answers with source citations
- Difficulty levels: single-hop, 2-hop, 3-hop, 4-hop
- Categories: code questions, test questions, architecture questions, experiment questions

#### Step 4.2: Baseline Implementations
Run the same QA dataset through four configurations:

| # | System | Description |
|---|--------|-------------|
| 1 | **NEXUS** | Graph traversal → evidence → small reasoning model |
| 2 | **Classic RAG** | Embed documents → top-K chunks → same reasoning model |
| 3 | **Hybrid** | Graph traversal + RAG chunks → combined evidence → reasoning model |
| 4 | **LLM-only** | Same reasoning model with no external knowledge (closed-book) |

#### Step 4.3: Evaluation Metrics
For each system, measure:

| Metric | Why it matters |
|--------|---------------|
| **Answer accuracy** | Primary: did we answer correctly? |
| **Hallucination rate** | What % of claims are unsupported by sources? |
| **Evidence precision** | How much of the evidence pack was actually used? |
| **Context size** | How much data (bytes) was fed to the reasoning model? |
| **Latency (end-to-end)** | Time from question to answer |
| **Traversal latency** | Time for graph traversal specifically |
| **RAM usage** | Memory footprint of knowledge store |
| **Source traceability** | Can we point to the exact source of each claim? |
| **Answer completeness** | Did we address all parts of the question? |

#### Step 4.4: Expected Results (Hypothesis)
Based on the architectural analysis, we expect:

| Metric | NEXUS | RAG | LLM-only |
|--------|-------|-----|----------|
| Answer accuracy | Highest | Medium | Lowest |
| Hallucination rate | Lowest | Medium | Highest |
| Context size | Smallest (~1-2KB) | Medium (~5-10KB) | N/A |
| Latency | Medium (traversal + inference) | Low (retrieval + inference) | Lowest |
| Source traceability | Full path + sources | Chunk similarity | None |
| Multi-hop performance | Best (explicit paths) | Medium (LLM infers) | Worst |

**Phase 4 Deliverables:**
- [ ] QA dataset (200-500 questions with ground truth)
- [ ] Four-system comparison benchmark
- [ ] Full metrics report
- [ ] Decision: does NEXUS substantively beat RAG on domain-specific QA?

### 3.5 Phase 5: Production-Ready NEXUS (Weeks 17-24)

**Goal:** Package NEXUS as a usable system with incremental knowledge updates.

#### Step 5.1: Incremental Graph Updates
- New document → re-run entity/relation extraction → merge into graph
- Changed code → detect modified functions → update relevant nodes
- New experiment results → create nodes, link to prior experiments
- **No retraining needed** — graph updates are O(1) per node/edge

#### Step 5.2: Persistent Graph Storage
- Move from in-memory to persistent (SQLite + JSONB or KuzuDB)
- Graph versioning: track when facts were added/removed
- Export/import for portability

#### Step 5.3: API / Interface
- CLI: `nexus ask "Why does the DHM migration test fail?"`
- Python API: `nexus.query("...")` → returns (answer, evidence_paths, sources)
- Optional: simple web UI for graph exploration

#### Step 5.4: Optimization
- Precompute frequently-traversed subgraphs
- Cache common query → evidence pack mappings
- Quantize reasoning model to INT4 for CPU deployment
- Profile and optimize traversal for graphs up to 1M nodes

#### Step 5.5: Documentation & Release
- Architecture documentation
- Ingestion pipeline documentation
- API reference
- Benchmark results
- Comparison to RAG baseline
- Getting started guide for new domains

**Phase 5 Deliverables:**
- [ ] Incremental update pipeline
- [ ] Persistent graph storage
- [ ] CLI + Python API
- [ ] Performance benchmarks on 100K+ node graphs
- [ ] Full documentation
- [ ] Reproducible benchmark suite

---

## Part 4: Repository Structure (Target)

```
SAM-architecture-research/
├── README.md                          # NEXUS vision, quick start
├── ANALYSIS_AND_ROADMAP.md            # This document
│
├── docs/
│   ├── graph-memory.md                # Graph as knowledge store (data model, node/edge types)
│   ├── graph-reasoning.md             # Graph traversal as reasoning (paths, scoring, evidence)
│   ├── rag-vs-graph-nexus.md          # Detailed comparison: RAG vs NEXUS
│   ├── architecture.md                # Full NEXUS architecture
│   ├── ingestion.md                   # How knowledge enters the graph
│   └── experiments.md                 # Experiment plans and results
│
├── nexus/                             # NEXUS implementation (NEW)
│   ├── __init__.py
│   ├── graph/
│   │   ├── store.py                   # Graph data structure (nodes, edges, properties)
│   │   ├── traversal.py               # BFS/DFS + beam search + edge filtering
│   │   └── scoring.py                 # Path scoring algorithms
│   ├── ingestion/
│   │   ├── entity_extractor.py        # Rule-based + LLM entity extraction
│   │   ├── relation_extractor.py      # Rule-based + LLM relation extraction
│   │   ├── deduplicator.py            # Fuzzy entity deduplication
│   │   └── normalizer.py              # Entity name canonicalization
│   ├── query/
│   │   ├── parser.py                  # Query → entity list + intent
│   │   └── disambiguator.py           # Entity disambiguation
│   ├── reasoning/
│   │   ├── evidence_builder.py        # Graph paths → structured evidence
│   │   ├── prompt_template.py         # Evidence → LLM prompt
│   │   └── verifier.py                # Answer → fact-check against evidence
│   └── cli.py                         # CLI interface
│
├── experiments/
│   ├── entity-extraction/             # Entity extraction quality experiments
│   ├── relation-extraction/           # Relation extraction quality experiments
│   ├── graph-traversal/               # Traversal quality & performance experiments
│   └── path-ranking/                  # Path scoring & ranking experiments
│
├── benchmarks/
│   ├── qa-dataset/                    # Domain-specific QA dataset
│   ├── graph-eval/                    # Graph-specific evaluation metrics
│   └── rag-baseline/                  # RAG baseline implementation for comparison
│
├── sam-lm/                            # Original SAM code (ARCHIVED — reference only)
│   └── ...
│
└── configs/                           # NEXUS configuration files
    ├── graph_config.yaml
    ├── ingestion_config.yaml
    └── reasoning_config.yaml
```

---

## Part 5: Key Research Questions (Unanswered)

These are the critical unknowns that the Phase 1-2 experiments must answer:

| # | Question | How to answer |
|---|----------|--------------|
| Q1 | Can entity/relation extraction from code docs + issues achieve >85% accuracy? | Phase 1: measure extraction precision/recall on labeled data |
| Q2 | Is graph traversal significantly faster than embedding-based retrieval at scale? | Phase 2: benchmark traversal vs vector search at 10K, 100K, 1M nodes |
| Q3 | Does explicit graph evidence reduce hallucination rate compared to RAG? | Phase 4: measure hallucination rate across NEXUS vs RAG on same QA set |
| Q4 | Can a <1B param reasoning model effectively use structured evidence? | Phase 3: fine-tune and evaluate on evidence → answer task |
| Q5 | Is graph maintenance (updates, dedup) manageable without manual curation? | Phase 5: run incremental ingestion over 4 weeks of new project artifacts |
| Q6 | Does graph-first reasoning generalize to new domains without architecture changes? | Phase 5+: apply NEXUS to a second domain (different codebase) |
| Q7 | Where does the graph approach break? (What types of questions need text, not graph?) | Phase 4: categorize failure cases from benchmark |

---

## Part 6: Immediate Next Actions (This Week)

1. **Create graph data model** — implement `nexus/graph/store.py` with typed nodes & edges
2. **Populate from existing experiments** — extract all experiment results as graph nodes
3. **Implement basic entity extraction** — rule-based extraction from markdown docs
4. **Run first traversal demo** — question → entity lookup → 1-hop traversal → answer
5. **Create QA dataset scaffold** — 200-300 questions about the project as initial test set

---

## Appendix A: Decision Gates

```
Phase 1 → Phase 2:   Entity extraction >80% accuracy on 50-example labeled set
Phase 2 → Phase 3:   Graph traversal returns relevant paths for >70% of test queries
Phase 3 → Phase 4:   End-to-end accuracy >60% on domain QA (baseline: LLM-only ~30%)
Phase 4 → Phase 5:   NEXUS beats RAG on ≥3 of 9 evaluation metrics with statistical significance
Phase 5 → Release:   All benchmarks documented, pipeline reproducible, API stable
```

## Appendix B: Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Entity extraction too noisy for useful graph traversal | Medium | High | Start with rule-based extraction from structured sources; add LLM extraction only for unstructured text |
| Graph traversal latency too high at scale | Medium | Medium | Index frequently-traversed paths; use beam search; limit traversal depth |
| Small reasoning model can't effectively use structured evidence | Low-Medium | High | Fine-tune specifically on evidence → answer task; fall back to larger model if needed |
| Graph becomes stale without continuous ingestion | Medium | Medium | Design incremental update from day 1; automate ingestion from CI/CD |
| Domain-specific QA not representative of real use cases | Low | Medium | Include diverse question types; validate with real user questions from issue tracker |

---

*This document supersedes all previous SAM roadmap documents. The SAM classic experiments (0–0.13A) are archived in `sam-lm/` and their findings are incorporated into the NEXUS design rationale above.*

---

## Phase 4 Status — 2026-07-09 (HISTORICAL)

> **Historical artifact.** The 200-question paired run below predates the
> registered Stage 0 / Stage 2 / Stage 3 / Pointer-Copy / comparison-plan
> stack-v1 evidence. Do **not** treat these FAIL rows as the current stack
> verdict. Authoritative status: [`STACK_RESULTS.md`](STACK_RESULTS.md) and
> [`docs/stack-v1-freeze.md`](docs/stack-v1-freeze.md). The open entity-resolution
> root-cause item remains valid follow-up work.

**Results file**: `benchmarks/results/phase4_paired_20260709_183954Z.json`
**Git commit**: e6e000f

### Checkpoint Results

| Checkpoint | Target | Actual | Status |
|---|---|---|---|
| Answer rate | ≥ 90% | 74.5% (149/200) | ❌ FAIL |
| Entity resolution | ≥ 88.5% | 51.5% (103/200) | ❌ FAIL |
| Hallucination | ≤ 19.25% | 15.63% | ✅ PASS |
| Paired accuracy vs RAG | > prev 35.98% | 24.17% (W=33, L=1, T=166) | ❌ FAIL |

### Measured Causes

1. **Entity resolution regression (88.5% → 51.5%)**: The type-prior-as-tiebreaker fix (Phase 1) did not restore the pre-regression 88.5% rate. Investigation needed: the diagnostic script reported 100% resolution on held-out questions while the benchmark shows 51.5% — these may measure different things (entity spotting vs usable entity IDs that yield paths).

2. **RAG accuracy drop (33.6% → 9.25%)**: The Phase 3 arm guard correctly prevents the RAG arm from accessing graph evidence. The previous 33.6% RAG accuracy was inflated by evidence-blind baseline scoring against lenient metrics. The 9.25% represents honest evidence-blind RAG accuracy.

3. **Answer rate below target (74.5% vs 90%)**: Cascade Level 1 recovers 115/200 questions. Level 3 synth fallback recovers 34 more but with 27.8% accuracy. Level 0 (51 questions, no entity resolution) cannot be recovered by cascade — these are genuine entity resolution failures.

4. **W/L/T = 33/1/166**: On the 34 questions where answers differ, NEXUS dominates (33 wins, 1 loss). But 166 questions are ties (both score 0) because of low answer rates on both arms.

### Open Items for Phase 5

- Router held-out quality (currently unvalidated beyond n=15/15)
- Oracle-test rebuild (force generation, inject into synth parse section)
- Latency budget experiment (relevance-rank truncation)
- Entity resolution root-cause: why does spot_entities + find_entity_by_keywords resolve only 51.5% when the diagnostic reports 100%?
