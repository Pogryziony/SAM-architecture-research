# Graph Memory — NEXUS Knowledge Store

How NEXUS stores knowledge as an explicit graph of entities, relations, and sources.

---

## Why a Graph?

Traditional RAG stores knowledge as **document chunks + embeddings**:
```
chunk_1 → embedding_1 → cosine_similarity(query_embedding)
chunk_2 → embedding_2 → cosine_similarity(query_embedding)
...
```

This works for single-hop fact retrieval but fails for multi-hop reasoning because:
- "Similar text" ≠ "causally connected"
- The retriever doesn't know that chunk_A depends on chunk_B
- Multi-hop chains require the LLM to infer connections from raw text

A graph stores knowledge as **nodes (entities) + edges (relationships)**:
```
DHM ──[validates]──► MigrationTest ──[blocked_by]──► Bug_#42
                                                  ──[caused_by]──► StatusChange
```

The graph structure **explicitly** encodes what RAG must **infer** from text.

## Data Model

### Node Types

| Type | What it represents | Properties | Example |
|------|-------------------|------------|---------|
| **Entity** | Domain object, system, component | name, type, status, metadata | `DHM`, `DataHub`, `WorkOrder` |
| **Concept** | Abstract idea, process, state | name, description, domain | `Migration`, `Visibility`, `StatusChange` |
| **Document** | Source text file | path, title, last_modified | `docs/architecture.md` |
| **CodeFile** | Source code file | path, language, lines | `src/migration_service.py` |
| **Function** | Specific function/method | name, file, signature | `validate_migration()` |
| **TestCase** | A test | name, status, failure_rate | `MigrationDataTest` |
| **Bug** | Known issue/defect | id, status, severity, description | `BUG: Elements visibility` |
| **Decision** | Design/architectural choice | description, rationale, date | "Use async queue for DHM migration" |
| **Requirement** | Specification requirement | id, description, priority | "Must support partial migration" |
| **Experiment** | Research experiment | id, hypothesis, result, date | "Experiment 0.11 — chain-set retrieval" |
| **Metric** | Measured/quantitative value | name, value, unit, timestamp | `all_required@32 = 100%` |

### Edge Types (Relationships)

| Relation | Meaning | Direction | Example | Confidence |
|----------|---------|-----------|---------|------------|
| **depends_on** | A requires B to function | A → B | `MigrationTest → DataHub` | 0.95 |
| **caused_by** | A was triggered/created by B | A → B | `Bug → StatusChange` | 0.90 |
| **validates** | A confirms/verifies B | A → B | `MigrationTest → DHM_migration` | 1.0 |
| **contradicts** | A conflicts with B | A → B | `TestResult → Requirement` | 0.85 |
| **implements** | A realizes B in code | A → B | `Function → Decision` | 0.90 |
| **mentioned_in** | A appears/referenced in B | A → B | `Entity → Document` | 0.80 |
| **derived_from** | A was created/derived from B | A → B | `Experiment → Hypothesis` | 0.85 |
| **related_to** | General association (weak) | A ↔ B | `Entity → Concept` | 0.50 |
| **replaces** | A supersedes/deprecates B | A → B | `NewFunction → OldFunction` | 0.95 |
| **blocked_by** | A cannot proceed until B is resolved | A → B | `MigrationTest → Bug` | 0.90 |

### Confidence Scores

Every edge has a confidence score [0.0, 1.0]:
- **1.0**: Verified fact (e.g., code explicitly imports/calls, test explicitly validates)
- **0.9-0.99**: Strongly inferred (e.g., bug report mentions test, experiment measures metric)
- **0.7-0.89**: Moderately inferred (e.g., document mentions entity, code comment references concept)
- **0.5-0.69**: Weakly inferred (e.g., co-occurrence, topic similarity)
- **<0.5**: Speculative — not used for evidence, only for exploration

## Knowledge Record Format

```json
{
  "node_id": "MigrationDataTest",
  "type": "TestCase",
  "properties": {
    "status": "failing",
    "last_run": "2026-06-15T10:30:00Z",
    "failure_rate": 1.0,
    "file": "tests/test_migration.py",
    "line": 142
  },
  "edges_out": [
    {
      "type": "validates",
      "target": "DHM_migration",
      "confidence": 1.0,
      "source": "tests/test_migration.py:142 (decorator @validate_dhm)"
    },
    {
      "type": "depends_on",
      "target": "DataHub_visibility",
      "confidence": 0.95,
      "source": "tests/test_migration.py:145 (assert datahub.visible)"
    },
    {
      "type": "blocked_by",
      "target": "BUG_Elements_visibility",
      "confidence": 0.90,
      "source": "issue-123 (comment: 'test blocked by Elements visibility bug')"
    }
  ],
  "edges_in": [
    {
      "type": "mentioned_in",
      "source": "migration_spec.md",
      "confidence": 0.85,
      "source": "docs/migration_spec.md:45-67"
    }
  ],
  "sources": [
    "tests/test_migration.py:140-160",
    "issue-123",
    "docs/migration_spec.md:45-67"
  ],
  "created_at": "2026-06-15T10:00:00Z",
  "updated_at": "2026-07-01T14:00:00Z"
}
```

## Graph Construction Pipeline

### Step 1: Ingestion Sources

| Source | What we extract |
|--------|----------------|
| **Issue tracker** (GitHub Issues) | Bug, Decision, Requirement nodes; mentioned_in, blocked_by, caused_by edges |
| **Test results** (JSON/XML) | TestCase nodes with status, failure_rate; validates edges |
| **Experiment reports** (Markdown) | Experiment, Metric nodes; derived_from, validates edges |
| **Codebase** (Python/JS/etc.) | CodeFile, Function nodes; implements, depends_on, replaces edges |
| **Documentation** (Markdown) | Document, Concept nodes; mentioned_in, related_to edges |
| **Config files** (YAML/JSON) | Entity nodes with properties; depends_on edges |
| **Commit history** (Git) | Change metadata; replaces, caused_by edges |

### Step 2: Entity Extraction

**Rule-based** (for structured sources):
- Test files: parse test function names → TestCase nodes
- Code: AST parsing → Function, Class, Module nodes
- Config: YAML/JSON keys → Entity nodes
- Issue titles: regex patterns for component names, error codes

**LLM-based** (for unstructured sources):
- Prompt template:
  ```
  Extract all entities from the following text. For each entity,
  determine its type (Entity, Concept, Bug, Decision, Requirement)
  and list any relationships to other entities with their types.
  Output as JSON.
  ```
- Model: small, fast (Phi-3-mini, Llama-3.2-3B)
- Post-processing: deduplication, normalization

### Step 3: Relation Extraction

**Rule-based:**
- Code AST: `import X` → depends_on; `@test_decorator` → validates; function call → depends_on
- Test results: test name contains entity name → validates
- Issue cross-references: `#123` → mentioned_in; "blocks #456" → blocked_by
- Git: file rename → replaces; `Fixes #123` → caused_by (inverse)

**LLM-based:**
- Prompt template:
  ```
  For the following text, identify relationships between entities:
  - Entity A [relation_type] Entity B
  Valid relation types: depends_on, caused_by, validates, contradicts,
  implements, related_to, blocked_by, replaces
  Output confidence (0-1) for each relation. Output as JSON.
  ```

### Step 4: Deduplication & Normalization

- **Name normalization**: `DHM`, `dhm`, `Dhm`, `DataHub Migration` → canonical `DHM`
- **Fuzzy matching**: Levenshtein distance < 3 → same entity
- **Type disambiguation**: `Migration` could be Concept or TestCase — resolved by context
- **Merge strategy**: newer properties override older; edges are union (with confidence averaging)

### Step 5: Source Annotation

Every node and edge carries source pointers:
- File path + line range for code
- Document path + section for docs
- Issue/PR number for tracker
- Experiment ID + run timestamp for experiments
- Git commit SHA for version tracking

This ensures full traceability: "Where did this fact come from?"

## Graph Storage

### Prototype (Phase 1-2): In-Memory Python

```python
class GraphStore:
    nodes: dict[str, Node]       # node_id → Node
    edges_out: dict[str, list[Edge]]  # node_id → outgoing edges
    edges_in: dict[str, list[Edge]]   # node_id → incoming edges
    type_index: dict[str, list[str]]  # node_type → [node_ids]
    name_index: dict[str, str]        # normalized_name → node_id
```

### Production (Phase 4+): Persistent

Options:
- **KuzuDB**: Embedded graph DB, columnar, SQL-like queries, no server needed
- **Neo4j**: Full graph DB with Cypher, but requires server
- **SQLite + JSONB**: Simple, portable, good for <1M nodes

Recommendation: **KuzuDB** for Phase 3+ — embedded, fast, property graph model, no server.

## Graph Update Model

NEXUS supports incremental, non-destructive updates:

```
New document arrives
     ↓
Extract entities & relations
     ↓
For each entity:
  → If exists in graph: update properties, add new edges (don't remove old)
  → If new: create node with edges
     ↓
Update timestamps
     ↓
Graph version incremented
```

**Key property: updates are additive.** Old facts are not deleted — they are superseded by `replaces` edges. This preserves the history of knowledge evolution in the graph itself.

## Why This Beats Vector Stores

| Property | Vector Store (RAG) | Graph Store (NEXUS) |
|----------|-------------------|---------------------|
| Multi-hop retrieval | LLM must infer chains from text | Traverse explicit edges |
| Noise | Semantically-similar but irrelevant chunks | Only structurally-connected nodes |
| Update | Re-index entire collection | Add/remove individual nodes |
| Traceability | "Similar to query" | "Edge X with confidence Y, from source Z" |
| Reasoning support | LLM must connect disparate chunks | Graph provides explicit connection structure |
| Storage format | Dense vectors (fixed dimension) | Sparse graph (variable connectivity) |
| Query model | k-NN search | Traversal + filtering |
