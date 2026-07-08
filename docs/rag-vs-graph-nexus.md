# RAG vs Graph-SAM (NEXUS) — Detailed Comparison

Why NEXUS is not "just RAG with a different retriever."

---

## The Core Difference (in one sentence)

> **RAG searches for similar text. NEXUS traverses explicit relationships.**

## How Each System Answers: "Why does the DHM migration test fail after WO status change?"

### RAG Approach

```
Step 1: Embed question → query vector [0.23, -0.45, 0.67, ...]
Step 2: Cosine similarity against 10,000 document chunk embeddings
Step 3: Retrieve top-5 most similar chunks:
  Chunk 1 (score: 0.87): "The DHM migration process involves transferring data..."
  Chunk 2 (score: 0.82): "Migration tests are run automatically on each PR..."
  Chunk 3 (score: 0.79): "WO status changes trigger several downstream events..."
  Chunk 4 (score: 0.75): "Known issue: DataHub Elements visibility after status change"
  Chunk 5 (score: 0.71): "Test setup requires proper DataHub configuration..."
Step 4: LLM receives: "Answer the question using these chunks: [Chunk 1...Chunk 5]"
Step 5: LLM outputs: "The DHM migration test fails because..." 
  (hopefully connecting Chunk 3 + Chunk 4 correctly)
```

**Problems:**
- Chunk 1 is about DHM migration in general — background noise
- Chunk 2 is about test CI — irrelevant
- Chunk 3 + Chunk 4 contain the answer, but the LLM must:
  - Recognize they're connected
  - Infer the causal chain from raw text
  - Not get distracted by Chunks 1, 2, 5
- No way to verify the answer against sources

### NEXUS Approach

```
Step 1: Parse question → entities: [DHM, MigrationTest, WorkOrder, status_change]
Step 2: Locate entry nodes:
  MigrationDataTest [TestCase, status=failing]
  WO_status_change [Concept]
Step 3: Traverse from MigrationDataTest (backward, intent=causal_explanation):
  MigrationDataTest →[blocked_by, 0.90]→ BUG_Elements_visibility
  BUG_Elements_visibility →[caused_by, 0.85]→ WO_status_change
Step 4: Path score: 0.87 (edge_confidence × type_weight × coverage)
Step 5: Evidence pack:
  {
    "path": "MigrationDataTest --[blocked_by]--> BUG_Elements_visibility --[caused_by]--> WO_status_change",
    "facts": [
      "MigrationDataTest is blocked by BUG_Elements_visibility",
      "BUG_Elements_visibility is caused by WO status change"
    ],
    "sources": ["issue-123:45", "tests/test_migration.py:145"]
  }
Step 6: LLM receives structured evidence (NOT raw chunks)
Step 7: LLM outputs: "The DHM migration test fails because it is blocked by a 
  known bug (Elements visibility issue), which is caused by the WO status change."
Step 8: Verifier checks: "MigrationDataTest" ✓ in evidence, "blocked_by" ✓ in evidence,
  "BUG_Elements_visibility" ✓ in evidence, "caused_by WO_status_change" ✓ in evidence
```

**Advantages:**
- No irrelevant chunks to distract the LLM
- The causal chain is explicit — LLM doesn't need to infer it
- Every claim is verifiable against the evidence pack
- Context size: ~500 tokens vs ~2500 for RAG

## Systematic Comparison

### 1. Knowledge Representation

| Aspect | RAG | NEXUS |
|--------|-----|-------|
| Storage unit | Text chunks (~500 tokens each) | Typed nodes with properties + typed edges |
| Organization | Flat vector index | Directed graph with semantic edge types |
| Search mechanism | Cosine similarity to query embedding | Graph traversal with edge type filtering |
| Multi-hop structure | Implicit (must be inferred from chunks) | Explicit (edges ARE the hops) |
| Knowledge density | Low (full text, mostly filler words) | High (properties, relations, no prose) |
| Update granularity | Re-index entire document | Add/update individual nodes and edges |

### 2. Retrieval Quality

| Scenario | RAG | NEXUS |
|----------|-----|-------|
| Single-hop fact lookup | Good (similar text → correct chunk) | Good (entity lookup → properties) |
| Multi-hop reasoning | Poor (chunks may not contain the chain) | Good (traverse explicit edges) |
| Causal questions ("Why X?") | Requires LLM to infer causality from text | Traverse caused_by, blocked_by edges |
| Dependency questions ("What depends on X?") | Chunks may mention X but not all dependents | Traverse depends_on edges → complete set |
| Comparative questions ("X vs Y") | Retrieve chunks about X and Y separately | Traverse both subgraphs + diff |
| Questions with negations ("Why doesn't X work?") | Hard — "doesn't work" is vague for similarity | Traverse blocked_by, contradicts edges |
| Ambiguous entities ("Migration" — concept or test?) | All chunks mentioning "migration" mixed together | Type-disambiguated nodes; clearer results |

### 3. Noise Characteristics

| Noise type | RAG | NEXUS |
|-----------|-----|-------|
| Semantically similar but irrelevant | HIGH — cosine similarity finds related topics | LOW — only structurally connected nodes |
| Outdated information | Medium — old chunks persist unless re-indexed | Low — `replaces` edges mark superseded facts |
| Missing connections | N/A (LLM must infer) | Medium — missing edges = broken traversal |
| Wrong connections | N/A (LLM may hallucinate connections) | Low-Medium — incorrect edges from extraction errors |
| Context pollution | HIGH — many chunks, LLM must filter | LOW — evidence pack is pre-filtered |

### 4. LLM Role

| Aspect | RAG | NEXUS |
|--------|-----|-------|
| What the LLM does | Read chunks → understand → connect → answer | Read structured evidence → articulate → answer |
| Reasoning burden | HIGH — must infer connections from raw text | LOW — connections are explicit in evidence |
| Context window usage | HIGH — 2K-10K tokens of chunks | LOW — 0.5K-2K tokens of structured facts |
| Hallucination risk | HIGH — noisy context + inference burden | LOW — clean evidence + verifier |
| Model size needed | Larger is better (more reasoning capacity) | Small is sufficient (verbalization only) |

### 5. Operational Characteristics

| Metric | RAG | NEXUS |
|--------|-----|-------|
| Indexing cost | O(N) — embed all chunks | O(E + R) — extract entities and relations |
| Indexing quality dependency | Embedding model quality | Entity/relation extraction quality |
| Retrieval latency | ~10-50ms (vector search) | ~10-100ms (graph traversal, depth-dependent) |
| Storage per fact | ~1-3KB (chunk with surrounding text) | ~100-500 bytes (node + edges) |
| Update cost | Re-index changed documents | Add/modify nodes and edges |
| Cold start | Need documents to embed | Need entities and relations to extract |
| Domain transfer | Retrain/switch embedding model | Rewrite extraction rules/prompts |

### 6. Failure Modes

| Failure | RAG | NEXUS |
|---------|-----|-------|
| Missed relevant info | Chunks not retrieved (low similarity score) | Edges missing (extraction failed) |
| Retrieved irrelevant info | Semantically similar but not useful chunks | Wrong edge type or spurious edge |
| LLM hallucination | LLM invents facts not in chunks | LLM invents facts not in evidence (caught by verifier) |
| Stale knowledge | Old chunks not re-indexed | Old edges not updated (mitigated by `replaces`) |
| Ambiguity | All meanings of entity retrieved together | Entity disambiguation may fail |

## When RAG Is Better

RAG is preferable when:
1. **Knowledge is predominantly textual/narrative** — stories, tutorials, explanations where the text IS the knowledge
2. **Graph construction cost is prohibitive** — if entity/relation extraction quality is poor, the graph is unreliable
3. **Queries are exploratory** — "Tell me about X" rather than specific factual questions
4. **Domain has few structured relationships** — loosely connected facts without clear causal/dependency structure
5. **Time to first answer is critical** — RAG can work immediately with any document collection

## When NEXUS Is Better

NEXUS is preferable when:
1. **Domain has rich, explicit structure** — codebases, systems, APIs, processes with clear dependencies
2. **Multi-hop reasoning is common** — "Why does X affect Y through Z?"
3. **Traceability is required** — need to prove where each answer came from
4. **Hallucination is unacceptable** — regulated domains, engineering decisions
5. **Knowledge evolves incrementally** — continuous updates, not batch re-indexing
6. **CPU/RAM budget is constrained** — graph operations don't need GPU

## The Hybrid Approach

For production systems, a hybrid RAG + NEXUS approach may be optimal:

```
Question
    ↓
┌───────────────────────────────────────┐
│  NEXUS: Graph traversal               │
│  → Structured evidence (entities,      │
│    relations, sources)                 │
└───────────────┬───────────────────────┘
                ↓
         Evidence pack
                ↓
┌───────────────────────────────────────┐
│  RAG: Vector search                   │
│  → Contextual detail (documentation    │
│    excerpts, code comments, examples)  │
└───────────────┬───────────────────────┘
                ↓
         Combined context
                ↓
┌───────────────────────────────────────┐
│  LLM: Reasoning + articulation        │
│  → Graph evidence for structure       │
│  → RAG chunks for detail              │
└───────────────────────────────────────┘
```

Graph provides the **skeleton** (what relates to what, why). RAG provides the **flesh** (detailed explanations, examples, nuances).

## Experimental Validation Plan

To validate that NEXUS beats RAG for domain-specific QA, we will:

1. **Create a QA dataset** of 200-500 questions about the SAM/NEXUS project itself
2. **Run all questions through:**
   - NEXUS (graph traversal + evidence + small LLM)
   - Classic RAG (chunk embeddings + top-K + same LLM)
   - Hybrid (graph evidence + RAG chunks + LLM)
   - LLM-only (closed-book, same LLM, no retrieval)
3. **Measure:** accuracy, hallucination rate, context size, latency, source traceability
4. **Analyze:** when does each approach fail? What types of questions favor which approach?

This experiment is the critical validation of the NEXUS thesis.
