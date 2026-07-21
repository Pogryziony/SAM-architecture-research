# Graph Reasoning — Traversal as the Reasoning Engine

How NEXUS uses graph traversal as the primary reasoning mechanism, before any LLM is involved.

---

## Core Principle

> **Reasoning does not start from text generation. It starts from graph traversal.**

In NEXUS, the "reasoning" step is not an LLM call — it's a structured walk through the knowledge graph. The LLM's role is to articulate what the traversal discovered, not to figure out what to discover.

## The Reasoning Pipeline

```
1. PARSE          Question → entities, intent, constraints
2. LOCATE         Find entry nodes in the graph
3. TRAVERSE       Walk edges from entry nodes (with direction, type, depth constraints)
4. SCORE          Rank paths by relevance, confidence, recency
5. SELECT         Choose top-K paths
6. BUILD          Construct structured evidence pack from selected paths
7. REASON         Small LLM converts evidence → natural language answer
8. VERIFY         Check answer against evidence; flag unsupported claims
```

Steps 1-6 are non-LLM, CPU-native operations. Only steps 7-8 involve a language model.

### Traversal budgets

`NEXUSConfig.max_expanded_edges` (default 10_000),
`max_expanded_nodes` (default 5_000), and optional `max_traversal_ms`
(default `0` = disabled) bound beam search expansion. When a budget is
exhausted, `TraversalStats.truncated` is set with a reason
(`max_expanded_edges` / `max_expanded_nodes` / `max_traversal_ms`). Truncation
means the search is incomplete — the reasoning audit must not recommend an
unconditional answer. See `nexus/graph/traversal.py`,
`tests/test_traversal_budgets.py`, and the synthetic campaign
`benchmarks/run_traversal_budget_campaign.py`.

## Step 1: Question Parsing

Input: *"Why does the DHM migration test not pass after changing WO status?"*

Output:
```json
{
  "entities": ["DHM", "MigrationTest", "WorkOrder", "status_change"],
  "intent": "causal_explanation",
  "constraints": {
    "direction": "backward",      // "why" → trace causes
    "max_depth": 4,
    "entity_types": ["TestCase", "Bug", "Entity", "Concept"]
  }
}
```

### Intent Classification

| Intent | What it means | Traversal direction |
|--------|--------------|-------------------|
| `causal_explanation` | "Why does X happen?" | Backward from X (caused_by, blocked_by) |
| `impact_analysis` | "What does X affect?" | Forward from X (validates, depends_on) |
| `factual_lookup` | "What is X?" | 1-hop from X (all edge types) |
| `comparison` | "X vs Y" | Separate traversals + diff |
| `diagnostic` | "Why is X failing?" | Backward until Bug or Decision found |
| `dependency_chain` | "What does X depend on?" | Forward/backward along depends_on |

## Step 2: Entity Location

Map extracted entity names to graph node IDs.

**Strategy:**
1. **Exact match** on normalized name
2. **Fuzzy match** (Levenshtein < 3) with type preference
3. **Acronym expansion**: `DHM` → `DataHubMigration`
4. **Contextual disambiguation**: if `Migration` could be `Concept:Migration` or `TestCase:MigrationDataTest`, prefer the one with edges matching the query intent

## Step 3: Graph Traversal

### Algorithm: Beam Search with Edge-Type Filtering

```python
def traverse(
    graph: GraphStore,
    entry_nodes: list[str],
    intent: Intent,
    max_depth: int = 4,
    beam_width: int = 5,
    edge_type_weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> list[Path]:
    """
    Beam search traversal from entry nodes.
    
    At each hop:
    1. Expand all current paths by one edge
    2. Score expanded paths
    3. Keep top beam_width paths
    
    Returns ranked list of paths.
    """
```

### Edge Type Weights (Default)

These weights determine which edge types are preferred during traversal:

| Edge Type | Weight | Rationale |
|-----------|--------|-----------|
| `caused_by` | 1.0 | Causal chains are the strongest reasoning signal |
| `blocked_by` | 0.95 | Blockers directly explain failures |
| `depends_on` | 0.85 | Dependencies reveal structural relationships |
| `validates` | 0.80 | Validation relationships confirm/test behavior |
| `contradicts` | 0.75 | Contradictions are strong negative signals |
| `implements` | 0.70 | Implementation tells us "how" |
| `derived_from` | 0.60 | Conceptual derivation (e.g. decision → supporting concept) |
| `replaces` | 0.55 | Historical context |
| `sub_experiment` | 0.50 | Structural parentage (run → experiment, metric → run); not scored in relation F1 |
| `related_to` | 0.30 | Weak association — used only if no stronger edges |
| `mentioned_in` | 0.20 | Co-occurrence — weakest signal, mainly for source tracing |

### Direction Strategy

| Query type | Direction | Rationale |
|-----------|-----------|-----------|
| "Why X?" | Backward (incoming edges) | Trace causes upstream |
| "What does X affect?" | Forward (outgoing edges) | Trace consequences downstream |
| "How is X related to Y?" | Bidirectional BFS | Find shortest connecting path |
| "What depends on X?" | Reverse depends_on | Dependency analysis |

### Depth Limiting

| Depth | What it captures | Example |
|-------|-----------------|---------|
| 1 | Direct relationships | DHM → MigrationTest |
| 2 | One-hop chains | DHM → MigrationTest → Bug |
| 3 | Two-hop chains | DHM → MigrationTest → Bug → StatusChange |
| 4 | Deep causal chains | DHM → MigrationTest → Bug → StatusChange → WorkOrder |

Default max_depth = 4. Can be increased for specific diagnostic queries.

Expansion is also bounded by `max_expanded_edges`, `max_expanded_nodes`, and
optional `max_traversal_ms`. Exhaustion sets `TraversalStats.truncated` and the
reasoning audit must not recommend an unconditional answer.

## Step 4: Path Scoring

Each path gets a composite score:

```python
def score_path(path: Path, query_entities: set[str]) -> float:
    """
    Composite path score combining:
    - Edge confidence (product of all edge confidences)
    - Edge type relevance (product of edge type weights)
    - Entity coverage (fraction of query entities covered)
    - Path length penalty (shorter paths preferred)
    - Source recency (newer sources preferred)
    """
    edge_score = prod(edge.confidence for edge in path.edges)
    type_score = prod(EDGE_TYPE_WEIGHTS[edge.type] for edge in path.edges)
    coverage = len(path.nodes ∩ query_entities) / len(query_entities)
    length_penalty = 1.0 / (1.0 + 0.1 * len(path.edges))  # mild decay
    
    # Recency bonus: prefer paths with recently-updated sources
    max_age_days = max((now - node.updated_at).days for node in path.nodes)
    recency = max(0.5, 1.0 - max_age_days / 365)
    
    return edge_score * type_score * coverage * length_penalty * recency
```

## Step 5: Path Selection

From all scored paths:
1. Remove duplicate paths (same node sequence, different order)
2. Remove subsumed paths (if path A is a prefix of path B, keep the more informative one)
3. Select top-K (default K=3)
4. Ensure diversity: if multiple paths share the same root cause, keep the highest-scored one

## Step 6: Evidence Building

Transform selected paths into structured evidence:

```json
{
  "question": "Why does the DHM migration test not pass after changing WO status?",
  "paths": [
    {
      "score": 0.87,
      "nodes": [
        {"id": "MigrationDataTest", "type": "TestCase", "status": "failing"},
        {"id": "BUG_Elements_visibility", "type": "Bug", "severity": "high"},
        {"id": "WO_status_change", "type": "Concept"}
      ],
      "edges": [
        {"type": "blocked_by", "from": "MigrationDataTest", "to": "BUG_Elements_visibility", "confidence": 0.9},
        {"type": "caused_by", "from": "BUG_Elements_visibility", "to": "WO_status_change", "confidence": 0.85}
      ]
    }
  ],
  "facts": [
    "MigrationDataTest is blocked by BUG_Elements_visibility (confidence: 0.9)",
    "BUG_Elements_visibility is caused by WO status change (confidence: 0.85)"
  ],
  "sources": [
    {"path": "issue-123", "excerpt": "Test blocked by Elements visibility bug after WO status change"},
    {"path": "tests/test_migration.py:145", "excerpt": "assert datahub.elements_visible()"},
    {"path": "docs/migration_spec.md:45-67", "excerpt": "Migration depends on DataHub element visibility"}
  ]
}
```

### Evidence Pack Design Principles

1. **Structured, not raw**: JSON, not markdown/text
2. **Small**: ~1-2KB, not 10-20KB of raw chunks
3. **Source-annotated**: every fact has a verifiable source
4. **Confidence-weighted**: the model knows which facts are more reliable
5. **Self-contained**: the evidence pack has everything needed to answer

## Step 7: Reasoning Model

The evidence pack is converted to a prompt and fed to a small LLM:

```
SYSTEM: You are a precise reasoning assistant. You receive structured evidence
from a knowledge graph. Answer ONLY based on the provided evidence.
If the evidence is insufficient, say "Insufficient evidence to answer."
Do not invent facts. Cite sources when possible.

QUESTION: Why does the DHM migration test not pass after changing WO status?

EVIDENCE:
Path 1 (score: 0.87):
  MigrationDataTest [TestCase, status=failing]
    → blocked_by → BUG_Elements_visibility [Bug, severity=high] (confidence: 0.90)
    → caused_by → WO_status_change [Concept] (confidence: 0.85)

Facts:
  - MigrationDataTest is blocked by BUG_Elements_visibility
  - BUG_Elements_visibility is caused by WO status change

Sources:
  - issue-123: "Test blocked by Elements visibility bug after WO status change"
  - test_migration.py:145: assert datahub.elements_visible()
  - migration_spec.md:45-67: "Migration depends on DataHub element visibility"

ANSWER:
```

The reasoning model:
- Receives ~500 tokens of structured evidence (vs 2000-5000 tokens of raw chunks in RAG)
- Can trace the exact chain: Test → blocked_by → Bug → caused_by → StatusChange
- Has sources for every claim
- Is explicitly instructed not to hallucinate

## Step 8: Verification

Rule-based verifier (not another LLM):

```python
def verify(answer: str, evidence: EvidencePack) -> VerificationResult:
    """
    Check that all factual claims in the answer are supported by evidence.
    """
    claims = extract_claims(answer)  # Simple NLP: split on sentences, find entity mentions
    
    unsupported = []
    for claim in claims:
        entities_in_claim = find_entities(claim, evidence.all_entity_names())
        relations_in_claim = find_relations(claim, evidence.all_relation_types())
        
        if not evidence.supports(entities_in_claim, relations_in_claim):
            unsupported.append(claim)
    
    return VerificationResult(
        supported=len(claims) - len(unsupported),
        unsupported=unsupported,
        hallucination_rate=len(unsupported) / len(claims) if claims else 0
    )
```

If hallucination_rate > threshold (0.2), flag answer and return "Insufficient evidence" instead.

## Why This Beats RAG Reasoning

| Reasoning Aspect | RAG | NEXUS |
|-----------------|-----|-------|
| **How chains are found** | LLM infers connections from raw text chunks | Traversal explicitly walks edges |
| **Context quality** | Noisy (similar ≠ relevant) | Clean (structurally connected only) |
| **Context size** | 2K-5K tokens (multiple chunks) | 0.5K-1K tokens (structured paths) |
| **Multi-hop** | LLM must track entities across chunks | Path IS the chain |
| **Explainability** | "The model generated this" | "Path: A→B→C, sources: X, Y, Z" |
| **Hallucination surface** | Large (LLM generates from noisy context) | Small (LLM only verbalizes structured facts) |
| **Verifiability** | Manual (read all chunks, check answer) | Automatic (verifier checks claims against evidence) |

## Key Design Decisions

1. **Traversal before LLM**: The expensive reasoning (finding connections) happens in the graph, not in the LLM
2. **Structured evidence**: The LLM receives facts and relations, not raw text — less room for interpretation errors
3. **Verifier is rule-based**: We don't use an LLM to check an LLM — the verifier is deterministic
4. **Small LLM is sufficient**: Since the LLM only verbalizes, not reasons from scratch, a <1B param model can work
5. **CPU-first**: Graph traversal is a CPU operation — no GPU needed for steps 1-6
