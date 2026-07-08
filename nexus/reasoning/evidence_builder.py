"""
NEXUS evidence builder — converts graph traversal paths into structured,
compact JSON evidence packs suitable for LLM consumption.

Each evidence pack contains:
  - The original question
  - Structured path data (nodes, edges with direction)
  - Human-readable fact strings with confidence scores
  - Unique source references
"""

from __future__ import annotations

import json
from typing import Any

from nexus.graph import Path, Node
from nexus.graph.store import InMemoryGraphStore


def _node_summary(node: Node) -> dict[str, Any]:
    """Build a compact dict representation of a node."""
    props = dict(node.properties)
    # Include key fields, trim long strings
    summary: dict[str, Any] = {"id": node.id, "type": node.type}
    # Add the most informative property
    for key in ("description", "key_finding", "title", "question", "name"):
        if key in props and props[key]:
            val = props[key]
            if isinstance(val, str) and len(val) > 200:
                val = val[:197] + "..."
            summary[key] = val
    return summary


def _edge_summary(step) -> dict[str, Any]:
    """Build a compact dict for an edge with direction info."""
    return {
        "type": step.edge.type,
        "from": step.from_node,
        "to": step.to_node,
        "confidence": round(step.edge.confidence, 2),
        "reversed": step.reversed,
    }


def _fact_from_step(step, graph: InMemoryGraphStore) -> str:
    """Build a human-readable fact string from a single path step."""
    from_node = graph.get_node(step.from_node)
    to_node = graph.get_node(step.to_node)

    from_name = from_node.id if from_node else step.from_node
    to_name = to_node.id if to_node else step.to_node

    # Choose a readable property if available
    for key in ("name", "display_name", "title"):
        if from_node and key in from_node.properties:
            from_name = from_node.properties[key]
            break
    for key in ("name", "display_name", "title"):
        if to_node and key in to_node.properties:
            to_name = to_node.properties[key]
            break

    relation_map = {
        "depends_on": ("depends on", "is a dependency of"),
        "caused_by": ("is caused by", "causes"),
        "blocked_by": ("is blocked by", "blocks"),
        "validates": ("validates", "is validated by"),
        "contradicts": ("contradicts", "is contradicted by"),
        "implements": ("implements", "is implemented by"),
        "derived_from": ("is derived from", "supports"),
        "replaces": ("replaces", "is replaced by"),
        "related_to": ("is related to", "is related to"),
        "mentioned_in": ("is mentioned in", "mentions"),
    }

    fwd, rev = relation_map.get(step.edge.type, (step.edge.type.replace("_", " "), step.edge.type.replace("_", " ")))
    rel_text = rev if step.reversed else fwd
    confidence = round(step.edge.confidence, 2)
    return f"{from_name} {rel_text} {to_name} (confidence: {confidence:.2f})"


# ── Type-aware fact priority ──
# Priority scores for node types given a question intent.
# Higher score = more relevant to the question, surfaced first.
# Maps (node_type, intent_group) -> priority.
_NODE_FACT_PRIORITY: dict[tuple[str, str], int] = {
    # causal_explanation / diagnostic → explain WHY
    ("Concept", "causal_explanation"): 20,
    ("Bug", "causal_explanation"): 15,
    ("Decision", "causal_explanation"): 12,
    ("Concept", "diagnostic"): 20,
    ("Bug", "diagnostic"): 18,
    ("Decision", "diagnostic"): 12,
    # factual_lookup → key findings and concepts are most relevant
    ("Experiment", "factual_lookup"): 20,
    ("Concept", "factual_lookup"): 18,
    ("Metric", "factual_lookup"): 14,
    # comparison → all typed nodes equal
    ("Experiment", "comparison"): 15,
    ("Concept", "comparison"): 15,
    ("Bug", "comparison"): 15,
    ("Decision", "comparison"): 15,
    ("Metric", "comparison"): 15,
}

# Extra priority boost for Concept nodes directly linked via validates/contradicts edges.
_VALIDATES_CONCEPT_BOOST = 10


def _get_node_fact_priority(node_type: str, question_intent: str) -> int:
    """Return a priority score for a node type given the question intent.

    Higher = more relevant, surfaced first in KEY FINDINGS.
    Falls back to 5 for unlisted type/intent combinations.
    """
    # Map intent aliases
    intent = question_intent
    if intent in ("causal_explanation", "dependency_chain", "impact_analysis"):
        intent = "causal_explanation"
    elif intent == "comparison":
        intent = "comparison"
    elif intent == "factual_lookup":
        intent = "factual_lookup"
    elif intent == "diagnostic":
        intent = "diagnostic"

    return _NODE_FACT_PRIORITY.get((node_type, intent), 5)


def _extract_node_facts(
    paths: list[Path],
    graph: InMemoryGraphStore,
    question_intent: str = "factual_lookup",
    target_entity: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract curated key_finding/description properties from unique nodes
    across all traversal paths.

    These are HIGH-CONFIDENCE facts because they were manually curated.

    Type-aware: prioritizes nodes based on their role in the question intent.
    Concept nodes linked via validates/contradicts edges get extra priority
    and a special annotation.

    Additionally, for nodes that appear in the path steps, we proactively
    look up their outgoing validates/contradicts edges and include the
    target Concept node facts — even if the validates-edge path was pruned
    by beam search. This ensures questions like "What concept does X validate?"
    always surface the relevant Concept descriptions.

    Returns a list of {text, confidence, source} dicts.
    """
    seen_nodes: set[str] = set()
    # Track which Concept nodes are directly linked by validates/contradicts edges
    validates_concepts: set[str] = set()

    # ── Pass 1: discover validates/contradicts → Concept across ALL path edges ──
    for path in paths:
        for step in path.steps:
            edge_type = step.edge.type
            # Detect validates/contradicts edges → mark target Concept nodes
            if edge_type in ("validates", "contradicts"):
                target_id = step.to_node
                target_node = graph.get_node(target_id)
                if target_node and target_node.type == "Concept":
                    validates_concepts.add(target_id)

    # ── Pass 2: proactively discover validates/contradicts edges from
    #            entry-point nodes (first nodes of each path) to Concept
    #            nodes. This handles beam-search pruning: even when the
    #            validates-edge path doesn't make the top-N cut, we still
    #            surface the Concept that the experiment was designed to
    #            validate. ──
    entry_node_ids: set[str] = set()
    for path in paths:
        if path.steps:
            entry_node_ids.add(path.steps[0].from_node)

    for node_id in entry_node_ids:
        # Look at outgoing validates/contradicts edges from this entry node
        for edge in graph.get_edges(node_id, "out"):
            if edge.type in ("validates", "contradicts") and edge.source == node_id:
                target_node = graph.get_node(edge.target)
                if target_node and target_node.type == "Concept":
                    validates_concepts.add(edge.target)

    # ── Pass 3: build priority-ranked fact list ──
    raw_facts: list[dict[str, Any]] = []

    for path in paths:
        for step in path.steps:
            for node_id in (step.from_node, step.to_node):
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                node = graph.get_node(node_id)
                if node is None:
                    continue
                props = node.properties
                # Prefer key_finding (Experiment nodes), then description (Concept nodes)
                value = props.get("key_finding") or props.get("description")
                if value and isinstance(value, str) and value.strip():
                    text = f"{node_id}: {value}"
                    # Determine source
                    source = props.get("title") or props.get("name")
                    # Calculate priority
                    priority = _get_node_fact_priority(node.type, question_intent)
                    # Boost validates-linked Concept nodes
                    if node_id in validates_concepts and node.type == "Concept":
                        priority += _VALIDATES_CONCEPT_BOOST
                        text = f"[This concept is directly validated by the experiment] {text}"
                    raw_facts.append({
                        "text": text,
                        "confidence": 1.0,
                        "source": source or node_id,
                        "confidence_label": "HIGH (manually curated)",
                        "_priority": priority,
                    })

    # ── Pass 4: include validates-linked Concept descriptions even if the
    #            Concept didn't appear in any path step (proactive discovery) ──
    #            BUT only if no target_entity filter or the concept matches it.
    for concept_id in sorted(validates_concepts):
        if concept_id in seen_nodes:
            continue  # Already included
        # If target_entity is set, skip concepts not matching it
        if target_entity and target_entity.lower() not in concept_id.lower():
            continue
        node = graph.get_node(concept_id)
        if node is None:
            continue
        props = node.properties
        value = props.get("description")
        if value and isinstance(value, str) and value.strip():
            priority = _get_node_fact_priority("Concept", question_intent) + _VALIDATES_CONCEPT_BOOST
            text = f"[This concept is directly validated by the experiment] {concept_id}: {value}"
            source = props.get("title") or props.get("name")
            raw_facts.append({
                "text": text,
                "confidence": 1.0,
                "source": source or concept_id,
                "confidence_label": "HIGH (manually curated)",
                "_priority": priority,
            })

    # If target_entity is specified, filter to only facts mentioning it.
    # If filtering removes everything, fall back to all facts — the prompt
    # template will handle second-level filtering.
    if target_entity:
        target_lower = target_entity.lower()
        filtered = [
            f for f in raw_facts
            if target_lower in f.get("text", "").lower()
        ]
        if filtered:
            raw_facts = filtered
        # else: keep all facts as fallback

    # Sort by priority descending, then by source for stability
    raw_facts.sort(key=lambda f: (-f["_priority"], f["source"]))

    # Strip internal _priority key before returning
    for f in raw_facts:
        del f["_priority"]

    return raw_facts


def build_evidence(
    question: str,
    paths: list[Path],
    graph: InMemoryGraphStore,
    max_paths: int = 5,
    max_facts_per_path: int = 10,
    question_intent: str = "factual_lookup",
    target_entity: str | None = None,
) -> str:
    """
    Build a structured JSON evidence pack from traversal paths.

    Args:
        question: The original natural language question
        paths: Ranked traversal paths (best first)
        graph: The graph store for node lookups
        max_paths: Maximum number of paths to include
        max_facts_per_path: Max facts per path
        question_intent: Detected intent (causal_explanation, factual_lookup, etc.)
        target_entity: If provided, filters node_facts to only those
                       mentioning this entity (used for factual questions)

    Returns:
        JSON string with evidence pack
    """
    evidence: dict[str, Any] = {
        "question": question,
        "paths": [],
        "node_facts": [],
        "facts": [],
        "sources": [],
    }

    all_sources: set[str] = set()
    all_facts: list[str] = []

    for path in paths[:max_paths]:
        if not path.steps:
            continue

        path_data: dict[str, Any] = {
            "score": round(path.score, 3),
            "length": path.length,
            "nodes": [],
            "edges": [],
        }

        # Collect unique nodes along the path
        seen_nodes: set[str] = set()
        for step in path.steps:
            for node_id in (step.from_node, step.to_node):
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    node = graph.get_node(node_id)
                    if node:
                        path_data["nodes"].append(_node_summary(node))
                        for src in node.sources:
                            all_sources.add(src)

        # Add edges
        for step in path.steps[:max_facts_per_path]:
            path_data["edges"].append(_edge_summary(step))
            # Build fact string
            fact = _fact_from_step(step, graph)
            all_facts.append(fact)
            # Add edge evidence as source
            if step.edge.evidence:
                all_sources.add(step.edge.evidence)

        evidence["paths"].append(path_data)

    # Extract curated node facts (key_finding/description) — placed BEFORE
    # edge-based facts because they are more reliable (manually curated).
    evidence["node_facts"] = _extract_node_facts(
        paths, graph, question_intent, target_entity
    )

    evidence["facts"] = all_facts
    evidence["sources"] = sorted(all_sources)

    return json.dumps(evidence, indent=2, ensure_ascii=False)


def build_evidence_pack(
    question: str,
    paths: list[Path],
    graph: InMemoryGraphStore,
    question_intent: str = "factual_lookup",
    target_entity: str | None = None,
) -> dict[str, Any]:
    """
    Build and return the evidence pack as a Python dict (no JSON serialization).

    Useful for programmatic access or further processing.
    """
    raw = build_evidence(
        question, paths, graph,
        question_intent=question_intent, target_entity=target_entity,
    )
    return json.loads(raw)
