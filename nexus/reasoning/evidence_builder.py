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
import re
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
    question: str = "",
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

    Relevance scoring: each fact is scored by word overlap with the question.
    Facts mentioning the target_entity get a bonus. This replaces the old
    binary target_entity filter which was too aggressive.

    Returns a list of {text, confidence, source} dicts, sorted by relevance.
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

    # ── Relevance scoring: word overlap with question + target_entity bonus ──
    # This replaces the old binary target_entity filter which was too aggressive.
    question_words: set[str] = set()
    if question:
        question_words = set(re.findall(r'\w+', question.lower()))

    for f in raw_facts:
        fact_text = f.get("text", "").lower()
        # Score by word overlap with question
        if question_words:
            fact_words = set(re.findall(r'\w+', fact_text))
            overlap = len(question_words & fact_words)
        else:
            overlap = 0
        # Target entity match gets a strong bonus
        entity_bonus = 0
        if target_entity and target_entity.lower() in fact_text:
            entity_bonus = 20
        f["_relevance"] = overlap + entity_bonus

    # Sort by relevance descending, then priority descending, then source
    raw_facts.sort(key=lambda f: (-f["_relevance"], -f["_priority"], f["source"]))

    # Strip internal keys before returning
    for f in raw_facts:
        del f["_priority"]
        del f["_relevance"]

    return raw_facts


# ── Confidence signals for routing ──

_NUMERIC_INTENT_PATTERNS: dict[str, list[str]] = {
    "accuracy": [r"%", r"\d+(?:\.\d+)?\s*%"],
    "parameter": [r"\bmillion\b", r"\d{5,}"],
    "slot": [r"\d+\s*slot"],
    "layer": [r"\d+\s*layer", r"\d+"],
    "dimension": [r"\d+"],
    "size": [r"\d+"],
    "token": [r"\d+\s*token", r"\d+[kKmM]"],
    "batch": [r"\d+\s*batch", r"\d+"],
    "head": [r"\d+\s*head", r"\d+"],
    "rate": [r"%", r"\d+(?:\.\d+)?\s*%"],
}


def _compute_numeric_match(question: str, node_facts: list[dict[str, Any]], facts: list[str]) -> float:
    """Score 0-1: how well evidence numbers match the question's numeric intent.

    Parses the question for numeric intent signals (accuracy→%, parameters→million,
    slots→slot-numbers) and checks whether the evidence text contains numbers
    in the matching category.

    Returns:
        1.0 = evidence has matching numbers for all detected intents
        0.5 = evidence has some numbers or partial match
        0.0 = no numbers or no matching numbers
    """
    q_lower = question.lower()

    # Combine all evidence text into one searchable string
    evidence_text = " ".join(
        [nf.get("text", "") for nf in node_facts] + [str(f) for f in facts]
    )

    # Check if evidence has any numbers at all
    has_any_numbers = bool(re.search(r"\d", evidence_text))
    if not has_any_numbers:
        return 0.0

    # Detect which specific numeric intents are present in the question
    detected_intents: list[tuple[str, list[str]]] = []
    for keyword, patterns in _NUMERIC_INTENT_PATTERNS.items():
        if keyword in q_lower:
            detected_intents.append((keyword, patterns))

    if not detected_intents:
        # No specific numeric intent detected, but evidence has numbers → partial
        return 0.5

    # For each detected intent, check if evidence has matching numbers
    match_count = 0
    for _keyword, patterns in detected_intents:
        for pat in patterns:
            if re.search(pat, evidence_text, re.IGNORECASE):
                match_count += 1
                break

    if match_count == len(detected_intents):
        return 1.0
    elif match_count > 0:
        return 0.5
    else:
        return 0.0


def _compute_has_key_finding(
    target_entity: str | None,
    graph: InMemoryGraphStore,
) -> float:
    """Check whether the target entity has a key_finding property.

    Returns 1.0 if the entity has key_finding, 0.0 otherwise.
    Falls back to 0.5 if no target_entity is specified (unknown).
    """
    if target_entity is None:
        return 0.5  # Unknown — neutral signal
    node = graph.get_node(target_entity)
    if node is None:
        return 0.0
    props = node.properties
    if props.get("key_finding") and isinstance(props["key_finding"], str) and props["key_finding"].strip():
        return 1.0
    return 0.0


def _compute_confidence_signals(
    question: str,
    paths: list[Path],
    graph: InMemoryGraphStore,
    evidence: dict[str, Any],
    target_entity: str | None,
    question_intent: str,
) -> dict[str, float]:
    """Compute confidence signals for routing decisions.

    These signals are available BEFORE generation and indicate how
    likely the evidence is to contain a correct answer.
    """
    node_facts = evidence.get("node_facts", [])
    all_facts = evidence.get("facts", [])

    # 1. Numeric match score — does evidence contain numbers relevant to the question?
    numeric_match = _compute_numeric_match(question, node_facts, all_facts)

    # 2. Key finding presence — does the target entity have curated key_finding?
    #    If target_entity is not provided, try to find it from the evidence paths.
    resolved_target = target_entity
    if resolved_target is None:
        resolved_target = _resolve_target_entity_from_evidence(question, evidence)

    has_key_finding = _compute_has_key_finding(resolved_target, graph)

    # 3. Path count signal — more paths = more evidence context
    path_count_signal = min(len(paths) / 5.0, 1.0)

    return {
        "numeric_match": round(numeric_match, 2),
        "has_key_finding": round(has_key_finding, 2),
        "path_count_signal": round(path_count_signal, 2),
    }


def _resolve_target_entity_from_evidence(
    question: str,
    evidence: dict[str, Any],
) -> str | None:
    """Find the target entity from evidence paths without graph access.

    Uses word overlap between question words and node IDs in the evidence paths.
    Splits node IDs on both word boundaries and underscores for compound IDs
    like Exp_0_6_Validation_oracle_memory.
    """
    q_lower = question.lower()
    q_words = set(re.findall(r"\w+", q_lower))

    best_match: str | None = None
    best_score = 0

    all_node_ids: set[str] = set()
    for path_data in evidence.get("paths", []):
        for node in path_data.get("nodes", []):
            nid = node.get("id", "")
            if nid:
                all_node_ids.add(nid)

    for nid in all_node_ids:
        nid_lower = nid.lower()
        # Direct match: node ID appears as substring in question
        if nid_lower in q_lower:
            score = len(nid_lower)
            if score > best_score:
                best_score = score
                best_match = nid
                continue

        # Word-level overlap: split on underscores AND word boundaries
        nid_words = set(re.findall(r"[a-zA-Z0-9]+", nid_lower))
        meaningful = {w for w in nid_words if len(w) > 2}
        overlap = len(meaningful & q_words)
        if overlap > 0 and overlap > best_score:
            best_score = overlap
            best_match = nid

    return best_match


def _collect_numbers_by_metric(
    paths: list[Path],
    graph: InMemoryGraphStore,
) -> dict[str, list[dict[str, Any]]]:
    """Collect all numbers grouped by metric name for easy lookup.

    Iterates over all unique nodes in the traversal paths. For each node:
    - If it has a 'metrics' dict property, group each metric→value pair
    - If it's a Metric-type node with 'name' and 'value' properties, include those too
    - Also proactively follows derived_from edges to collect Metric nodes that
      are connected to traversed experiment/run nodes but may not appear in
      the top traversal paths.

    Returns a dict like:
      {"accuracy": [{"entity": "Exp_0_6_Validation", "value": "99.87%"}],
       "recall": [{"entity": "Exp_0_12_Selection", "value": "96.6%"}]}
    """
    by_metric: dict[str, list[dict[str, Any]]] = {}
    seen_entities: set[str] = set()
    # Track experiment/run node IDs to proactively collect their metric nodes
    exp_node_ids: set[str] = set()

    for path in paths:
        for step in path.steps:
            for node_id in (step.from_node, step.to_node):
                if node_id in seen_entities:
                    continue
                seen_entities.add(node_id)
                node = graph.get_node(node_id)
                if node is None:
                    continue

                # Track experiment/run nodes for proactive metric collection
                if node.type == "Experiment":
                    exp_node_ids.add(node_id)

                # Case 1: node has a 'metrics' dict (e.g., experiment nodes)
                metrics = node.properties.get("metrics")
                if metrics and isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if key not in by_metric:
                            by_metric[key] = []
                        by_metric[key].append({"entity": node_id, "value": str(value)})

                # Case 2: Metric-type node with name/value properties
                if node.type == "Metric":
                    metric_name = node.properties.get("name")
                    metric_value = node.properties.get("value")
                    if metric_name and metric_value is not None:
                        key = metric_name.lower().replace(" ", "_")
                        if key not in by_metric:
                            by_metric[key] = []
                        unit = node.properties.get("unit", "")
                        val = metric_value
                        # Scale percentages: metrics.json stores 0.9158 for 91.58%
                        if unit == "%" and isinstance(val, (int, float)):
                            val = round(val * 100, 2)
                        val_str = f"{val}{unit}"
                        by_metric[key].append({"entity": node_id, "value": val_str})

    # ── Proactive collection: follow derived_from edges to collect Metric nodes ──
    # from experiment/run nodes that appeared in traversal paths.
    for exp_id in exp_node_ids:
        for edge in graph.get_edges(exp_id, "in"):
            if edge.type != "derived_from":
                continue
            neighbor_id = edge.source  # derived_from source → target
            if neighbor_id in seen_entities:
                continue
            seen_entities.add(neighbor_id)
            neighbor = graph.get_node(neighbor_id)
            if neighbor is None:
                continue

            # If the neighbor is a Metric node, collect it
            if neighbor.type == "Metric":
                metric_name = neighbor.properties.get("name")
                metric_value = neighbor.properties.get("value")
                if metric_name and metric_value is not None:
                    key = metric_name.lower().replace(" ", "_")
                    if key not in by_metric:
                        by_metric[key] = []
                    unit = neighbor.properties.get("unit", "")
                    val = metric_value
                    if unit == "%" and isinstance(val, (int, float)):
                        val = round(val * 100, 2)
                    val_str = f"{val}{unit}"
                    by_metric[key].append({"entity": neighbor_id, "value": val_str})

            # If the neighbor is a run node, also follow ITS incoming derived_from
            # to find its metric children
            if neighbor.type == "Experiment":
                for inner_edge in graph.get_edges(neighbor_id, "in"):
                    if inner_edge.type != "derived_from":
                        continue
                    inner_id = inner_edge.source
                    if inner_id in seen_entities:
                        continue
                    seen_entities.add(inner_id)
                    inner_node = graph.get_node(inner_id)
                    if inner_node and inner_node.type == "Metric":
                        m_name = inner_node.properties.get("name")
                        m_value = inner_node.properties.get("value")
                        if m_name and m_value is not None:
                            key = m_name.lower().replace(" ", "_")
                            if key not in by_metric:
                                by_metric[key] = []
                            unit = inner_node.properties.get("unit", "")
                            val = m_value
                            if unit == "%" and isinstance(val, (int, float)):
                                val = round(val * 100, 2)
                            val_str = f"{val}{unit}"
                            by_metric[key].append({"entity": inner_id, "value": val_str})

    return by_metric


def _collect_numbers(
    paths: list[Path],
    graph: InMemoryGraphStore,
) -> list[dict[str, Any]]:
    """
    Collect all structured metric→value pairs from all nodes in the evidence.

    For each unique node across all paths, extracts the 'metrics' property
    (if present) and returns a flat list of {entity, metric_name: value} entries.
    Also proactively follows derived_from edges to collect Metric-type nodes.

    This creates a machine-readable NUMBERS table that the prompt template
    can render as simple key-value pairs for the model.
    """
    numbers: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    exp_node_ids: set[str] = set()

    for path in paths:
        for step in path.steps:
            for node_id in (step.from_node, step.to_node):
                if node_id in seen_entities:
                    continue
                seen_entities.add(node_id)
                node = graph.get_node(node_id)
                if node is None:
                    continue

                # Track experiment/run nodes for proactive metric collection
                if node.type == "Experiment":
                    exp_node_ids.add(node_id)

                # Case 1: node has a 'metrics' dict (e.g., experiment nodes)
                metrics = node.properties.get("metrics")
                if metrics and isinstance(metrics, dict) and metrics:
                    entry: dict[str, Any] = {"entity": node_id}
                    entry.update(metrics)
                    numbers.append(entry)

                # Case 2: Metric-type node with name/value properties
                if node.type == "Metric":
                    metric_name = node.properties.get("name")
                    metric_value = node.properties.get("value")
                    if metric_name and metric_value is not None:
                        unit = node.properties.get("unit", "")
                        val = metric_value
                        if unit == "%" and isinstance(val, (int, float)):
                            val = round(val * 100, 2)
                        entry = {
                            "entity": node_id,
                            metric_name.lower().replace(" ", "_"): f"{val}{unit}",
                        }
                        numbers.append(entry)

    # ── Proactive collection: follow derived_from edges to Metric nodes ──
    for exp_id in exp_node_ids:
        for edge in graph.get_edges(exp_id, "in"):
            if edge.type != "derived_from":
                continue
            neighbor_id = edge.source
            if neighbor_id in seen_entities:
                continue
            seen_entities.add(neighbor_id)
            neighbor = graph.get_node(neighbor_id)
            if neighbor is None:
                continue

            if neighbor.type == "Metric":
                m_name = neighbor.properties.get("name")
                m_value = neighbor.properties.get("value")
                if m_name and m_value is not None:
                    unit = neighbor.properties.get("unit", "")
                    val = m_value
                    if unit == "%" and isinstance(val, (int, float)):
                        val = round(val * 100, 2)
                    entry = {
                        "entity": neighbor_id,
                        m_name.lower().replace(" ", "_"): f"{val}{unit}",
                    }
                    numbers.append(entry)

            # If neighbor is a run node, also collect its metric children
            if neighbor.type == "Experiment":
                for inner_edge in graph.get_edges(neighbor_id, "in"):
                    if inner_edge.type != "derived_from":
                        continue
                    inner_id = inner_edge.source
                    if inner_id in seen_entities:
                        continue
                    seen_entities.add(inner_id)
                    inner_node = graph.get_node(inner_id)
                    if inner_node and inner_node.type == "Metric":
                        inner_name = inner_node.properties.get("name")
                        inner_value = inner_node.properties.get("value")
                        if inner_name and inner_value is not None:
                            unit = inner_node.properties.get("unit", "")
                            val = inner_value
                            if unit == "%" and isinstance(val, (int, float)):
                                val = round(val * 100, 2)
                            entry = {
                                "entity": inner_id,
                                inner_name.lower().replace(" ", "_"): f"{val}{unit}",
                            }
                            numbers.append(entry)

    return numbers


def _collect_neighbor_key_findings(
    paths: list[Path],
    graph: InMemoryGraphStore,
    target_entity: str | None = None,
    question: str = "",
) -> list[dict[str, Any]]:
    """
    Include key_findings/descriptions from 1-hop neighbor nodes.

    For each unique node that appears in the paths, look at all
    directly connected neighbors (nodes connected by a single edge).
    If a neighbor is an Experiment or Concept with key_finding/description,
    include it — this catches cases where the answer lives in a
    connected node, not the target entity directly.

    Relevance scoring via word overlap with question orders
    results from most to least relevant.

    Returns a list of {entity, type, text} dicts, sorted by relevance.
    """
    neighbor_facts: list[dict[str, Any]] = []
    seen_neighbors: set[str] = set()
    # Track all nodes that are already IN the paths (don't repeat them)
    in_path_nodes: set[str] = set()
    for path in paths:
        for step in path.steps:
            in_path_nodes.add(step.from_node)
            in_path_nodes.add(step.to_node)

    # Map entity_id to its node for faster lookup
    entity_map: dict[str, Node] = {}
    for path in paths:
        for step in path.steps:
            for node_id in (step.from_node, step.to_node):
                if node_id not in entity_map:
                    node = graph.get_node(node_id)
                    if node:
                        entity_map[node_id] = node

    # Also include the target_entity if it's not already in the paths
    if target_entity and target_entity not in in_path_nodes:
        node = graph.get_node(target_entity)
        if node:
            entity_map[target_entity] = node

    # For each entity in evidence, look at 1-hop neighbors
    for entity_id in list(entity_map.keys()):
        for edge in graph.get_edges(entity_id, "both"):
            neighbor_id = edge.target if edge.source == entity_id else edge.source
            if (neighbor_id in seen_neighbors 
                or neighbor_id in in_path_nodes 
                or neighbor_id == entity_id):
                continue

            neighbor = graph.get_node(neighbor_id)
            if neighbor is None:
                continue
            # Include Experiment, Concept, and Decision neighbors with key info
            if neighbor.type not in ("Experiment", "Concept", "Decision"):
                continue
            props = neighbor.properties
            text = props.get("key_finding") or props.get("description")
            if not text or not isinstance(text, str) or not text.strip():
                continue

            seen_neighbors.add(neighbor_id)
            # Compute relevance score via question word overlap
            relevance = 0
            if question:
                qw = set(re.findall(r'\w+', question.lower()))
                nw = set(re.findall(r'\w+', text.lower()))
                relevance = len(qw & nw)
                if target_entity and target_entity.lower() in neighbor_id.lower():
                    relevance += 15  # strong bonus for target match
            neighbor_facts.append({
                "entity": neighbor_id,
                "type": neighbor.type,
                "text": text,
                "_relevance": relevance,
            })

    # Sort by relevance descending
    neighbor_facts.sort(key=lambda nf: -nf["_relevance"])
    # Strip internal key
    for nf in neighbor_facts:
        del nf["_relevance"]

    return neighbor_facts


def _collect_source_snippets(
    paths: list[Path],
    graph: InMemoryGraphStore,
    question: str = "",
) -> list[dict[str, Any]]:
    """Collect source_snippets from the top-2 relevance-ranked nodes.

    Iterates over all unique nodes in the traversal paths, scores each by
    word overlap with the question text, and returns the top 2 snippets
    (entity ID, snippet text, source path).
    Nodes without a ``source_snippet`` property are silently skipped.
    """
    scored: list[tuple[Node, int]] = []
    seen: set[str] = set()

    for path in paths:
        for step in path.steps:
            for node_id in (step.from_node, step.to_node):
                if node_id in seen:
                    continue
                seen.add(node_id)
                node = graph.get_node(node_id)
                if node is None:
                    continue
                snippet = node.properties.get("source_snippet", "")
                if not snippet:
                    continue
                relevance = 0
                if question:
                    qw = set(re.findall(r'\w+', question.lower()))
                    sw = set(re.findall(r'\w+', snippet.lower()))
                    relevance = len(qw & sw)
                scored.append((node, relevance))

    scored.sort(key=lambda x: -x[1])

    snippets: list[dict[str, Any]] = []
    for node, _ in scored[:2]:
        snippet = node.properties.get("source_snippet", "")
        if snippet:
            snippets.append({
                "entity": node.id,
                "text": snippet,
                "source": node.sources[0] if node.sources else "",
            })
    return snippets


def build_evidence(
    question: str,
    paths: list[Path],
    graph: InMemoryGraphStore,
    max_paths: int = 7,
    max_facts_per_path: int = 20,
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
        "numbers": [],
        "numbers_by_metric": {},
        "neighbor_facts": [],
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
        paths, graph, question_intent, target_entity, question
    )

    evidence["facts"] = all_facts
    evidence["sources"] = sorted(all_sources)

    # ── NUMBERS section: flat, machine-readable table of all numbers ──
    evidence["numbers"] = _collect_numbers(paths, graph)
    # Cap at 80 entries — enough to cover most experiments without bloat
    if len(evidence["numbers"]) > 80:
        evidence["numbers"] = evidence["numbers"][:80]

    # ── NUMBERS_BY_METRIC: grouped by metric name for easy model lookup ──
    evidence["numbers_by_metric"] = _collect_numbers_by_metric(paths, graph)
    # Cap each metric to at most 15 entries to prevent extreme bloat
    for key in list(evidence["numbers_by_metric"].keys()):
        if len(evidence["numbers_by_metric"][key]) > 15:
            evidence["numbers_by_metric"][key] = evidence["numbers_by_metric"][key][:15]

    # ── Neighbor key_findings: 1-hop neighbor facts for enriched context ──
    evidence["neighbor_facts"] = _collect_neighbor_key_findings(
        paths, graph, target_entity, question
    )

    # ── Source snippets: top-2 relevance-ranked nodes with source_snippet ──
    evidence["snippets"] = _collect_source_snippets(paths, graph, question)

    # ── Add confidence signals for routing ──
    evidence["confidence_signals"] = _compute_confidence_signals(
        question, paths, graph, evidence, target_entity, question_intent,
    )

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


def build_zero_hop_pack(
    graph: InMemoryGraphStore,
    entity_ids: list[str],
    question: str = "",
) -> dict[str, Any]:
    """Build a minimal evidence pack from resolved entity nodes — no traversal.

    For each entity, extracts key_finding, description, and metrics directly
    from the node properties.  Used by the Tier 3 cascade fallback when both
    filtered (tier 1) and unfiltered (tier 2) evidence produce LLM refusals.

    Returns a dict matching the evidence_pack schema (node_facts, numbers,
    numbers_by_metric, paths, facts, neighbour_facts, sources).
    """
    node_facts: list[dict[str, Any]] = []
    numbers: list[dict[str, Any]] = []
    metrics_aggregated: dict[str, Any] = {}
    sources: list[str] = []

    for eid in entity_ids:
        node = graph.get_node(eid)
        if node is None:
            continue
        props = node.properties or {}
        kf = props.get("key_finding", "")
        desc = props.get("description", "")
        if kf:
            src = node.sources[0] if node.sources else ""
            node_facts.append({
                "text": f"[{eid}] {kf}",
                "source": src,
                "confidence": 0.8,
            })
            if src:
                sources.append(src)
        elif desc:
            src = node.sources[0] if node.sources else ""
            node_facts.append({
                "text": f"[{eid}] {desc}",
                "source": src,
                "confidence": 0.5,
            })
            if src:
                sources.append(src)
        node_metrics = props.get("metrics", {})
        if node_metrics and isinstance(node_metrics, dict):
            entry: dict[str, Any] = {"entity": eid}
            entry.update(node_metrics)
            numbers.append(entry)
            for mk, mv in node_metrics.items():
                if mk not in metrics_aggregated:
                    metrics_aggregated[mk] = []
                metrics_aggregated[mk].append(mv)

    if not node_facts and not numbers:
        return {}  # Caller should handle the empty-pack case

    return {
        "question": question,
        "node_facts": node_facts,
        "numbers": numbers,
        "numbers_by_metric": (
            {k: v for k, v in metrics_aggregated.items()} if metrics_aggregated else {}
        ),
        "paths": [],
        "facts": [],
        "neighbor_facts": [],
        "sources": sorted(set(sources)),
    }
