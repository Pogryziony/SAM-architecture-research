"""Deterministic natural question templates for synthetic data generation.

Replaces the repetitive "What is <alias>?" templates with diverse natural
question styles matching factual, diagnostic, comparison, and multi-hop intents.

All templates are deterministic — same graph always produces same questions.
No frozen test inspection.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any


SEED = 20260710

# ── Template catalog ──

FACTUAL_TEMPLATES = [
    "What was the result of {name}?",
    "What did {name} demonstrate?",
    "What was the key finding of {name}?",
    "Describe {name}.",
    "What is {alias}?",
    "What is the significance of {name}?",
    "What was the purpose of {name}?",
    "Summarize {name}.",
    "What was discovered in {name}?",
    "What did {name} show?",
]

DIAGNOSTIC_TEMPLATES = [
    "Why did {name} produce {finding_snippet}?",
    "How does {name} relate to the overall research?",
    "What is the significance of the finding that {finding}?",
    "Why is {name} important?",
    "What does the result of {name} imply?",
]

COMPARISON_TEMPLATES = [
    "Compare {name_a} and {name_b}.",
    "How does {name_a} differ from {name_b}?",
    "What is the relationship between {name_a} and {name_b}?",
    "Contrast {name_a} with {name_b}.",
]

MULTI_HOP_TEMPLATES = [
    "How does {name_a} influence {name_b}?",
    "What is the connection between {name_a} and {name_b}?",
    "How does {name_b} depend on {name_a}?",
    "Explain the relationship between {name_a} and {name_b}.",
]


def _short_name(node_id: str, node: Any, max_words: int = 4) -> str:
    """Get a short display name for a node."""
    name = node_id.replace("_", " ")
    if hasattr(node, "aliases") and node.aliases:
        name = str(node.aliases[0])
    props = getattr(node, "properties", {}) or {}
    display = str(props.get("display_name", props.get("name", "")))
    if display:
        name = display
    return " ".join(name.split()[:max_words])


def _first_sentence(text: str, max_words: int = 8) -> str:
    """Extract a short snippet from text."""
    words = text.split()
    return " ".join(words[:max_words])


def _pair_id(question: str, entity_ids: list[str]) -> str:
    payload = question + "\0" + "\0".join(sorted(entity_ids))
    return "gen_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def generate_natural_pairs(graph: Any) -> list[dict[str, Any]]:
    """Generate deterministic natural-language training pairs from the graph.

    Sources and sampling:
    - Factual: alias/key-finding nodes (15% of total)
    - Diagnostic: nodes with findings (included in factual)
    - Comparison: pairs of experiments with depends_on/derived_from edges (25% graph-mined paraphrases include these)
    - Multi-hop: edge-connected pairs (10% relation examples)

    Returns:
        List of pair dicts with id, question, answer, entities, intent, category,
        source_id, label_source, confidence.
    """
    rng = random.Random(SEED)
    pairs: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    # Collect nodes with aliases and findings
    nodes_with_aliases = []
    nodes_with_findings = []
    edge_pairs = []

    for node_id, node in sorted(graph._nodes.items()):
        aliases = [str(a).strip() for a in getattr(node, "aliases", []) if str(a).strip()]
        props = getattr(node, "properties", {}) or {}
        finding = str(props.get("key_finding", "")).strip()

        if aliases:
            nodes_with_aliases.append((node_id, node, aliases))
        if finding:
            nodes_with_findings.append((node_id, node, finding))

        # Collect edge-connected pairs for comparison/multi-hop
        for edge in graph.get_outgoing(node_id):
            if edge.target in graph._nodes and edge.type in ("depends_on", "derived_from", "validates"):
                edge_pairs.append((node_id, edge.target, edge.type))

    rng.shuffle(nodes_with_aliases, random=lambda: rng.random())
    rng.shuffle(nodes_with_findings, random=lambda: rng.random())
    rng.shuffle(edge_pairs, random=lambda: rng.random())

    # ── Factual: alias/key-finding questions ──
    for node_id, node, aliases in nodes_with_aliases:
        template = rng.choice(FACTUAL_TEMPLATES)
        name = _short_name(node_id, node)
        alias = aliases[0]
        props = getattr(node, "properties", {}) or {}
        finding = str(props.get("key_finding", ""))
        question = template.format(name=name, alias=alias, finding_snippet=_first_sentence(finding))
        key = (question, (node_id,))
        pairs[key] = {
            "id": _pair_id(question, [node_id]),
            "question": question,
            "answer": finding or node_id,
            "question_type": "factual",
            "entities": [node_id],
            "intent": "factual_lookup",
            "category": "factual",
            "source_id": f"graph:v3:{node_id}:natural_factual",
            "label_source": "exact_alias",
            "confidence": 1.0,
        }

    # ── Diagnostic: findings-based questions ──
    for node_id, node, finding in nodes_with_findings:
        template = rng.choice(DIAGNOSTIC_TEMPLATES)
        name = _short_name(node_id, node)
        question = template.format(
            name=name,
            finding=finding,
            finding_snippet=_first_sentence(finding),
        )
        key = (question, (node_id,))
        pairs[key] = {
            "id": _pair_id(question, [node_id]),
            "question": question,
            "answer": finding,
            "question_type": "diagnostic",
            "entities": [node_id],
            "intent": "diagnostic",
            "category": "diagnostic",
            "source_id": f"graph:v3:{node_id}:natural_diagnostic",
            "label_source": "key_finding",
            "confidence": 0.95,
        }

    # ── Comparison / Multi-hop: edge-connected pairs ──
    for node_a, node_b, edge_type in edge_pairs[:200]:  # Cap at 200 pairs
        node_a_obj = graph.get_node(node_a)
        node_b_obj = graph.get_node(node_b)
        if node_a_obj is None or node_b_obj is None:
            continue
        is_comparison = edge_type in ("depends_on", "validates")
        templates = COMPARISON_TEMPLATES if is_comparison else MULTI_HOP_TEMPLATES
        template = rng.choice(templates)
        name_a = _short_name(node_a, node_a_obj)
        name_b = _short_name(node_b, node_b_obj)
        question = template.format(name_a=name_a, name_b=name_b)
        entity_ids = sorted([node_a, node_b])
        key = (question, tuple(entity_ids))
        pairs[key] = {
            "id": _pair_id(question, entity_ids),
            "question": question,
            "answer": edge_type,
            "question_type": "comparison" if is_comparison else "multi-hop",
            "entities": entity_ids,
            "intent": "comparison" if is_comparison else "multi_hop",
            "category": "comparative" if is_comparison else "multi-hop",
            "source_id": f"graph:v3:{node_a}:{edge_type}:{node_b}:natural_relation",
            "label_source": "weak",
            "confidence": 0.85,
        }

    return list(pairs.values())


def generate_balanced_dataset(
    real_questions: list[dict[str, Any]],
    graph: Any,
    real_ratio: float = 0.50,
    paraphrase_ratio: float = 0.25,
    alias_ratio: float = 0.15,
    relation_ratio: float = 0.10,
) -> list[dict[str, Any]]:
    """Generate a source-balanced training dataset.

    Proportions (preregistered):
    - 50% real questions (train.jsonl)
    - 25% natural graph-mined paraphrases
    - 15% alias/key-finding examples
    - 10% relation examples

    Args:
        real_questions: Questions from train.jsonl.
        graph: InMemoryGraphStore instance.
        real_ratio, paraphrase_ratio, alias_ratio, relation_ratio: Source proportions.

    Returns:
        Combined list of training pairs with source labels.
    """
    total = sum((real_ratio, paraphrase_ratio, alias_ratio, relation_ratio))
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Source ratios sum to {total}, expected 1.0")

    natural_pairs = generate_natural_pairs(graph)

    # Split natural pairs into paraphrase, alias/key-finding, and relation sources
    paraphrase_pairs = [p for p in natural_pairs if "natural_factual" in p["source_id"] or "natural_diagnostic" in p["source_id"]]
    alias_pairs = [p for p in natural_pairs if "natural_factual" in p["source_id"]]
    relation_pairs = [p for p in natural_pairs if "natural_relation" in p["source_id"]]

    # Determine counts based on real questions
    real_count = len(real_questions)
    total_target = int(real_count / real_ratio)
    paraphrase_count = int(total_target * paraphrase_ratio)
    alias_count = int(total_target * alias_ratio)
    relation_count = int(total_target * relation_ratio)

    # Mark source in each record
    for record in real_questions:
        record["source"] = "real_train"
    for record in paraphrase_pairs[:paraphrase_count]:
        record["source"] = "graph_mined_paraphrase"
    for record in alias_pairs[:alias_count]:
        record["source"] = "graph_alias_keyfinding"
    for record in relation_pairs[:relation_count]:
        record["source"] = "graph_relation"

    combined = (
        real_questions
        + paraphrase_pairs[:paraphrase_count]
        + alias_pairs[:alias_count]
        + relation_pairs[:relation_count]
    )

    return combined
