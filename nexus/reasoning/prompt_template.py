"""
NEXUS prompt template — converts structured evidence into a clean text prompt
for the reasoning model.

The prompt is designed to be compact and directive: the model only sees
pre-digested evidence, never raw text. This keeps the context small enough
for local models while preserving provenance.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _detect_question_type(question: str) -> str:
    """Detect the question type from the question text.

    Returns one of: 'factual', 'diagnostic', 'comparative', 'multi-hop'
    """
    q_lower = question.lower().strip()

    # Comparative patterns first (more specific)
    comparative_patterns = [
        r'\bcompare\b', r'\bvs\b', r'\bversus\b',
        r'\bdifference between\b', r'\bdifferences between\b',
        r'\bwhich is (better|worse|higher|lower|faster|slower)\b',
    ]
    for pat in comparative_patterns:
        if re.search(pat, q_lower):
            return "comparative"

    # Diagnostic patterns
    diagnostic_patterns = [
        r'\bwhy\b', r'\bwhat caused\b', r'\bwhat cause\b',
        r'\bhow did\b', r'\breason(s)? (for|behind)\b',
        r'\bexplain why\b', r'\bwhat led to\b',
    ]
    for pat in diagnostic_patterns:
        if re.search(pat, q_lower):
            return "diagnostic"

    # Multi-hop patterns
    multi_hop_patterns = [
        r'\bhow does\b', r'\bwalk through\b', r'\bexplain the chain\b',
        r'\bstep by step\b', r'\bhow are\b', r'\bwhat is the relationship\b',
        r'\bhow (is|are|do)\b',
    ]
    for pat in multi_hop_patterns:
        if re.search(pat, q_lower):
            return "multi-hop"

    # Factual patterns (what, how many, which, when, where)
    factual_patterns = [
        r'\bwhat (is|are|was|were)\b', r'\bwhat\'s\b',
        r'\bhow many\b', r'\bhow much\b',
        r'\bwhich\b', r'\bwhen\b', r'\bwhere\b',
        r'\bwho\b', r'\bname the\b', r'\blist the\b',
    ]
    for pat in factual_patterns:
        if re.search(pat, q_lower):
            return "factual"

    return "factual"


def _filter_relevant_nodes(
    paths: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    """Filter paths to only include nodes that are relevant to the question.

    Relevance is computed as word overlap between question words and the
    node's id + description/key_finding. If filtering removes all nodes,
    the original paths are returned unchanged.
    """
    question_words = set(re.findall(r'\w+', question.lower()))
    if not question_words:
        return paths

    filtered: list[dict[str, Any]] = []
    for path_data in paths:
        relevant_nodes: list[dict[str, Any]] = []
        for node in path_data.get("nodes", []):
            nid = (node.get("id", "") or "").lower()
            # Collect text to check overlap against
            text_parts = [nid]
            for key in ("key_finding", "description", "title", "name"):
                val = node.get(key, "")
                if val and isinstance(val, str):
                    text_parts.append(val.lower())
            combined = " ".join(text_parts)
            combined_words = set(re.findall(r'\w+', combined))
            overlap = len(question_words & combined_words)
            if overlap > 0:
                relevant_nodes.append(node)

        if relevant_nodes:
            filtered_path = dict(path_data)
            filtered_path["nodes"] = relevant_nodes
            filtered.append(filtered_path)

    # If filtering removed everything, return original
    if not filtered:
        return paths

    return filtered


def _find_question_entity(
    question: str,
    paths: list[dict[str, Any]],
    node_facts: list[dict[str, Any]] | None = None,
) -> str | None:
    """Find which entity (node ID) the question is specifically about.

    Checks whether any node ID (or a significant portion) appears
    directly in the question text, or has strong word overlap with it.
    Also checks node_facts text content for word overlap with the question
    to find entities whose descriptions match the question.
    Returns the matching node ID, or None if no clear match.
    """
    q_lower = question.lower()
    best_match: str | None = None
    best_score = 0

    # Collect all unique node IDs across paths
    all_node_ids: set[str] = set()
    for path_data in paths:
        for node in path_data.get("nodes", []):
            nid = node.get("id", "")
            if nid:
                all_node_ids.add(nid)

    q_words = set(re.findall(r'\w+', q_lower))

    for nid in all_node_ids:
        nid_lower = nid.lower()
        # Direct match: node ID appears as a substring in the question
        if nid_lower in q_lower:
            score = len(nid_lower)  # longer match = better
            if score > best_score:
                best_score = score
                best_match = nid
                continue

        # Word-level overlap between node ID parts and question words
        nid_words = set(re.findall(r"[a-zA-Z0-9]+", nid_lower))
        # Remove common prefix words like "Exp_", "Decision_", "Concept_"
        meaningful = {w for w in nid_words if len(w) > 2}
        overlap = len(meaningful & q_words)
        if overlap > 0 and overlap > best_score:
            best_score = overlap
            best_match = nid

    # Also check node_facts text for entities whose facts match the question
    if node_facts and best_score < 3:
        for nf in node_facts:
            text = nf.get("text", "")
            if not text:
                continue
            # Extract entity ID from the text (format: "EntityID: description")
            match = re.match(r'^(\S+):', text)
            if not match:
                continue
            nid = match.group(1)
            nid_lower = nid.lower()
            text_lower = text.lower()
            # Score by word overlap between question and fact text
            fact_words = set(re.findall(r'\w+', text_lower))
            overlap = len(q_words & fact_words)
            # Bonus for entity name appearing in question
            nid_words_set = set(re.findall(r'[a-zA-Z0-9]+', nid_lower))
            meaningful_nid = {w for w in nid_words_set if len(w) > 2}
            entity_overlap = len(meaningful_nid & q_words)
            total_score = overlap + entity_overlap * 5
            if total_score > best_score:
                best_score = total_score
                best_match = nid

    return best_match


def _format_path_steps(paths: list[dict[str, Any]]) -> list[str]:
    """Format path steps as directional arrow chains."""
    lines: list[str] = []
    for i, path_data in enumerate(paths, 1):
        edges = path_data.get("edges", [])
        if not edges:
            continue
        parts: list[str] = [f"  Path {i} (score: {path_data.get('score', 0):.3f}):"]
        # Build the chain
        chain_parts: list[str] = []
        for j, edge in enumerate(edges):
            arrow = "<--" if edge.get("reversed") else "-->"
            if j == 0:
                chain_parts.append(edge["from"])
            chain_parts.append(f" --[{edge['type']}](conf:{edge.get('confidence', 1.0):.2f}){arrow} {edge['to']}")
        parts.append("    " + "  ".join(chain_parts) if len(chain_parts) == 1 else "")
        # If chain_parts has odd number (from + arrow alternation), format properly
        if chain_parts:
            formatted = ""
            for k, chunk in enumerate(chain_parts):
                if k % 2 == 1:
                    formatted += chunk + " "
                else:
                    formatted += chunk + " "
            parts.append("    " + formatted.strip())
        lines.append("\n".join(parts))
    return lines


def _format_nodes(paths: list[dict[str, Any]]) -> list[str]:
    """Extract and format unique nodes from all paths."""
    seen: set[str] = set()
    lines: list[str] = []
    for path_data in paths:
        for node in path_data.get("nodes", []):
            nid = node.get("id", "")
            if nid in seen:
                continue
            seen.add(nid)
            ntype = node.get("type", "unknown")
            # Pick the most descriptive property
            desc = ""
            for key in ("description", "key_finding", "title", "name"):
                if key in node and node[key]:
                    desc = f" — {node[key]}"
                    break
            if not desc:
                desc = f" — {nid}"
            lines.append(f"  - [{ntype}] {nid}{desc}")
    return lines


def _format_node_details(paths: list[dict[str, Any]]) -> list[str]:
    """
    Extract and format node descriptions in a plain, readable format.
    This helps small models easily find facts like numbers, findings, etc.
    """
    seen: set[str] = set()
    details: list[tuple[str, str]] = []
    for path_data in paths:
        for node in path_data.get("nodes", []):
            nid = node.get("id", "")
            if nid in seen:
                continue
            seen.add(nid)
            # Collect all descriptive properties
            for key in ("key_finding", "description", "title", "name"):
                if key in node and node[key] and node[key] != nid:
                    details.append((nid, node[key]))
                    break
    if not details:
        return []
    lines = ["  Key findings from evidence nodes:"]
    for nid, text in details:
        # Clean up the text a bit
        clean = text.replace("\n", " ").strip()
        if len(clean) > 300:
            clean = clean[:297] + "..."
        lines.append(f"  - {nid}: {clean}")
    return lines


def _format_sources(sources: list[str]) -> str:
    """Format sources as a numbered reference list."""
    if not sources:
        return "  (none)"
    lines = []
    for i, src in enumerate(sources, 1):
        # Extract filename from path
        filename = src.split("/")[-1] if "/" in src else src.split("\\")[-1] if "\\" in src else src
        lines.append(f"  [{i}] {filename}  ({src})")
    return "\n".join(lines)


def _render_metric_numbers(
    parts: list[str],
    numbers_by_metric: dict[str, list[dict[str, Any]]],
    question: str,
) -> None:
    """Render the NUMBERS BY METRIC section for a specific question's metric term."""
    from nexus.query.parser import extract_metric_term
    metric_term = extract_metric_term(question)
    if metric_term:
        matching_entries = numbers_by_metric.get(metric_term, [])
        if matching_entries:
            parts.append(f"\n  Numbers for '{metric_term}':")
            for entry in matching_entries[:5]:
                entity = entry.get("entity", "")
                value = entry.get("value", "")
                parts.append(f"  - [{entity}] {value}")
        else:
            for key, entries in numbers_by_metric.items():
                if key.startswith(metric_term) or metric_term.startswith(key):
                    parts.append(f"\n  Numbers for '{key}':")
                    for entry in entries[:5]:
                        entity = entry.get("entity", "")
                        value = entry.get("value", "")
                        parts.append(f"  - [{entity}] {value}")
                    break


def build_prompt(question: str, evidence_json: str) -> str:
    """
    Build a clean text prompt from a question and JSON evidence pack.

    Args:
        question: The natural language question
        evidence_json: JSON string from evidence_builder.build_evidence()

    Returns:
        A formatted prompt string ready for model input
    """
    try:
        evidence = json.loads(evidence_json)
    except (json.JSONDecodeError, TypeError):
        evidence = {"paths": [], "facts": [], "sources": []}

    paths = evidence.get("paths", [])
    facts = evidence.get("facts", [])
    sources = evidence.get("sources", [])
    numbers = evidence.get("numbers", [])
    numbers_by_metric = evidence.get("numbers_by_metric", {})
    neighbor_facts = evidence.get("neighbor_facts", [])

    node_facts = evidence.get("node_facts", [])

    # Detect question type for targeted instructions
    q_type = _detect_question_type(question)

    # Find which entity the question is specifically about
    target_entity = _find_question_entity(question, paths, node_facts) if q_type == "factual" else None

    parts: list[str] = []

    # System instruction — tuned for small local models, varies by question type
    _type_instructions = {
        "factual": (
            "SYSTEM: You are a precise reasoning assistant. "
            "You receive structured evidence from a knowledge graph. "
            "Answer ONLY about the specific entity mentioned in the question. "
            "Start your answer with the entity name and the fact. "
            "Example: 'The oracle memory experiment achieved 99.87% accuracy.' "
            "For factual questions: answer in EXACTLY ONE sentence with the specific value. "
            "Do NOT list dependency chains unless the question asks about dependencies. "
            "Do NOT mention other experiments' findings. "
            "Quote specific numbers when available. "
            "If evidence truly lacks the answer, say \"Insufficient evidence to answer.\" "
            "Do not invent facts."
        ),
        "diagnostic": (
            "SYSTEM: You are a precise reasoning assistant. "
            "You receive structured evidence from a knowledge graph. "
            "Explain the causal chain from the evidence. "
            "Show how entities are connected and why. "
            "Quote specific findings. "
            "Do NOT list dependency chains unless the question asks about dependencies. "
            "If evidence truly lacks the answer, say \"Insufficient evidence to answer.\" "
            "Do not invent facts."
        ),
        "comparative": (
            "SYSTEM: You are a precise reasoning assistant. "
            "You receive structured evidence from a knowledge graph. "
            "State the comparison clearly. Mention which is higher/better/different and by how much. "
            "Maximum two sentences. "
            "Quote specific numbers. "
            "Do NOT list dependency chains. "
            "If evidence truly lacks the answer, say \"Insufficient evidence to answer.\" "
            "Do not invent facts."
        ),
        "multi-hop": (
            "SYSTEM: You are a precise reasoning assistant. "
            "You receive structured evidence from a knowledge graph. "
            "Walk through the chain of evidence step by step. "
            "Explain how each entity connects to the next to answer the question. "
            "Quote specific numbers when available. "
            "If evidence truly lacks the answer, say \"Insufficient evidence to answer.\" "
            "Do not invent facts."
        ),
    }
    parts.append(_type_instructions.get(q_type, _type_instructions["factual"]))

    # Question
    parts.append(f"\nQUESTION: {question}")

    # IMPORTANT constraint — varies by question type
    if q_type == "factual":
        parts.append(
            "\nIMPORTANT: Answer ONLY what the question asks. "
            "EXACTLY ONE sentence with the specific value. "
            "Do not list other experiments or dependency chains."
        )
    elif q_type == "comparative":
        parts.append(
            "\nIMPORTANT: Maximum TWO sentences. State the comparison and numbers. "
            "Do not list dependency chains."
        )
    else:
        parts.append(
            "\nIMPORTANT: Answer ONLY the question asked. Do not list all evidence. "
            "Maximum 3 sentences."
        )

    # Evidence section
    parts.append("\nEVIDENCE:")

    if not paths and not facts and not node_facts:
        parts.append("  (No evidence found in the knowledge graph.)")
    else:
        # All node_facts are pre-sorted by relevance in the evidence builder.
        # For factual questions, we highlight target-entity-specific facts first,
        # then show supporting context.

        filtered_paths = _filter_relevant_nodes(paths, question)

        if q_type == "factual" and target_entity:
            # ── Factual + target entity: show target-specific facts prominently ──
            target_facts = [
                nf for nf in node_facts
                if target_entity.lower() in nf.get("text", "").lower()
            ]
            other_facts = [
                nf for nf in node_facts
                if target_entity.lower() not in nf.get("text", "").lower()
            ]

            parts.append("\n  The answer to your question is in these facts:")
            if target_facts:
                for nf in target_facts:
                    text = nf.get("text", "")
                    if text:
                        parts.append(f"  - {text}")
            else:
                parts.append("  (look for the fact most relevant to the question)")
                for nf in node_facts:
                    text = nf.get("text", "")
                    if text:
                        parts.append(f"  - {text}")

            # NUMBERS BY METRIC for the question's metric term
            if numbers_by_metric:
                _render_metric_numbers(parts, numbers_by_metric, question)

            # Neighbor facts for additional context
            if neighbor_facts:
                parts.append("\n  Related entity findings:")
                for nf in neighbor_facts:
                    entity = nf.get("entity", "")
                    etype = nf.get("type", "")
                    text = nf.get("text", "")
                    if text:
                        parts.append(f"  - [{etype}] {entity}: {text}")

            # Other relevant facts (non-target but still question-relevant)
            if other_facts:
                parts.append("\n  Other relevant findings:")
                for nf in other_facts[:3]:  # cap to save tokens
                    text = nf.get("text", "")
                    if text:
                        parts.append(f"  - {text}")

        else:
            # ── Non-factual or no target entity: show all key facts ──
            if node_facts:
                parts.append("\n  Key evidence (most relevant facts):")
                for nf in node_facts:
                    text = nf.get("text", "")
                    if text:
                        parts.append(f"  - {text}")

            # NUMBERS BY METRIC for the question's metric term
            if numbers_by_metric:
                _render_metric_numbers(parts, numbers_by_metric, question)

            # Neighbor facts
            if neighbor_facts:
                parts.append("\n  Related entity findings:")
                for nf in neighbor_facts:
                    entity = nf.get("entity", "")
                    etype = nf.get("type", "")
                    text = nf.get("text", "")
                    if text:
                        parts.append(f"  - [{etype}] {entity}: {text}")

        # ── KEY NUMBERS section: flat, scannable number table (for all q types) ──
        if numbers:
            parts.append("\n  Key metrics from evidence:")
            for num_entry in numbers:
                entity = num_entry.get("entity", "")
                kv_pairs = [
                    f"{k}: {v}" for k, v in num_entry.items()
                    if k != "entity"
                ]
                if kv_pairs:
                    parts.append(f"  - [{entity}] {' | '.join(kv_pairs)}")

        # ── SOURCE CONTEXT: paragraphs from source documents ──
        snippets = evidence.get("snippets", [])
        if snippets:
            parts.append("\n  SOURCE CONTEXT (from source documents):")
            for snip in snippets[:2]:
                entity = snip.get("entity", "")
                text = snip.get("text", "")
                ctx = snip.get("context", "")
                if text:
                    if len(text) > 400:
                        text = text[:397] + "..."
                    parts.append(f"  [{entity}] {text}")
                if ctx:
                    if len(ctx) > 500:
                        ctx = ctx[:497] + "..."
                    parts.append(f"  [{entity} context] {ctx}")

        # ── Edge-based facts (relation chains) ──
        if facts:
            parts.append("\n  Relation chains:")
            for fact in facts:
                parts.append(f"  - {fact}")

        # ── Supporting evidence (for non-factual only) ──
        if q_type != "factual":
            node_details = _format_node_details(filtered_paths)
            if node_details:
                parts.append("\n  Supporting evidence:")
                parts.extend(node_details)

            if filtered_paths:
                parts.append("\n  Knowledge graph paths:")
                path_lines = _format_path_steps(filtered_paths)
                parts.extend(path_lines)

        # Sources
        if sources:
            parts.append(f"\n  Sources ({len(sources)} total):")
            parts.append(_format_sources(sources))

    # Answer prompt
    parts.append("\nANSWER:")

    return "\n".join(parts)
