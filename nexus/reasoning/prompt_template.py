"""
NEXUS prompt template — converts structured evidence into a clean text prompt
for the reasoning model.

The prompt is designed to be compact and directive: the model only sees
pre-digested evidence, never raw text. This keeps the context small enough
for local models while preserving provenance.
"""

from __future__ import annotations

import json
from typing import Any


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

    node_facts = evidence.get("node_facts", [])

    parts: list[str] = []

    # System instruction — tuned for small local models
    parts.append(
        "SYSTEM: You are a precise reasoning assistant. "
        "You receive structured evidence from a knowledge graph. "
        "The \"KEY FINDINGS\" section contains manually curated, high-confidence facts. "
        "Use those facts to answer the question. "
        "Quote specific numbers when available. "
        "If evidence truly lacks the answer, say \"Insufficient evidence to answer.\" "
        "Do not invent facts."
    )

    # Question
    parts.append(f"\nQUESTION: {question}")

    # Evidence section
    parts.append("\nEVIDENCE:")

    if not paths and not facts and not node_facts:
        parts.append("  (No evidence found in the knowledge graph.)")
    else:
        # KEY FINDINGS (curated) — highest-confidence facts first
        if node_facts:
            parts.append("\n  KEY FINDINGS (curated — high confidence):")
            for nf in node_facts:
                text = nf.get("text", "")
                if text:
                    parts.append(f"  - {text}")

        # Node details — the MOST IMPORTANT section for small models
        # Place it after KEY FINDINGS so the model sees the actual facts immediately
        node_details = _format_node_details(paths)
        if node_details:
            parts.append("\n  Node details (read these facts to answer the question):")
            parts.extend(node_details)

        # Path chains
        if paths:
            parts.append("\n  Knowledge graph paths:")
            path_lines = _format_path_steps(paths)
            parts.extend(path_lines)

        # Facts (human-readable relations)
        if facts:
            parts.append("\n  Relation facts:")
            for fact in facts:
                parts.append(f"  - {fact}")

        # Sources
        if sources:
            parts.append(f"\n  Sources ({len(sources)} total):")
            parts.append(_format_sources(sources))

    # Answer prompt
    parts.append("\nANSWER:")

    return "\n".join(parts)
