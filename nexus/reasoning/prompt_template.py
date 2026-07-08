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

    parts: list[str] = []

    # System instruction
    parts.append(
        "SYSTEM: You are a precise reasoning assistant. "
        "You receive structured evidence from a knowledge graph. "
        "Answer ONLY based on the provided evidence. "
        "If the evidence is insufficient, say \"Insufficient evidence to answer.\" "
        "Do not invent facts. Cite sources when possible."
    )

    # Question
    parts.append(f"\nQUESTION: {question}")

    # Evidence section
    parts.append("\nEVIDENCE:")

    if not paths and not facts:
        parts.append("  (No evidence found in the knowledge graph.)")
    else:
        # Path chains
        if paths:
            parts.append("\n  Knowledge graph paths:")
            path_lines = _format_path_steps(paths)
            parts.extend(path_lines)

        # Facts (human-readable)
        if facts:
            parts.append("\n  Extracted facts:")
            for fact in facts:
                parts.append(f"  - {fact}")

        # Sources
        if sources:
            parts.append(f"\n  Sources ({len(sources)} total):")
            parts.append(_format_sources(sources))

    # Answer prompt
    parts.append("\nANSWER:")

    return "\n".join(parts)
