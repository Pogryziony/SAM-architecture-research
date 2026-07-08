"""
Rule-based relation extraction from markdown documents.

Extracts typed edges between entities using regex patterns.
No LLM dependency -- purely heuristic.

Edge types extracted:
  depends_on, caused_by, validates, contradicts, implements,
  mentioned_in, derived_from, related_to, replaces, blocked_by
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def extract_relations(
    text: str,
    source_path: str,
    entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract typed edges from markdown text using rule-based patterns.
    
    Returns:
        (edges, new_entities) where:
        - edges: list of {source_name, target_name, edge_type, confidence, evidence}
        - new_entities: list of entity dicts discovered during relation extraction
    """
    edges = []
    new_entities = []
    entity_names = {e["name"] for e in entities}
    entity_names_lower = {n.lower(): n for n in entity_names}

    def find_or_create(name: str, etype: str = "Entity") -> str:
        """Find an entity by name, or create a new one and return its name."""
        norm = name.strip().lower()
        # Exact match
        if norm in entity_names_lower:
            return entity_names_lower[norm]
        # Normalized match with underscores
        norm_us = norm.replace(" ", "_").replace("-", "_")
        for ename in entity_names:
            if ename.lower().replace(" ", "_").replace("-", "_") == norm_us:
                return ename
        # Substring match
        for ename in entity_names:
            en_lower = ename.lower()
            if norm in en_lower or en_lower in norm:
                if len(norm) >= 4 and len(en_lower) >= 4:
                    return ename
        # Create new entity
        if len(name.strip()) >= 2 and name.strip() not in entity_names:
            entity_names.add(name)
            entity_names_lower[name.lower()] = name
            new_entities.append({
                "name": name, "type": etype,
                "source": source_path, "line": 0,
            })
        return name

    def add_edge(source: str, target: str, edge_type: str, confidence: float, evidence: str):
        src = find_or_create(source)
        tgt = find_or_create(target)
        if src != tgt and src and tgt:
            edges.append({
                "source_name": src, "target_name": tgt,
                "edge_type": edge_type, "confidence": confidence,
                "evidence": evidence,
            })

    # ── Pattern 1: Experiment dependencies ──
    # "Experiment 0.X depends on / builds on / extends Experiment 0.Y"
    exp_patterns = [
        (r'(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?[A-Z]?)?)\s+(?:builds?\s+on|depends?\s+on|extends?|follows?)\s+(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?[A-Z]?)?)',
         0.95, "depends_on", "Explicit dependency"),
        (r'(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?[A-Z]?)?)\s+(?:validates?|confirms?|proves?)\s+(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?[A-Z]?)?)',
         0.85, "validates", "Validation relationship"),
    ]
    for pattern, conf, etype, desc in exp_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            add_edge(f"Experiment {m.group(1)}", f"Experiment {m.group(2)}",
                     etype, conf, f"{desc} in {source_path}")

    # ── Pattern 2: "Previous experiment" / "Experiment 0.10 results" ──
    prev_exp = re.finditer(
        r'(?:the\s+)?(?:previous|prior)\s+experiment',
        text, re.IGNORECASE
    )
    for m in prev_exp:
        # Try to determine which experiment the current file is about
        source_match = re.findall(
            r'(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?[A-Z]?)?)',
            source_path, re.IGNORECASE
        )
        if source_match:
            exp_num = source_match[-1]
            # The "previous" experiment is the one before this
            parts = re.findall(r'\d+', exp_num)
            if parts:
                major = int(parts[0])
                if len(parts) >= 2:
                    minor = int(parts[1])
                    if minor > 0:
                        prev_num = f"{major}.{minor - 1}"
                        # Handle special cases
                        if exp_num == "0.2":
                            prev_num = "0"
                        elif exp_num == "0.10":
                            prev_num = "0.9"
                        add_edge(f"Experiment {exp_num}", f"Experiment {prev_num}",
                                 "depends_on", 0.85,
                                 f"References previous experiment in {source_path}")

    # ── Pattern 3: Verb patterns between entities ──
    # Only patterns with confidence >= 0.85 survive.
    verb_configs = [
        # (pattern, edge_type, confidence)
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:validates?|confirms?|proves?)\s+(?:the\s+)?`([^`]+)`', "validates", 0.80),
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:is\s+validated\s+by|is\s+confirmed\s+by)\s+(?:the\s+)?`([^`]+)`', "validates", 0.80),
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:implements?|provides?)\s+(?:the\s+)?`([^`]+)`', "implements", 0.80),
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:causes?|leads?\s+to|results?\s+in)\s+(?:the\s+)?`([^`]+)`', "caused_by", 0.75),
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:is\s+caused\s+by|is\s+due\s+to)\s+(?:the\s+)?`([^`]+)`', "caused_by", 0.75),
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:replaces?|supersedes?)\s+(?:the\s+)?`([^`]+)`', "replaces", 0.85),
        (r'(?:the\s+)?`([^`]+)`\s+(?:contradicts?|conflicts?\s+with)\s+(?:the\s+)?`([^`]+)`', "contradicts", 0.85),
        (r'(?:the\s+)?`([^`]+)`\s+(?:is\s+blocked\s+by)\s+(?:the\s+)?`([^`]+)`', "blocked_by", 0.85),
        (r'(?:the\s+)?`([^`]+)`\s+(?:blocks?)\s+(?:the\s+)?`([^`]+)`', "blocked_by", 0.85),
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:mentions?|references?)\s+(?:the\s+)?`([^`]+)`', "mentioned_in", 0.70),
        # DROPPED: low precision — (r'(?:the\s+)?`([^`]+)`\s+(?:derives?\s+from|is\s+derived\s+from)\s+(?:the\s+)?`([^`]+)`', "derived_from", 0.80),
    ]
    for pattern, etype, conf in verb_configs:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s, t = m.group(1).strip(), m.group(2).strip()
            if len(s) >= 3 and len(t) >= 3:
                add_edge(s, t, etype, conf,
                         f"Pattern '{etype}' in {source_path} (line ~{text[:m.start()].count(chr(10))+1})")

    # ── Pattern 4: Non-backtick verb patterns ──
    # DROPPED: low precision — bold verb patterns generate ~40% noise edges
    # bold_verb_configs = [ ... ] — all dropped; confidence < 0.85

    # ── Pattern 5: Section header X mentions backtick ref Y ──
    # DROPPED: generic co-occurrence — produces massive noise edge volume
    # header_blocks = re.finditer( ... ) — confidence 0.50, too noisy

    # ── Pattern 6: Cross-document markdown references ──
    # DROPPED: low precision — generic cross-reference noise
    # cross_refs = re.finditer( ... ) — confidence 0.50

    # ── Pattern 7: Arrow chains in timelines ──
    # "X ──► Y ──► Z" or "X -> Y -> Z"
    arrow_text = text.replace('\u2500\u2500\u25ba', ' --> ').replace('\u2500\u25ba', ' -> ')
    arrow_text = arrow_text.replace('\u2192', ' -> ')  # → arrow
    arrow_lines = [l for l in arrow_text.split('\n') if ' --> ' in l or ' -> ' in l]
    for line in arrow_lines:
        parts = re.split(r'\s*(?:-->\s*|->\s*)', line)
        parts = [p.strip() for p in parts if p.strip()]
        for i in range(len(parts) - 1):
            src = re.sub(r'\([^)]*\)', '', parts[i]).strip()
            tgt = re.sub(r'\([^)]*\)', '', parts[i+1]).strip()
            if len(src) >= 3 and len(tgt) >= 3:
                add_edge(src, tgt, "depends_on", 0.85,
                         f"Timeline arrow in {source_path}")

    # ── Pattern 8: Sequential experiment mentions ──
    # DROPPED: low precision — produces 1000+ noise edges from unrelated co-mentions
    # Sequential experiment mention edges — confidence 0.40, far too noisy

    # ── Pattern 9: "X is the bottleneck" ──
    for m in re.finditer(
        r'`([^`]+)`\s+is\s+(?:the\s+)?(?:primary\s+)?(?:critical\s+)?bottleneck',
        text, re.IGNORECASE
    ):
        add_edge(m.group(1), "overall_progress", "blocked_by", 0.85,
                 f"Bottleneck identified in {source_path}")

    # ── Pattern 10: Metric validation ──
    # DROPPED: low precision — "X achieves Y" generates too many unrelated edges
    # Metric validation — confidence 0.75, dropped

    # ── Pattern 11: "X >> Y" comparison ──
    # DROPPED: low precision — generic comparison with 0.40 confidence
    # X >> Y comparison — confidence 0.40, dropped

    # ── Pattern 12: "X is Y" definitions ──
    # DROPPED: low precision — 0.55 confidence, generic definitions
    # "X is a Y" definitions — dropped

    # ── Pattern 13: Glossary cross-references ──
    # DROPPED: low precision — glossary see-also and contrast produce low-signal edges
    # Glossary cross-references — confidence 0.40–0.50, dropped

    # ── Pattern 14: "Gate X: PASS" / "Gate X: FAIL" ──
    for m in re.finditer(r'Gate\s+([A-Z\d]+)\s*:\s*(PASS|FAIL)', text, re.IGNORECASE):
        gate_result = m.group(2).upper()
        # Find the experiment context
        before = text[:m.start()]
        nearby_exp = re.findall(
            r'(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?)?)',
            before[-200:], re.IGNORECASE
        )
        if nearby_exp:
            gate_entity = f"Gate {m.group(1)}"
            exp_entity = f"Experiment {nearby_exp[-1]}"
            if gate_result == "PASS":
                add_edge(exp_entity, gate_entity, "validates", 0.85,
                         f"Gate pass in {source_path}")
            else:
                add_edge(gate_entity, exp_entity, "blocked_by", 0.85,
                         f"Gate fail in {source_path}")

    # ── Pattern 15: Experiment headers with descriptions ──
    # DROPPED: low precision — "## Experiment X — Description" produces
    # unreliable edges (confidence 0.60, description as pseudo-entity)
    # Experiment header description edges — dropped

    # ── Pattern 16: Sentence-level entity co-occurrence ──
    # DROPPED: generic co-occurrence patterns — this was the #1 noise
    # source generating thousands of 0.35-confidence "related_to" edges
    # Sentence-level co-occurrence — dropped

    # ── Pattern 17: Bold text contains backtick-wrapped entities ──
    # DROPPED: bold+backtick proximity — produces the bulk of low-precision
    # 0.30–0.45 confidence edges
    # Bold+backtick co-occurrence — dropped

    # ── Pattern 18: "X achieves/eliminates/reaches Y" without backtick on X ──
    # DROPPED: low precision — plain-verb patterns with 0.45–0.55 confidence
    # Plain-verb patterns without backtick — dropped

    # ── Deduplicate ──
    unique_edges = []
    seen = set()
    for e in edges:
        key = (e["source_name"], e["target_name"], e["edge_type"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges, new_entities
