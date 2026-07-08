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
                                 "depends_on", 0.80,
                                 f"References previous experiment in {source_path}")

    # ── Pattern 3: Verb patterns between entities ──
    verb_configs = [
        # (pattern, edge_type, confidence)
        (r'(?:the\s+)?`([^`]+)`\s+(?:validates?|confirms?|proves?)\s+(?:the\s+)?`([^`]+)`', "validates", 0.80),
        (r'(?:the\s+)?`([^`]+)`\s+(?:is\s+validated\s+by|is\s+confirmed\s+by)\s+(?:the\s+)?`([^`]+)`', "validates", 0.80),
        (r'(?:the\s+)?`([^`]+)`\s+(?:implements?|provides?)\s+(?:the\s+)?`([^`]+)`', "implements", 0.80),
        (r'(?:the\s+)?`([^`]+)`\s+(?:causes?|leads?\s+to|results?\s+in)\s+(?:the\s+)?`([^`]+)`', "caused_by", 0.75),
        (r'(?:the\s+)?`([^`]+)`\s+(?:is\s+caused\s+by|is\s+due\s+to)\s+(?:the\s+)?`([^`]+)`', "caused_by", 0.75),
        (r'(?:the\s+)?`([^`]+)`\s+(?:replaces?|supersedes?)\s+(?:the\s+)?`([^`]+)`', "replaces", 0.85),
        (r'(?:the\s+)?`([^`]+)`\s+(?:contradicts?|conflicts?\s+with)\s+(?:the\s+)?`([^`]+)`', "contradicts", 0.80),
        (r'(?:the\s+)?`([^`]+)`\s+(?:is\s+blocked\s+by)\s+(?:the\s+)?`([^`]+)`', "blocked_by", 0.85),
        (r'(?:the\s+)?`([^`]+)`\s+(?:blocks?)\s+(?:the\s+)?`([^`]+)`', "blocked_by", 0.80),
        (r'(?:the\s+)?`([^`]+)`\s+(?:mentions?|references?)\s+(?:the\s+)?`([^`]+)`', "mentioned_in", 0.70),
        (r'(?:the\s+)?`([^`]+)`\s+(?:derives?\s+from|is\s+derived\s+from)\s+(?:the\s+)?`([^`]+)`', "derived_from", 0.80),
    ]
    for pattern, etype, conf in verb_configs:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Only add if both sides look like meaningful names (>2 chars, not just numbers)
            s, t = m.group(1).strip(), m.group(2).strip()
            if len(s) >= 3 and len(t) >= 3:
                add_edge(s, t, etype, conf,
                         f"Pattern '{etype}' in {source_path} (line ~{text[:m.start()].count(chr(10))+1})")

    # ── Pattern 4: Non-backtick verb patterns ──
    # Same verbs but without backtick requirement: "X validates Y"
    bold_verb_configs = [
        (r'\*\*(.+?)\*\*\s+(?:validates?|confirms?)\s+\*\*(.+?)\*\*', "validates", 0.80),
        (r'\*\*(.+?)\*\*\s+(?:is\s+caused\s+by)\s+\*\*(.+?)\*\*', "caused_by", 0.75),
        (r'\*\*(.+?)\*\*\s+(?:implements?)\s+\*\*(.+?)\*\*', "implements", 0.80),
        (r'\*\*(.+?)\*\*\s+(?:is\s+blocked\s+by)\s+\*\*(.+?)\*\*', "blocked_by", 0.85),
    ]
    for pattern, etype, conf in bold_verb_configs:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s, t = m.group(1).strip(), m.group(2).strip()
            if len(s) >= 3 and len(t) >= 3:
                add_edge(s, t, etype, conf,
                         f"Bold pattern '{etype}' in {source_path}")

    # ── Pattern 5: Section header X mentions backtick ref Y ──
    header_blocks = re.finditer(
        r'^#{1,3}\s+(.+?)$\n+(.+?)(?=\n#{1,3}\s|\Z)',
        text, re.MULTILINE | re.DOTALL
    )
    for m in header_blocks:
        header = m.group(1).strip()
        body = m.group(2)
        backtick_refs = re.findall(r'`([A-Za-z_][A-Za-z0-9_\.]+)`', body)
        for ref in backtick_refs:
            if ref.lower() != header.lower():
                add_edge(header, ref, "related_to", 0.50,
                         f"Section mentions {ref} in {source_path}")

    # ── Pattern 6: Cross-document markdown references ──
    cross_refs = re.finditer(r'\[([^\]]+)\]\(([^)]+\.md)\)', text)
    for m in cross_refs:
        ref_text = m.group(1)
        ref_path = m.group(2)
        doc_name = Path(ref_path).stem.replace("-", " ").title()
        add_edge(ref_text, doc_name, "related_to", 0.50,
                 f"Cross-reference to {ref_path} in {source_path}")

    # ── Pattern 7: Arrow chains in timelines ──
    # "X ──► Y ──► Z" or "X -> Y -> Z"
    arrow_text = text.replace('\u2500\u2500\u25ba', ' --> ').replace('\u2500\u25ba', ' -> ')
    arrow_text = arrow_text.replace('\u2192', ' -> ')  # → arrow
    # Now split on arrows
    arrow_lines = [l for l in arrow_text.split('\n') if ' --> ' in l or ' -> ' in l]
    for line in arrow_lines:
        # Split by arrow sequences
        parts = re.split(r'\s*(?:-->\s*|->\s*)', line)
        parts = [p.strip() for p in parts if p.strip()]
        for i in range(len(parts) - 1):
            src = re.sub(r'\([^)]*\)', '', parts[i]).strip()
            tgt = re.sub(r'\([^)]*\)', '', parts[i+1]).strip()
            if len(src) >= 3 and len(tgt) >= 3:
                add_edge(src, tgt, "depends_on", 0.70,
                         f"Timeline arrow in {source_path}")

    # ── Pattern 8: Sequential experiment mentions ──
    exp_nums = []
    for m in re.finditer(
        r'(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?[A-Z]?)?)',
        text, re.IGNORECASE
    ):
        num = m.group(1)
        if num not in exp_nums:
            exp_nums.append(num)

    if len(exp_nums) >= 2:
        for i in range(len(exp_nums) - 1):
            src = f"Experiment {exp_nums[i]}"
            tgt = f"Experiment {exp_nums[i+1]}"
            add_edge(src, tgt, "related_to", 0.40,
                     f"Sequential mention in {source_path}")

    # ── Pattern 9: "X is the bottleneck" ──
    for m in re.finditer(
        r'`([^`]+)`\s+is\s+(?:the\s+)?(?:primary\s+)?(?:critical\s+)?bottleneck',
        text, re.IGNORECASE
    ):
        add_edge(m.group(1), "overall_progress", "blocked_by", 0.70,
                 f"Bottleneck identified in {source_path}")

    # ── Pattern 10: Metric validation ──
    # "X achieves Y% accuracy", "Rec@8 = 99.3%", etc.
    for m in re.finditer(
        r'(?:achieves?|reaches?|shows?|scores?)\s+([\d]+(?:\.[\d]+)?%?)\s+(?:accuracy|precision|recall|F1|overall)',
        text, re.IGNORECASE
    ):
        before = text[:m.start()]
        nearby = re.findall(
            r'(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?)?)',
            before[-500:], re.IGNORECASE
        )
        if nearby:
            add_edge(f"Experiment {nearby[-1]}", f"{m.group(1)}% accuracy",
                     "validates", 0.75,
                     f"Metric validation in {source_path}")

    # ── Pattern 11: "X >> Y" comparison ──
    for m in re.finditer(r'`([^`]+)`\s*>>\s*`([^`]+)`', text):
        add_edge(m.group(1), m.group(2), "related_to", 0.40,
                 f"Comparison in {source_path}")

    # ── Pattern 12: "X is Y" definitions ──
    for m in re.finditer(r'`([^`]+)`\s+is\s+(?:a|an|the)\s+`([^`]+)`', text, re.IGNORECASE):
        add_edge(m.group(1), m.group(2), "related_to", 0.55,
                 f"Definition in {source_path}")

    # ── Pattern 13: Glossary cross-references ──
    glossary_terms = re.finditer(
        r'^###\s+(.+?)$\n+(.+?)(?=\n###\s|\Z)',
        text, re.MULTILINE | re.DOTALL
    )
    for m in glossary_terms:
        term = m.group(1).strip()
        body = m.group(2)
        # "See X", "See also X"
        for ref in re.findall(r'(?:See|see)\s+(?:also\s+)?(?:the\s+)?`([^`]+)`', body):
            add_edge(term, ref, "related_to", 0.40,
                     f"Glossary see-also in {source_path}")
        # "Contrast with X"
        for ref in re.findall(
            r'(?:contrast|not\s+the\s+same|different\s+from)\s+(?:with\s+)?`([^`]+)`',
            body, re.IGNORECASE
        ):
            add_edge(term, ref, "contradicts", 0.50,
                     f"Glossary contrast in {source_path}")

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
                add_edge(gate_entity, exp_entity, "blocked_by", 0.75,
                         f"Gate fail in {source_path}")

    # ── Pattern 15: Experiment headers with descriptions ──
    # "## Experiment 0.6 — Full Validation" 
    for m in re.finditer(
        r'^#{1,3}\s+(?:Experiment|Exp)\s*([\d]+(?:\.[\d]+[A-Z]?[A-Z]?)?)\s*[—–-]\s*(.+?)$',
        text, re.MULTILINE
    ):
        exp_num = m.group(1)
        desc = m.group(2).strip()
        exp_entity = f"Experiment {exp_num}"
        if len(desc) >= 3:
            add_edge(exp_entity, desc, "implements", 0.60,
                     f"Experiment header in {source_path}")

    # ── Pattern 16: Sentence-level entity co-occurrence ──
    # If two or more extracted entities appear in the same sentence, create
    # low-confidence "related_to" edges between them.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        sent_entities = [e for e in entities if e["name"] in sent]
        if len(sent_entities) >= 2:
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    add_edge(sent_entities[i]["name"], sent_entities[j]["name"],
                             "related_to", 0.35,
                             f"Sentence co-occurrence in {source_path}")

    # ── Pattern 17: Bold text contains backtick-wrapped entities ──
    # "**SAM `oracle_memory` achieves ... `core_only`**" → SAM related_to oracle_memory
    bold_blocks = re.finditer(r'\*\*(.+?)\*\*', text)
    for bm in bold_blocks:
        bold_content = bm.group(1)
        # Find bold entity (text before first backtick or all bold text)
        bold_entity_match = re.match(r'([^*`\n]{2,})', bold_content)
        if bold_entity_match:
            bold_entity = bold_entity_match.group(1).strip()
            # Find backtick-wrapped entities inside the bold block
            backtick_entities = re.findall(r'`([^`]+)`', bold_content)
            for bt_entity in backtick_entities:
                if len(bt_entity) >= 2 and bold_entity.lower() != bt_entity.lower():
                    add_edge(bold_entity, bt_entity, "related_to", 0.45,
                             f"Bold+backtick co-occurrence in {source_path}")
                    add_edge(bt_entity, bold_entity, "validates", 0.30,
                             f"Bold+backtick reverse in {source_path}")

    # ── Pattern 18: "X achieves/eliminates/reaches Y" without backtick on X ──
    plain_verb_configs = [
        (r'(?:the\s+)?([A-Za-z][A-Za-z0-9_\s-]{3,30}?)\s+(?:achieves?|reaches?|eliminates?)\s+(?:the\s+)?`([^`]+)`', "validates", 0.55),
        (r'(?:the\s+)?([A-Za-z][A-Za-z0-9_\s-]{3,30}?)\s+(?:fails?\s+to\s+set|does\s+not\s+set|did\s+not\s+set)\s+(?:the\s+)?`([^`]+)`', "caused_by", 0.50),
        (r'(?:the\s+)?([A-Za-z][A-Za-z0-9_\s-]{3,30}?)\s+(?:tries?\s+to|attempts?\s+to)\s+(?:the\s+)?`([^`]+)`', "implements", 0.45),
    ]
    for pattern, etype, conf in plain_verb_configs:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s, t = m.group(1).strip(), m.group(2).strip()
            if len(s) >= 3 and len(t) >= 3:
                add_edge(s, t, etype, conf,
                         f"Plain-verb pattern '{etype}' in {source_path}")

    # ── Deduplicate ──
    unique_edges = []
    seen = set()
    for e in edges:
        key = (e["source_name"], e["target_name"], e["edge_type"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges, new_entities
