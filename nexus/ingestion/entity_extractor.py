"""
Rule-based and LLM-assisted entity extraction from project artifacts.

Sources supported:
- Markdown documents
- Python source code (AST)
- JSON/YAML config files
- Experiment result files
- GitHub issues (API)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nexus.ingestion.normalizer import normalize_entity_name
from nexus.ingestion.deduplicator import deduplicate_entities

# ---------------------------------------------------------------------------
# Stop-list: generic header titles that should never become entities
# ---------------------------------------------------------------------------
_GENERIC_HEADERS: set[str] = {
    "introduction", "overview", "conclusion", "summary", "references",
    "getting started", "prerequisites", "installation", "usage", "examples",
    "appendix", "acknowledgements", "faq", "troubleshooting", "setup",
    "abstract", "table of contents", "toc", "related work", "future work",
    "contributing", "license", "changelog", "configuration", "background",
    "methodology", "discussion", "limitations", "next steps",
    "simple version", "root cause", "modified files",
    "medium-term next steps", "longer-term vision items",
    "immediate next step", "immediate next step: experiment",
    "executive verdict",
    "bugs found and fixed",
    "why sam is not", "why sam is not \"just rag\"",
    "what cpu-first sparse-memory architecture means",
    "what \"cpu-first sparse-memory architecture\" means",
    "current status of components",
    "memory integration modes",
}

# ---------------------------------------------------------------------------
# Domain vocabulary: small curated list to match plain-text mentions
# that would otherwise be missed (built from sam-lm/docs/glossary.md).
# ---------------------------------------------------------------------------
_DOMAIN_TERMS: list[tuple[str, str]] = [
    # (regex-pattern, entity-type)
    # -- experiment running modes
    (r'\bcore.only\b', "Concept"),
    (r'\boracle.memory\b', "Concept"),
    (r'\bretrieved.memory\b', "Concept"),
    (r'\brandom.memory\b', "Concept"),
    (r'\boracle.text.memory\b', "Concept"),
    (r'\boracle.filter.diagnostic\b', "Concept"),
    (r'\boracle.filter\b', "Concept"),
    (r'\boracle.plus.distractors\b', "Concept"),
    (r'\blearned.selector\b', "Concept"),
    (r'\bslot.selector\b', "Concept"),
    # -- retrieval / indexing
    (r'\bchain.set\b', "Concept"),
    (r'\bdual.encoder\b', "Concept"),
    (r'\bproduct.key.memory\b', "Concept"),
    (r'\bproduct.key.associative.memory\b', "Concept"),
    (r'\bgated.integration\b', "Concept"),
    (r'\bgated.sum\b', "Concept"),
    (r'\bconcat.projection\b', "Concept"),
    (r'\bintegrate.gated\b', "Function"),
    (r'\bforced.gate.1\b', "Concept"),
    # -- metrics with @-syntax
    (r'\b(all_required|any_required|coverage|Recall|recall)@\d+\b', "Metric"),
    # -- other important terms (case-sensitive for acronyms)
    (r'\bPKM\b', "Concept"),
    (r'\bRAG\b', "Concept"),
    (r'\btopK\b', "Metric"),
    (r'\bmmap\b', "Concept"),
    (r'\bInfoNCE\b', "Concept"),
    (r'\bBCE loss\b', "Concept"),
    # -- general concept terms (multi-word, capitalized patterns)
    (r'\b(sparse memory|dense weights|latent memory|synthetic dataset|residual stream|memory bank|query vector|memory block|memory norm|memory vector|memory integration|hard negative|controlled distractor)\b', "Concept"),
    # -- multi-hop terminology
    (r'\b\d+-hop(?:\s+reasoning)?\b', "Concept"),
    # -- oracle/memory mode combos
    (r'\b(oracle_gap|memory_gain|dense.open.book|dense baseline)\b', "Concept"),
    # -- hyphenated multi-word concepts
    (r'\b(retrieval.score.weighted.averaging|live.slot.only.negative.sampling|uniform.weighted.averaging|multi.hop.retrieval|multi.positive.loss|forced.gate)\b', "Concept"),
    # -- capitalization-sensitive concept names
    (r'\b(Ternary core quantization|Adaptive multi.hop)\b', "Concept"),
    # -- BERT / model short references
    (r'\bSAM\b', "Entity"),
    # -- domain-specific metric terms (only when near percentage thresholds)
    (r'>\d+% (recall|precision)\b', "Metric"),
]


def extract_from_markdown(text: str, source_path: str) -> list[dict[str, Any]]:
    """
    Extract entities from markdown text using rule-based patterns.

    Detects:
    - Section headers as potential entity/concept names (noise-filtered)
    - Code references (backtick-wrapped)
    - Plain-text entity mentions (CamelCase, snake_case, domain terms)
    """
    entities: list[dict[str, Any]] = []

    # ---- Section headers (## Title) ----
    for match in re.finditer(r'^#{1,3}\s+(.+)$', text, re.MULTILINE):
        raw_title = match.group(1).strip()
        title = _clean_header(raw_title)
        if not title:
            continue
        # Skip generic / noise headers (exact or prefix match)
        title_lower = title.lower()
        skip = False
        for gh in _GENERIC_HEADERS:
            if title_lower == gh or title_lower.startswith(gh + ":") or title_lower.startswith(gh + " --"):
                skip = True
                break
        if skip:
            continue
        # Skip purely numeric / single-letter headers
        if re.fullmatch(r'[\d.]+\s*([a-zA-Z]?)?', title):
            continue
        entities.append({
            "name": title,
            "type": _infer_type_from_header(title),
            "source": source_path,
            "line": text[:match.start()].count('\n') + 1,
        })

    # ---- Backtick-wrapped code references ----
    for match in re.finditer(r'`([A-Za-z_][A-Za-z0-9_\.@\-]+)`', text):
        ref = match.group(1)
        if len(ref) >= 3 and not ref.startswith("http"):
            entities.append({
                "name": ref,
                "type": _infer_type_from_name(ref),
                "source": source_path,
                "line": text[:match.start()].count('\n') + 1,
            })

    # Extended: dotted access like `slot_ids.clamp` or `model.memory_mode`
    for match in re.finditer(r'`([A-Za-z_][A-Za-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]+)`', text):
        ref = match.group(1)
        entities.append({
            "name": ref,
            "type": _infer_type_from_name(ref),
            "source": source_path,
            "line": text[:match.start()].count('\n') + 1,
        })

    # Extended: identifiers with parens like `read_slot_values()`
    for match in re.finditer(r'`([A-Za-z_][A-Za-z0-9_]+\(\))`', text):
        ref = match.group(1)
        entities.append({
            "name": ref,
            "type": _infer_type_from_name(ref),
            "source": source_path,
            "line": text[:match.start()].count('\n') + 1,
        })

    # ---- Plain-text entity mentions ----
    entities.extend(_extract_plain_text_entities(text, source_path))

    # ---- Deduplicate ----
    return deduplicate_entities(entities)


def extract_from_code(file_path: str) -> list[dict[str, Any]]:
    """
    Extract entities from Python source code using AST.

    Detects:
    - Function/class definitions
    - Import statements
    - Decorator references
    """
    entities = []

    try:
        import ast
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                entities.append({
                    "name": node.name,
                    "type": "Function",
                    "source": file_path,
                    "line": node.lineno,
                })
            elif isinstance(node, ast.ClassDef):
                entities.append({
                    "name": node.name,
                    "type": "CodeFile" if "Test" in node.name else "Entity",
                    "source": file_path,
                    "line": node.lineno,
                })
    except (SyntaxError, Exception):
        pass

    return entities


def extract_from_experiment_results(metrics_path: str) -> list[dict[str, Any]]:
    """
    Extract Experiment and Metric nodes from experiment result files.
    """
    entities = []

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flatten metrics
        for key, value in _flatten_dict(data):
            if isinstance(value, (int, float)):
                entities.append({
                    "name": key,
                    "type": "Metric",
                    "properties": {"value": value},
                    "source": metrics_path,
                })
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    return entities


def _infer_type_from_header(header: str) -> str:
    header_lower = header.lower()
    if any(w in header_lower for w in ("test", "test case")):
        return "TestCase"
    if any(w in header_lower for w in ("bug", "issue", "error", "fix")):
        return "Bug"
    if any(w in header_lower for w in ("experiment", "result", "finding")):
        return "Experiment"
    if any(w in header_lower for w in ("metric", "measure", "score")):
        return "Metric"
    if any(w in header_lower for w in ("decision", "design", "choice")):
        return "Decision"
    if any(w in header_lower for w in ("requirement", "spec", "must", "should")):
        return "Requirement"
    if any(w in header_lower for w in ("document", "readme", "guide")):
        return "Document"
    return "Concept"


def _infer_type_from_name(name: str) -> str:
    name_lower = name.lower()
    if "test" in name_lower:
        return "TestCase"
    if name.endswith(".py") or name.endswith(".js") or name.endswith(".ts"):
        return "CodeFile"
    if name.endswith(".md"):
        return "Document"
    if name.endswith("()"):
        return "Function"
    if "." in name:
        return "Concept"
    return "Entity"


# ---------------------------------------------------------------------------
# Header cleaning
# ---------------------------------------------------------------------------

def _clean_header(raw: str) -> str:
    """
    Strip numbered prefixes and trailing noise from section headers.

    Examples:
      "7.1 Core Issue: Query Projection Mismatch"
        -> "Core Issue: Query Projection Mismatch"
      "2.3.1 Results"
        -> "Results"
      "1. Introduction"
        -> "Introduction"
      "3. Chain Oracle-Filter Diagnostic"
        -> "Chain Oracle-Filter Diagnostic"
    """
    # Remove leading numbering patterns like "7.1", "2.3.1", "1.", "3. "
    cleaned = re.sub(r'^[\d]+(?:\.[\d]+)*\.?\s*', '', raw).strip()
    if not cleaned:
        return ""
    # Remove bold/italic markdown markers and backticks
    cleaned = re.sub(r'\*{1,3}', '', cleaned).strip()
    cleaned = cleaned.replace('`', '').strip()
    return cleaned


# ---------------------------------------------------------------------------
# Common English stop-words (do not extract these as entities)
# ---------------------------------------------------------------------------
_COMMON_WORDS: set[str] = {
    "the", "and", "for", "was", "not", "are", "with", "that", "this",
    "from", "has", "been", "its", "but", "all", "can", "had", "have",
    "use", "new", "one", "two", "way", "each", "set", "run", "see",
    "end", "may", "via", "also", "only", "very", "any", "our", "per",
    "had", "did", "due", "now", "get", "how", "why", "who", "put",
    "big", "old", "key", "top", "low", "few", "ago", "yet", "own",
    "off", "out", "too", "far", "the", "its", "his", "her", "etc",
}


def _extract_plain_text_entities(text: str, source_path: str) -> list[dict[str, Any]]:
    """
    Extract entity mentions that the backtick-based extractor misses.

    Handles ONLY targeted patterns:
      - Experiment references ("Experiment 0.6")
      - Bug references ("Bug 1")
      - Gate references ("Gate 1")
      - File references (sam/..., configs/...)
      - Bold-emphasized concepts (**X**)
      - Domain terms from curated vocabulary (outside backticks only)
    """
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _line_of(pos: int) -> int:
        return text[:pos].count('\n') + 1

    def _add(name: str, etype: str, pos: int) -> None:
        norm = normalize_entity_name(name)
        if len(norm) < 2:
            return
        if norm in _COMMON_WORDS:
            return
        if norm in seen:
            return
        seen.add(norm)
        entities.append({
            "name": name,
            "type": etype,
            "source": source_path,
            "line": _line_of(pos),
        })

    # --- Experiment references: "Experiment 0.X", "Exp 0.XY" ---
    for m in re.finditer(
        r'(?:Experiment|Exp)[_\s]*([\d]+(?:\.[\d]+[A-Z]?)?)',
        text, re.IGNORECASE,
    ):
        _add(f"Experiment {m.group(1)}", "Experiment", m.start())

    # --- Bug references: "Bug 1", "Bug 2" ---
    for m in re.finditer(r'\bBug (\d+)\b', text):
        _add(f"Bug {m.group(1)}", "Bug", m.start())

    # --- Gate references: "Gate 1", "Gate 3" ---
    for m in re.finditer(r'\bGate (\d+)\b', text):
        _add(f"Gate {m.group(1)}", "Decision", m.start())

    # --- File references ---
    for m in re.finditer(r'(sam/[\w/]+\.(?:py|yaml|yml|json))', text):
        _add(m.group(1), "CodeFile", m.start())
    for m in re.finditer(r'configs/([\w_]+\.ya?ml)', text):
        _add(m.group(1), "CodeFile", m.start())

    # --- Bold-emphasized concepts: **X** ---
    for m in re.finditer(r'\*\*([^*\n]{5,50})\*\*', text):
        bold = m.group(1).strip()
        # Strip backtick wrappers from bold content (e.g. **`core_only`**)
        bold = bold.strip('`').strip()
        if re.search(r'[.?!]$', bold):
            continue
        if bold.rstrip().endswith(':'):
            continue
        if re.match(r'^[\d.,%+\-=]', bold):
            continue
        if _is_inside_backtick(text, m.start(), m.end()):
            continue
        if re.match(r'^(the|and|but|that|this|these|those|for|with|from|it|is|are|was|were|be|a|an|if|when|how|use|no|not|into|small|large|every|each)\b',
                     bold, re.IGNORECASE):
            continue
        _add(bold, "Concept", m.start())

    # --- Domain terms (outside backticks only) ---
    for pattern, etype in _DOMAIN_TERMS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                name = m.group(1)
            except IndexError:
                name = m.group(0)
            start, end = m.span()
            if _is_inside_backtick(text, start, end):
                continue
            _add(name, etype, m.start())

    # --- Domain terms from inside long backtick spans ---
    # e.g. `oracle_gap = accuracy(oracle_memory) - accuracy(retrieved_memory)`
    # We extract individual domain terms like oracle_memory that appear as
    # tokens within longer inline-code expressions (span > 30 chars).
    for bt_match in re.finditer(r'`([^`]+)`', text):
        code = bt_match.group(1)
        if len(code) <= 30:
            continue  # skip short backtick refs (already handled above)
        bt_start = bt_match.start()
        for pattern, etype in _DOMAIN_TERMS:
            for m in re.finditer(pattern, code, re.IGNORECASE):
                try:
                    name = m.group(1)
                except IndexError:
                    name = m.group(0)
                pos = bt_start + 1 + m.start()
                _add(name, etype, pos)

    return entities


def _is_inside_backtick(text: str, start: int, end: int) -> bool:
    """Return True if the span [start, end) lies within backtick-enclosed text."""
    before = text[:start]
    # Count unmatched backticks in the preceding text
    backtick_count = before.count('`')
    # If odd number, we're inside a backtick span
    if backtick_count % 2 != 0:
        return True
    # Also check if the character immediately after the match is a closing
    # backtick — not after stripping whitespace (which falsely flags opening
    # backticks that happen to appear next in the text).
    if end < len(text) and text[end] == '`':
        return True
    return False


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dict into (key, value) pairs."""
    items = []
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, full_key))
        else:
            items.append((full_key, v))
    return items
