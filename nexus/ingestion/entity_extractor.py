"""
Rule-based and LLM-assisted entity extraction from project artifacts.

Sources supported:
- Markdown documents
- Python source code (AST)
- JSON/YAML config files
- Experiment result files
- GitHub issues (API)

The markdown extractor is designed to be corpus-agnostic:
- Section headers are filtered aggressively (only ProperName/Experiment patterns pass)
- Domain-agnostic patterns: backticks, file paths, CONFIG_KEYS, proper names
- Quality filter: stopwords, noise patterns, minimum letter content
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

    # ---- Section headers (## Title) — only extract significant ones ---- 
    for match in re.finditer(r'^#{1,3}\s+(.+)$', text, re.MULTILINE):
        raw_title = match.group(1).strip()
        title = _clean_header(raw_title)
        if not title:
            continue
        # Skip generic / noise headers
        if _is_generic_heading(title):
            continue
        # Only extract if header looks significant (ProperName, CodeRef, Experiment)
        if not _is_significant_header(title):
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

    # ---- Filter noise after extraction ----
    entities = _filter_noise_entities(entities)

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
    if any(w in header_lower for w in ("tech", "stack", "architecture", "component", "service")):
        return "Technology"
    return "Concept"


def _infer_type_from_name(name: str) -> str:
    name_lower = name.lower()
    if "test" in name_lower:
        return "TestCase"
    if name.endswith(".py") or name.endswith(".js") or name.endswith(".ts") or name.endswith(".cs"):
        return "CodeFile"
    if name.endswith(".jsx") or name.endswith(".tsx") or name.endswith(".java") or name.endswith(".go"):
        return "CodeFile"
    if name.endswith(".md") or name.endswith(".txt") or name.endswith(".rst"):
        return "Document"
    if name.endswith(".yaml") or name.endswith(".yml") or name.endswith(".json"):
        return "CodeFile"
    if name.endswith("()"):
        return "Function"
    if "." in name:
        return "Concept"
    return "Entity"


def _is_generic_heading(name: str) -> bool:
    """
    Return True if the name looks like a generic section heading that should
    NOT be extracted as an entity.

    Checks:
      - Exact match against generic header list
      - Starts with generic prefix + colon/dash
      - Is a single common word
      - Looks like a sentence (>6 words, ends with punctuation)
    """
    name_lower = name.lower().strip()
    # Exact match against generic headers
    if name_lower in _GENERIC_HEADERS:
        return True
    # Starts with generic prefix (e.g., "introduction: ...", "setup -- ...")
    for gh in _GENERIC_HEADERS:
        if name_lower == gh or name_lower.startswith(gh + ":") or name_lower.startswith(gh + " --"):
            return True
    # Single common word
    word_count = len(name_lower.split())
    if word_count == 1 and name_lower in _COMMON_WORDS:
        return True
    # Looks like a sentence (too long, natural language)
    if word_count > 6 and re.search(r'[.?!]$', name.strip()):
        return True
    # Pure question headers
    if re.match(r'^(what|how|why|when|where|who|is|are|can|do|does|should|will)\b', name_lower):
        return True
    return False


def _is_significant_header(title: str) -> bool:
    """
    Return True if the header title is significant enough to extract as an entity.

    Significant patterns:
      - Contains a CamelCase identifier (2+ capital letters)
      - Contains experiment/version references (Experiment X.Y, v2.0)
      - Contains code references (backtick patterns)
      - Contains CONFIG_KEY patterns
      - Is a proper name (2-4 consecutive capitalized words)

    Also rejects task-description headers (starting with verbs like Check, Update, Fix, etc.)
    """
    # Reject task-description headers
    task_verbs = {
        "check", "update", "edit", "fix", "implement", "add", "remove",
        "delete", "migrate", "deploy", "create", "configure", "set up",
        "install", "upgrade", "downgrade", "refactor", "rewrite", "replace",
        "move", "rename", "restructure", "clean", "cleanup", "optimize",
        "improve", "change", "modify", "review", "test", "verify", "validate",
        "build", "run", "start", "stop", "restart", "enable", "disable",
    }
    first_word = title.strip().split()[0].lower() if title.strip() else ""
    if first_word in task_verbs:
        return False
    # Contains CamelCase (two or more capital letters in a single word)
    if re.search(r'[A-Z][a-z]+[A-Z]', title):
        return True
    # Contains code references (function_name, ClassName)
    if re.search(r'[a-z_]+\(\)', title):
        return True
    # Contains experiment references
    if re.search(r'(?:Experiment|Exp|Bug|Gate)\s*[\d]+(?:\.[\d]+)?', title, re.IGNORECASE):
        return True
    # Contains version references
    if re.search(r'\bv?\d+\.\d+', title):
        return True
    # Contains proper names (2-4 consecutive capitalized words)
    if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', title):
        return True
    # Contains CONFIG_KEY patterns
    if re.search(r'[A-Z][A-Z_]{2,}[A-Z]', title):
        return True
    return False

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
# Common English words (broad stoplist: ~200 most frequent words)
# Entity names matching these are noise and are filtered out.
# ---------------------------------------------------------------------------
_COMMON_WORDS: set[str] = {
    # Top 50 most common English words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    # Next 50
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    # Next 50
    "great", "must", "such", "here", "high", "own", "old", "right", "still",
    "off", "need", "try", "each", "found", "long", "ask", "last", "same", "may",
    "between", "called", "keep", "very", "left", "few", "while", "along", "might",
    "close", "seem", "next", "open", "begin", "got", "run", "walk", "help", "turn",
    "start", "show", "hear", "play", "move", "live", "mean", "pull", "push", "end",
    # More common noise words
    "put", "due", "per", "via", "yet", "ago", "far", "big", "top", "low",
    "set", "cut", "let", "add", "had", "did", "has", "was", "were", "been",
    "are", "itself", "himself", "herself", "themselves", "something", "anything",
    "nothing", "everything", "someone", "anyone", "everyone", "much", "many",
    "more", "less", "really", "quite", "almost", "rather", "enough", "too",
    "perhaps", "maybe", "often", "always", "never", "ever", "already",
    "above", "below", "through", "around", "throughout", "within", "without",
    "during", "before", "after", "until", "since", "upon", "across",
    "etc", "eg", "ie", "vs", "aka", "note", "yes", "no", "ok", "okay",
    "please", "thanks", "welcome", "done", "using", "based",
}
_STOP_WORDS: set[str] = _COMMON_WORDS  # alias for backward compatibility


def _is_valid_entity(name: str) -> bool:
    """
    Return True if the entity name is valid (not noise).

    Rejects:
      - Names shorter than 3 characters
      - Names that are pure numeric (e.g. "12", "0.5")
      - Names that contain no letters (pure punctuation/numbers)
      - Names that are stop words / common English words
      - Names containing pipe characters (markdown table residue)
      - Names containing backticks (formatting artifacts)
      - Names that look like URLs (http://, https://, www.)
      - Names that look like email addresses (@)
      - Names that are version numbers (v1.2.3, 10.x)
      - Names that are dates (YYYY-MM-DD, DD/MM/YYYY)
      - Names that look like generic section headings
    """
    stripped = name.strip()
    if len(stripped) < 3:
        return False
    if "|" in stripped:
        return False
    if "`" in stripped:
        return False
    # Pure numeric (with optional decimal point and percent sign)
    if re.fullmatch(r"[\d]+(?:\.[\d]+)?%?", stripped):
        return False
    # Must contain at least one letter
    if not re.search(r"[a-zA-Z]", stripped):
        return False
    # Checkbox patterns: [x], [ ], ✓, ☐, ☑
    if re.match(r'^\[[ x✓☐☑]\]', stripped):
        return False
    # Boolean literals
    if stripped.lower() in {"true", "false", "null", "none", "undefined"}:
        return False
    # Very long names (> 80 chars) are likely sentences or table residue
    if len(stripped) > 80:
        return False
    # URL patterns
    if re.match(r'https?://', stripped, re.IGNORECASE):
        return False
    if re.match(r'www\.', stripped, re.IGNORECASE):
        return False
    # Email addresses
    if '@' in stripped and ('.' in stripped.split('@')[-1] if '@' in stripped else False):
        return False
    # Version numbers: v1.2.3, v10, 2.0.1
    if re.fullmatch(r'v?\d+\.\d+(?:\.\d+)?(?:[a-z]\w*)?', stripped, re.IGNORECASE):
        return False
    # Dates: 2024-03-15, 2024/03/15, 15.03.2024
    if re.fullmatch(r'\d{2,4}[-/.]\d{1,2}[-/.]\d{1,4}', stripped):
        return False
    # Common English word (check lowercase)
    if stripped.lower() in _COMMON_WORDS:
        return False
    # Filter generic headings (even after cleaning)
    if _is_generic_heading(stripped):
        return False
    return True


def _filter_noise_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Post-extraction noise filter.  Rejects entities whose source line contains
    a pipe character (`|`) — a strong signal the entity came from a table row
    — and entities that fail _is_valid_entity().
    """
    clean: list[dict[str, Any]] = []
    for entity in entities:
        name = entity.get("name", "")
        if not _is_valid_entity(name):
            continue
        clean.append(entity)
    return clean


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
        if re.match(r'^(the|and|but|that|this|these|those|for|with|from|it|is|are|was|were|be|a|an|if|when|how|use|no|not|into|small|large|every|each|can|has|have|also|only|very|any|our|per|may|via)\b',
                     bold, re.IGNORECASE):
            continue
        _add(bold, "Concept", m.start())

    # --- DOMAIN-AGNOSTIC PATTERNS ---

    # CONFIG_KEY patterns: UPPER_CASE_WITH_UNDERSCORES (min 2 parts)
    # e.g. AZURE_SUBSCRIPTION_ID, JWT_ACCESS_TOKEN, SMS_API_KEY
    for m in re.finditer(r'\b([A-Z][A-Z0-9]*(?:_[A-Z][A-Z0-9]*){1,})\b', text):
        key = m.group(1)
        if len(key) >= 6 and not key.startswith("HTTP"):
            start, end = m.span()
            if _is_inside_backtick(text, start, end):
                continue
            _add(key, "Entity", m.start())

    # Generic file paths: dir/file.ext, src/main.py, docs/guide.md
    for m in re.finditer(r'\b([\w][\w./_-]*\.(?:py|js|ts|jsx|tsx|yaml|yml|json|md|cs|css|html|xml|sh|ps1|sql|env|cfg|ini|toml|dockerfile|txt))\b', text, re.IGNORECASE):
        fpath = m.group(1)
        if len(fpath) >= 5 and "/" in fpath:
            start, end = m.span()
            if _is_inside_backtick(text, start, end):
                continue
            etype = "Document" if fpath.endswith(".md") else "CodeFile"
            _add(fpath, etype, m.start())

    # Proper names: 2-4 consecutive capitalized words (not at start of sentence)
    # e.g. "App Router", "Docker Compose", "GitHub Actions"
    # Must not be preceded by sentence-start markers
    for m in re.finditer(r'(?<![.!?\n]\s)(?<![.!?\n])(?<!\A)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text):
        name = m.group(1)
        if len(name) >= 6:
            # Skip if it's a common English phrase
            if name.lower() in _COMMON_WORDS:
                continue
            start, end = m.span()
            if _is_inside_backtick(text, start, end):
                continue
            _add(name, "Entity", m.start())

    # Technology/version combos: "React 19", "PostgreSQL 16", ".NET 10"
    # Matches capitalized word followed by a number (version)
    for m in re.finditer(r'\b([A-Z][a-zA-Z.]*(?:\s+[A-Z][a-zA-Z.]*){0,2})\s+(\d+(?:\.\d+)?)\b', text):
        tech = m.group(1).strip()
        ver = m.group(2)
        if len(tech) >= 3 and tech.lower() not in _COMMON_WORDS:
            # Skip months and other noise
            if tech.lower() in {"january", "february", "march", "april", "may", "june",
                                 "july", "august", "september", "october", "november", 
                                 "december", "chapter", "section", "step", "part", "phase",
                                 "level", "stage", "type", "case", "option"}:
                continue
            start, end = m.span()
            if _is_inside_backtick(text, start, end):
                continue
            name = f"{tech} {ver}"
            _add(name, "Technology", m.start())
            # Also add just the tech name without version
            _add(tech, "Technology", m.start())

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
