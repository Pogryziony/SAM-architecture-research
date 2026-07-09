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


def _extract_metrics(text: str) -> dict[str, str]:
    """
    Extract metric→value pairs from text.

    Detects patterns like:
      - "99.87% accuracy", "100% recall", "precision: 50%"
      - "1,650 slots", "853 vocabulary tokens", "19,000 training examples"
      - "99.87%" (standalone percentage — stores as "accuracy" if ambiguous)
      - "precision 50%, recall 96.6%"
      - "Rec@8", "all_required@32 = 100%"
      - "+8 distractors", "3-hop collapses at +16 (39%)"

    Returns a dict of metric_name → value, e.g.:
      {"accuracy": "99.87%", "recall": "96.6%", "slot": "1650"}
    """
    metrics: dict[str, str] = {}

    # ── Pattern 1: Metric names with percentage values ──
    # "recall 96.6%, precision 50%", "accuracy of 99.87%", "99.0% Rec@8"
    pct_pairs = [
        # recall/accuracy/precision/f1/coverage WORD followed by NUMBER%
        (re.compile(
            r'(recall|accuracy|precision|f1|coverage)\s+(\d{1,3}(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ), True),
        # NUMBER% accuracy/recall/precision (e.g., "99.87% accuracy", "100% recall")
        (re.compile(
            r'(\d{1,3}(?:\.\d+)?)\s*%\s*(accuracy|recall|precision|f1|coverage)',
            re.IGNORECASE,
        ), False),
        # accuracy/recall OF NUMBER% (e.g., "accuracy of 99.87%")
        (re.compile(
            r'(accuracy|recall|precision|f1|coverage)\s+of\s+(\d{1,3}(?:\.\d+)?)\s*%',
            re.IGNORECASE,
        ), True),
    ]
    for pat, metric_first in pct_pairs:
        for m in pat.finditer(text):
            if metric_first:
                metric_name = m.group(1).lower()
                val_str = m.group(2)
            else:
                val_str = m.group(1)
                metric_name = m.group(2).lower()
            key = metric_name.lower()
            metrics[key] = f"{val_str}%"

    # ── Pattern 2: "X slots/tokens/examples/parameters/distractors" ──
    unit_pattern = re.compile(
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*'
        r'(?:live\s+)?'
        r'(slots?|tokens?|examples?|parameters?|params?|subkeys?|'
        r'distractors?|vocabulary|hops?|layers?|heads?|dimensions?|'
        r'batch_size|batch|epochs?|steps?|'
        r'training\s+examples?)',
        re.IGNORECASE,
    )
    for m in unit_pattern.finditer(text):
        val_str = m.group(1).replace(',', '')
        unit = m.group(2).lower()
        # Normalize plurals (but keep 's' for short units)
        if len(unit) > 3:
            unit = unit.rstrip('s')
        key = unit.replace(' ', '_')
        metrics[key] = val_str

    # ── Pattern 3: "+N distractors", "+N hop" ──
    plus_pattern = re.compile(
        r'\+(\d+)\s+(distractors?|hops?)',
        re.IGNORECASE,
    )
    for m in plus_pattern.finditer(text):
        val_str = m.group(1)
        unit = m.group(2).lower().rstrip('s')
        key = f"plus_{unit}"
        metrics[key] = val_str

    # ── Pattern 4: "X-hop" with percentage ──
    # "3-hop collapses at +16 (39%)"
    hop_pct_pattern = re.compile(
        r'(\d+)[- ]?hop.*?\((\d{1,3}(?:\.\d+)?)\)',
        re.IGNORECASE,
    )
    for m in hop_pct_pattern.finditer(text):
        hop_count = m.group(1)
        pct = m.group(2)
        metrics[f"{hop_count}_hop_accuracy"] = f"{pct}%"

    # ── Pattern 5: "X% N-hop accuracy" ──
    hop_acc_pattern = re.compile(
        r'(\d{1,3}(?:\.\d+)?)\s*%\s*'
        r'(\d+)[- ]?hop\s+accuracy',
        re.IGNORECASE,
    )
    for m in hop_acc_pattern.finditer(text):
        val_str = m.group(1)
        hop_num = m.group(2)
        metrics[f"{hop_num}_hop_accuracy"] = f"{val_str}%"

    # ── Pattern 6: "@ notation" metrics: all_required@32, Rec@8, recall@32 ──
    at_pattern = re.compile(
        r'(all_required|recall|rec|precision|prec|coverage|cov|f1)@(\d+)\s*[=:]\s*'
        r'(\d{1,3}(?:\.\d+)?)\s*%?',
        re.IGNORECASE,
    )
    for m in at_pattern.finditer(text):
        metric_name = m.group(1).lower()
        at_val = m.group(2)
        metric_val = m.group(3)
        key = f"{metric_name}@{at_val}"
        metrics[key] = f"{metric_val}%"

    # Also catch Rec@N without explicit value (just the notation)
    rec_at_ref = re.compile(
        r'(all_required|recall|rec|rec)@(\d+)',
        re.IGNORECASE,
    )
    for m in rec_at_ref.finditer(text):
        metric_name = m.group(1).lower()
        at_val = m.group(2)
        key = f"{metric_name}@{at_val}"
        # Only add if not already captured
        if key not in metrics:
            # Look for a nearby percentage value
            nearby = re.search(
                rf'{re.escape(m.group(0))}.*?(\d{{1,3}}(?:\.\d+)?)\s*%',
                text, re.IGNORECASE,
            )
            if nearby:
                metrics[key] = f"{nearby.group(1)}%"

    # ── Pattern 7: "Gate X PASSED" / "Gate X FAILED" ──
    gate_pattern = re.compile(
        r'gate\s+(\d+)\s+(passed|failed)',
        re.IGNORECASE,
    )
    for m in gate_pattern.finditer(text):
        gate_num = m.group(1)
        status = m.group(2).upper()
        metrics[f"gate_{gate_num}"] = status

    # ── Pattern 8: K=XX notation ──
    k_pattern = re.compile(r'\b[kK]=(\d+)\b')
    for m in k_pattern.finditer(text):
        metrics['k'] = m.group(1)

    # ── Pattern 9: Standalone percentages (e.g., "99.87%", "68.74%") ──
    # Only capture if we haven't already tagged them with a named metric.
    # Try to find context keywords near the percentage rather than using
    # "accuracy"/"metric_N" defaults which conflate unrelated numbers.
    standalone_pct = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')
    unnamed_count = 0
    for m in standalone_pct.finditer(text):
        val_str = m.group(1)
        # Check if this value is already captured under a named metric
        already_captured = any(v == f"{val_str}%" for v in metrics.values())
        if not already_captured:
            unnamed_count += 1
            # Look at text BEFORE the percentage for context keywords
            prefix_text = text[:m.start()].lower()
            # Extract the last 80 chars for tight context
            prefix_context = prefix_text[-80:] if len(prefix_text) > 80 else prefix_text
            # Known metric sense words to try (ordered by specificity)
            sense_words = [
                "accuracy", "recall", "precision", "f1", "coverage",
                "oracle", "core", "retrieved", "random", "baseline",
                "text", "latent", "dual", "chain", "selector",
            ]
            # Find the nearest sense word in the prefix context
            nearest_word = ""
            nearest_pos = -1
            for word in sense_words:
                pos = prefix_context.rfind(word)
                if pos > nearest_pos:
                    nearest_pos = pos
                    nearest_word = word
            if nearest_word and nearest_pos >= 0:
                key = f"{nearest_word}_{'accuracy' if nearest_word not in ('accuracy', 'recall', 'precision', 'f1', 'coverage') else ''}"
                key = key.rstrip('_')
            else:
                key = f"value_{unnamed_count}"
            # Only add if this key isn't already taken
            if key not in metrics:
                metrics[key] = f"{val_str}%"

    # ── Pattern 10: "X million" (e.g., "15.7 million") ──
    million_pattern = re.compile(
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*million',
        re.IGNORECASE,
    )
    for m in million_pattern.finditer(text):
        val_str = m.group(1).replace(',', '')
        metrics['million'] = val_str

    return metrics


def extract_from_markdown(text: str, source_path: str) -> list[dict[str, Any]]:
    """
    Extract entities from markdown text using rule-based patterns.

    Detects:
    - Section headers as potential entity/concept names (noise-filtered)
    - Code references (backtick-wrapped)
    - Plain-text entity mentions (CamelCase, snake_case, domain terms)

    Each entity may include an extracted 'metrics' property from surrounding
    text (e.g., "99.87% accuracy", "1,650 slots").
    """
    # Extract all metrics from the full text first (for entity association)
    # Also split text into lines for per-entity window extraction
    text_lines = text.split('\n')
    
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
        line_num = text[:match.start()].count('\n') + 1
        # Extract metrics from a window around this entity (the paragraph below the header)
        ctx_start = max(0, line_num - 1)
        ctx_end = min(len(text_lines), line_num + 3)
        ctx_text = '\n'.join(text_lines[ctx_start:ctx_end])
        entity_metrics = _extract_metrics(ctx_text)
        source_snippet = _extract_source_snippet(text, line_num, title)
        props: dict[str, Any] = {}
        if entity_metrics:
            props["metrics"] = entity_metrics
        if source_snippet:
            props["source_snippet"] = source_snippet
        entities.append({
            "name": title,
            "type": _infer_type_from_header(title),
            "source": source_path,
            "line": line_num,
            "properties": props,
        })

    # ---- Backtick-wrapped code references ----
    for match in re.finditer(r'`([A-Za-z_][A-Za-z0-9_\.@\-]+)`', text):
        ref = match.group(1)
        if len(ref) >= 3 and not ref.startswith("http"):
            line_num = text[:match.start()].count('\n') + 1
            source_snippet = _extract_source_snippet(text, line_num, ref)
            props: dict[str, Any] = {}
            if source_snippet:
                props["source_snippet"] = source_snippet
            entities.append({
                "name": ref,
                "type": _infer_type_from_name(ref),
                "source": source_path,
                "line": line_num,
                "properties": props,
            })

    # Extended: dotted access like `slot_ids.clamp` or `model.memory_mode`
    for match in re.finditer(r'`([A-Za-z_][A-Za-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]+)`', text):
        ref = match.group(1)
        line_num = text[:match.start()].count('\n') + 1
        source_snippet = _extract_source_snippet(text, line_num, ref)
        props: dict[str, Any] = {}
        if source_snippet:
            props["source_snippet"] = source_snippet
        entities.append({
            "name": ref,
            "type": _infer_type_from_name(ref),
            "source": source_path,
            "line": line_num,
            "properties": props,
        })

    # Extended: identifiers with parens like `read_slot_values()`
    for match in re.finditer(r'`([A-Za-z_][A-Za-z0-9_]+\(\))`', text):
        ref = match.group(1)
        line_num = text[:match.start()].count('\n') + 1
        source_snippet = _extract_source_snippet(text, line_num, ref)
        props: dict[str, Any] = {}
        if source_snippet:
            props["source_snippet"] = source_snippet
        entities.append({
            "name": ref,
            "type": _infer_type_from_name(ref),
            "source": source_path,
            "line": line_num,
            "properties": props,
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


def _extract_table_snippet(text_lines: list[str], match_line_idx: int) -> str:
    """Extract table header row + entity row for table context snippets."""
    parts: list[str] = []

    # Find table header row — look backwards for the column-header row
    # (the line just above the separator `|---|---|` or the first non-separator |...| row)
    header_idx: int | None = None
    for i in range(match_line_idx - 1, max(match_line_idx - 10, -1), -1):
        line = text_lines[i].strip()
        if not line:
            break
        if re.match(r'^\|[\s\-:]+\|', line):
            # This is a separator row — header is above it
            header_idx = i - 1
            break
        if line.startswith('|') and line.endswith('|'):
            header_idx = i

    # Build snippet: header + separator + entity row
    if header_idx is not None and header_idx >= 0:
        parts.append(text_lines[header_idx].strip())
        sep_idx = header_idx + 1
        if sep_idx < len(text_lines) and re.match(r'^\|[\s\-:]+\|', text_lines[sep_idx].strip()):
            parts.append(text_lines[sep_idx].strip())
    parts.append(text_lines[match_line_idx].strip())

    return '\n'.join(parts)


def _extract_source_snippet(text: str, line_num: int, entity_name: str = "") -> str:
    """Extract ±3 sentences of surrounding context for an entity match.

    If the entity appears in a markdown table row, the table header and
    entity row are returned instead.  Falls back to ±3 raw lines when
    sentence splitting produces no match.
    """
    if not text or line_num <= 0:
        return ""

    text_lines = text.split('\n')
    if line_num > len(text_lines):
        return ""

    match_line_idx = line_num - 1  # 0-indexed

    # -- Detect table context (lines containing | as table-cell markers) --
    is_table = False
    for offset in range(-5, 6):
        idx = match_line_idx + offset
        if 0 <= idx < len(text_lines):
            stripped = text_lines[idx].strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                is_table = True
                break

    if is_table:
        return _extract_table_snippet(text_lines, match_line_idx)

    # -- Sentence-based extraction (±3 sentences) --
    # Compute character offset of the match line
    char_pos = 0
    for i in range(match_line_idx):
        char_pos += len(text_lines[i]) + 1  # +1 for '\n'

    # Split into sentences (handles . ! ? as sentence boundaries)
    sentences: list[str] = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return ""

    # Find which sentence contains the match position
    target_sent_idx = -1
    sent_pos = 0
    for i, sent in enumerate(sentences):
        sent_end = sent_pos + len(sent)
        if sent_pos <= char_pos < sent_end:
            target_sent_idx = i
            break
        sent_pos = sent_end + 1  # +1 for the split separator space

    if target_sent_idx < 0:
        # Fallback: return ±3 raw lines around the match
        start = max(0, match_line_idx - 3)
        end = min(len(text_lines), match_line_idx + 4)
        return '\n'.join(text_lines[start:end]).strip()

    # Take ±3 sentences
    start_sent = max(0, target_sent_idx - 3)
    end_sent = min(len(sentences), target_sent_idx + 4)

    snippet = ' '.join(sentences[start_sent:end_sent]).strip()
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet


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
        line = _line_of(pos)
        source_snippet = _extract_source_snippet(text, line, name)
        props: dict[str, Any] = {}
        if source_snippet:
            props["source_snippet"] = source_snippet
        entities.append({
            "name": name,
            "type": etype,
            "source": source_path,
            "line": line,
            "properties": props,
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


# ---------------------------------------------------------------------------
# Automatic alias extraction from corpus text
# ---------------------------------------------------------------------------

# Words so generic they should never be auto-alias candidates
_GENERIC_ALIAS_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "and", "or", "but",
    "not", "no", "if", "then", "else", "so", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "we",
    "you", "i", "me", "my", "our", "your", "their", "about",
    "also", "just", "only", "very", "too", "all", "some", "any",
    "each", "every", "more", "most", "other", "such", "same",
    "here", "there", "now", "then", "when", "where", "how",
    "what", "which", "who", "why", "report", "experiment",
    "version", "summary", "result", "results", "overview",
    "findings", "finding", "conclusion", "analysis",
}


def _is_generic_alias(alias: str) -> bool:
    """Return True if the alias is too generic to be useful."""
    alias_lower = alias.lower().strip()
    words = alias_lower.split()
    # Too short
    if len(alias_lower) < 4:
        return True
    # All stopwords
    if all(w in _GENERIC_ALIAS_WORDS for w in words):
        return True
    # Single generic word
    if len(words) == 1 and alias_lower in _GENERIC_ALIAS_WORDS:
        return True
    # Looks like a sentence
    if len(words) > 5 and re.search(r'[.?!]$', alias.strip()):
        return True
    return False


def _extract_header_aliases(
    entity_name: str,
    source_text: str,
) -> list[str]:
    """Find section headers in source text that relate to the entity and use them as aliases."""
    name_lower = entity_name.lower()
    name_words = set(re.findall(r'[a-z0-9]{3,}', name_lower))

    if len(name_words) < 1:
        return []

    # Words that are too common in headers to indicate a meaningful relationship
    _common_header_words = {
        "experiment", "report", "summary", "result", "results",
        "analysis", "overview", "finding", "findings", "status",
        "gate", "gates", "gateway",
    }

    aliases: list[str] = []
    for match in re.finditer(r'^#{1,3}\s+(.+)$', source_text, re.MULTILINE):
        raw_title = match.group(1).strip()
        title = _clean_header(raw_title)
        if not title or len(title) < 4:
            continue
        if _is_generic_heading(title):
            continue
        title_lower = title.lower()
        # Skip headers that are about gates or are too generic/structural
        if re.search(r'\bgate[s]?\s*(\d+|—|–|-)', title_lower):
            continue
        # Skip headers that are purely about metrics/measurements
        if re.match(r'^(metrics|measurements|results|comparison|comparisons)\b', title_lower):
            continue
        title_words = set(re.findall(r'[a-z0-9]{3,}', title_lower))

        # Strategy 1: Entity name appears as substring in header
        if name_lower in title_lower or title_lower in name_lower:
            if not _is_generic_alias(title_lower):
                aliases.append(title_lower)
        # Strategy 2: At least 2 significant words shared (exclude common header words)
        else:
            significant_shared = (name_words & title_words) - _common_header_words
            if len(significant_shared) >= 2:
                if not _is_generic_alias(title_lower):
                    aliases.append(title_lower)
        # Strategy 3: Entity name minus "Experiment X.Y" prefix appears in header
        stripped_name = re.sub(r'^(experiment|exp)\s+[\d.]+\s*', '', name_lower, flags=re.IGNORECASE).strip()
        if stripped_name and len(stripped_name) > 4 and stripped_name in title_lower:
            if not _is_generic_alias(title_lower):
                aliases.append(title_lower)

    return aliases


def _extract_bold_aliases(
    entity_name: str,
    source_text: str,
) -> list[str]:
    """Find bold/strong text near entity mentions and use them as aliases."""
    name_lower = entity_name.lower()
    aliases: list[str] = []

    # Find all bold spans
    bold_spans = list(re.finditer(r'\*\*([^*\n]{3,60})\*\*', source_text))

    # Find entity mentions in text — use exact entity name match for high precision
    entity_exact_positions = [m.start() for m in re.finditer(
        re.escape(name_lower), source_text, re.IGNORECASE,
    )]
    # Also try constituent words (for partial name matching)
    name_words = re.findall(r'[a-z0-9]{3,}', name_lower)
    entity_word_positions: list[int] = []
    if len(name_words) >= 2:
        word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in name_words if len(w) >= 3) + r')\b',
            re.IGNORECASE,
        )
        entity_word_positions = [m.start() for m in word_pattern.finditer(source_text)]

    # Patterns to reject as bold aliases
    _numeric_pattern = re.compile(r'^[\d.,%+\-=()]+$')
    _table_value_pattern = re.compile(r'^[\d]+(?:\.[\d]+)?\s*%?$')
    _pass_fail_pattern = re.compile(r'^(pass|fail|passed|failed|✓|✗)$', re.IGNORECASE)
    _gate_pattern = re.compile(r'^gate\b', re.IGNORECASE)
    _metric_header = re.compile(r'^(metrics|measurements|comparison|comparisons)\b', re.IGNORECASE)

    for bold_match in bold_spans:
        bold_text = bold_match.group(1).strip()
        bold_text_lower = bold_text.lower().strip('`').strip()
        bold_pos = bold_match.start()

        # Skip purely numeric/percentage bold text (table values)
        if _numeric_pattern.match(bold_text_lower):
            continue
        if _table_value_pattern.match(bold_text_lower):
            continue
        if _pass_fail_pattern.match(bold_text_lower):
            continue
        if _gate_pattern.match(bold_text_lower):
            continue
        if _metric_header.match(bold_text_lower):
            continue
        if _is_generic_alias(bold_text_lower):
            continue
        if re.search(r'[.?!]$', bold_text):
            continue
        # Skip single-word bold that is just a number with unit
        if re.match(r'^[\d.]+x?\s*(improvement|infinite|boost)$', bold_text_lower, re.IGNORECASE):
            continue

        # Check if this bold text is near an EXACT entity name mention (within 150 chars)
        near_exact = any(abs(bold_pos - ep) < 150 for ep in entity_exact_positions)
        # Check proximity to word-level mentions only if entity has ≥2 significant words
        near_word = any(abs(bold_pos - ep) < 100 for ep in entity_word_positions)

        # Also check if bold text itself contains entity name or its words
        contains_entity = name_lower in bold_text_lower or any(
            w in bold_text_lower for w in name_words if len(w) >= 4
        )

        if near_exact or (near_word and contains_entity):
            if not _is_generic_alias(bold_text_lower):
                aliases.append(bold_text_lower)

    return aliases


def _extract_numeric_experiment_aliases(entity_name: str) -> list[str]:
    """Generate numeric variant aliases for Experiment entities.

    e.g., "Experiment 0.6" → ["experiment 0.6", "exp 0.6", "experiment 0 6", "exp 0 6"]
    """
    aliases: list[str] = []
    name_lower = entity_name.lower().strip()

    # Match patterns like "Experiment 0.6", "Exp 0.11", "Experiment 0.13A"
    m = re.match(
        r'(experiment|exp)\s+([\d]+)\.([\d]+[a-z]?)\s*(.*)',
        name_lower, re.IGNORECASE,
    )
    if m:
        prefix = m.group(1).lower()
        major = m.group(2)
        minor = m.group(3)
        suffix = m.group(4).strip() if m.group(4) else ""

        if suffix:
            # e.g., "experiment 0.6 validation" → "full validation"
            # But skip very long suffixes
            if len(suffix) <= 30:
                # Base variants
                variants = [
                    f"experiment {major}.{minor}",
                    f"exp {major}.{minor}",
                    f"experiment {major} {minor}",
                    f"exp {major} {minor}",
                    f"experiment_{major}_{minor}",
                    f"exp_{major}_{minor}",
                ]
            else:
                variants = [
                    f"experiment {major}.{minor}",
                    f"exp {major}.{minor}",
                    f"experiment_{major}_{minor}",
                    f"exp_{major}_{minor}",
                ]
        else:
            variants = [
                f"experiment {major}.{minor}",
                f"exp {major}.{minor}",
                f"experiment {major} {minor}",
                f"exp {major} {minor}",
                f"experiment_{major}_{minor}",
                f"exp_{major}_{minor}",
            ]

        aliases.extend(v for v in variants if v not in aliases)

    # Also handle simple "Experiment X" without minor version
    m2 = re.match(r'(experiment|exp)\s+(\d+)\s*$', name_lower, re.IGNORECASE)
    if m2:
        major = m2.group(2)
        variants = [
            f"experiment {major}",
            f"exp {major}",
        ]
        aliases.extend(v for v in variants if v not in aliases)

    return aliases


def _extract_property_aliases(
    properties: dict[str, Any] | None = None,
) -> list[str]:
    """Extract alias candidates from entity properties (key_finding, description).

    Uses simple noun-phrase extraction: finds capitalized multi-word phrases
    and key technical terms.
    """
    if not properties:
        return []

    text_parts: list[str] = []
    for prop_name in ("key_finding", "description"):
        val = properties.get(prop_name, "")
        if isinstance(val, str) and val:
            text_parts.append(val)

    if not text_parts:
        return []

    combined = " ".join(text_parts)
    aliases: list[str] = []

    # Strategy A: Extract multi-word capitalized phrases (likely domain terms)
    # e.g., "Chain-Set BCE Retriever" → alias
    for m in re.finditer(
        r'\b([A-Z][a-z]+(?:\s*[-—–]\s*[A-Z][a-z]+)+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]{2,})?)',
        combined,
    ):
        phrase = m.group(1).strip()
        if len(phrase) >= 5 and not _is_generic_alias(phrase):
            aliases.append(phrase.lower().replace("—", " ").replace("–", " ").replace("-", " "))

    # Strategy B: Extract noun phrases — pairs of consecutive non-trivial words
    # For key_finding like "Chain-set BCE retriever solves multi-hop"
    # → "chain set", "bce retriever" (filtering out verb-final bigrams)
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9]{2,}', combined)
    stop_words = {"the", "and", "but", "for", "with", "from", "that",
                   "this", "not", "are", "was", "has", "had", "can",
                   "all", "its", "our", "via", "per", "out", "any",
                   "each", "use", "new", "old", "two", "one", "over",
                   "more", "very", "also", "just", "than", "then",
                   "only", "vs", "not", "nor", "yet", "now", "much",
                   "many", "into", "some", "like", "well"}
    # Verbs and other words that don't make good alias components
    verb_endings = {"solves", "solve", "shows", "show", "proves", "prove",
                    "finds", "find", "makes", "make", "gives", "give",
                    "gets", "get", "takes", "take", "works", "work",
                    "does", "goes", "go", "see", "saw", "say", "said",
                    "let", "put", "set", "run", "ran", "test", "tests",
                    "confirm", "confirms", "confirming",
                    "means", "mean", "needs", "need", "fix", "fixes",
                    "tolerates", "tolerate", "collapses", "collapse",
                    "improves", "improve", "reaches", "reach",
                    "achieves", "achieve", "requires", "require",
                    "validates", "validate", "enables", "enable",
                    "eliminates", "eliminate", "provides", "provide"}
    # Single measurements that are too generic
    metric_words = {"recall", "accuracy", "precision", "f1", "loss",
                    "coverage", "score", "rate", "performance"}

    # Bigrams of significant words
    for i in range(len(words) - 1):
        w1 = words[i].lower()
        w2 = words[i + 1].lower()
        if w1 in stop_words or w2 in stop_words:
            continue
        # Skip bigrams starting or ending with a verb (e.g., "solves multi", "retriever solves")
        if w1 in verb_endings or w2 in verb_endings:
            continue
        # Skip bigrams that are just metric pairs (e.g., "accuracy recall")
        if w1 in metric_words and w2 in metric_words:
            continue
        # Skip bigrams of two short words (< 4 chars each)
        if len(w1) < 4 and len(w2) < 4:
            continue
        bigram = f"{w1} {w2}"
        if not _is_generic_alias(bigram) and len(bigram) >= 5:
            aliases.append(bigram)

    return aliases


def _extract_auto_aliases(
    entity_name: str,
    source_text: str = "",
    entity_type: str = "Entity",
    properties: dict[str, Any] | None = None,
    max_aliases: int = 5,
) -> list[str]:
    """
    Extract automatic aliases from corpus text for an entity.

    Uses multiple strategies:
    1. Section headers that mention or relate to the entity
    2. Bold/strong text near entity mentions
    3. Numeric variants for Experiment entities
    4. Key noun phrases from entity properties (key_finding, description)

    Args:
        entity_name: The display name of the entity
        source_text: The full text of the source document (for strategies 1-2)
        entity_type: The entity type (e.g., "Experiment", "Concept")
        properties: Optional dict of entity properties (for strategy 4)
        max_aliases: Maximum number of aliases to return

    Returns:
        A list of auto-generated aliases (max max_aliases).
    """
    all_aliases: list[str] = []

    if source_text:
        # Strategy 1: Section headers
        header_aliases = _extract_header_aliases(entity_name, source_text)
        all_aliases.extend(header_aliases)

        # Strategy 2: Bold text near entity mentions
        if len(all_aliases) < max_aliases * 2:  # Allow buffer for dedup
            bold_aliases = _extract_bold_aliases(entity_name, source_text)
            all_aliases.extend(bold_aliases)

    # Strategy 3: Numeric variants for Experiment entities
    if entity_type == "Experiment":
        num_aliases = _extract_numeric_experiment_aliases(entity_name)
        all_aliases.extend(num_aliases)

    # Strategy 4: Property-based aliases
    if properties:
        prop_aliases = _extract_property_aliases(properties)
        all_aliases.extend(prop_aliases)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for alias in all_aliases:
        norm = alias.lower().strip().replace(" ", "_").replace("-", "_")
        # Also skip aliases that are just the entity name itself
        entity_norm = entity_name.lower().strip().replace(" ", "_").replace("-", "_")
        if norm == entity_norm:
            continue
        if norm not in seen:
            seen.add(norm)
            deduped.append(alias)

    return deduped[:max_aliases]
