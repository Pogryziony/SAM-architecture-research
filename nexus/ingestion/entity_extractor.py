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


def extract_from_markdown(text: str, source_path: str) -> list[dict[str, Any]]:
    """
    Extract entities from markdown text using rule-based patterns.

    Detects:
    - Section headers as potential entity/concept names
    - Code references (backtick-wrapped)
    - Explicit entity mentions in structured sections
    """
    entities = []

    # Extract section headers (## Title)
    for match in re.finditer(r'^#{1,3}\s+(.+)$', text, re.MULTILINE):
        title = match.group(1).strip()
        # Skip generic headers
        if title.lower() in ("introduction", "overview", "conclusion", "summary", "references"):
            continue
        entities.append({
            "name": title,
            "type": _infer_type_from_header(title),
            "source": source_path,
            "line": text[:match.start()].count('\n') + 1,
        })

    # Extract backtick-wrapped code references
    for match in re.finditer(r'`([A-Za-z_][A-Za-z0-9_\.]+)`', text):
        ref = match.group(1)
        if len(ref) > 3 and not ref.startswith("http"):
            entities.append({
                "name": ref,
                "type": _infer_type_from_name(ref),
                "source": source_path,
                "line": text[:match.start()].count('\n') + 1,
            })

    return entities


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
    if "test" in name.lower():
        return "TestCase"
    if name.endswith(".py") or name.endswith(".js") or name.endswith(".ts"):
        return "CodeFile"
    if name.endswith(".md"):
        return "Document"
    return "Entity"


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
