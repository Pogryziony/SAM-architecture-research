"""Architecture enforcement: nexus/ must not import stack/."""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
NEXUS_DIR = REPO_ROOT / "nexus"

_ALLOWED: dict[str, set[str]] = {
    "nexus/pipeline/runner.py": {
        "stack.encoder.entity_ranker_v3",
        "stack.encoder.canonical_mapping",
        "stack.encoder.entity_text",
    },
    "nexus/query/parser.py": {
        "stack.normalization.lemmatizer",
        "stack.dialogue.state",
    },
}


def test_nexus_never_imports_stack():
    violations: list[str] = []
    for py_file in sorted(NEXUS_DIR.rglob("*.py")):
        rel = str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
        allowed = _ALLOWED.get(rel, set())
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for node in ast.walk(ast.parse(source, filename=rel)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("stack") and alias.name not in allowed:
                        violations.append(f"{rel}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("stack") and node.module not in allowed:
                    violations.append(f"{rel}:{node.lineno}: from {node.module}")
    assert not violations, (
        "ARCHITECTURE VIOLATION: nexus/ -> stack/\n" +
        "\n".join(f"  {v}" for v in violations)
    )
