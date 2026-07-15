"""Architecture enforcement: nexus/ must not import stack/."""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
NEXUS_DIR = REPO_ROOT / "nexus"

def test_nexus_never_imports_stack():
    violations: list[str] = []
    for py_file in sorted(NEXUS_DIR.rglob("*.py")):
        rel = str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for node in ast.walk(ast.parse(source, filename=rel)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("stack"):
                        violations.append(f"{rel}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("stack"):
                    violations.append(f"{rel}:{node.lineno}: from {node.module}")
    assert not violations, (
        "ARCHITECTURE VIOLATION: nexus/ -> stack/\n" +
        "\n".join(f"  {v}" for v in violations)
    )
