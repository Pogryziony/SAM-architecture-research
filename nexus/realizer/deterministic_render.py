"""Stage 7 deterministic realization gates (partial, zero-LLM).

Renders a structured answer from proof steps and verifies every generated
statement maps to a proof step. Identical structured input yields identical
output.
"""

from __future__ import annotations

from typing import Any, Sequence


def render_from_proof_steps(proof_steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Deterministically render L1 statements from proof steps.

    Multi-step answers use an oracle-friendly ``Yes. A R B, and B R C.`` shape
    so relation / multi-hop gold can score without an LLM.
    """
    statements: list[str] = []
    mappings: list[dict[str, Any]] = []
    cores: list[str] = []
    for index, step in enumerate(proof_steps):
        source = str(step.get("from_node") or step.get("source") or "")
        relation = str(step.get("relation") or "")
        target = str(step.get("to_node") or step.get("target") or "")
        if not (source and relation and target):
            continue
        core = f"{source} {relation} {target}"
        text = f"{core}."
        statements.append(text)
        cores.append(core)
        mappings.append(
            {
                "statement_index": len(statements) - 1,
                "statement": text,
                "proof_step_id": step.get("step_id") or f"proof_{index}",
                "from_node": source,
                "relation": relation,
                "to_node": target,
            }
        )
    if not cores:
        answer = ""
    elif len(cores) == 1:
        answer = f"Yes. {cores[0]}."
    else:
        answer = "Yes. " + ", and ".join(cores) + "."
    return {
        "answer": answer,
        "statements": statements,
        "statement_proof_map": mappings,
        "backend": "deterministic_render_v1",
    }


def validate_statement_proof_coverage(render: dict[str, Any]) -> list[str]:
    """Fail if any statement lacks a proof mapping."""
    errors: list[str] = []
    statements = list(render.get("statements") or [])
    mappings = list(render.get("statement_proof_map") or [])
    covered = {int(item.get("statement_index", -1)) for item in mappings}
    for index, statement in enumerate(statements):
        if index not in covered:
            errors.append(f"unmapped statement[{index}]: {statement}")
        else:
            mapped = next(
                item for item in mappings if int(item.get("statement_index", -1)) == index
            )
            if not mapped.get("proof_step_id"):
                errors.append(f"statement[{index}] missing proof_step_id")
    return errors
