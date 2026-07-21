"""Stage 7 deterministic realization gates."""

from __future__ import annotations

from nexus.realizer.deterministic_render import (
    render_from_proof_steps,
    validate_statement_proof_coverage,
)


def test_identical_structured_input_yields_identical_output():
    steps = [
        {
            "step_id": "p1",
            "from_node": "Exp_A",
            "relation": "validates",
            "to_node": "Concept_B",
        }
    ]
    a = render_from_proof_steps(steps)
    b = render_from_proof_steps(steps)
    assert a == b
    assert a["answer"] == "Exp_A validates Concept_B."
    assert validate_statement_proof_coverage(a) == []


def test_unmapped_statement_is_detected():
    render = {
        "statements": ["orphan claim."],
        "statement_proof_map": [],
    }
    errors = validate_statement_proof_coverage(render)
    assert errors
