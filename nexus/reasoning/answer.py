"""
NEXUS end-to-end answer pipeline.

Single entry point: answer_question() runs the full pipeline:
  parse → traverse → evidence → prompt → model → verify

Handles edge cases: empty graph, no entities found, no paths, etc.
"""

from __future__ import annotations

from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import traverse_with_intent
from nexus.query.parser import parse_question
from nexus.reasoning.evidence_builder import build_evidence, build_evidence_pack
from nexus.reasoning.prompt_template import build_prompt
from nexus.reasoning.model_interface import (
    DummyModel, ModelInterface, get_available_model,
)
from nexus.reasoning.verifier import Verifier, VerificationResult


def answer_question(
    question: str,
    graph: InMemoryGraphStore,
    model: ModelInterface | None = None,
    verifier: Verifier | None = None,
    max_depth: int = 4,
    beam_width: int = 5,
    max_paths: int = 5,
) -> dict[str, Any]:
    """
    Run the complete NEXUS pipeline on a natural language question.

    Pipeline:
      1. Parse the question → spot entities, detect intent
      2. Traverse the graph → beam search from entry nodes
      3. Build evidence pack → structured JSON from paths
      4. Build prompt → format evidence for the model
      5. Generate answer → run the model
      6. Verify answer → check against evidence for hallucinations

    Args:
        question: Natural language question
        graph: Populated graph store
       model: ModelInterface instance (defaults to auto-detected best model)
        verifier: Verifier instance (defaults to Verifier with 0.2 threshold)
        max_depth: Maximum traversal depth
        beam_width: Beam width for search
        max_paths: Maximum paths to include in evidence

    Returns:
        Dict with keys:
            - question: original question
            - answer: model-generated answer text
            - evidence_pack: dict with paths, facts, sources
            - verification: VerificationResult
            - parsed_query: ParsedQuery from the parser
            - path_count: number of traversal paths found
    """
    if model is None:
       model = get_available_model()
    if verifier is None:
        verifier = Verifier(hallucination_threshold=0.2)

    result: dict[str, Any] = {
        "question": question,
        "answer": "",
        "evidence_pack": {},
        "verification": None,
        "parsed_query": None,
        "path_count": 0,
    }

    # ── Edge case: empty graph ──
    if graph.node_count == 0:
        result["answer"] = "Insufficient evidence to answer. The knowledge graph is empty."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        return result

    # ── Step 1: Parse ──
    parsed = parse_question(question, graph, cutoff=0.6)
    result["parsed_query"] = parsed

    # Edge case: no entities found
    if not parsed.entity_ids:
        result["answer"] = "Insufficient evidence to answer. No relevant entities found in the question."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        return result

    # ── Step 2: Traverse ──
    query_entities = set(parsed.entity_ids)
    paths = traverse_with_intent(
        graph=graph,
        entry_nodes=parsed.entity_ids,
        query_entities=query_entities,
        intent=parsed.intent,
        max_depth=max_depth,
        beam_width=beam_width,
    )
    result["path_count"] = len(paths)

    # Edge case: no paths found
    if not paths:
        result["answer"] = "Insufficient evidence to answer. No traversal paths found connecting the identified entities."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        return result

    # ── Step 3: Build evidence ──
    evidence_json = build_evidence(question, paths, graph, max_paths=max_paths)
    evidence_pack = build_evidence_pack(question, paths, graph)
    result["evidence_pack"] = evidence_pack

    # ── Step 4: Build prompt ──
    prompt = build_prompt(question, evidence_json)

    # ── Step 5: Generate answer ──
    answer = model.generate(prompt)
    result["answer"] = answer

    # ── Step 6: Verify ──
    verification = verifier.verify(answer, evidence_pack)
    result["verification"] = verification

    return result


def run_smoke_test():
    """Quick smoke test of the full pipeline with 3 questions on the real graph."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from nexus.graph.store import InMemoryGraphStore
    from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR

    print("=" * 70)
    print("NEXUS Pipeline Smoke Test -- answer_question()")
    print("=" * 70)

    # Populate graph
    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)
    print(f"\nGraph: {graph.node_count} nodes, {graph.edge_count} edges\n")

    questions = [
        "What was the key finding of the chain-aware retrieval experiment?",
        "Why did the project pivot to NEXUS?",
        "What showed that the selector is the bottleneck?",
    ]

    model = DummyModel()
    verifier = Verifier(hallucination_threshold=0.2)

    for i, q in enumerate(questions, 1):
        print(f"--- Question {i} ---")
        print(f"Q: {q}")

        result = answer_question(q, graph, model=model, verifier=verifier)

        parsed = result["parsed_query"]
        if parsed:
            print(f"Intent: {parsed.intent}, Entities: {parsed.entity_ids}")
        print(f"Paths found: {result['path_count']}")

        print(f"\nAnswer:")
        print(result["answer"])

        v = result["verification"]
        if v:
            status = "PASS" if v.passed else "FAIL"
            print(f"\nVerification: {status} | supported={v.supported_count}, "
                  f"unsupported={len(v.unsupported_claims)}, "
                  f"rate={v.hallucination_rate:.2f}")
            if v.unsupported_claims:
                print("  Unsupported claims:")
                for claim in v.unsupported_claims:
                    print(f"    - {claim[:100]}{'...' if len(claim) > 100 else ''}")

        print()

    print("=" * 70)
    print("Smoke test complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
