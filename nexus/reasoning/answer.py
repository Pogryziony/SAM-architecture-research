"""
NEXUS end-to-end answer pipeline.

Single entry point: answer_question() runs the full pipeline:
  parse → traverse → evidence → prompt → model → verify

Handles edge cases: empty graph, no entities found, no paths, etc.
"""

from __future__ import annotations

import re
import time
from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import traverse_with_intent
from nexus.query.parser import parse_question
from nexus.reasoning.evidence_builder import build_evidence, build_evidence_pack
from nexus.reasoning.prompt_template import build_prompt, _find_question_entity
from nexus.reasoning.model_interface import (
    DummyModel, ModelInterface, get_available_model,
)
from nexus.reasoning.verifier import Verifier, VerificationResult
from nexus.reasoning.post_edit import edit_answer
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG


def answer_question(
    question: str,
    graph: InMemoryGraphStore,
    model: ModelInterface | None = None,
    verifier: Verifier | None = None,
    max_depth: int | None = None,
    beam_width: int | None = None,
    max_paths: int = 7,
    config: NEXUSConfig = DEFAULT_CONFIG,
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
       verifier: Verifier instance (defaults to Verifier with threshold from config)
       max_depth: Maximum traversal depth (default from config)
       beam_width: Beam width for search (default from config)
       max_paths: Maximum paths to include in evidence
       config: NEXUSConfig with tunable parameters

    Returns:
       Dict with keys:
           - question: original question
           - answer: model-generated answer text
           - evidence_pack: dict with paths, facts, sources
           - verification: VerificationResult
           - parsed_query: ParsedQuery from the parser
           - path_count: number of traversal paths found
    """
    if max_depth is None:
       max_depth = config.max_depth
    if beam_width is None:
       beam_width = config.beam_width

    if model is None:
       model = get_available_model()
    if verifier is None:
       verifier = Verifier(hallucination_threshold=config.hallucination_threshold)

    result: dict[str, Any] = {
        "question": question,
        "answer": "",
        "raw_answer": "",
        "evidence_pack": {},
        "verification": None,
        "parsed_query": None,
        "path_count": 0,
        "post_edit_changes": None,
    }

    # Per-step timing breakdown
    timing: dict[str, float] = {}

    # ── Edge case: empty graph ──
    if graph.node_count == 0:
        result["answer"] = "Insufficient evidence to answer. The knowledge graph is empty."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        result["timing"] = timing
        return result

    # ── Step 1: Parse ──
    t0 = time.perf_counter()
    parsed = parse_question(question, graph, cutoff=0.6, config=config)
    timing["parse_time"] = round(time.perf_counter() - t0, 6)

    result["parsed_query"] = parsed
    result["entity_resolution_method"] = parsed.resolution_method

    # Edge case: no entities found
    if not parsed.entity_ids:
        result["answer"] = "Insufficient evidence to answer. No relevant entities found in the question."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        result["timing"] = timing
        return result

    # ── Step 2: Traverse ──
    t0 = time.perf_counter()
    query_entities = set(parsed.entity_ids)
    paths = traverse_with_intent(
        graph=graph,
        entry_nodes=parsed.entity_ids,
        query_entities=query_entities,
        intent=parsed.intent,
        max_depth=max_depth,
        beam_width=beam_width,
        config=config,
    )
    timing["traverse_time"] = round(time.perf_counter() - t0, 6)
    result["path_count"] = len(paths)

    # Edge case: no paths found
    if not paths:
        result["answer"] = "Insufficient evidence to answer. No traversal paths found connecting the identified entities."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        result["timing"] = timing
        return result

    # ── Step 3: Build evidence ──
    t0 = time.perf_counter()
    # Determine target entity for factual questions to filter evidence.
    # Use _find_question_entity from prompt_template which handles compound
    # node IDs (e.g., "chainretrieval" matching "chain" + "retriever").
    target_entity = None
    if parsed.intent == "factual_lookup":
        # Build minimal node dicts from paths for _find_question_entity
        node_dicts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in paths:
            nodes: list[dict[str, Any]] = []
            for step in p.steps:
                for nid in (step.from_node, step.to_node):
                    if nid not in seen:
                        seen.add(nid)
                        nodes.append({"id": nid})
            if nodes:
                node_dicts.append({"nodes": nodes})
        target_entity = _find_question_entity(question, node_dicts)

    evidence_json = build_evidence(
        question, paths, graph, max_paths=max_paths,
        question_intent=parsed.intent, target_entity=target_entity,
    )
    evidence_pack = build_evidence_pack(
        question, paths, graph,
        question_intent=parsed.intent, target_entity=target_entity,
    )
    timing["evidence_time"] = round(time.perf_counter() - t0, 6)
    result["evidence_pack"] = evidence_pack

    # ── Step 4: Build prompt ──
    t0 = time.perf_counter()
    prompt = build_prompt(question, evidence_json)
    timing["prompt_time"] = round(time.perf_counter() - t0, 6)

    # ── Step 5: Generate answer ──
    t0 = time.perf_counter()
    raw_answer = model.generate(prompt)
    timing["generate_time"] = round(time.perf_counter() - t0, 6)
    result["raw_answer"] = raw_answer

    # ── Step 5.5: Post-edit — fix hallucinated numbers (disabled by default) ──
    # Post-edit masks the model's true accuracy; enable only for explicit experiments.
    t0 = time.perf_counter()
    if config.post_edit_enabled:
        post_edit_result = edit_answer(raw_answer, evidence_pack)
        timing["post_edit_time"] = round(time.perf_counter() - t0, 6)
        answer = post_edit_result["answer"]
        result["post_edit_changes"] = {
            "numbers_fixed": post_edit_result["numbers_fixed"],
            "numbers_removed": post_edit_result["numbers_removed"],
            "changes": post_edit_result["changes"],
        }
    else:
        answer = raw_answer
        timing["post_edit_time"] = 0.0
        result["post_edit_changes"] = None
    result["answer"] = answer

    # ── Step 5.6: Cascade fallback — if LLM refuses but we have paths,
    #               retry with unfiltered evidence (no target_entity filter) ──
    if "insufficient evidence" in answer.lower() and len(paths) > 0:
        t_retry = time.perf_counter()
        evidence_json_retry = build_evidence(
            question, paths, graph, max_paths=max_paths,
            question_intent=parsed.intent, target_entity=None,
        )
        evidence_pack_retry = build_evidence_pack(
            question, paths, graph,
            question_intent=parsed.intent, target_entity=None,
        )
        prompt_retry = build_prompt(question, evidence_json_retry)
        raw_answer_retry = model.generate(prompt_retry)

        if config.post_edit_enabled:
            post_edit_retry = edit_answer(raw_answer_retry, evidence_pack_retry)
            answer_retry = post_edit_retry["answer"]
        else:
            answer_retry = raw_answer_retry

        timing["cascade_retry_time"] = round(time.perf_counter() - t_retry, 6)

        # Use the retry answer, updating evidence_pack and prompt
        evidence_pack = evidence_pack_retry
        evidence_json = evidence_json_retry
        prompt = prompt_retry
        raw_answer = raw_answer_retry
        answer = answer_retry

        result["answer"] = answer
        result["raw_answer"] = raw_answer
        result["evidence_pack"] = evidence_pack

    # ── Step 6: Verify ──
    t0 = time.perf_counter()
    verification = verifier.verify(answer, evidence_pack)
    timing["verify_time"] = round(time.perf_counter() - t0, 6)
    result["verification"] = verification

    # Store timing breakdown and prompt tokens for cost estimation
    result["timing"] = timing
    result["prompt_text"] = prompt

    return result


def run_smoke_test():
    """Quick smoke test of the full pipeline with 3 questions on the real graph."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from nexus.graph.store import InMemoryGraphStore
    from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
    from nexus.utils.config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG

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
    verifier = Verifier(hallucination_threshold=config.hallucination_threshold)

    for i, q in enumerate(questions, 1):
        print(f"--- Question {i} ---")
        print(f"Q: {q}")

        result = answer_question(q, graph, model=model, verifier=verifier)

        parsed = result["parsed_query"]
        if parsed:
            print(f"Intent: {parsed.intent}, Entities: {parsed.entity_ids}")
        print(f"Paths found: {result['path_count']}")

        timing = result.get("timing", {})
        if timing:
            print(f"Timing: parse={timing.get('parse_time', 0)*1000:.0f}ms, "
                  f"traverse={timing.get('traverse_time', 0)*1000:.0f}ms, "
                  f"evidence={timing.get('evidence_time', 0)*1000:.0f}ms, "
                  f"prompt={timing.get('prompt_time', 0)*1000:.0f}ms, "
                  f"generate={timing.get('generate_time', 0)*1000:.0f}ms, "
                  f"verify={timing.get('verify_time', 0)*1000:.0f}ms")

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
