"""
NEXUS end-to-end answer pipeline.

Single entry point: answer_question() runs the full pipeline:
  parse → traverse → evidence → prompt → model → verify

Handles edge cases: empty graph, no entities found, no paths, etc.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.graph import EDGE_TYPES
from nexus.graph.traversal import traverse_with_intent
from nexus.query.parser import parse_question
from nexus.reasoning.evidence_builder import (
    build_evidence, build_evidence_pack, build_zero_hop_pack,
)
from nexus.reasoning.prompt_template import build_prompt, _find_question_entity
from nexus.reasoning.model_interface import (
    DummyModel, ModelInterface, SynthesizingModel, get_available_model,
)
from nexus.reasoning.verifier import Verifier, VerificationResult
from nexus.reasoning.audit import build_reasoning_audit
from nexus.reasoning.post_edit import edit_answer
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG

# ── Insufficiency detection patterns ────────────────────────────────
_INSUFFICIENCY_PATTERNS = [
    "insufficient evidence",
    "not enough evidence",
    "cannot answer from the evidence",
    "no evidence",
    "unable to determine",
]


def _explicit_relation_claim(question: str) -> tuple[str, str, str] | None:
    """Parse the benchmark/API form ``Does A have the rel relation to B?``.

    This narrow parser is intentionally fail-closed and only accepts the
    registered relation vocabulary.  It prevents an absent relation from
    being answered with unrelated facts about either endpoint.
    """
    match = re.fullmatch(
        r"\s*does\s+(.+?)\s+have\s+the\s+([a-z_]+)\s+relation\s+to\s+(.+?)\?\s*",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    source, relation, target = match.group(1), match.group(2).casefold(), match.group(3)
    if relation not in EDGE_TYPES:
        return None
    return source, relation, target


def _graph_has_relation(graph: InMemoryGraphStore, source: str, relation: str, target: str) -> bool:
    return any(edge.type == relation and edge.target == target for edge in graph.get_outgoing(source))


def is_insufficient_answer(answer: str) -> bool:
    """Detect refusal / insufficiency in a model answer using known patterns."""
    lower = answer.lower()
    return any(pat in lower for pat in _INSUFFICIENCY_PATTERNS)


def _attach_reasoning_audit(
    result: dict[str, Any],
    graph: InMemoryGraphStore,
    paths: list[Any],
    config: NEXUSConfig,
    max_paths: int,
) -> dict[str, Any]:
    """Attach the deterministic audit without changing answer semantics."""
    audit = build_reasoning_audit(
        paths[:max_paths],
        graph,
        result.get("evidence_pack", {}),
        result.get("verification"),
        result.get("answer", ""),
        answer_threshold=config.readiness_answer_threshold,
        conditional_threshold=config.readiness_conditional_threshold,
    )
    result["reasoning_audit"] = audit.to_dict()
    return result


def _tier3_generate_answer(
    question: str,
    direct_pack: dict[str, Any],
    model: ModelInterface,
    verifier: Verifier,
    config: NEXUSConfig,
) -> str:
    """Generate a Tier 3 answer from a 0-hop evidence pack.

    Routes to the configured tier3_backend:
      - "synth" (default): SynthesizingModel — template-based, never refuses.
      - "llm_no_refusal": Standard LLM model (passed as ``model``).
    """
    if config.tier3_backend == "synth":
        # Build a prompt the synthesizer can parse.
        # Include metric term from question for metric-aware selection.
        try:
            from nexus.query.parser import extract_metric_term
            metric_term = extract_metric_term(question) or ""
        except ImportError:
            metric_term = ""
        prompt_direct = _build_synth_prompt(question, direct_pack, metric_term)
        synth = SynthesizingModel()
        return synth.generate(prompt_direct)
    else:
        # llm_no_refusal: standard LLM with the evidence pack.
        # For A/B testing later — currently uses the same build_prompt
        # but a future prompt template variant can omit the insufficiency
        # instruction.
        evidence_json_direct = json.dumps(direct_pack, indent=2)
        prompt_direct = build_prompt(question, evidence_json_direct)
        if config.post_edit_enabled:
            post_edit_direct = edit_answer(model.generate(prompt_direct), direct_pack)
            return post_edit_direct["answer"]
        else:
            return model.generate(prompt_direct)


def _build_synth_prompt(
    question: str,
    pack: dict[str, Any],
    metric_term: str = "",
) -> str:
    """Build a prompt the SynthesizingModel can parse from a 0-hop evidence pack.

    Formats node_facts, numbers, and (if metric_term is given) numbers_by_metric
    into sections the synthesizer's section extractors already understand.
    """
    lines: list[str] = []
    lines.append(f"QUESTION: {question}")
    lines.append("")

    # Key findings (maps to "Key findings from evidence nodes:" section)
    node_facts = pack.get("node_facts", [])
    if node_facts:
        lines.append("Key findings from evidence nodes:")
        for nf in node_facts:
            lines.append(f"- {nf['text']}")
        lines.append("")

    # Numbers — include both raw and metric-keyed
    numbers = pack.get("numbers", [])
    if numbers:
        lines.append("Extracted facts:")
        for num in numbers:
            parts = []
            for k, v in num.items():
                parts.append(f"{k}: {v}")
            lines.append(f"- {', '.join(parts)}")
        lines.append("")

    # Metric-specific numbers (Phase 4 metric-aware selection)
    if metric_term and metric_term in pack.get("numbers_by_metric", {}):
        values = pack["numbers_by_metric"][metric_term]
        lines.append(f"Metric: {metric_term} = {values}")
        lines.append("")

    # Paths — empty for 0-hop
    lines.append("Knowledge graph paths:")
    lines.append("(No paths available — evidence extracted directly from entity nodes)")
    lines.append("")

    lines.append("ANSWER:")
    return "\n".join(lines)


def _merge_resolved_entity_evidence(
    evidence_pack: dict[str, Any],
    direct_pack: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Surface question-matching facts from resolved but disconnected nodes."""
    if not direct_pack:
        return evidence_pack
    stopwords = {
        "the", "and", "for", "from", "what", "which", "that", "this", "with",
        "does", "did", "was", "were", "how", "many", "about", "have", "into",
    }
    q_terms = {
        token for token in re.findall(r"[A-Za-zÀ-ž0-9_%+-]+", question.casefold())
        if len(token) >= 3 and token not in stopwords
    }
    matching: list[tuple[int, dict[str, Any]]] = []
    for fact in direct_pack.get("node_facts", []):
        text = str(fact.get("text", ""))
        terms = set(re.findall(r"[A-Za-zÀ-ž0-9_%+-]+", text.casefold()))
        overlap = len(q_terms & terms)
        if overlap >= 2:
            matching.append((overlap, fact))
    matching.sort(key=lambda item: item[0], reverse=True)

    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in [item[1] for item in matching[:3]] + list(evidence_pack.get("node_facts", [])):
        normalized = re.sub(r"\s+", " ", str(fact.get("text", ""))).strip().casefold()
        if normalized and normalized not in seen:
            combined.append(fact)
            seen.add(normalized)
        if len(combined) >= 8:
            break
    evidence_pack["node_facts"] = combined
    evidence_pack["sources"] = sorted(set(evidence_pack.get("sources", [])) | set(direct_pack.get("sources", [])))
    return evidence_pack


def answer_question(
    question: str,
    graph: InMemoryGraphStore,
    model: ModelInterface | None = None,
    verifier: Verifier | None = None,
    max_depth: int | None = None,
    beam_width: int | None = None,
    max_paths: int = 7,
    config: NEXUSConfig = DEFAULT_CONFIG,
    embedding_index=None,
    dialogue_state=None,
    entry_nodes_override: list[str] | None = None,
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
       embedding_index: Optional NodeEmbeddingIndex for semantic entity resolution.
       dialogue_state: Optional DialogueState for anaphora/ellipsis resolution.
       entry_nodes_override: When provided, use these entity IDs for traversal
          instead of the parser's results. Parser is still called for intent
          detection. The override controls which entities actually reach
          traversal and evidence building.
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
        "path_scores": [],
        "post_edit_changes": None,
        "cascade_level": 0,
        "resolution_confidence": 0.0,
        "reasoning_audit": {},
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
        return _attach_reasoning_audit(result, graph, [], config, max_paths)

    # ── Step 1: Parse ──
    t0 = time.perf_counter()
    parsed = parse_question(question, graph, cutoff=0.6, config=config,
                            embedding_index=embedding_index,
                            dialogue_state=dialogue_state)
    timing["parse_time"] = round(time.perf_counter() - t0, 6)

    result["parsed_query"] = parsed
    result["entity_resolution_method"] = parsed.resolution_method
    result["entities_resolved"] = bool(parsed.entity_ids)

    # ── Apply entry_nodes_override for external resolvers (ER3, etc.) ──
    if entry_nodes_override is not None:
        parsed.entity_ids = list(entry_nodes_override)[:config.max_entry_nodes]
        result["entity_resolution_method"] = "external_override"
        result["parsed_query"] = parsed

    # Map resolution method to a confidence score
    _resolution_confidence_map = {"alias": 1.0, "embedding": 0.8, "fuzzy": 0.6, "none": 0.0}
    result["resolution_confidence"] = _resolution_confidence_map.get(
        parsed.resolution_method, 0.0,
    )

    # Edge case: no entities found
    if not parsed.entity_ids:
        result["answer"] = "Insufficient evidence to answer. No relevant entities found in the question."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, [], config, max_paths)

    explicit_relation = _explicit_relation_claim(question)
    if explicit_relation and not _graph_has_relation(graph, *explicit_relation):
        result["answer"] = "Insufficient evidence for that relation."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, [], config, max_paths)

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
    result["path_scores"] = [round(path.score, 6) for path in paths]

    # Class B fix: path_count == 0 but entities resolved (e.g., fuzzy match, no
    # outgoing edges).  Build a 0-hop evidence pack directly from entity nodes
    # instead of refusing.
    if not paths and parsed.entity_ids:
        direct_pack = build_zero_hop_pack(graph, parsed.entity_ids, question=question)
        if direct_pack:
            answer_direct = _tier3_generate_answer(
                question, direct_pack, model, verifier, config,
            )
            result["answer"] = answer_direct
            result["raw_answer"] = answer_direct
            result["evidence_pack"] = direct_pack
            result["cascade_level"] = 3
            result["verification"] = verifier.verify(answer_direct, direct_pack)
            result["timing"] = timing
            return _attach_reasoning_audit(result, graph, [], config, max_paths)

    # Edge case: no paths found
    if not paths:
        result["answer"] = "Insufficient evidence to answer. No traversal paths found connecting the identified entities."
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, [], config, max_paths)

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
    direct_pack = build_zero_hop_pack(graph, parsed.entity_ids, question=question)
    evidence_pack = _merge_resolved_entity_evidence(evidence_pack, direct_pack, question)
    evidence_json = json.dumps(evidence_pack, indent=2, ensure_ascii=False)
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

    # Cascade level tracking: tier 1 (filtered evidence) succeeded
    if not is_insufficient_answer(answer):
        result["cascade_level"] = 1

    # ── Step 5.6: Cascade fallback — if LLM refuses but we have paths,
    #               retry with unfiltered evidence (no target_entity filter) ──
    if is_insufficient_answer(answer) and len(paths) > 0:
        t_retry = time.perf_counter()
        evidence_json_retry = build_evidence(
            question, paths, graph, max_paths=max_paths,
            question_intent=parsed.intent, target_entity=None,
        )
        evidence_pack_retry = build_evidence_pack(
            question, paths, graph,
            question_intent=parsed.intent, target_entity=None,
        )
        direct_pack_retry = build_zero_hop_pack(graph, parsed.entity_ids, question=question)
        evidence_pack_retry = _merge_resolved_entity_evidence(
            evidence_pack_retry, direct_pack_retry, question,
        )
        evidence_json_retry = json.dumps(evidence_pack_retry, indent=2, ensure_ascii=False)
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

        # Cascade level tracking: tier 2 (unfiltered evidence) succeeded
        if not is_insufficient_answer(answer):
            result["cascade_level"] = 2

    # ── Step 5.7: Tier 3 cascade — 0-hop evidence from entity nodes ──
    # When both tier 1 (filtered) and tier 2 (unfiltered) produce refusals
    # AND path_count > 0 (Class A refusals), build evidence directly from
    # the resolved entity nodes' own properties — no traversal needed.
    if is_insufficient_answer(answer) and len(paths) > 0:
        t_level3 = time.perf_counter()
        direct_pack = build_zero_hop_pack(graph, parsed.entity_ids, question=question)

        if direct_pack:
            answer = _tier3_generate_answer(
                question, direct_pack, model, verifier, config,
            )

            timing["cascade_level3_time"] = round(time.perf_counter() - t_level3, 6)

            result["answer"] = answer
            result["raw_answer"] = answer
            result["evidence_pack"] = direct_pack
            result["cascade_level"] = 3
        else:
            timing["cascade_level3_time"] = round(time.perf_counter() - t_level3, 6)

    # ── Step 6: Verify ──
    t0 = time.perf_counter()
    verification = verifier.verify(answer, evidence_pack)
    timing["verify_time"] = round(time.perf_counter() - t0, 6)
    result["verification"] = verification

    # Store timing breakdown and prompt tokens for cost estimation
    result["timing"] = timing
    if "prompt_text" not in result:
        result["prompt_text"] = prompt

    return _attach_reasoning_audit(result, graph, paths, config, max_paths)


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
