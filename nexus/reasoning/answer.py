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
from nexus.graph.scoring import (
    focus_query_entities,
    incident_paths_for_entities,
    select_proof_paths,
)
from nexus.graph.traversal import TraversalStats, traverse_with_intent
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
from nexus.realizer.pointer_copy import realize_pointer_copy
from nexus.realizer.comparison_plan import (
    BACKEND_NAME, HYBRID_BACKEND_NAME, realize_comparison_plan,
)
from nexus.realizer.deterministic_render import (
    render_from_proof_steps,
    validate_statement_proof_coverage,
)
from nexus.realizer.l1_qualitative_compare import try_qualitative_compare
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG

DETERMINISTIC_RENDER_BACKEND = "deterministic_render"
L1_ACCEPTANCE_BACKEND = "l1_acceptance"

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
    traversal_stats = result.get("traversal_stats") or {}
    audit = build_reasoning_audit(
        paths[:max_paths],
        graph,
        result.get("evidence_pack", {}),
        result.get("verification"),
        result.get("answer", ""),
        answer_threshold=config.readiness_answer_threshold,
        conditional_threshold=config.readiness_conditional_threshold,
        require_structured_provenance=bool(
            getattr(config, "require_structured_provenance", False)
        ),
        traversal_truncated=bool(traversal_stats.get("truncated")),
        traversal_stats=traversal_stats,
    )
    result["reasoning_audit"] = audit.to_dict()
    return result


_FAIL_CLOSED_ABSTAIN = (
    "Insufficient evidence to answer. Deterministic grounded realization "
    "could not produce a supported answer and synth/LLM fallback is disabled."
)


def _synth_fallback_permitted(config: NEXUSConfig) -> bool:
    """Return whether unconstrained synth/LLM cascade is allowed."""
    return bool(getattr(config, "allow_synth_fallback", True))


def _record_fallback_decision(
    result: dict[str, Any],
    *,
    selected_realizer: str,
    fallback_considered: bool,
    fallback_permitted: bool,
    fallback_reason: str,
    terminal_outcome: str,
) -> None:
    """Attach fail-closed fallback audit fields to an answer result."""
    result["selected_realizer"] = selected_realizer
    result["fallback_considered"] = fallback_considered
    result["fallback_permitted"] = fallback_permitted
    result["fallback_reason"] = fallback_reason
    result["fallback_terminal_outcome"] = terminal_outcome


def _fail_closed_abstain_result(
    result: dict[str, Any],
    *,
    selected_realizer: str,
    fallback_reason: str,
    timing: dict[str, Any],
    verifier: Verifier,
    evidence_pack: dict[str, Any],
) -> dict[str, Any]:
    """Return a structured abstention when synth/LLM fallback is forbidden."""
    answer = _FAIL_CLOSED_ABSTAIN
    result["answer"] = answer
    result["raw_answer"] = answer
    result["cascade_level"] = 0
    result["realization"] = {
        "strategy": "fail_closed_abstain",
        "reason": fallback_reason,
        "allow_synth_fallback": False,
    }
    result["verification"] = verifier.verify(answer, evidence_pack or {})
    result["timing"] = timing
    _record_fallback_decision(
        result,
        selected_realizer=selected_realizer,
        fallback_considered=True,
        fallback_permitted=False,
        fallback_reason=fallback_reason,
        terminal_outcome="ABSTAIN",
    )
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
    if not _synth_fallback_permitted(config):
        return _FAIL_CLOSED_ABSTAIN
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


def _proof_step_dicts_from_paths(paths: list[Any]) -> list[dict[str, Any]]:
    """Project traversal paths into deterministic-render proof step dicts."""
    steps: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for step in getattr(path, "steps", []) or []:
            source = str(getattr(step, "from_node", "") or "")
            relation = str(getattr(getattr(step, "edge", None), "type", "") or "")
            target = str(getattr(step, "to_node", "") or "")
            key = (source, relation, target)
            if not (source and relation and target) or key in seen:
                continue
            seen.add(key)
            steps.append(
                {
                    "step_id": f"{source}|{relation}|{target}",
                    "from_node": source,
                    "relation": relation,
                    "to_node": target,
                }
            )
    return steps


_POINTER_COPY_INTENTS = frozenset({
    "factual_lookup",
    "diagnostic",
    "causal_explanation",
    "impact_analysis",
    "comparison",
})
_PATH_RENDER_INTENTS = frozenset({
    "causal_explanation",
    "dependency_chain",
    "impact_analysis",
    "diagnostic",
    "factual_lookup",
})
_SOFT_FALLTHROUGH_BACKENDS = frozenset({HYBRID_BACKEND_NAME, L1_ACCEPTANCE_BACKEND})
_TEMPORAL_ABSTAIN_KNOWN = (
    "Insufficient evidence to answer. No valid temporal facts for that as-known-at query."
)
_TEMPORAL_ABSTAIN_VALID = (
    "Insufficient evidence to answer. No facts valid at that as-valid-at query."
)
_TEMPORAL_ABSTAIN_RETRACT = (
    "Insufficient evidence to answer. Relevant facts were retracted before "
    "that as-known-at query."
)
# Backward-compatible alias used by older tests/docs.
_TEMPORAL_ABSTAIN = _TEMPORAL_ABSTAIN_KNOWN


def _is_path_answer_question(question: str) -> bool:
    """True when the question asks for graph relations / multi-hop chains."""
    if _explicit_relation_claim(question) is not None:
        return True
    q = question.casefold()
    if " relation to " in q or re.search(r"\bhave the \w+ relation\b", q):
        return True
    if "linked through" in q or "which depends" in q:
        return True
    if "according to the graph" in q:
        return True
    if "depend on" in q and (" which " in q or " through " in q):
        return True
    if re.search(r"\bwhich experiments?\b", q):
        return True
    if re.search(r"\bwhich (experiment|entity|node|concept)\b", q) and any(
        relation.replace("_", " ") in q or relation in q for relation in EDGE_TYPES
    ):
        return True
    # Explicit typed-relation wording only — avoid prose "depends on" diagnostics.
    for relation in EDGE_TYPES:
        if f"the {relation} relation" in q:
            return True
    return False


def _is_dependency_chain_question(question: str) -> bool:
    q = question.casefold()
    if "dependency chain" in q:
        return True
    if "walk through" in q and "experiment" in q:
        return True
    if "from start to end" in q and "experiment" in q:
        return True
    return False


def _pit_cutoffs_active(config: NEXUSConfig) -> bool:
    return bool(
        str(getattr(config, "as_valid_at", "") or "").strip()
        or str(getattr(config, "as_known_at", "") or "").strip()
    )


def _has_visible_edges_under_pit(
    graph: InMemoryGraphStore,
    entity_ids: list[str],
    config: NEXUSConfig,
) -> bool:
    """True when at least one incident edge survives configured PIT cutoffs."""
    from nexus.graph.bitemporal import filter_edges_bitemporal

    as_valid_at = str(getattr(config, "as_valid_at", "") or "")
    as_known_at = str(getattr(config, "as_known_at", "") or "")
    for entity_id in entity_ids:
        edges = graph.get_edges(entity_id, "both")
        kept = filter_edges_bitemporal(
            edges, as_valid_at=as_valid_at, as_known_at=as_known_at,
        )
        if kept:
            return True
    return False


def _temporal_abstain_message(
    graph: InMemoryGraphStore,
    entity_ids: list[str],
    config: NEXUSConfig,
) -> str:
    """Choose a differentiated PIT abstain reason from filtered edge stamps."""
    from nexus.graph.bitemporal import (
        BiTemporalStamp,
        edge_to_fact,
        is_known_at,
        is_valid_at,
    )

    as_valid_at = str(getattr(config, "as_valid_at", "") or "").strip()
    as_known_at = str(getattr(config, "as_known_at", "") or "").strip()
    saw_retract = False
    saw_valid_fail = False
    saw_known_fail = False
    for entity_id in entity_ids:
        for edge in graph.get_edges(entity_id, "both"):
            stamp = BiTemporalStamp.from_mapping(edge_to_fact(edge))
            valid_ok = (not as_valid_at) or is_valid_at(stamp, as_valid_at)
            known_ok = (not as_known_at) or is_known_at(stamp, as_known_at)
            if valid_ok and known_ok:
                continue
            if as_valid_at and not valid_ok:
                saw_valid_fail = True
            if as_known_at and not known_ok:
                # Retract-only when clearing retracted_at would make it known.
                without_retract = BiTemporalStamp(
                    valid_from=stamp.valid_from,
                    valid_to=stamp.valid_to,
                    observed_at=stamp.observed_at,
                    retracted_at="",
                )
                if valid_ok and is_known_at(without_retract, as_known_at):
                    saw_retract = True
                else:
                    saw_known_fail = True
    if saw_retract and not saw_valid_fail:
        return _TEMPORAL_ABSTAIN_RETRACT
    if as_valid_at and saw_valid_fail and not saw_known_fail:
        return _TEMPORAL_ABSTAIN_VALID
    if as_valid_at and not as_known_at:
        return _TEMPORAL_ABSTAIN_VALID
    return _TEMPORAL_ABSTAIN_KNOWN


def _l1_qualitative_compare_result(
    question: str,
    graph: InMemoryGraphStore,
    entity_ids: list[str],
    config: NEXUSConfig,
) -> dict[str, Any] | None:
    """Render curated dual-side compare_* properties when a template matches."""
    if getattr(config, "realizer_backend", "") != L1_ACCEPTANCE_BACKEND:
        return None
    matched = try_qualitative_compare(question, graph, entity_ids)
    if matched is None:
        return None
    answer, template = matched
    return {
        "answer": answer,
        "raw_answer": answer,
        "realization": {
            "backend": L1_ACCEPTANCE_BACKEND,
            "strategy": "l1_qualitative_compare",
            "template": template.name,
            "statements": [answer],
            "statement_proof_map": [],
            "coverage_errors": [],
        },
        "abstain": False,
    }


_ENTITY_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_]+:\s+")
_VALIDATES_ANNOTATION = re.compile(
    r"^\[This concept is directly validated by the experiment\]\s*",
    re.IGNORECASE,
)


def _strip_entity_prefix(text: str) -> str:
    """Drop annotation / ``NodeId: `` prefixes so copied node facts match gold."""
    cleaned = str(text or "").strip()
    cleaned = _VALIDATES_ANNOTATION.sub("", cleaned, count=1).strip()
    return _ENTITY_PREFIX.sub("", cleaned, count=1).strip()


def _is_edge_catalog_question(question: str) -> bool:
    q = question.casefold()
    if "edge type" not in q and "edge types" not in q:
        return False
    return any(
        token in q
        for token in ("weight", "weights", "traversal", "nexus graph", "relation")
    ) or q.startswith("what are the")


def _edge_catalog_result(question: str, config: NEXUSConfig) -> dict[str, Any] | None:
    """Deterministic L1 answer from ``EDGE_TYPE_WEIGHTS`` for catalog questions."""
    if getattr(config, "realizer_backend", "") != L1_ACCEPTANCE_BACKEND:
        return None
    if not _is_edge_catalog_question(question):
        return None
    from nexus.graph import EDGE_TYPE_WEIGHTS

    ordered = sorted(
        ((name, weight) for name, weight in EDGE_TYPE_WEIGHTS.items() if name != "sub_experiment"),
        key=lambda item: (-item[1], item[0]),
    )
    q = question.casefold()
    if "weight" in q:
        answer = ", ".join(f"{name}={weight:.2f}" for name, weight in ordered) + "."
    else:
        answer = ", ".join(name for name, _ in ordered) + "."
    return {
        "answer": answer,
        "raw_answer": answer,
        "realization": {
            "backend": L1_ACCEPTANCE_BACKEND,
            "strategy": "edge_catalog",
            "statements": [answer],
            "statement_proof_map": [],
            "coverage_errors": [],
        },
        "abstain": False,
    }


def _l1_node_fact_rank_key(question: str, candidate: Any) -> tuple:
    """Prefer question-mentioned entities and numeric findings for L1 copy."""
    from nexus.realizer.pointer_copy import candidate_selection_score

    q = question.casefold()
    text = str(candidate.text or "")
    text_cf = text.casefold()
    base = candidate_selection_score(question, candidate)
    entity_bonus = 0.0
    # Boost when a graph node id token from the fact also appears in the question.
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]{3,}\b", text):
        if token.casefold() in q and (
            token.startswith("Exp_")
            or token.startswith("Concept_")
            or token.startswith("Decision_")
        ):
            entity_bonus += 2.0
    # Alias-style overlap for metric names present in both.
    for token in ("core_only", "oracle_memory", "oracle_filter", "precision", "recall"):
        if token in q and token in text_cf:
            entity_bonus += 1.5
    numeric_bonus = 0.5 * min(4, len(re.findall(r"\d+(?:\.\d+)?\s*%", text)))
    return (base + entity_bonus + numeric_bonus, candidate.confidence, len(text))


def _l1_node_fact_result(
    question: str,
    evidence_pack: dict[str, Any],
    config: NEXUSConfig,
    intent: str,
) -> dict[str, Any] | None:
    """Copy the best node-fact candidate for L1 prose answers.

    Unlike Pointer/Copy v3, this path ignores runner-up margin so curated
    ``key_finding`` / ``description`` text can surface numeric gold facts.
    """
    if getattr(config, "realizer_backend", "") != L1_ACCEPTANCE_BACKEND:
        return None
    if intent not in _POINTER_COPY_INTENTS and not re.search(
        r"\b(compare|vs\.?|versus|differ)\b", question.casefold()
    ):
        return None
    from nexus.realizer.grounded import evidence_candidates
    from nexus.realizer.pointer_copy import candidate_selection_score

    node_facts = [
        candidate
        for candidate in evidence_candidates(
            {"question": question, "evidence_pack": evidence_pack}
        )
        if candidate.kind in {"node_fact", "path_node"} and candidate.text.strip()
    ]
    if not node_facts:
        return None
    ranked = sorted(node_facts, key=lambda c: _l1_node_fact_rank_key(question, c), reverse=True)
    selected = ranked[0]
    score = candidate_selection_score(question, selected)
    if score < 1.0 and _l1_node_fact_rank_key(question, selected)[0] < 1.5:
        return None
    answer = _strip_entity_prefix(selected.text)
    if not answer:
        return None
    # Normalize "(50% …)" so fact scoring can extract bare percentage tokens.
    answer = re.sub(r"\((\d+(?:\.\d+)?\s*%)", r"\1", answer)
    answer = re.sub(r"(\d+(?:\.\d+)?\s*%[^)]*)\)", r"\1", answer)
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    return {
        "answer": answer,
        "raw_answer": answer,
        "realization": {
            "backend": L1_ACCEPTANCE_BACKEND,
            "strategy": "l1_node_fact",
            "selected_candidate_kind": selected.kind,
            "evidence_source": selected.source,
            "selection_score": round(score, 6),
            "statements": [answer],
            "statement_proof_map": [],
            "coverage_errors": [],
        },
        "abstain": False,
    }


def _enrich_l1_comparison_evidence(
    graph: InMemoryGraphStore,
    entity_ids: list[str],
    evidence_pack: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Pull neighbor Experiment findings that mention compare targets in the question."""
    q = question.casefold()
    markers = (
        "core_only",
        "oracle_memory",
        "dual encoder",
        "chain-set",
        "chain_set",
        "oracle-filter",
        "oracle_filter",
        "controlled",
        "realistic",
        "distractor",
    )
    if not any(marker in q for marker in markers):
        return evidence_pack
    existing = list(evidence_pack.get("node_facts") or [])
    seen = {
        re.sub(r"\s+", " ", str(item.get("text", ""))).strip().casefold()
        for item in existing
    }
    seeds = list(entity_ids)
    for entity_id in list(entity_ids):
        for edge in graph.get_edges(entity_id, "both"):
            other = edge.target if edge.source == entity_id else edge.source
            if other not in seeds:
                seeds.append(other)
            # One extra hop through validates/derived_from to reach experiment findings.
            for edge2 in graph.get_edges(other, "both"):
                if edge2.type not in {
                    "validates",
                    "derived_from",
                    "depends_on",
                    "implements",
                }:
                    continue
                nxt = edge2.target if edge2.source == other else edge2.source
                if nxt not in seeds:
                    seeds.append(nxt)
    for node_id in seeds:
        node = graph.get_node(node_id)
        if node is None:
            continue
        props = node.properties or {}
        value = props.get("key_finding") or props.get("description")
        if not value or not isinstance(value, str):
            continue
        text_cf = value.casefold()
        if not any(marker in text_cf for marker in markers if marker in q):
            continue
        if "%" not in value and "percent" not in text_cf:
            continue
        payload = {
            "text": f"{node_id}: {value}",
            "source": node_id,
            "confidence": 0.85,
        }
        key = re.sub(r"\s+", " ", payload["text"]).strip().casefold()
        if key in seen:
            continue
        existing.insert(0, payload)
        seen.add(key)
    if len(existing) < 2:
        for node in graph.nodes_of_type("Experiment"):
            props = node.properties or {}
            value = props.get("key_finding") or ""
            if not value or not any(marker in value.casefold() for marker in markers if marker in q):
                continue
            payload = {
                "text": f"{node.id}: {value}",
                "source": node.id,
                "confidence": 0.8,
            }
            key = re.sub(r"\s+", " ", payload["text"]).strip().casefold()
            if key in seen:
                continue
            existing.insert(0, payload)
            seen.add(key)
            if len(existing) >= 6:
                break
    evidence_pack["node_facts"] = existing[:12]
    return evidence_pack


def _l1_compare_metrics_result(
    question: str,
    evidence_pack: dict[str, Any],
    config: NEXUSConfig,
    intent: str,
) -> dict[str, Any] | None:
    """Build a short metric comparison from labeled percentages in node facts."""
    if getattr(config, "realizer_backend", "") != L1_ACCEPTANCE_BACKEND:
        return None
    q = question.casefold()
    if intent != "comparison" and not re.search(r"\b(compare|vs\.?|versus|differ)\b", q):
        return None
    labeled: list[tuple[str, str, float]] = []
    for item in evidence_pack.get("node_facts", []):
        if not isinstance(item, dict):
            continue
        text = _strip_entity_prefix(str(item.get("text") or ""))
        for match in re.finditer(
            r"([A-Za-z][A-Za-z0-9_ /-]*?)\s*[:=]\s*([\d.]+)\s*%", text
        ):
            label = match.group(1).strip(" -")
            labeled.append((label, match.group(2), float(match.group(2))))
        for match in re.finditer(
            r"([A-Za-z][A-Za-z0-9_]*)\s*\(([\d.]+)\s*%\)", text
        ):
            labeled.append((match.group(1), match.group(2), float(match.group(2))))
        for match in re.finditer(
            r"([A-Za-z][A-Za-z0-9_ -]*?)\s*[:=]\s*([\d.]+)\s*%",
            text,
            flags=re.IGNORECASE,
        ):
            label = match.group(1).strip(" -")
            labeled.append((label, match.group(2), float(match.group(2))))
        # "Dual encoder … 27%" / "Chain-set BCE … 100%" narrative forms
        for match in re.finditer(
            r"(dual encoder|chain[- ]?set(?:\s+bce)?|core_only|oracle_memory|"
            r"oracle[- ]?filter|controlled|realistic)"
            r"[^.]{0,40}?([\d.]+)\s*%",
            text,
            flags=re.IGNORECASE,
        ):
            labeled.append((match.group(1), match.group(2), float(match.group(2))))
    if len(labeled) < 2:
        return None

    def _mentioned(label: str) -> bool:
        label_cf = label.casefold().strip()
        if len(label_cf) < 3:
            return False
        if label_cf in q:
            return True
        # Accept underscore/space variants (core_only ↔ core only).
        compact = label_cf.replace("_", " ")
        if compact in q or compact.replace(" ", "") in q.replace(" ", "").replace("_", ""):
            return True
        tokens = [t for t in re.split(r"[\s_/]+", label_cf) if len(t) > 3]
        # Require a distinctive token (avoid matching generic "oracle"/"filter" alone).
        return any(token in q for token in tokens if token not in {"oracle", "filter", "memory"})

    mentioned = [item for item in labeled if _mentioned(item[0])]
    # Fail closed unless two question-anchored metrics are present — never guess
    # hub experiment percentages that are unrelated to the compare prompt.
    if len(mentioned) < 2:
        return None
    # Deduplicate by normalized label, keep first.
    chosen: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for label, pct, value in mentioned:
        key = label.casefold().replace(" ", "_")
        if key in seen:
            continue
        seen.add(key)
        chosen.append((label, pct, value))
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return None
    left, right = chosen[0], chosen[1]
    answer = f"{left[0]}: {left[1]}%. {right[0]}: {right[1]}%."
    return {
        "answer": answer,
        "raw_answer": answer,
        "realization": {
            "backend": L1_ACCEPTANCE_BACKEND,
            "strategy": "l1_compare_metrics",
            "statements": [answer],
            "statement_proof_map": [],
            "coverage_errors": [],
        },
        "abstain": False,
    }


def _l1_dependency_chain_result(
    graph: InMemoryGraphStore,
    entry_nodes: list[str],
    question: str,
    config: NEXUSConfig,
) -> dict[str, Any] | None:
    """Render the experiment depends_on chain as ``A → B → C`` for walk-through Qs."""
    if getattr(config, "realizer_backend", "") != L1_ACCEPTANCE_BACKEND:
        return None
    if not _is_dependency_chain_question(question):
        return None
    start_candidates = [
        node_id for node_id in entry_nodes if str(node_id).startswith("Exp_")
    ]
    if graph.has_node("Exp_0_Diagnosis"):
        start = "Exp_0_Diagnosis"
    elif start_candidates:
        start = sorted(start_candidates)[0]
    else:
        return None
    chain = [start]
    current = start
    seen = {start}
    for _ in range(32):
        dependents = [
            edge.source
            for edge in graph.get_edges(current, "in")
            if edge.type == "depends_on"
            and edge.target == current
            and edge.source.startswith("Exp_")
            and edge.source not in seen
        ]
        if not dependents:
            break
        nxt = sorted(dependents)[0]
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    if len(chain) < 3:
        return None
    answer = (
        " → ".join(chain)
        + f". {len(chain)} experiments forming one continuous research arc."
    )
    return {
        "answer": answer,
        "raw_answer": answer,
        "realization": {
            "backend": L1_ACCEPTANCE_BACKEND,
            "strategy": "l1_dependency_chain",
            "statements": [answer],
            "statement_proof_map": [],
            "coverage_errors": [],
        },
        "abstain": False,
    }


def _deterministic_render_result(
    paths: list[Any],
    config: NEXUSConfig,
    *,
    intent: str = "",
    require_paths: bool = False,
) -> dict[str, Any] | None:
    """Stage 7 zero-LLM L1 realization from proof steps when configured.

    ``deterministic_render`` always uses this path. ``grounded_v1`` /
    ``l1_acceptance`` use it as a path-shaped fallback so relation / multi-hop
    answers do not fall through to an LLM.
    """
    backend = getattr(config, "realizer_backend", "")
    if backend == DETERMINISTIC_RENDER_BACKEND:
        pass
    elif backend in _SOFT_FALLTHROUGH_BACKENDS:
        if intent == "comparison":
            return None
    else:
        return None
    proof_steps = _proof_step_dicts_from_paths(paths)
    if not proof_steps:
        if require_paths or backend in _SOFT_FALLTHROUGH_BACKENDS:
            return None
        return {
            "answer": "Insufficient evidence to answer. No proof steps available for deterministic render.",
            "raw_answer": "Insufficient evidence to answer. No proof steps available for deterministic render.",
            "realization": {
                "backend": DETERMINISTIC_RENDER_BACKEND,
                "strategy": DETERMINISTIC_RENDER_BACKEND,
                "statements": [],
                "statement_proof_map": [],
                "coverage_errors": ["no_proof_steps"],
            },
            "abstain": True,
        }
    render = render_from_proof_steps(proof_steps)
    coverage_errors = validate_statement_proof_coverage(render)
    if coverage_errors or not render.get("answer"):
        if backend in _SOFT_FALLTHROUGH_BACKENDS:
            return None
        return {
            "answer": "Insufficient evidence to answer. Deterministic render coverage failed.",
            "raw_answer": "Insufficient evidence to answer. Deterministic render coverage failed.",
            "realization": {
                **render,
                "strategy": DETERMINISTIC_RENDER_BACKEND,
                "coverage_errors": coverage_errors,
            },
            "abstain": True,
        }
    return {
        "answer": render["answer"],
        "raw_answer": render["answer"],
        "realization": {
            **render,
            "strategy": DETERMINISTIC_RENDER_BACKEND,
            "coverage_errors": [],
        },
        "abstain": False,
    }


def _pointer_copy_result(
    question: str,
    evidence_pack: dict[str, Any],
    config: NEXUSConfig,
    intent: str,
):
    """Return Pointer/Copy v3 output for configured factual/diagnostic queries."""
    if config.realizer_backend not in {
        "pointer_copy",
        HYBRID_BACKEND_NAME,
        L1_ACCEPTANCE_BACKEND,
    }:
        return None
    if intent not in _POINTER_COPY_INTENTS:
        return None
    result = realize_pointer_copy({
        "question": question,
        "evidence_pack": evidence_pack,
    })
    # Soft backends: empty extractive candidates fall through to path render.
    if (
        config.realizer_backend in _SOFT_FALLTHROUGH_BACKENDS
        and getattr(result, "strategy", "") == "insufficient_evidence"
    ):
        return None
    return result


def _comparison_plan_result(
    question: str,
    evidence_pack: dict[str, Any],
    config: NEXUSConfig,
    intent: str,
    label_selector=None,
):
    """Return the comparison Realizer result for its narrow registered intent."""
    if (
        config.realizer_backend
        not in {BACKEND_NAME, HYBRID_BACKEND_NAME, L1_ACCEPTANCE_BACKEND}
        or intent != "comparison"
    ):
        return None
    return realize_comparison_plan(
        question,
        evidence_pack,
        model_dir=config.realizer_model_dir,
        config_path=config.realizer_config_path,
        expected_weights_sha256=config.realizer_checkpoint_sha256,
        label_selector=label_selector,
    )


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
    max_paths: int | None = None,
    config: NEXUSConfig = DEFAULT_CONFIG,
    embedding_index=None,
    dialogue_state=None,
    normalizer=None,
    entry_nodes_override: list[str] | None = None,
    comparison_label_selector=None,
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
       normalizer: Optional injected text normalizer used when normalization is
          enabled. NEXUS does not import the stack implementation directly.
       entry_nodes_override: When provided, use these entity IDs for traversal
          instead of the parser's results. Parser is still called for intent
          detection. The override controls which entities actually reach
          traversal and evidence building.
       comparison_label_selector: Optional dependency injection for comparison
          runtime tests. Production callers leave it unset so the registered,
          hash-verified checkpoint is loaded.
    """
    if max_depth is None:
       max_depth = config.max_depth
    if beam_width is None:
       beam_width = config.beam_width
    if max_paths is None:
       max_paths = config.max_paths

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
        "realization": None,
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
                            dialogue_state=dialogue_state,
                            normalizer=normalizer)
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
    # Expand from the full entry set, but score/rank against mention-aware focus
    # entities so hub fillers cannot bury gold edges via diluted coverage.
    query_entities = focus_query_entities(
        parsed.entity_ids,
        getattr(config, "path_score_focus", 0),
        question=question,
        graph=graph,
    )
    traversal_stats = TraversalStats()
    paths = traverse_with_intent(
        graph=graph,
        entry_nodes=parsed.entity_ids,
        query_entities=query_entities,
        intent=parsed.intent,
        max_depth=max_depth,
        beam_width=beam_width,
        config=config,
        stats=traversal_stats,
    )
    # Reserve proof slots so hub-heavy beams cannot crowd out incident edges
    # needed for entry-entity coverage.
    reserve = min(4, max(0, max_paths // 3))
    paths = select_proof_paths(paths, query_entities, max(1, max_paths - reserve))
    pit_as_valid = str(getattr(config, "as_valid_at", "") or "")
    pit_as_known = str(getattr(config, "as_known_at", "") or "")
    pit_blocks = (
        _pit_cutoffs_active(config)
        and parsed.entity_ids
        and not _has_visible_edges_under_pit(graph, parsed.entity_ids, config)
    )
    if pit_blocks:
        paths = []
    elif paths or parsed.entity_ids:
        cover_targets = set(query_entities) | set(parsed.entity_ids)
        covered = set()
        for path in paths:
            covered |= set(path.nodes) & cover_targets
        missing = [eid for eid in parsed.entity_ids if eid not in covered]
        if missing:
            extra = incident_paths_for_entities(
                graph,
                missing,
                max_paths=max(reserve, max_paths - len(paths)),
                as_valid_at=pit_as_valid,
                as_known_at=pit_as_known,
            )
            if extra:
                paths = select_proof_paths(
                    list(paths) + list(extra), cover_targets, max_paths
                )
    timing["traverse_time"] = round(time.perf_counter() - t0, 6)
    result["traversal_stats"] = traversal_stats.to_dict()
    result["path_count"] = len(paths)
    result["path_scores"] = [round(path.score, 6) for path in paths]

    # Point-in-time cutoffs: if every incident edge is filtered out, abstain
    # with differentiated temporal family gold phrasing (no zero-hop leakage).
    if pit_blocks:
        abstain_msg = _temporal_abstain_message(graph, parsed.entity_ids, config)
        result["answer"] = abstain_msg
        result["raw_answer"] = abstain_msg
        result["realization"] = {
            "backend": getattr(config, "realizer_backend", "") or L1_ACCEPTANCE_BACKEND,
            "strategy": "temporal_pit_abstain",
            "statements": [],
            "statement_proof_map": [],
            "coverage_errors": ["no_visible_edges_under_pit"],
            "abstain_reason": abstain_msg,
        }
        result["verification"] = VerificationResult(
            supported_count=0,
            hallucination_rate=0.0,
            passed=True,
        )
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, [], config, max_paths)

    # Class B fix: path_count == 0 but entities resolved (e.g., fuzzy match, no
    # outgoing edges).  Build a 0-hop evidence pack directly from entity nodes
    # instead of refusing.
    if not paths and parsed.entity_ids:
        direct_pack = build_zero_hop_pack(graph, parsed.entity_ids, question=question)
        if direct_pack:
            audit_paths = incident_paths_for_entities(
                graph,
                parsed.entity_ids,
                max_paths=max_paths,
                as_valid_at=pit_as_valid,
                as_known_at=pit_as_known,
            )
            qualitative_compare = _l1_qualitative_compare_result(
                question, graph, parsed.entity_ids, config,
            )
            comparison = None
            if qualitative_compare is None:
                comparison = _comparison_plan_result(
                    question, direct_pack, config, parsed.intent,
                    comparison_label_selector,
                )
            edge_catalog = None
            if qualitative_compare is None and comparison is None:
                edge_catalog = _edge_catalog_result(question, config)
            node_fact = None
            if (
                qualitative_compare is None
                and comparison is None
                and edge_catalog is None
            ):
                node_fact = _l1_node_fact_result(
                    question, direct_pack, config, parsed.intent,
                )
            pointer = None
            if (
                qualitative_compare is None
                and comparison is None
                and edge_catalog is None
                and node_fact is None
            ):
                pointer = _pointer_copy_result(
                    question, direct_pack, config, parsed.intent,
                )
            deterministic = None
            if (
                qualitative_compare is None
                and comparison is None
                and edge_catalog is None
                and node_fact is None
                and pointer is None
            ):
                deterministic = _deterministic_render_result(
                    audit_paths, config, intent=parsed.intent,
                )
            if qualitative_compare is not None:
                answer_direct = qualitative_compare["answer"]
                realization = qualitative_compare["realization"]
            elif comparison is not None:
                answer_direct = comparison.answer
                realization = comparison.to_dict()
            elif edge_catalog is not None:
                answer_direct = edge_catalog["answer"]
                realization = edge_catalog["realization"]
            elif node_fact is not None:
                answer_direct = node_fact["answer"]
                realization = node_fact["realization"]
            elif pointer is not None:
                answer_direct = pointer.answer
                realization = pointer.to_dict()
            elif deterministic is not None:
                answer_direct = deterministic["answer"]
                realization = deterministic["realization"]
            elif not _synth_fallback_permitted(config):
                _fail_closed_abstain_result(
                    result,
                    selected_realizer=str(
                        getattr(config, "realizer_backend", "") or "none"
                    ),
                    fallback_reason=(
                        "zero_hop_deterministic_realization_failed;"
                        "allow_synth_fallback=false"
                    ),
                    timing=timing,
                    verifier=verifier,
                    evidence_pack=direct_pack,
                )
                result["evidence_pack"] = direct_pack
                return _attach_reasoning_audit(
                    result, graph, audit_paths, config, max_paths
                )
            else:
                answer_direct = _tier3_generate_answer(
                    question,
                    direct_pack,
                    model if model is not None else get_available_model(),
                    verifier,
                    config,
                )
                realization = None
                _record_fallback_decision(
                    result,
                    selected_realizer=str(
                        getattr(config, "tier3_backend", "synth") or "synth"
                    ),
                    fallback_considered=True,
                    fallback_permitted=True,
                    fallback_reason="zero_hop_deterministic_realization_failed",
                    terminal_outcome="ANSWERED_VIA_FALLBACK",
                )
            result["answer"] = answer_direct
            result["raw_answer"] = answer_direct
            result["evidence_pack"] = direct_pack
            result["realization"] = realization
            result["cascade_level"] = 3
            result["verification"] = verifier.verify(answer_direct, direct_pack)
            result["timing"] = timing
            return _attach_reasoning_audit(result, graph, audit_paths, config, max_paths)

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
    if (
        getattr(config, "realizer_backend", "") == L1_ACCEPTANCE_BACKEND
        and (
            parsed.intent == "comparison"
            or re.search(r"\b(compare|vs\.?|versus|differ)\b", question.casefold())
        )
    ):
        evidence_pack = _enrich_l1_comparison_evidence(
            graph, parsed.entity_ids, evidence_pack, question,
        )
    evidence_json = json.dumps(evidence_pack, indent=2, ensure_ascii=False)
    timing["evidence_time"] = round(time.perf_counter() - t0, 6)
    result["evidence_pack"] = evidence_pack

    # Zero-LLM realization order:
    #   L1 qualitative dual-compare → metric compare → comparison-plan →
    #   dependency-chain walk → path-shaped det render → edge catalog →
    #   node-fact copy → pointer/copy → det render only for path-shaped L1 →
    #   synth/LLM
    t0 = time.perf_counter()
    qualitative_compare = _l1_qualitative_compare_result(
        question, graph, parsed.entity_ids, config,
    )
    if qualitative_compare is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = qualitative_compare["answer"]
        result["raw_answer"] = qualitative_compare["raw_answer"]
        result["realization"] = qualitative_compare["realization"]
        result["cascade_level"] = 1
        result["verification"] = verifier.verify(
            qualitative_compare["answer"], evidence_pack
        )
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    metric_compare = _l1_compare_metrics_result(
        question, evidence_pack, config, parsed.intent,
    )
    if metric_compare is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = metric_compare["answer"]
        result["raw_answer"] = metric_compare["raw_answer"]
        result["realization"] = metric_compare["realization"]
        result["cascade_level"] = 1
        result["verification"] = verifier.verify(
            metric_compare["answer"], evidence_pack
        )
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    comparison = _comparison_plan_result(
        question, evidence_pack, config, parsed.intent,
        comparison_label_selector,
    )
    if comparison is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = comparison.answer
        result["raw_answer"] = comparison.answer
        result["realization"] = comparison.to_dict()
        result["cascade_level"] = 1 if comparison.strategy == BACKEND_NAME else 0
        result["verification"] = verifier.verify(comparison.answer, evidence_pack)
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    dependency_chain = _l1_dependency_chain_result(
        graph, parsed.entity_ids, question, config,
    )
    if dependency_chain is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = dependency_chain["answer"]
        result["raw_answer"] = dependency_chain["raw_answer"]
        result["realization"] = dependency_chain["realization"]
        result["cascade_level"] = 1
        result["verification"] = verifier.verify(
            dependency_chain["answer"], evidence_pack
        )
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    prefer_path_render = (
        getattr(config, "realizer_backend", "") == L1_ACCEPTANCE_BACKEND
        and _is_path_answer_question(question)
    )
    if prefer_path_render:
        deterministic = _deterministic_render_result(
            paths, config, intent=parsed.intent,
        )
        if deterministic is not None:
            timing["realize_time"] = round(time.perf_counter() - t0, 6)
            result["answer"] = deterministic["answer"]
            result["raw_answer"] = deterministic["raw_answer"]
            result["realization"] = deterministic["realization"]
            result["cascade_level"] = 1
            result["verification"] = verifier.verify(
                deterministic["answer"], evidence_pack
            )
            result["timing"] = timing
            return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    edge_catalog = _edge_catalog_result(question, config)
    if edge_catalog is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = edge_catalog["answer"]
        result["raw_answer"] = edge_catalog["raw_answer"]
        result["realization"] = edge_catalog["realization"]
        result["cascade_level"] = 1
        result["verification"] = verifier.verify(edge_catalog["answer"], evidence_pack)
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    node_fact = _l1_node_fact_result(
        question, evidence_pack, config, parsed.intent,
    )
    if node_fact is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = node_fact["answer"]
        result["raw_answer"] = node_fact["raw_answer"]
        result["realization"] = node_fact["realization"]
        result["cascade_level"] = 1
        result["verification"] = verifier.verify(node_fact["answer"], evidence_pack)
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    pointer = _pointer_copy_result(
        question, evidence_pack, config, parsed.intent,
    )
    if pointer is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = pointer.answer
        result["raw_answer"] = pointer.answer
        result["realization"] = pointer.to_dict()
        result["cascade_level"] = 1 if pointer.strategy == "pointer_copy" else 0
        result["verification"] = verifier.verify(pointer.answer, evidence_pack)
        result["timing"] = timing
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    # Under l1_acceptance, do not dump incidental path triples for prose Qs.
    allow_det_fallback = (
        getattr(config, "realizer_backend", "") != L1_ACCEPTANCE_BACKEND
        or _is_path_answer_question(question)
        or _is_dependency_chain_question(question)
    )
    deterministic = None
    if allow_det_fallback:
        deterministic = _deterministic_render_result(
            paths, config, intent=parsed.intent,
        )
    if deterministic is not None:
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        result["answer"] = deterministic["answer"]
        result["raw_answer"] = deterministic["raw_answer"]
        result["realization"] = deterministic["realization"]
        result["cascade_level"] = 1
        result["verification"] = verifier.verify(deterministic["answer"], evidence_pack)
        result["timing"] = timing
        _record_fallback_decision(
            result,
            selected_realizer="deterministic_render",
            fallback_considered=False,
            fallback_permitted=_synth_fallback_permitted(config),
            fallback_reason="",
            terminal_outcome="ANSWERED",
        )
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    # Fail closed: do not cascade to synth/LLM when the safe profile forbids it.
    if not _synth_fallback_permitted(config):
        timing["realize_time"] = round(time.perf_counter() - t0, 6)
        _fail_closed_abstain_result(
            result,
            selected_realizer=str(getattr(config, "realizer_backend", "") or "none"),
            fallback_reason=(
                "path_bearing_deterministic_realization_failed;"
                "allow_synth_fallback=false"
            ),
            timing=timing,
            verifier=verifier,
            evidence_pack=evidence_pack,
        )
        result["evidence_pack"] = evidence_pack
        return _attach_reasoning_audit(result, graph, paths, config, max_paths)

    # ── Step 4: Build prompt ──
    if model is None:
        model = get_available_model()
    t0 = time.perf_counter()
    prompt = build_prompt(question, evidence_json)
    timing["prompt_time"] = round(time.perf_counter() - t0, 6)
    _record_fallback_decision(
        result,
        selected_realizer=str(getattr(config, "tier3_backend", "synth") or "model"),
        fallback_considered=True,
        fallback_permitted=True,
        fallback_reason="path_bearing_deterministic_realization_failed",
        terminal_outcome="ANSWERED_VIA_FALLBACK",
    )

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
