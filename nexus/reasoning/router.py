"""
NEXUS Router — intelligently routes questions to synthesizer (template-based,
near-zero cost) or LLM (for complex reasoning).

Key insight: 80%+ of the QA dataset can be handled by template-based synthesis
when the SynthesizingModel is expanded with comparative, chain, and definition
methods. For these, template-based synthesis achieves comparable accuracy to
LLM at ~0 generation cost and ~400× faster.

Phase 5: Data-driven routing via learned decision table.
The decision table maps (intent, has_matching_metric, estimated_hops) → best_arm
based on per-question paired accuracy data from both arms. When a lookup has no
match, falls back to the legacy hand-weighted confidence score.

Target: >90% of questions route to synthesizer, achieving near-zero cost for
the vast majority of queries.

Decision logic:
  1. Load learned decision table from router_policy.json
  2. Extract signals from evidence pack: intent, has_matching_metric, estimated_hops
  3. Lookup table → synthesizer or llm with per-arm accuracy stats
  4. If no match in table, fall back to weighted confidence score (legacy)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import traverse_with_intent
from nexus.query.parser import parse_question, ParsedQuery
from nexus.reasoning.evidence_builder import build_evidence, build_evidence_pack
from nexus.reasoning.prompt_template import build_prompt, _find_question_entity
from nexus.reasoning.model_interface import (
    ModelInterface,
    SynthesizingModel,
    get_available_model,
)
from nexus.reasoning.verifier import Verifier, VerificationResult
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG

# Path to the learned decision table (relative to repo root)
_POLICY_PATH = Path(__file__).parent / "router_policy.json"


def _make_table_key(intent: str, has_matching_metric: bool, estimated_hops: int) -> str:
    """Build a lookup key matching the decision table format."""
    return f"{intent}|{int(has_matching_metric)}|{estimated_hops}"


class Router:
    """Routes questions to 'synthesizer' or 'llm' based on question type and evidence.

    The synthesizer route uses SynthesizingModel (template-based, ~0 cost, ~10ms).
    The LLM route uses OllamaModel or similar (actual inference, higher cost, slower).

    Phase 5: Uses a learned decision table (router_policy.json) for routing.
    Falls back to hand-weighted confidence score when the table has no match.
    """

    def __init__(self, policy_path: Path | str | None = None):
        """Initialize the router, loading the learned decision table if available.

        Args:
            policy_path: Path to router_policy.json. Defaults to the file
                         next to this module (nexus/reasoning/router_policy.json).
        """
        self._policy_path = Path(policy_path) if policy_path else _POLICY_PATH
        self._decision_table: dict[str, dict[str, Any]] = {}
        self._table_loaded = False
        self._load_policy()

    def _load_policy(self) -> None:
        """Load the learned decision table from router_policy.json."""
        try:
            if self._policy_path.exists():
                with open(self._policy_path, "r", encoding="utf-8") as f:
                    policy = json.load(f)
                self._decision_table = policy.get("decision_table", {})
                self._table_loaded = True
        except (json.JSONDecodeError, OSError):
            pass  # Silently fall back to legacy scoring

    def route(
        self,
        question: str,
        parsed_query: ParsedQuery,
        evidence_pack: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """
        Decide whether to route to 'synthesizer' or 'llm'.

        Phase 5 priority:
          1. Decision table lookup (learned from per-question paired data)
          2. Fall back to weighted confidence score (legacy hand-tuning)

        Args:
            question: The original natural language question.
            parsed_query: ParsedQuery with intent and entity resolution.
            evidence_pack: Evidence pack dict (must contain confidence_signals).

        Returns:
            (route: str, reason: str) — route is "synthesizer" or "llm".
        """
        # ── Extract confidence signals from evidence pack ──
        if evidence_pack is None:
            return "llm", "no evidence pack available"

        signals = evidence_pack.get("confidence_signals")
        if signals is None:
            node_facts = evidence_pack.get("node_facts", [])
            if len(node_facts) > 0:
                return "synthesizer", "has key_findings (fallback)"
            return "llm", "no confidence signals — no evidence"

        numeric_match = signals.get("numeric_match", 0.0)
        has_key_finding = signals.get("has_key_finding", 0.0)
        path_count_signal = signals.get("path_count_signal", 0.0)

        # ── Phase 5: Decision table lookup ──
        if self._table_loaded and self._decision_table:
            table_route = self._route_by_table(
                question, parsed_query, numeric_match
            )
            if table_route is not None:
                return table_route

        # ── Legacy: Weighted confidence score (fallback) ──
        confidence_score = (
            0.4 * numeric_match
            + 0.4 * has_key_finding
            + 0.2 * path_count_signal
        )

        if confidence_score >= 0.6:
            reason = (
                f"fallback conf={confidence_score:.2f} "
                f"(num={numeric_match:.1f}, kf={has_key_finding:.1f}, "
                f"paths={path_count_signal:.1f})"
            )
            return "synthesizer", reason
        else:
            reason = (
                f"fallback low conf={confidence_score:.2f} "
                f"(num={numeric_match:.1f}, kf={has_key_finding:.1f}, "
                f"paths={path_count_signal:.1f})"
            )
            return "llm", reason

    def _route_by_table(
        self,
        question: str,
        parsed_query: ParsedQuery,
        numeric_match: float,
    ) -> tuple[str, str] | None:
        """Try to route using the learned decision table. Returns None if no match."""
        intent = parsed_query.intent
        has_matching_metric = numeric_match > 0.0
        estimated_hops = self._estimate_hops(question)
        if estimated_hops < 1:
            estimated_hops = 1

        key = _make_table_key(intent, has_matching_metric, estimated_hops)
        entry = self._decision_table.get(key)

        if entry is None:
            return None

        best_arm = entry.get("best_arm", "llm")
        synth_acc = entry.get("synth_accuracy", 0.0)
        llm_acc = entry.get("llm_accuracy", 0.0)

        reason = (
            f"decision_table: synth_accuracy={synth_acc:.2f}, "
            f"llm_accuracy={llm_acc:.2f} "
            f"(intent={intent}, metric={has_matching_metric}, hops={estimated_hops})"
        )
        return best_arm, reason

    @staticmethod
    def _has_key_finding(evidence_pack: dict[str, Any]) -> bool:
        """Check if evidence pack contains curated node facts (key_findings)."""
        node_facts = evidence_pack.get("node_facts", [])
        return len(node_facts) > 0

    @staticmethod
    def _has_comparable_entities(evidence_pack: dict[str, Any]) -> bool:
        """Check if evidence has ≥2 Experiment nodes with numeric key_findings.

        For comparative questions, we need at least two entities with
        measurable (numeric) outcomes to synthesize a comparison.
        """
        node_facts = evidence_pack.get("node_facts", [])
        exp_count = 0
        for nf in node_facts:
            text = nf.get("text", "")
            # Check if it's an Experiment node with numeric data
            if Router._is_experiment_with_numbers(text):
                exp_count += 1
                if exp_count >= 2:
                    return True
        return False

    @staticmethod
    def _is_experiment_with_numbers(text: str) -> bool:
        """Check if a node_fact text indicates an Experiment with numeric findings."""
        # Experiment nodes start with "Exp_" in their ID
        exp_pattern = re.search(r'^\[?.*?\]?\s*(Exp_\w+)', text)
        if not exp_pattern:
            return False
        # Must contain numeric data
        return bool(re.search(r'\d+(?:\.\d+)?%', text) or re.search(r'\b\d+(?:\.\d+)?\b', text))

    @staticmethod
    def _has_multi_edge_paths(evidence_pack: dict[str, Any]) -> bool:
        """Check if evidence pack has paths with ≥2 edges (multi-hop traversal)."""
        paths = evidence_pack.get("paths", [])
        for path_data in paths:
            edges = path_data.get("edges", [])
            if len(edges) >= 2:
                return True
        return False

    @staticmethod
    def _has_causal_edges(evidence_pack: dict[str, Any]) -> bool:
        """Check if evidence has causal edges (caused_by, blocked_by).

        Looks in both relation facts and path edges.
        """
        # Check relation facts for causal keywords
        facts = evidence_pack.get("facts", [])
        for fact in facts:
            if isinstance(fact, str) and re.search(r'\b(caused_by|blocked_by)\b', fact):
                return True
        # Check path edges for causal types
        paths = evidence_pack.get("paths", [])
        for path_data in paths:
            for edge in path_data.get("edges", []):
                if edge.get("type") in ("caused_by", "blocked_by"):
                    return True
        return False

    @staticmethod
    def _is_simple_factual(question: str) -> bool:
        """Check if question is a simple factual lookup by structure."""
        q_lower = question.lower()
        patterns = [
            r"\bwhat\s+is\b",
            r"\bwhat\s+was\b",
            r"\bwhat\s+are\b",
            r"\bwhat\s+were\b",
            r"\bhow\s+many\b",
            r"\bwhich\b",
        ]
        return any(re.search(p, q_lower) for p in patterns)

    @staticmethod
    def _estimate_hops(question: str) -> int:
        """
        Estimate reasoning hops from question structure.

        Multi-hop indicators suggest the answer requires connecting
        multiple pieces of evidence (e.g., causal chains, comparisons).
        """
        q_lower = question.lower()
        multi_hop_indicators = [
            r"\bwhy\b",
            r"\bbecause\b",
            r"\bcaused\b",
            r"\bled\s+to\b",
            r"\bdepends?\s+on\b",
            r"\baffects?\b",
            r"\bimpact\b",
            r"\binfluence\b",
            r"\bcompare\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bdifference\b",
            r"\bbetween\b.*\band\b",
            r"\bhow\s+(do|does|to|can)\b",
            r"\bdiagnose\b",
            r"\bdebug\b",
        ]
        for pattern in multi_hop_indicators:
            if re.search(pattern, q_lower):
                return 2
        return 1


class RoutedPipeline:
    """
    Full NEXUS pipeline with intelligent routing between synthesizer and LLM.

    Pipeline flow:
      1. Parse question → entity resolution + intent detection
      2. Traverse graph → beam search from entry nodes
      3. Build evidence → structured JSON with node facts + relation facts
      4. Route → decide synthesizer vs LLM based on question + evidence
      5. Generate → template synthesis (~0 cost) or LLM inference
      6. Verify → hallucination check against evidence

    The key win: for factual 1-hop questions (~63% of dataset), the
    synthesizer produces answers at ~0 generation cost and ~10ms latency.
    """

    def __init__(
        self,
        graph: InMemoryGraphStore,
        router: Router | None = None,
        synthesizer_model: ModelInterface | None = None,
        llm_model: ModelInterface | None = None,
        verifier: Verifier | None = None,
        config: NEXUSConfig = DEFAULT_CONFIG,
    ):
        self.graph = graph
        self.router = router or Router()
        self.synthesizer_model = synthesizer_model or SynthesizingModel()
        self._llm_model = llm_model
        self.verifier = verifier or Verifier(
            hallucination_threshold=config.hallucination_threshold
        )
        self.config = config

    def _ensure_llm(self) -> ModelInterface:
        if self._llm_model is None:
            self._llm_model = get_available_model()
        return self._llm_model

    def answer(
        self,
        question: str,
        max_depth: int | None = None,
        beam_width: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the full routed pipeline on a question.

        Returns a dict with:
            - question, answer, evidence_pack, verification, parsed_query
            - path_count, routed_to, route_reason
            - generation_latency_s: time spent in model.generate()
        """
        if max_depth is None:
            max_depth = self.config.max_depth
        if beam_width is None:
            beam_width = self.config.beam_width

        result: dict[str, Any] = {
            "question": question,
            "answer": "",
            "evidence_pack": {},
            "verification": None,
            "parsed_query": None,
            "path_count": 0,
            "routed_to": "",
            "route_reason": "",
            "generation_latency_s": 0.0,
        }

        # ── Edge case: empty graph ──
        if self.graph.node_count == 0:
            result["answer"] = (
                "Insufficient evidence to answer. The knowledge graph is empty."
            )
            result["verification"] = VerificationResult(
                supported_count=0, hallucination_rate=0.0, passed=True
            )
            return result

        # ── Step 1: Parse ──
        parsed = parse_question(question, self.graph, cutoff=0.6, config=self.config)
        result["parsed_query"] = parsed

        if not parsed.entity_ids:
            result["answer"] = (
                "Insufficient evidence to answer. "
                "No relevant entities found in the question."
            )
            result["verification"] = VerificationResult(
                supported_count=0, hallucination_rate=0.0, passed=True
            )
            return result

        # ── Step 2: Traverse ──
        from nexus.graph.scoring import focus_query_entities

        query_entities = focus_query_entities(
            parsed.entity_ids, getattr(self.config, "path_score_focus", 0)
        )
        paths = traverse_with_intent(
            graph=self.graph,
            entry_nodes=parsed.entity_ids,
            query_entities=query_entities,
            intent=parsed.intent,
            max_depth=max_depth,
            beam_width=beam_width,
            config=self.config,
        )
        result["path_count"] = len(paths)

        if not paths:
            result["answer"] = (
                "Insufficient evidence to answer. "
                "No traversal paths found connecting the identified entities."
            )
            result["verification"] = VerificationResult(
                supported_count=0, hallucination_rate=0.0, passed=True
            )
            return result

        # ── Step 3: Build evidence ──
        # Determine target entity for factual questions to feed confidence signals.
        target_entity = None
        if parsed.intent == "factual_lookup":
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

        evidence_pack = build_evidence_pack(
            question, paths, self.graph, question_intent=parsed.intent,
            target_entity=target_entity,
        )
        result["evidence_pack"] = evidence_pack

        # ── Step 4: Route ──
        route, reason = self.router.route(question, parsed, evidence_pack)
        result["routed_to"] = route
        result["route_reason"] = reason

        # ── Step 5: Generate ──
        evidence_json = build_evidence(
            question, paths, self.graph, question_intent=parsed.intent,
            target_entity=target_entity,
        )
        prompt = build_prompt(question, evidence_json)

        t0 = time.perf_counter()
        if route == "synthesizer":
            model = self.synthesizer_model
        else:
            model = self._ensure_llm()
        answer = model.generate(prompt)
        gen_latency = time.perf_counter() - t0

        result["answer"] = answer
        result["generation_latency_s"] = round(gen_latency, 4)

        # ── Step 6: Verify ──
        verification = self.verifier.verify(answer, evidence_pack)
        result["verification"] = verification

        return result
