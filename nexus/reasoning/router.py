"""
NEXUS Router — intelligently routes questions to synthesizer (template-based,
near-zero cost) or LLM (for complex reasoning).

Key insight: 63% of the QA dataset consists of 1-hop factual questions where
the answer is directly present in the key_finding property of evidence nodes.
For these, template-based synthesis achieves comparable accuracy to LLM at
~0 generation cost and ~400× faster.

Decision logic:
  1. factual_lookup intent + evidence has key_finding → synthesizer
  2. Simple "what is"/"how many"/"which" with <=1 hop → synthesizer
  3. No key_finding in evidence → LLM (nothing to synthesize from)
  4. Everything else (multi-hop, causal, diagnostic) → LLM
"""

from __future__ import annotations

import re
import time
from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import traverse_with_intent
from nexus.query.parser import parse_question, ParsedQuery
from nexus.reasoning.evidence_builder import build_evidence, build_evidence_pack
from nexus.reasoning.prompt_template import build_prompt
from nexus.reasoning.model_interface import (
    ModelInterface,
    SynthesizingModel,
    get_available_model,
)
from nexus.reasoning.verifier import Verifier, VerificationResult
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG


class Router:
    """Routes questions to 'synthesizer' or 'llm' based on question type and evidence.

    The synthesizer route uses SynthesizingModel (template-based, ~0 cost, ~10ms).
    The LLM route uses OllamaModel or similar (actual inference, higher cost, slower).
    """

    def route(
        self,
        question: str,
        parsed_query: ParsedQuery,
        evidence_pack: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """
        Decide whether to route to 'synthesizer' or 'llm'.

        Args:
            question: The original natural language question.
            parsed_query: ParsedQuery with intent and entity resolution.
            evidence_pack: Evidence pack dict (None if pre-traversal — assumes key_finding exists).

        Returns:
            (route: str, reason: str) — route is "synthesizer" or "llm".
        """
        # Check whether the evidence contains curated key_findings
        has_kf = self._has_key_finding(evidence_pack) if evidence_pack is not None else True

        # ── Rule 1: factual_lookup intent → synthesizer candidate ──
        if parsed_query.intent == "factual_lookup":
            if has_kf:
                return "synthesizer", "factual_lookup intent"
            else:
                return "llm", "factual_lookup but no key_finding in evidence"

        # ── Rule 2: Simple what-is / how-many / which questions with ≤1 hop ──
        if self._is_simple_factual(question) and self._estimate_hops(question) <= 1:
            if has_kf:
                return "synthesizer", "simple factual question (<=1 hop)"
            else:
                return "llm", "simple factual but no key_finding in evidence"

        # ── Rule 3: No key_finding → LLM (nothing to synthesize from) ──
        if not has_kf:
            return "llm", "no key_finding in evidence"

        # ── Rule 4: Everything else → LLM ──
        return "llm", "complex or multi-hop question"

    @staticmethod
    def _has_key_finding(evidence_pack: dict[str, Any]) -> bool:
        """Check if evidence pack contains curated node facts (key_findings)."""
        node_facts = evidence_pack.get("node_facts", [])
        return len(node_facts) > 0

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
        query_entities = set(parsed.entity_ids)
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
        evidence_pack = build_evidence_pack(
            question, paths, self.graph, question_intent=parsed.intent
        )
        result["evidence_pack"] = evidence_pack

        # ── Step 4: Route ──
        route, reason = self.router.route(question, parsed, evidence_pack)
        result["routed_to"] = route
        result["route_reason"] = reason

        # ── Step 5: Generate ──
        evidence_json = build_evidence(
            question, paths, self.graph, question_intent=parsed.intent
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
