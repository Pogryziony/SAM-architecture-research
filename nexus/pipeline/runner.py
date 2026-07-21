"""Canonical NEXUS v1 pipeline runner.

Single entry point that executes the full pipeline:
  parse → entity resolution → traverse → evidence → answer → verify

Serializes complete per-question diagnostics for auditability.
Never relies on DEFAULT_CONFIG alone.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from nexus.graph.store import InMemoryGraphStore
from nexus.query.parser import parse_question, ParsedQuery
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import ModelInterface, get_available_model
from nexus.reasoning.verifier import Verifier
from nexus.utils.config import DEFAULT_CONFIG

from nexus.pipeline.config import ProductionNEXUSConfig, validate_config
from nexus.pipeline.entity_resolver import (
    EntityResolver,
    ResolutionResult,
    coerce_resolution_result,
)


@dataclass
class QuestionResult:
    """Complete per-question diagnostic record."""
    question_id: str = ""
    question: str = ""
    question_hash: str = ""
    parsed_intent: str = ""
    predicted_entities: list[str] = field(default_factory=list)
    entity_resolution_method: str = ""
    candidate_pool_size: int = 0
    resolution_candidates: list[dict[str, Any]] = field(default_factory=list)
    resolver_name: str = ""
    resolver_version: str = ""
    resolver_threshold: float | None = None
    resolver_rejection_reason: str = ""
    resolver_latency_ms: float = 0.0
    resolver_context_latency_ms: float = 0.0
    selected_entry_nodes: list[str] = field(default_factory=list)
    graph_paths_count: int = 0
    path_scores: list[float] = field(default_factory=list)
    evidence_pack_keys: list[str] = field(default_factory=list)
    evidence_pack: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    raw_answer: str = ""
    verifier_passed: bool = False
    hallucination_rate: float = 0.0
    cascade_level: int = 0
    failure_category: str = ""
    per_stage_latency_ms: dict[str, float] = field(default_factory=dict)
    lexical_fallback_used: bool = False
    reasoning_readiness_score: float = 0.0
    reasoning_action: str = "abstain"
    proof_steps_count: int = 0
    counter_evidence_count: int = 0
    provenance_coverage: float = 0.0
    proof_valid: bool = False
    reasoning_audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    config_hash: str = ""
    source_sha: str = ""
    questions_total: int = 0
    questions_answered: int = 0
    questions_failed: int = 0
    total_latency_ms: float = 0.0
    per_question: list[QuestionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    evaluation_mode: str = "predicted"


class NEXUSRunner:
    """Canonical NEXUS v1 pipeline runner.

    Usage:
        config = ProductionNEXUSConfig.lexical_only()
        runner = NEXUSRunner(graph, config)
        result = runner.run(questions)
    """

    def __init__(
        self,
        graph: InMemoryGraphStore,
        config: ProductionNEXUSConfig | None = None,
        model: ModelInterface | None = None,
        entity_resolver: EntityResolver | None = None,
        normalizer: Callable[[str], str] | None = None,
        dialogue_state: Any = None,
    ):
        self.graph = graph
        self.config = config or ProductionNEXUSConfig.lexical_only()
        self.model = model
        self._verifier: Verifier | None = None
        self._entity_resolver = entity_resolver
        self._normalizer = normalizer
        self._dialogue_state = dialogue_state

    @property
    def verifier(self) -> Verifier:
        if self._verifier is None:
            self._verifier = Verifier(
                hallucination_threshold=self.config.hallucination_threshold
            )
        return self._verifier

    def run(
        self,
        questions: list[dict[str, Any]],
        source_sha: str = "",
    ) -> PipelineResult:
        """Run the pipeline on a list of questions.

        Each question dict must have 'id' and 'question'.
        """
        errors = validate_config(self.config)
        if (
            self.config.pipeline_id.entity_ranker_v3_enabled
            and self._entity_resolver is None
        ):
            errors.append(
                "entity_ranker_v3 is enabled but no EntityResolver was injected"
            )
        if errors:
            return PipelineResult(
                config_hash=self.config.config_hash,
                source_sha=source_sha,
                errors=errors,
                evaluation_mode="predicted",
            )

        if self.model is None:
            self.model = get_available_model()
        model = self.model
        pipeline_start = time.perf_counter()

        results: list[QuestionResult] = []
        answered = 0
        failed = 0

        for record in questions:
            qid = str(record.get("id", ""))
            question = str(record["question"])
            qr = self._run_single(
                qid,
                question,
                model,
                config_override=self._config_for_record(record),
            )
            results.append(qr)
            if qr.failure_category:
                failed += 1
            else:
                answered += 1

        total_latency = (time.perf_counter() - pipeline_start) * 1000

        return PipelineResult(
            config_hash=self.config.config_hash,
            source_sha=source_sha,
            questions_total=len(questions),
            questions_answered=answered,
            questions_failed=failed,
            total_latency_ms=round(total_latency, 3),
            per_question=results,
            evaluation_mode="predicted",
        )

    def run_oracle(
        self,
        questions: list[dict[str, Any]],
        source_sha: str = "",
    ) -> PipelineResult:
        """Evaluate NEXUS with gold entry entities, independently of SAM/ER3.

        Every record must contain a non-empty ``gold_entities`` list.  This
        mode is fail-closed so an incomplete benchmark cannot silently fall
        back to lexical entity resolution.
        """
        errors = validate_config(self.config)
        for index, record in enumerate(questions):
            gold_entities = record.get("gold_entities")
            if not isinstance(gold_entities, list) or not gold_entities:
                qid = str(record.get("id", index))
                errors.append(f"question {qid} missing non-empty gold_entities")
        if errors:
            return PipelineResult(
                config_hash=self.config.config_hash,
                source_sha=source_sha,
                errors=errors,
                evaluation_mode="oracle",
            )

        if self.model is None:
            self.model = get_available_model()
        model = self.model
        pipeline_start = time.perf_counter()
        results: list[QuestionResult] = []
        answered = 0
        failed = 0
        for record in questions:
            qr = self._run_single(
                str(record.get("id", "")),
                str(record["question"]),
                model,
                entry_nodes_override=[str(item) for item in record["gold_entities"]],
                config_override=self._config_for_record(record),
            )
            results.append(qr)
            if qr.failure_category:
                failed += 1
            else:
                answered += 1

        total_latency = (time.perf_counter() - pipeline_start) * 1000
        return PipelineResult(
            config_hash=self.config.config_hash,
            source_sha=source_sha,
            questions_total=len(questions),
            questions_answered=answered,
            questions_failed=failed,
            total_latency_ms=round(total_latency, 3),
            per_question=results,
            evaluation_mode="oracle",
        )

    def _config_for_record(self, record: dict[str, Any]):
        """Apply per-question bi-temporal cutoffs when the oracle record carries them."""
        as_known_at = str(record.get("as_known_at") or "").strip()
        as_valid_at = str(record.get("as_valid_at") or "").strip()
        if not as_known_at and not as_valid_at:
            return None
        return replace(
            self.config,
            as_known_at=as_known_at,
            as_valid_at=as_valid_at,
        )

    def _run_single(
        self,
        qid: str,
        question: str,
        model: ModelInterface,
        entry_nodes_override: list[str] | None = None,
        config_override: Any | None = None,
    ) -> QuestionResult:
        qr = QuestionResult(
            question_id=qid,
            question=question,
            question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
        )
        active_config = config_override if config_override is not None else self.config

        try:
            resolution: ResolutionResult | None = None
            # ── Entity resolution: injected resolver, oracle, or lexical ──
            if entry_nodes_override is not None:
                oracle_entities = list(entry_nodes_override)[:active_config.max_entry_nodes]
                qr.predicted_entities = oracle_entities
                qr.entity_resolution_method = "oracle"
                qr.resolver_name = "oracle"
                qr.resolver_version = "1"
                qr.candidate_pool_size = len(oracle_entities)
                qr.selected_entry_nodes = oracle_entities
                qr.lexical_fallback_used = False
                result = answer_question(
                    question,
                    self.graph,
                    model=model,
                    verifier=self.verifier,
                    config=active_config,
                    normalizer=self._normalizer,
                    dialogue_state=self._dialogue_state,
                    entry_nodes_override=oracle_entities,
                )
                parsed = result.get("parsed_query")
                if parsed:
                    qr.parsed_intent = parsed.intent
            elif self._entity_resolver is not None:
                resolution = coerce_resolution_result(
                    self._entity_resolver.resolve(question, self.graph),
                    resolver_name=self._entity_resolver.__class__.__name__,
                )
                selected = resolution.selected_entity_ids[:active_config.max_entry_nodes]
                qr.predicted_entities = list(resolution.selected_entity_ids)
                qr.entity_resolution_method = resolution.resolver_name
                qr.resolver_name = resolution.resolver_name
                qr.resolver_version = resolution.resolver_version
                qr.resolver_threshold = resolution.threshold
                qr.resolver_rejection_reason = resolution.rejection_reason
                qr.resolver_latency_ms = resolution.latency_ms
                qr.resolver_context_latency_ms = resolution.context_latency_ms
                qr.resolution_candidates = [
                    {"entity_id": item.entity_id, "score": item.score}
                    for item in resolution.candidates
                ]
                qr.candidate_pool_size = resolution.candidate_pool_size
                qr.selected_entry_nodes = selected
                qr.lexical_fallback_used = resolution.fallback_used

                result = answer_question(
                    question, self.graph, model=model,
                    verifier=self.verifier, config=active_config,
                    normalizer=self._normalizer,
                    dialogue_state=self._dialogue_state,
                    entry_nodes_override=selected,
                )
                parsed = result.get("parsed_query")
                if parsed:
                    qr.parsed_intent = parsed.intent
            else:
                result_raw = answer_question(
                    question, self.graph, model=model,
                    verifier=self.verifier, config=active_config,
                    normalizer=self._normalizer,
                    dialogue_state=self._dialogue_state,
                )
                parsed = result_raw.get("parsed_query")
                if parsed:
                    qr.parsed_intent = parsed.intent
                    qr.predicted_entities = list(parsed.entity_ids)
                    qr.entity_resolution_method = parsed.resolution_method
                    qr.selected_entry_nodes = list(
                        parsed.entity_ids[:active_config.max_entry_nodes]
                    )
                    qr.lexical_fallback_used = True

            # ── Continue with answer_question if lexical ──
            if entry_nodes_override is None and self._entity_resolver is None:
                result = result_raw

            qr.graph_paths_count = result.get("path_count", 0)
            if resolution is None and qr.candidate_pool_size == 0 and parsed:
                qr.candidate_pool_size = len(parsed.entity_ids)
            qr.path_scores = list(result.get("path_scores", []))
            qr.cascade_level = result.get("cascade_level", 0)
            qr.answer = result.get("answer", "")
            qr.raw_answer = result.get("raw_answer", "")

            verification = result.get("verification")
            if verification is not None:
                qr.verifier_passed = verification.passed
                qr.hallucination_rate = verification.hallucination_rate

            ep = result.get("evidence_pack", {})
            if isinstance(ep, dict):
                qr.evidence_pack_keys = sorted(ep.keys())
                qr.evidence_pack = ep

            timing = result.get("timing", {})
            if isinstance(timing, dict):
                qr.per_stage_latency_ms = {
                    k: round(v * 1000, 3) if isinstance(v, (int, float)) else 0
                    for k, v in timing.items()
                }

            audit = result.get("reasoning_audit", {})
            if isinstance(audit, dict):
                qr.reasoning_audit = audit
                qr.reasoning_readiness_score = float(
                    audit.get("readiness_score", 0.0)
                )
                qr.reasoning_action = str(
                    audit.get("recommended_action", "abstain")
                )
                qr.proof_steps_count = len(audit.get("proof_steps", []))
                qr.counter_evidence_count = len(audit.get("counter_evidence", []))
                qr.provenance_coverage = float(
                    audit.get("provenance_coverage", 0.0)
                )
                qr.proof_valid = bool(audit.get("proof_valid", False))

            # Check lexical fallback
            if (entry_nodes_override is None
                    and self._entity_resolver is None
                    and not self.config.enable_associative_encoder):
                qr.lexical_fallback_used = True

            # Failure categorization
            if not parsed or not parsed.entity_ids:
                qr.failure_category = "no_entities_resolved"
            elif qr.graph_paths_count == 0:
                qr.failure_category = "no_graph_paths"
            elif not qr.answer or "insufficient" in qr.answer.lower():
                qr.failure_category = "insufficient_answer"
            elif not qr.verifier_passed:
                qr.failure_category = "verifier_failed"

        except Exception as exc:
            qr.failure_category = f"exception:{exc.__class__.__name__}"

        return qr

    def serialize_result(
        self, result: PipelineResult, output_path: Path | None = None
    ) -> dict[str, Any]:
        """Serialize to JSON-safe dict. Optionally writes to *output_path*."""
        if output_path and output_path.exists():
            raise FileExistsError(f"Refusing to overwrite: {output_path}")

        data = {
            "pipeline": "nexus_v1",
            "evaluation_mode": result.evaluation_mode,
            "config_hash": result.config_hash,
            "source_sha": result.source_sha,
            "questions_total": result.questions_total,
            "questions_answered": result.questions_answered,
            "questions_failed": result.questions_failed,
            "total_latency_ms": result.total_latency_ms,
            "per_question": [
                {
                    "question_id": qr.question_id,
                    "question_hash": qr.question_hash,
                    "parsed_intent": qr.parsed_intent,
                    "predicted_entities": qr.predicted_entities,
                    "entity_resolution_method": qr.entity_resolution_method,
                    "candidate_pool_size": qr.candidate_pool_size,
                    "resolution_candidates": qr.resolution_candidates,
                    "resolver_name": qr.resolver_name,
                    "resolver_version": qr.resolver_version,
                    "resolver_threshold": qr.resolver_threshold,
                    "resolver_rejection_reason": qr.resolver_rejection_reason,
                    "resolver_latency_ms": qr.resolver_latency_ms,
                    "resolver_context_latency_ms": qr.resolver_context_latency_ms,
                    "selected_entry_nodes": qr.selected_entry_nodes,
                    "graph_paths_count": qr.graph_paths_count,
                    "path_scores": qr.path_scores,
                    "evidence_pack_keys": qr.evidence_pack_keys,
                    "evidence_pack": qr.evidence_pack,
                    "answer": qr.answer[:500],
                    "verifier_passed": qr.verifier_passed,
                    "hallucination_rate": qr.hallucination_rate,
                    "cascade_level": qr.cascade_level,
                    "failure_category": qr.failure_category,
                    "per_stage_latency_ms": qr.per_stage_latency_ms,
                    "lexical_fallback_used": qr.lexical_fallback_used,
                    "reasoning_readiness_score": qr.reasoning_readiness_score,
                    "reasoning_action": qr.reasoning_action,
                    "proof_steps_count": qr.proof_steps_count,
                    "counter_evidence_count": qr.counter_evidence_count,
                    "provenance_coverage": qr.provenance_coverage,
                    "proof_valid": qr.proof_valid,
                    "reasoning_audit": qr.reasoning_audit,
                }
                for qr in result.per_question
            ],
            "errors": result.errors,
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.rename(output_path)

        return data
