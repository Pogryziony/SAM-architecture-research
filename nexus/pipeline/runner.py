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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.query.parser import parse_question, ParsedQuery
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import ModelInterface, get_available_model
from nexus.reasoning.verifier import Verifier
from nexus.utils.config import DEFAULT_CONFIG

from nexus.pipeline.config import ProductionNEXUSConfig, validate_config


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
    selected_entry_nodes: list[str] = field(default_factory=list)
    graph_paths_count: int = 0
    path_scores: list[float] = field(default_factory=list)
    evidence_pack_keys: list[str] = field(default_factory=list)
    answer: str = ""
    raw_answer: str = ""
    verifier_passed: bool = False
    hallucination_rate: float = 0.0
    cascade_level: int = 0
    failure_category: str = ""
    per_stage_latency_ms: dict[str, float] = field(default_factory=dict)
    lexical_fallback_used: bool = False


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
    ):
        self.graph = graph
        self.config = config or ProductionNEXUSConfig.lexical_only()
        self.model = model
        self._verifier: Verifier | None = None
        self._er3_model: Any = None
        self._er3_tokenizer: Any = None

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
        if errors:
            return PipelineResult(
                config_hash=self.config.config_hash,
                source_sha=source_sha,
                errors=errors,
            )

        model = self.model or get_available_model()
        pipeline_start = time.perf_counter()

        results: list[QuestionResult] = []
        answered = 0
        failed = 0

        for record in questions:
            qid = str(record.get("id", ""))
            question = str(record["question"])
            qr = self._run_single(qid, question, model)
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
        )

    def _load_er3(self):
        """Lazy-load Entity Ranker V3 if enabled."""
        if self._er3_model is not None:
            return
        if not self.config.pipeline_id.entity_ranker_v3_enabled:
            return
        try:
            from stack.encoder.entity_ranker_v3 import load_ranker_v3
            model, tokenizer, _ = load_ranker_v3(
                self.config.pipeline_id.entity_ranker_v3_dir
            )
            model.eval()
            self._er3_model = model
            self._er3_tokenizer = tokenizer
        except Exception as exc:
            raise RuntimeError(
                f"Entity Ranker V3 enabled but failed to load from "
                f"{self.config.pipeline_id.entity_ranker_v3_dir}: {exc}"
            )

    def _resolve_entities_er3(self, question: str) -> list[str]:
        """Use Entity Ranker V3 for exhaustive canonical-vocabulary ranking."""
        import torch
        from stack.encoder.canonical_mapping import _is_canonical_id, build_canonical_mapping, apply_canonical_mapping
        from stack.encoder.entity_text import build_entity_text

        self._load_er3()
        if self._er3_model is None:
            return []

        # Build exhaustive canonical candidate pool
        all_canonical = sorted(
            nid for nid in self.graph._nodes
            if _is_canonical_id(str(nid)) and self.graph.get_node(nid) is not None
        )
        entity_texts = [build_entity_text(cid, self.graph) for cid in all_canonical]

        # Score with ER3
        offsets, indices = self._er3_tokenizer.tokenize_batch([question])
        with torch.no_grad():
            scores = self._er3_model(
                torch.tensor(indices), torch.tensor(offsets[:-1]),
                entity_texts, self._er3_tokenizer,
            )
        ranked_indices = torch.argsort(scores[0], descending=True).tolist()
        ranked_ids = [all_canonical[i] for i in ranked_indices]

        # Apply canonical mapping and cap at K=10
        mapping = build_canonical_mapping(self.graph)
        return apply_canonical_mapping(ranked_ids, mapping, top_k=10)

    def _run_single(
        self, qid: str, question: str, model: ModelInterface
    ) -> QuestionResult:
        qr = QuestionResult(
            question_id=qid,
            question=question,
            question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
        )

        try:
            # ── Entity resolution: ER3 or lexical parser ──
            if self.config.pipeline_id.entity_ranker_v3_enabled:
                er3_entities = self._resolve_entities_er3(question)
                qr.predicted_entities = er3_entities
                qr.entity_resolution_method = "entity_ranker_v3"
                qr.selected_entry_nodes = er3_entities[:self.config.max_entry_nodes]
                qr.lexical_fallback_used = False

                from nexus.query.parser import parse_question
                parsed = parse_question(question, self.graph, config=self.config)
                parsed.entity_ids = er3_entities
                qr.parsed_intent = parsed.intent
            else:
                result_raw = answer_question(
                    question, self.graph, model=model,
                    verifier=self.verifier, config=self.config,
                )
                parsed = result_raw.get("parsed_query")
                if parsed:
                    qr.parsed_intent = parsed.intent
                    qr.predicted_entities = list(parsed.entity_ids)
                    qr.entity_resolution_method = parsed.resolution_method
                    qr.selected_entry_nodes = list(parsed.entity_ids[:self.config.max_entry_nodes])
                    qr.lexical_fallback_used = True

            # ── Continue with answer_question for graph paths/evidence ──
            if self.config.pipeline_id.entity_ranker_v3_enabled:
                result = answer_question(
                    question, self.graph, model=model,
                    verifier=self.verifier, config=self.config,
                )
                result["parsed_query"] = parsed
            else:
                result = result_raw

            qr.graph_paths_count = result.get("path_count", 0)
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

            timing = result.get("timing", {})
            if isinstance(timing, dict):
                qr.per_stage_latency_ms = {
                    k: round(v * 1000, 3) if isinstance(v, (int, float)) else 0
                    for k, v in timing.items()
                }

            # Check lexical fallback
            if (not self.config.enable_associative_encoder
                    and not self.config.pipeline_id.entity_ranker_v3_enabled):
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
                    "selected_entry_nodes": qr.selected_entry_nodes,
                    "graph_paths_count": qr.graph_paths_count,
                    "path_scores": qr.path_scores,
                    "evidence_pack_keys": qr.evidence_pack_keys,
                    "answer": qr.answer[:500],
                    "verifier_passed": qr.verifier_passed,
                    "hallucination_rate": qr.hallucination_rate,
                    "cascade_level": qr.cascade_level,
                    "failure_category": qr.failure_category,
                    "per_stage_latency_ms": qr.per_stage_latency_ms,
                    "lexical_fallback_used": qr.lexical_fallback_used,
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
