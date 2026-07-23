"""
NEXUS QA Benchmark Harness (P3 -- Phase 3/4).

Compares the full NEXUS pipeline against an evidence-blind baseline on the QA dataset.
Key metrics: accuracy score, hallucination rate, answer rate, latency, verification pass rate.

Reproducibility: The benchmark graph is built deterministically from
populate_from_experiments + ingest_docs in fixed order. Results are exactly
reproducible from committed code with no non-deterministic components.

Model pinned to qwen2.5:latest for reproducibility — change only in controlled experiments.

Exact reproduction command:
    python benchmarks/run_benchmark.py --limit 50 --output benchmarks/results/my_run_TIMESTAMP.json

Usage:
    python benchmarks/run_benchmark.py --limit 50 --output benchmarks/results/my_run.json
    python benchmarks/run_benchmark.py --limit 100 --output benchmarks/results/my_run.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import (
    DummyModel, EvidenceBlindModel, ModelInterface,
    get_available_model, FallbackModel, SynthesizingModel,
)
from nexus.reasoning.verifier import Verifier, VerificationResult
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG

# Cost model for local-only pricing
from benchmarks.cost_model import (
    LocalCostModel, BlendedRouterCost,
    estimate_cost_per_1k, format_cost_comparison, format_router_cost_comparison,
    FRONTIER_PRICING,  # retained for historical reference
)
from benchmarks.compare_arms import compare_paired


# ---- Token counting (for conciseness) ----

def _count_tokens(text: str) -> int:
    """Simple word-count token estimation (split on whitespace)."""
    return len(text.split())


# ---- Question loader ----

def load_questions(jsonl_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load questions from the JSONL dataset, optionally limited to N questions."""
    questions: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit and limit > 0:
        questions = questions[:limit]
    return questions


# ---- Key-fact accuracy scoring ----

# Regex patterns for extracting key facts from text
_FACT_PATTERNS = [
    # Percentages: 99.87%, 100%, 50%, 96.6% (lookahead avoids consuming trailing punctuation)
    (re.compile(r'\b(\d+\.?\d*\s*%)(?=\s|$|[,.);])'), "percentage"),
    # Numbers with "million": 15.7 million, 19,000
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*million\b', re.IGNORECASE), "number+million"),
    # Numbers with common technical units: 1,650 slots, 19,000 examples, 853 vocab tokens
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:live\s+)?(slots?|examples?|tokens?|parameters?|params?|subkeys?|distractors?|vocabulary|hops?)\b', re.IGNORECASE), "number+unit"),
    # Standalone large numbers (>=100) — exclude digits part of percentages
    (re.compile(r'\b(\d{3,}(?:,\d{3})*(?:\.\d+)?)\b(?!\s*%)'), "large_number"),
    # @ notation: all_required@32, Rec@8
    (re.compile(r'\b(\w+@\d+)\b'), "at_notation"),
    # K= notation: K=32
    (re.compile(r'\b([Kk]=\d+)\b'), "k_notation"),
    # Named experiment IDs: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
    (re.compile(r'\b(Exp_\d+_\d+[A-Z]?_\w+)\b'), "experiment_id"),
    # Named concept IDs: Concept_SelectorBottleneck, Concept_ArchitectureWorks
    (re.compile(r'\b(Concept_\w+)\b'), "concept_id"),
    # Named decision IDs: Decision_PivotToNEXUS
    (re.compile(r'\b(Decision_\w+)\b'), "decision_id"),
    # Relation words: depends_on, validates, caused_by, contradicts, etc.
    (re.compile(r'\b(depends_on|validates|caused_by|contradicts|implements|mentioned_in|derived_from|related_to|replaces|blocked_by)\b', re.IGNORECASE), "relation"),
    # Key named modes: core_only, oracle_memory, retrieved_memory, random_memory, oracle_text_memory
    (re.compile(r'\b(core_only|oracle_memory|retrieved_memory|random_memory|oracle_text_memory|oracle_filter|oracle_text_memory|retrieved_memory_external_text_query)\b', re.IGNORECASE), "sam_mode"),
    # Gate references: Gate 1, Gate 2
    (re.compile(r'\b(Gate\s+\d+)\b', re.IGNORECASE), "gate_ref"),
]


def _extract_key_facts(text: str) -> set[str]:
    """Extract key facts from text using defined regex patterns.
    
    Returns a set of normalized fact strings suitable for set intersection.
    """
    facts: set[str] = set()
    for pattern, fact_type in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            # Normalize: lowercase, strip whitespace
            fact_str = match.group(0).strip().lower()
            # Normalize comma-separated numbers: 1,650 -> 1650
            fact_str = re.sub(r'(\d),(\d)', r'\1\2', fact_str)
            facts.add(fact_str)
    return facts


def _extract_numbers(text: str) -> set[float]:
    """Extract all numeric values from text: integers, decimals, percentages.

    Percentages are converted to decimals: "99.87%" -> 0.9987, "100%" -> 1.0.
    Comma-separated numbers are normalized: "1,650" -> 1650.0.
    Avoids false positives like extracting "1" from "1-hop".
    Returns a set of floats.
    """
    numbers: set[float] = set()

    # Tokenize on whitespace
    tokens: list[str] = re.findall(r'[^\s]+', text)
    percent_re = re.compile(r'^([\d,]+\.?\d*)\s*%$')

    for token in tokens:
        # Strip trailing punctuation that is not %
        while token and token[-1] in ',.;:)!?' and token[-1] != '%':
            token = token[:-1]
        if not token:
            continue

        # Check for percentage (e.g., "99.87%", "100%")
        pm = percent_re.match(token)
        if pm:
            try:
                val = float(pm.group(1).replace(',', ''))
                numbers.add(round(val / 100.0, 10))
            except ValueError:
                pass
            continue

        # Check if token is purely numeric (possibly with commas/decimals)
        # Use isdigit check on the stripped string to avoid matching "1-hop" etc.
        stripped = token.replace(',', '')
        if stripped.replace('.', '', 1).isdigit():
            try:
                val = float(stripped)
                numbers.add(val)
            except ValueError:
                pass

    return numbers


def _fuzzy_number_match(pred_nums: set[float], gt_nums: set[float]) -> tuple[int, int]:
    """Match predicted numbers against ground truth with 5% relative tolerance.

    Returns (matches, total_gt_nums).
    A predicted number matches a ground truth number if:
        abs(pred - gt) / max(abs(gt), 0.001) < 0.05  (5% relative tolerance)
        OR abs(pred - gt) < 0.001  (near-exact match)
    Each gt number is matched at most once (greedy best-match).
    """
    if not gt_nums:
        return 0, 0

    pred_list: list[float | None] = list(pred_nums)
    gt_list = sorted(gt_nums, reverse=True)
    matches = 0

    for gt in gt_list:
        best_idx: int = -1
        best_err: float = float('inf')

        for i, pred in enumerate(pred_list):
            if pred is None:
                continue
            denom = max(abs(gt), 0.001)
            rel_err = abs(pred - gt) / denom
            abs_err = abs(pred - gt)

            if rel_err < 0.05 or abs_err < 0.001:
                if rel_err < best_err:
                    best_err = rel_err
                    best_idx = i

        if best_idx >= 0:
            matches += 1
            pred_list[best_idx] = None  # mark as consumed

    return matches, len(gt_nums)


def compute_exact_fact_score(predicted_answer: str, ground_truth: str) -> float | None:
    """Compute key-fact match score using exact regex-based matching (original method).

    Returns None if ground truth has no extractable facts.
    """
    if "insufficient evidence" in predicted_answer.lower():
        return 0.0

    gt_facts = _extract_key_facts(ground_truth)
    pred_facts = _extract_key_facts(predicted_answer)

    if not gt_facts:
        return None

    intersection = gt_facts & pred_facts
    score = len(intersection) / len(gt_facts)
    return round(score, 4)


def compute_key_fact_score(
    predicted_answer: str, ground_truth: str, use_fuzzy: bool = True
) -> dict[str, Any]:
    """Compute key-fact match score with fuzzy numeric scoring.

    Returns a dict with:
        - fuzzy_accuracy: primary score (fuzzy numeric or exact regex fallback)
        - exact_accuracy: old exact regex score for comparison
        - scoring_detail: breakdown of numbers, matches, and entity overlap

    If the predicted answer says "Insufficient evidence", both scores = 0.0.
    If ground truth has no extractable facts, both scores are None.
    """
    # Build the "no score" detail skeleton
    empty_detail: dict[str, Any] = {
        "gt_numbers": [],
        "pred_numbers": [],
        "fuzzy_matches": 0,
        "total_gt": 0,
        "fuzzy_score": 0.0,
        "exact_score": 0.0,
        "entity_overlap": [],
    }

    if "insufficient evidence" in predicted_answer.lower():
        return {
            "fuzzy_accuracy": 0.0,
            "exact_accuracy": 0.0,
            "scoring_detail": empty_detail,
        }

    gt_facts = _extract_key_facts(ground_truth)
    pred_facts = _extract_key_facts(predicted_answer)

    if not gt_facts:
        empty_detail["fuzzy_score"] = None
        empty_detail["exact_score"] = None
        return {
            "fuzzy_accuracy": None,
            "exact_accuracy": None,
            "scoring_detail": empty_detail,
        }

    # Exact score (old regex-based method)
    intersection = gt_facts & pred_facts
    exact_score: float = round(len(intersection) / len(gt_facts), 4)

    # Fuzzy numeric scoring
    gt_nums = _extract_numbers(ground_truth)
    pred_nums = _extract_numbers(predicted_answer)
    entity_overlap = sorted(list(intersection))

    if gt_nums and use_fuzzy:
        fuzzy_matches, total_gt = _fuzzy_number_match(pred_nums, gt_nums)
        fuzzy_score: float | None = (
            round(fuzzy_matches / total_gt, 4) if total_gt > 0 else None
        )
        primary_accuracy: float = fuzzy_score if fuzzy_score is not None else exact_score
    else:
        # No numeric ground truth or fuzzy disabled: fall back to exact regex
        fuzzy_matches = 0
        total_gt = 0
        fuzzy_score = None
        primary_accuracy = exact_score

    return {
        "fuzzy_accuracy": primary_accuracy,
        "exact_accuracy": exact_score,
        "scoring_detail": {
            "gt_numbers": sorted(list(gt_nums)),
            "pred_numbers": sorted(list(pred_nums)),
            "fuzzy_matches": fuzzy_matches,
            "total_gt": total_gt,
            "fuzzy_score": fuzzy_score,
            "exact_score": exact_score,
            "entity_overlap": entity_overlap,
        },
    }


# ---- Graph construction ----


def build_benchmark_graph(config: NEXUSConfig = DEFAULT_CONFIG) -> tuple[InMemoryGraphStore, dict[str, Any]]:
    """Thin wrapper — canonical construction lives in nexus.ingestion.canonical_graph."""
    from nexus.ingestion.canonical_graph import build_canonical_sam_graph

    return build_canonical_sam_graph(config)


# ---- Pipeline runners ----

def run_nexus_pipeline(
    question_text: str,
    graph: InMemoryGraphStore,
    model: ModelInterface,
    verifier: Verifier,
    embedding_index: Any = None,
) -> dict[str, Any]:
    """
    Run the full NEXUS pipeline and return timing + metrics.
    
    Returns a dict with:
        - answer: the generated answer text
        - passed: whether verification passed
        - hallucination_rate: float 0.0-1.0
        - supported_count: number of supported claims
        - unsupported_count: number of unsupported claims
        - path_count: number of traversal paths found
        - is_insufficient: whether the answer says "Insufficient evidence"
        - latency_s: total wall-clock seconds
        - latency_breakdown: per-step timing dict
        - prompt_tokens: estimated prompt tokens
        - completion_tokens: estimated completion (answer) tokens
        - error: error message if pipeline crashed (None otherwise)
    """
    t0 = time.perf_counter()
    try:
        result = answer_question(
            question_text, graph, model=model, verifier=verifier,
            embedding_index=embedding_index,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "answer": f"[ERROR] {exc}",
            "passed": False,
            "hallucination_rate": 1.0,
            "supported_count": 0,
            "unsupported_count": 0,
            "path_count": 0,
            "is_insufficient": False,
            "latency_s": round(elapsed, 4),
            "latency_breakdown": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(exc),
            "entity_resolution_method": "none",
            "cascade_level": 0,
            "resolution_confidence": 0.0,
            "parsed_entity_ids": [],
            "post_edit_changes": None,
            "evidence_raw": "",
        }
    elapsed = time.perf_counter() - t0

    verif: VerificationResult | None = result.get("verification")
    answer = result.get("answer", "")

    is_insufficient = "insufficient evidence" in answer.lower()

    if verif is not None:
        passed = verif.passed
        hall_rate = verif.hallucination_rate
        supported = verif.supported_count
        unsupported = len(verif.unsupported_claims)
    else:
        passed = True
        hall_rate = 0.0
        supported = 0
        unsupported = 0

    # Per-step timing breakdown
    latency_breakdown = result.get("timing", {})

    # Token counting for cost estimation
    prompt_text = result.get("prompt_text", "")
    prompt_tokens = _count_tokens(prompt_text)
    completion_tokens = _count_tokens(answer)

    # Extract evidence facts for oracle test
    evidence_raw = ""
    evidence_pack = result.get("evidence_pack", {})
    if evidence_pack:
        # Flatten evidence_pack to text for fact extraction
        node_facts = evidence_pack.get("node_facts", [])
        facts = evidence_pack.get("facts", [])
        neighbor_facts = evidence_pack.get("neighbor_facts", [])
        numbers = evidence_pack.get("numbers", [])
        
        # Concatenate all evidence texts
        evidence_texts = []
        for nf in node_facts:
            evidence_texts.append(nf.get("text", ""))
        for f in facts:
            evidence_texts.append(f)
        for nf in neighbor_facts:
            evidence_texts.append(nf.get("text", ""))
        for num_entry in numbers:
            num_entry_str = "; ".join(f"{k}: {v}" for k, v in num_entry.items() if k != "entity")
            if num_entry_str:
                evidence_texts.append(num_entry_str)
        evidence_raw = " ".join(evidence_texts)

    return {
        "answer": answer,
        "passed": passed,
        "hallucination_rate": round(hall_rate, 4),
        "supported_count": supported,
        "unsupported_count": unsupported,
        "path_count": result.get("path_count", 0),
        "is_insufficient": is_insufficient,
        "latency_s": round(elapsed, 4),
        "latency_breakdown": latency_breakdown,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": None,
        "entity_resolution_method": result.get("entity_resolution_method", "none"),
        "cascade_level": result.get("cascade_level", 0),
        "resolution_confidence": result.get("resolution_confidence", 0.0),
        "parsed_entity_ids": (
            result["parsed_query"].entity_ids if result.get("parsed_query") else []
        ),
        "post_edit_changes": result.get("post_edit_changes"),
        "evidence_raw": evidence_raw,
    }


def run_baseline(
    question_text: str,
    model: ModelInterface,
) -> dict[str, Any]:
    """
    Run the evidence-blind baseline.
    
    The model receives the question WITHOUT any evidence from the
    knowledge graph — simulating a model that can only use general
    knowledge. Uses the same prompt structure as NEXUS but with
    evidence stripped out.
    """
    t0 = time.perf_counter()
    prompt = (
        "SYSTEM: You are a precise reasoning assistant. "
        "Answer based on your general knowledge. "
        "If you truly don't know, say so honestly.\n\n"
        f"QUESTION: {question_text}\n\n"
        "EVIDENCE:\n  (No evidence found in the knowledge graph.)\n\n"
        "ANSWER:"
    )
    try:
        answer = model.generate(prompt)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "answer": f"[ERROR] {exc}",
            "latency_s": round(elapsed, 4),
            "prompt_tokens": _count_tokens(prompt),
            "completion_tokens": 0,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - t0
    
    # Check if the answer shows evidence of comprehension
    is_insufficient = "insufficient evidence" in answer.lower()
    
    return {
        "answer": answer,
        "is_insufficient": is_insufficient,
        "latency_s": round(elapsed, 4),
        "prompt_tokens": _count_tokens(prompt),
        "completion_tokens": _count_tokens(answer),
        "error": None,
    }


def run_rag_retrieval(
    question_text: str,
    graph: InMemoryGraphStore,
    model: ModelInterface,
) -> dict[str, Any]:
    """Run simple keyword-based RAG retrieval + model generation.

    Uses the graph's keyword index (find_entity_by_keywords) to locate
    relevant entities, extracts their properties as evidence, and feeds
    them to the model as context.
    """
    t0 = time.perf_counter()

    # Keyword search in graph
    keyword_hits = graph.find_entity_by_keywords(question_text)
    evidence_texts: list[str] = []
    for nid, _ in keyword_hits[:10]:  # top 10 matching entities
        node = graph.get_node(nid)
        if node is None:
            continue
        props = node.properties
        name = props.get("name", nid)
        desc = props.get("description", "")
        key_finding = props.get("key_finding", "")
        parts = [name]
        if key_finding:
            parts.append(f"key_finding: {key_finding}")
        if desc:
            parts.append(desc)
        evidence_texts.append(". ".join(parts))

    evidence_raw = " | ".join(evidence_texts) if evidence_texts else ""
    retrieval_tokens = _count_tokens(evidence_raw)

    evidence_block = (
        evidence_raw if evidence_raw
        else "(No evidence found in the knowledge graph.)"
    )

    prompt = (
        "SYSTEM: You are a precise reasoning assistant. "
        "Answer based on the evidence provided. "
        "If the evidence is insufficient, say so honestly.\n\n"
        f"QUESTION: {question_text}\n\n"
        f"EVIDENCE:\n  {evidence_block}\n\n"
        "ANSWER:"
    )

    try:
        answer = model.generate(prompt)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "answer": f"[ERROR] {exc}",
            "is_insufficient": False,
            "latency_s": round(elapsed, 4),
            "prompt_tokens": _count_tokens(prompt),
            "completion_tokens": 0,
            "retrieval_tokens": retrieval_tokens,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - t0

    is_insufficient = "insufficient evidence" in answer.lower()

    return {
        "answer": answer,
        "is_insufficient": is_insufficient,
        "latency_s": round(elapsed, 4),
        "prompt_tokens": _count_tokens(prompt),
        "completion_tokens": _count_tokens(answer),
        "retrieval_tokens": retrieval_tokens,
        "error": None,
    }


def validate_benchmark_results(
    results: list[dict],
    config: dict,
    question_count: int | None = None,
    summary: dict | None = None,
    paired_comparison: dict | None = None,
    nexus_config_obj: Any = None,
    allow_experimental: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate benchmark results. Returns (hard_errors, warnings).

    Hard errors cause the run to be marked INVALID (exit code 1).
    Warnings are written into the output file as suspect flags.

    Args:
        results: List of per-question result dicts (two rows per question).
        config: Config header dict (arm_rag, arm_nexus, etc.).
        question_count: Expected number of unique questions.
        summary: Computed summary dict (from compute_summary).
        paired_comparison: Paired comparison dict (from compare_paired).
        nexus_config_obj: NEXUSConfig dataclass for config integrity checks.
        allow_experimental: If True, skip experimental flag guard.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check NEXUS arm
    nexus_results = [r for r in results if r.get("arm_mode") == "nexus"]
    nexus_tokens = [r.get("retrieval_tokens") or 0 for r in nexus_results]
    if config.get("arm_nexus") == "nexus" and not nexus_results:
        errors.append("NEXUS arm configured as 'nexus' but produced 0 result rows")
    if nexus_results and sum(nexus_tokens) == 0:
        errors.append("NEXUS arm: all retrieval_tokens == 0 — evidence pack empty")

    # Check RAG arm
    rag_results = [r for r in results if r.get("arm_mode") == "rag_retrieval"]
    rag_tokens = [r.get("retrieval_tokens") or 0 for r in rag_results]
    if rag_results and sum(rag_tokens) == 0:
        errors.append("RAG arm labeled rag_retrieval but all retrieval_tokens == 0 — check config")

    # ── Guard 1: RAG arm zero result rows (when configured) ──
    rag_arm_mode = config.get("arm_rag", "")
    if rag_arm_mode == "rag_retrieval" and len(rag_results) == 0:
        errors.append(
            f"RAG arm configured as 'rag_retrieval' but produced 0 result rows "
            f"({len(results)} total rows)"
        )

    # ── Guard 1b: Empty RAG summary ──
    # When rag_retrieval is configured, summary.baseline must be non-empty.
    if rag_arm_mode == "rag_retrieval" and summary is not None:
        rag_summary = summary.get("baseline")
        if not rag_summary:
            errors.append(
                "Empty RAG arm: summary.baseline is empty or missing — "
                "RAG arm produced no meaningful data"
            )

    # ── Guard 2: Row count mismatch ──
    if question_count is not None:
        arm_count = 2  # nexus + rag/baseline
        expected_rows = question_count * arm_count
        if len(results) != expected_rows:
            errors.append(
                f"Row count mismatch: expected {expected_rows} "
                f"({question_count} questions × {arm_count} arms), "
                f"got {len(results)}"
            )

    # ── Guard 3: paired_n == 0 or absent ──
    if paired_comparison is not None:
        paired_n = paired_comparison.get("paired_n", 0)
        if paired_n == 0:
            errors.append(
                "paired_n == 0: no question has both arms scored — "
                "paired comparison is impossible"
            )

    # ── Guard 4: Any arm answered count == 0 ──
    if summary is not None:
        nexus_answered = summary.get("nexus", {}).get("answered", -1)
        baseline_answered = summary.get("baseline", {})
        if not baseline_answered or baseline_answered.get("insufficient_evidence") is None:
            # evidence_blind arm has no "answered" — skip
            pass
        else:
            # For rag_retrieval, check that baseline has data (avg_accuracy present)
            if rag_arm_mode == "rag_retrieval" and baseline_answered.get("avg_accuracy") is None:
                errors.append(
                    "RAG arm answered count is effectively 0 — no scorable answers produced"
                )
        if nexus_answered == 0:
            errors.append(
                "NEXUS arm answered 0 questions — all questions had insufficient evidence or errors"
            )

    # ── Guard 5: Config integrity assertion ──
    if nexus_config_obj is not None and not allow_experimental:
        if nexus_config_obj.enable_cooccurrence_edges:
            errors.append(
                "Config integrity FAIL: enable_cooccurrence_edges is True. "
                "Re-run with --allow-experimental if this is intentional."
            )
        if nexus_config_obj.enable_embedding_er:
            errors.append(
                "Config integrity FAIL: enable_embedding_er is True. "
                "Re-run with --allow-experimental if this is intentional."
            )

    # ── Guard 6: Sanity band on avg_paths (WARNING only, not hard failure) ──
    # With experimental flags off, avg_paths_found >= 8 is flagged as "suspect"
    # (warning only — could be legitimate high-path graph from beam_width).
    if summary is not None and not allow_experimental:
        avg_paths = summary.get("nexus", {}).get("avg_paths_found", 0)
        if avg_paths >= 8:
            warnings.append(
                f"Suspect: avg_paths_found={avg_paths} >= 8 with experimental "
                f"flags off. This is suspicious — verify co-occurrence edges "
                f"and beam_width settings. Run flagged as suspect."
            )

    # Check both arms used same model backend
    nexus_models = set(r.get("nexus", {}).get("model", "") for r in results)
    rag_models = set(r.get("baseline", {}).get("model", "") for r in results)
    if nexus_models and rag_models and nexus_models != rag_models:
        errors.append(f"Model mismatch: NEXUS={nexus_models}, RAG={rag_models}")

    return errors, warnings


def validate_benchmark_artifact(path: str | Path) -> tuple[list[str], list[str]]:
    """Validate the exact JSON artifact emitted by a benchmark run.

    This is intentionally separate from validating in-memory rows: publication
    status must be based on the serialized artifact that reviewers will inspect.
    """
    artifact_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    if not artifact_path.exists():
        return [f"Artifact missing: {artifact_path}"], []
    if artifact_path.stat().st_size == 0:
        return [f"Artifact is zero-byte: {artifact_path}"], []
    try:
        with artifact_path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"Artifact unreadable: {exc}"], []
    if not isinstance(artifact, dict):
        return ["Artifact root must be an object"], []

    required = {"config", "graph_provenance", "summary", "paired_comparison", "results"}
    missing = sorted(required - set(artifact))
    if missing:
        return [f"Artifact missing required keys: {missing}"], []

    config = artifact["config"]
    provenance = artifact["graph_provenance"]
    if not isinstance(config, dict) or not config:
        errors.append("Artifact configuration missing or empty")
        config = {}
    if not isinstance(provenance, dict):
        errors.append("Artifact graph provenance missing or invalid")
        provenance = {}
    if "effective_config" not in provenance or "edge_type_counts" not in provenance:
        errors.append("Artifact provenance incomplete: effective graph configuration and edge counts are required")
    summary = artifact["summary"]
    if not isinstance(summary, dict) or not summary:
        errors.append("Artifact summary incomplete: summary is empty or invalid")
        summary = {}
    if not summary.get("nexus") or not summary.get("baseline"):
        errors.append("Artifact summary incomplete: both summary.nexus and summary.baseline are required")
    effective = provenance.get("effective_config", {})
    edge_counts = provenance.get("edge_type_counts", {})
    if isinstance(edge_counts, dict) and isinstance(provenance.get("edge_count"), int):
        if sum(value for value in edge_counts.values() if isinstance(value, int)) != provenance["edge_count"]:
            errors.append("Artifact graph edge-count metadata mismatch")
    for flag in (
        "enable_cooccurrence_edges", "enable_embedding_er",
        "enable_associative_encoder", "enable_normalization",
    ):
        if flag in config and flag in effective and config[flag] != effective[flag]:
            errors.append(f"Artifact config mismatch for {flag}: header={config[flag]} graph={effective[flag]}")
    if config.get("enable_cooccurrence_edges") is False:
        related_count = provenance.get("edge_type_counts", {}).get("related_to", 0)
        if related_count:
            errors.append(
                f"Artifact config mismatch: cooccurrence disabled but graph has {related_count} related_to edges"
            )

    results = artifact["results"]
    if not isinstance(results, list):
        errors.append("Artifact results are missing or invalid")
        results = []
    paired_comparison = artifact["paired_comparison"]
    if not isinstance(paired_comparison, dict):
        errors.append("Artifact paired comparison is missing or invalid")
        paired_comparison = {}
    expected_questions = summary.get("total_questions")
    row_errors, row_warnings = validate_benchmark_results(
        results, config,
        question_count=expected_questions,
        summary=summary,
        paired_comparison=paired_comparison,
    )
    errors.extend(row_errors)
    warnings.extend(row_warnings)
    return errors, warnings


# ---- Metrics computation ----

def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from benchmark results.

    Deduplicates rows by question_id to avoid double-counting when both
    NEXUS and RAG arm rows are present for the same question.
    """
    # Deduplicate: keep the first row per question_id
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        qid = r.get("question_id", "")
        if qid not in seen:
            seen.add(qid)
            deduped.append(r)
    results = deduped
    total = len(results)
    nexus_errors = sum(1 for r in results if r["nexus"].get("error"))
    baseline_errors = sum(1 for r in results if r["baseline"].get("error"))

    nexus_answered = sum(1 for r in results if not r["nexus"]["is_insufficient"] and not r["nexus"].get("error"))
    nexus_insufficient = sum(1 for r in results if r["nexus"]["is_insufficient"])
    nexus_passed = sum(1 for r in results if r["nexus"]["passed"] and not r["nexus"].get("error") and not r["nexus"]["is_insufficient"])
    nexus_hall_rates = [r["nexus"]["hallucination_rate"] for r in results if not r["nexus"].get("error")]
    nexus_latencies = [r["nexus"]["latency_s"] for r in results if not r["nexus"].get("error")]
    nexus_paths = [r["nexus"]["path_count"] for r in results if not r["nexus"].get("error")]

    # Post-edit statistics
    post_edit_fixed = sum(
        (r["nexus"].get("post_edit_changes") or {}).get("numbers_fixed", 0)
        for r in results if not r["nexus"].get("error")
    )
    post_edit_removed = sum(
        (r["nexus"].get("post_edit_changes") or {}).get("numbers_removed", 0)
        for r in results if not r["nexus"].get("error")
    )
    post_edit_total = post_edit_fixed + post_edit_removed

    baseline_latencies = [r["baseline"]["latency_s"] for r in results if not r["baseline"].get("error")]
    baseline_insufficient = sum(1 for r in results if r["baseline"]["is_insufficient"])

    # Token metrics
    nexus_prompt_tokens = [r["nexus"]["prompt_tokens"] for r in results if not r["nexus"].get("error")]
    nexus_completion_tokens = [r["nexus"]["completion_tokens"] for r in results if not r["nexus"].get("error")]
    baseline_prompt_tokens = [r["baseline"]["prompt_tokens"] for r in results if not r["baseline"].get("error")]
    baseline_completion_tokens = [r["baseline"]["completion_tokens"] for r in results if not r["baseline"].get("error")]

    # Latency breakdown aggregation
    latency_breakdowns = [
        r["nexus"]["latency_breakdown"]
        for r in results
        if not r["nexus"].get("error") and r["nexus"].get("latency_breakdown")
    ]
    avg_latency_breakdown: dict[str, float] = {}
    if latency_breakdowns:
        for key in latency_breakdowns[0]:
            vals = [lb[key] for lb in latency_breakdowns if key in lb]
            if vals:
                avg_latency_breakdown[key] = round(sum(vals) / len(vals), 6)

    # Accuracy scores (exclude None values — questions without extractable GT facts)
    nexus_accuracies = [
        r["nexus"]["accuracy"]
        for r in results
        if not r["nexus"].get("error") 
        and "accuracy" in r["nexus"] 
        and r["nexus"]["accuracy"] is not None
    ]
    nexus_exact_accuracies = [
        r["nexus"]["exact_accuracy"]
        for r in results
        if not r["nexus"].get("error") 
        and "exact_accuracy" in r["nexus"]
        and r["nexus"]["exact_accuracy"] is not None
    ]
    baseline_accuracies = [
        r["baseline"]["accuracy"]
        for r in results
        if not r["baseline"].get("error") 
        and "accuracy" in r["baseline"]
        and r["baseline"]["accuracy"] is not None
    ]
    baseline_exact_accuracies = [
        r["baseline"]["exact_accuracy"]
        for r in results
        if not r["baseline"].get("error") 
        and "exact_accuracy" in r["baseline"]
        and r["baseline"]["exact_accuracy"] is not None
    ]
    scorable_count = len(nexus_accuracies)  # questions with measurable ground truth

    # Number reproduction stats (from scoring_detail)
    num_gt_counts: list[int] = []
    num_matched_counts: list[int] = []
    for r in results:
        detail = r["nexus"].get("scoring_detail", {})
        if detail.get("total_gt", 0) > 0:
            num_gt_counts.append(detail["total_gt"])
            num_matched_counts.append(detail.get("fuzzy_matches", 0))
    total_gt_numbers = sum(num_gt_counts)
    total_matched_numbers = sum(num_matched_counts)
    number_recall_rate = round(total_matched_numbers / total_gt_numbers, 4) if total_gt_numbers > 0 else 0.0
    avg_numbers_in_gt = round(sum(num_gt_counts) / len(num_gt_counts), 2) if num_gt_counts else 0.0
    avg_numbers_matched = round(sum(num_matched_counts) / len(num_matched_counts), 2) if num_matched_counts else 0.0

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # Conciseness metrics
    conciseness_data = [
        r["nexus"]["conciseness"]
        for r in results
        if not r["nexus"].get("error") and "conciseness" in r["nexus"]
    ]
    avg_ans_tokens = round(avg([c["answer_tokens"] for c in conciseness_data]), 1)
    avg_c_ratio = round(avg([c["ratio"] for c in conciseness_data]), 2)
    verbose_count = sum(1 for c in conciseness_data if c["too_verbose"])
    verbose_rate = round(verbose_count / len(conciseness_data), 4) if conciseness_data else 0.0

    conciseness_summary = {
        "avg_answer_tokens": avg_ans_tokens,
        "avg_conciseness_ratio": avg_c_ratio,
        "verbose_count": verbose_count,
        "verbose_rate": verbose_rate,
    }

    # Per-hop accuracy breakdown
    accuracy_by_hops: dict[str, dict[str, Any]] = {}
    for r in results:
        if r["nexus"].get("error") or r["nexus"]["accuracy"] is None:
            continue
        h = str(r.get("hops", "?"))
        if h not in accuracy_by_hops:
            accuracy_by_hops[h] = {"count": 0, "accuracies": []}
        accuracy_by_hops[h]["count"] += 1
        accuracy_by_hops[h]["accuracies"].append(r["nexus"]["accuracy"])
    accuracy_by_hops = {
        hop: {"count": data["count"], "avg_accuracy": avg(data["accuracies"])}
        for hop, data in sorted(accuracy_by_hops.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
    }

    # Per-question-type accuracy breakdown
    accuracy_by_type: dict[str, dict[str, Any]] = {}
    for r in results:
        if r["nexus"].get("error") or r["nexus"]["accuracy"] is None:
            continue
        qt = r.get("question_type", "unknown")
        if qt not in accuracy_by_type:
            accuracy_by_type[qt] = {"count": 0, "accuracies": []}
        accuracy_by_type[qt]["count"] += 1
        accuracy_by_type[qt]["accuracies"].append(r["nexus"]["accuracy"])
    accuracy_by_type = {
        qt: {"count": data["count"], "avg_accuracy": avg(data["accuracies"])}
        for qt, data in sorted(accuracy_by_type.items())
    }

    # Entity resolution accuracy
    er_hits = sum(
        1 for r in results
        if not r["nexus"].get("error") and r["nexus"].get("entity_resolution_hit")
    )
    er_total = sum(
        1 for r in results
        if not r["nexus"].get("error") and r["nexus"].get("entity_resolution_hit") is not None
    )
    entity_resolution_rate = round(er_hits / er_total, 4) if er_total > 0 else 0.0
    
    # Entity accuracy: resolved entities match GT expected entities
    ea_hits = sum(
        1 for r in results
        if not r["nexus"].get("error") and r["nexus"].get("entity_accuracy") is True
    )
    ea_total = sum(
        1 for r in results
        if not r["nexus"].get("error") and r["nexus"].get("entity_accuracy") is not None
    )
    entity_accuracy_rate = round(ea_hits / ea_total, 4) if ea_total > 0 else 0.0

    # ── Entity resolution by split: first-30 vs remaining (held-out) ──
    # The first 30 questions have good alias coverage (manually aliased in
    # populate_from_experiments.py); the remaining 170 are held-out.
    first_30 = [r for r in results if r.get("question_id", "") in 
                [f"q{str(n).zfill(3)}" for n in range(1, 31)]]
    remaining = [r for r in results if r.get("question_id", "") not in
                 [f"q{str(n).zfill(3)}" for n in range(1, 31)]]

    def _er_rate(subset: list[dict]) -> dict[str, Any]:
        hits = sum(1 for r in subset if not r["nexus"].get("error") and r["nexus"].get("entity_resolution_hit"))
        total_er = sum(1 for r in subset if not r["nexus"].get("error") and r["nexus"].get("entity_resolution_hit") is not None)
        return {
            "hits": hits, "total": total_er,
            "rate": round(hits / total_er, 4) if total_er > 0 else 0.0,
        }

    entity_resolution_by_split = {
        "first_30": _er_rate(first_30),
        "remaining": _er_rate(remaining),
    }

    # ── Entity resolution method breakdown: alias vs fuzzy ──
    alias_results = [r for r in results if not r["nexus"].get("error")
                     and r["nexus"].get("entity_resolution_method") == "alias"]
    fuzzy_results = [r for r in results if not r["nexus"].get("error")
                     and r["nexus"].get("entity_resolution_method") == "fuzzy"]
    none_results = [r for r in results if not r["nexus"].get("error")
                    and r["nexus"].get("entity_resolution_method") == "none"]

    n_alias = len(alias_results)
    n_fuzzy = len(fuzzy_results)
    n_none = len(none_results)
    n_total_with_method = n_alias + n_fuzzy + n_none

    alias_hit_rate = round(n_alias / n_total_with_method, 4) if n_total_with_method > 0 else 0.0
    fuzzy_hit_rate = round(n_fuzzy / n_total_with_method, 4) if n_total_with_method > 0 else 0.0

    # Accuracy by resolution method
    def _acc_for(subset: list[dict], key: str = "accuracy") -> dict[str, Any]:
        accs = [r["nexus"][key] for r in subset
                if not r["nexus"].get("error")
                and key in r["nexus"]
                and r["nexus"][key] is not None]
        return {
            "count": len(accs),
            "avg_accuracy": round(avg(accs), 4),
        }

    accuracy_by_resolution_method = {
        "alias": _acc_for(alias_results),
        "fuzzy": _acc_for(fuzzy_results),
        "none": _acc_for(none_results),
    }

    return {
        "total_questions": total,
        "scorable_questions": scorable_count,
        "nexus_errors": nexus_errors,
        "baseline_errors": baseline_errors,
        "nexus": {
            "answered": nexus_answered,
            "insufficient_evidence": nexus_insufficient,
            "answer_rate": round(nexus_answered / total, 4) if total > 0 else 0.0,
            "verification_passed": nexus_passed,
            "verification_pass_rate": round(nexus_passed / nexus_answered, 4) if nexus_answered > 0 else 0.0,
            "avg_hallucination_rate": round(avg(nexus_hall_rates), 4),
            "min_hallucination_rate": round(min(nexus_hall_rates), 4) if nexus_hall_rates else 0.0,
            "max_hallucination_rate": round(max(nexus_hall_rates), 4) if nexus_hall_rates else 0.0,
            "avg_latency_s": round(avg(nexus_latencies), 4),
            "avg_latency_breakdown": avg_latency_breakdown,
            "avg_paths_found": round(avg(nexus_paths), 2),
            "avg_prompt_tokens": round(avg(nexus_prompt_tokens), 1),
            "avg_completion_tokens": round(avg(nexus_completion_tokens), 1),
            "avg_accuracy": round(avg(nexus_accuracies), 4),
            "avg_exact_accuracy": round(avg(nexus_exact_accuracies), 4),
            "min_accuracy": round(min(nexus_accuracies), 4) if nexus_accuracies else 0.0,
            "max_accuracy": round(max(nexus_accuracies), 4) if nexus_accuracies else 0.0,
        },
        "baseline": {
            "avg_latency_s": round(avg(baseline_latencies), 4),
            "insufficient_evidence": baseline_insufficient,
            "avg_prompt_tokens": round(avg(baseline_prompt_tokens), 1),
            "avg_completion_tokens": round(avg(baseline_completion_tokens), 1),
            "avg_accuracy": round(avg(baseline_accuracies), 4),
            "avg_exact_accuracy": round(avg(baseline_exact_accuracies), 4),
        },
        "number_reproduction": {
            "total_gt_numbers": total_gt_numbers,
            "total_matched_numbers": total_matched_numbers,
            "number_recall_rate": number_recall_rate,
            "avg_numbers_in_gt": avg_numbers_in_gt,
            "avg_numbers_matched": avg_numbers_matched,
        },
        "accuracy_by_hops": accuracy_by_hops,
        "accuracy_by_type": accuracy_by_type,
        "entity_resolution_rate": entity_resolution_rate,
        "entity_accuracy": entity_accuracy_rate,
        "entity_resolution": {
            "hits": er_hits,
            "total": er_total,
            "rate": entity_resolution_rate,
        },
        "entity_resolution_by_split": entity_resolution_by_split,
        "resolution_method_breakdown": {
            "alias_count": n_alias,
            "fuzzy_count": n_fuzzy,
            "none_count": n_none,
            "alias_hit_rate": alias_hit_rate,
            "fuzzy_hit_rate": fuzzy_hit_rate,
        },
        "accuracy_by_resolution_method": accuracy_by_resolution_method,
        "conciseness": conciseness_summary,
        "post_edit": {
            "numbers_fixed": post_edit_fixed,
            "numbers_removed": post_edit_removed,
            "total_interventions": post_edit_total,
        },
    }


# ---- Display ----

def print_comparison(summary: dict[str, Any]):
    """Print a human-readable comparison table."""
    n = summary["nexus"]
    b = summary["baseline"]
    total = summary["total_questions"]

    print()
    print("=" * 72)
    print("  NEXUS QA Benchmark -- Results Summary")
    print("=" * 72)
    print(f"  Questions benchmarked:  {total}")
    print(f"  Scorable (GT has facts): {summary.get('scorable_questions', total)}")
    print(f"  NEXUS errors:           {summary['nexus_errors']}")
    print(f"  Baseline errors:        {summary['baseline_errors']}")
    print()
    print("  -- Comparison: NEXUS (with evidence) vs Baseline (without evidence) --")
    print(f"  {'Metric':<38} {'NEXUS':>10} {'Baseline':>12}")
    print(f"  {'-'*38} {'-'*10} {'-'*12}")
    
    # Answer rate
    nexus_answer_rate = n["answer_rate"]
    nexus_ans_str = f"{nexus_answer_rate:.1%} ({n['answered']}/{total})"
    base_ans_str = f"{total - b['insufficient_evidence']}/{total}" if "insufficient_evidence" in b else "N/A"
    print(f"  {'Answer rate':<38} {nexus_ans_str:>10} {base_ans_str:>12}")
    
    # Insufficient evidence
    ins_str = f"{n['insufficient_evidence']}/{total}"
    base_ins_str = f"{b['insufficient_evidence']}/{total}" if "insufficient_evidence" in b else "N/A"
    print(f"  {'Insufficient evidence':<38} {ins_str:>10} {base_ins_str:>12}")
    
    # Fuzzy accuracy (primary metric)
    n_fuzzy = f"{n['avg_accuracy']:.2%}" if n.get("avg_accuracy") is not None else "N/A"
    b_fuzzy = f"{b['avg_accuracy']:.2%}" if b.get("avg_accuracy") is not None else "N/A"
    print(f"  {'Fuzzy acc (numeric tolerant)':<38} {n_fuzzy:>10} {b_fuzzy:>12}")
    
    # Exact accuracy (old regex, for comparison)
    n_exact = f"{n['avg_exact_accuracy']:.2%}" if n.get("avg_exact_accuracy") is not None else "N/A"
    b_exact = f"{b['avg_exact_accuracy']:.2%}" if b.get("avg_exact_accuracy") is not None else "N/A"
    print(f"  {'Exact acc (regex key-fact)':<38} {n_exact:>10} {b_exact:>12}")
    
    # Number reproduction stats
    if summary.get("number_reproduction"):
        nr = summary["number_reproduction"]
        n_recall = f"{nr['number_recall_rate']:.0%}"
        n_avg = f"avg {nr['avg_numbers_matched']} of {nr['avg_numbers_in_gt']} numbers ({n_recall} recall)"
        print(f"  {'Number reproduction':<38} {n_avg:>10}")
    
    # Hallucination rate
    hall_str = f"{n['avg_hallucination_rate']:.2%}"
    print(f"  {'Avg hallucination rate':<38} {hall_str:>10} {'N/A':>12}")

    # Post-edit statistics
    if summary.get("post_edit"):
        pe = summary["post_edit"]
        if pe["total_interventions"] > 0:
            pe_str = f"{pe['numbers_fixed']} fixed, {pe['numbers_removed']} removed ({pe['total_interventions']} total)"
            print(f"  {'Post-edit interventions':<38} {pe_str:>10}")
        else:
            print(f"  {'Post-edit interventions':<38} {'none':>10}")
    
    # Verification pass rate
    ver_p_str = f"{n['verification_pass_rate']:.1%} ({n['verification_passed']}/{n['answered']})"
    print(f"  {'Verification pass rate':<38} {ver_p_str:>10} {'N/A':>12}")
    
    # Latency
    n_lat = f"{n['avg_latency_s']:.3f}s"
    b_lat = f"{b['avg_latency_s']:.3f}s"
    print(f"  {'Avg latency':<38} {n_lat:>10} {b_lat:>12}")
    
    # Paths
    n_paths = f"{n['avg_paths_found']:.1f}"
    print(f"  {'Avg paths found':<38} {n_paths:>10} {'N/A':>12}")
    
    # Entity resolution rate
    if summary.get("entity_resolution"):
        er = summary["entity_resolution"]
        er_str = f"{er['rate']:.1%} ({er['hits']}/{er['total']})"
        print(f"  {'Entity resolution rate':<38} {er_str:>10} {'N/A':>12}")
    if summary.get("entity_accuracy"):
        ea = summary["entity_accuracy"]
        ea_str = f"{ea:.1%}" if isinstance(ea, float) else str(ea)
        print(f"  {'Entity accuracy (vs GT expected)':<38} {ea_str:>10} {'N/A':>12}")

    # Entity resolution by split (first-30 vs remaining)
    if summary.get("entity_resolution_by_split"):
        er_split = summary["entity_resolution_by_split"]
        print()
        print("  -- Entity Resolution by Split (overfitting check) --")
        if "first_30" in er_split:
            f30 = er_split["first_30"]
            f30_str = f"{f30['rate']:.1%} ({f30['hits']}/{f30['total']})"
            print(f"  {'First 30 (alias-covered)':<38} {f30_str:>10}")
        if "remaining" in er_split:
            rem = er_split["remaining"]
            rem_str = f"{rem['rate']:.1%} ({rem['hits']}/{rem['total']})"
            print(f"  {'Remaining (held-out)':<38} {rem_str:>10}")
        # Gap analysis
        if "first_30" in er_split and "remaining" in er_split:
            gap = er_split["first_30"]["rate"] - er_split["remaining"]["rate"]
            gap_str = f"{gap:.1%}"
            print(f"  {'Overfitting gap':<38} {gap_str:>10}")

    # Resolution method breakdown
    if summary.get("resolution_method_breakdown"):
        rmb = summary["resolution_method_breakdown"]
        print()
        print("  -- Resolution Method: Alias vs Fuzzy --")
        alias_str = f"{rmb['alias_hit_rate']:.1%} (n={rmb['alias_count']})"
        fuzzy_str = f"{rmb['fuzzy_hit_rate']:.1%} (n={rmb['fuzzy_count']})"
        none_str = f"n={rmb['none_count']}"
        print(f"  {'Alias-resolved questions':<38} {alias_str:>10}")
        print(f"  {'Fuzzy-resolved questions':<38} {fuzzy_str:>10}")
        print(f"  {'No resolution':<38} {none_str:>10}")

    # Accuracy by resolution method
    if summary.get("accuracy_by_resolution_method"):
        acc_by_method = summary["accuracy_by_resolution_method"]
        print()
        print("  -- Accuracy by Resolution Method --")
        print(f"  {'Method':<16} {'Count':>8} {'Avg Accuracy':>14}")
        print(f"  {'-'*16} {'-'*8} {'-'*14}")
        for method, data in acc_by_method.items():
            if data["count"] > 0:
                print(f"  {method:<16} {data['count']:>8} {data['avg_accuracy']:>13.2%}")
            else:
                print(f"  {method:<16} {data['count']:>8} {'N/A':>14}")

        # Overfitting assessment
        alias_acc = acc_by_method.get("alias", {}).get("avg_accuracy", 0) or 0
        fuzzy_acc = acc_by_method.get("fuzzy", {}).get("avg_accuracy", 0) or 0
        alias_n = acc_by_method.get("alias", {}).get("count", 0)
        fuzzy_n = acc_by_method.get("fuzzy", {}).get("count", 0)
        if alias_n > 0 and fuzzy_n > 0:
            acc_gap = alias_acc - fuzzy_acc
            print()
            if acc_gap > 0.15:
                print(f"  !! OVERFITTING: Alias-resolved questions score {acc_gap:.0%} higher --")
                print(f"    aliases are doing real work but on a specific question set.")
            elif acc_gap > 0.05:
                print(f"  !! MILD OVERFITTING: Alias advantage of {acc_gap:.0%} suggests")
                print(f"    aliases help but fuzzy matching works partially.")
            else:
                print(f"  OK: NO OVERFITTING: Accuracy similar regardless of resolution method.")
                print(f"    Fuzzy matching is working well enough; aliases are a speed optimization.")

    # Conciseness
    if summary.get("conciseness"):
        c = summary["conciseness"]
        c_str = f"avg {c['avg_answer_tokens']:.0f} tokens, {c['avg_conciseness_ratio']}x GT, {c['verbose_rate']:.0%} too verbose"
        print(f"  {'Conciseness':<38} {c_str:>10}")
    
    # ── Latency Breakdown (NEXUS) ──
    n_breakdown = n.get("avg_latency_breakdown", {})
    if n_breakdown:
        print()
        print("  LATENCY (NEXUS):")
        total_lat = n["avg_latency_s"]
        step_labels = [
            ("parse_time", "parse"),
            ("traverse_time", "traverse"),
            ("evidence_time", "evidence"),
            ("prompt_time", "prompt"),
            ("generate_time", "generate"),
            ("verify_time", "verify"),
        ]
        for key, label in step_labels:
            val_s = n_breakdown.get(key, 0.0)
            pct = (val_s / total_lat * 100) if total_lat > 0 else 0.0
            if val_s < 0.001:
                formatted = "<1ms"
            elif val_s < 1.0:
                formatted = f"{val_s*1000:.0f}ms"
            else:
                formatted = f"{val_s:.2f}s"
            is_dominant = pct > 50
            marker = "  <-- dominant" if is_dominant else ""
            print(f"    {label + ':':<14} {formatted:>8} ({pct:.1f}%){marker}")
        print(f"    {'TOTAL:':<14} {total_lat:.2f}s")

    # ── Token Metrics ──
    if n.get("avg_prompt_tokens") or b.get("avg_prompt_tokens"):
        print()
        print("  TOKEN USAGE (avg per question):")
        n_prompt = f"{n['avg_prompt_tokens']:.0f}" if n.get("avg_prompt_tokens") else "N/A"
        n_compl = f"{n['avg_completion_tokens']:.0f}" if n.get("avg_completion_tokens") else "N/A"
        b_prompt = f"{b['avg_prompt_tokens']:.0f}" if b.get("avg_prompt_tokens") else "N/A"
        b_compl = f"{b['avg_completion_tokens']:.0f}" if b.get("avg_completion_tokens") else "N/A"
        print(f"    {'NEXUS:':<14} {n_prompt:>8} prompt, {n_compl:>8} completion")
        print(f"    {'Baseline:':<14} {b_prompt:>8} prompt, {b_compl:>8} completion")

    # ── Cost Comparison ──
    if n.get("avg_prompt_tokens"):
        print()
        print("  COST (local-only, electricity-based):")
        print(f"    {'NEXUS + local (electricity)':<38} {'$0.00':>10}")
        print()

        # Show local-only cost comparison
        lines = format_cost_comparison(
            "NEXUS",
            n["avg_prompt_tokens"],
            n["avg_completion_tokens"],
            local=True,
        )
        for line in lines:
            print(line)

        print(f"\n  NEXUS + Router blended cost (80% synth -> $0):")
        # Auto-load from newest throughput results — no hardcoded throughput
        router_model = BlendedRouterCost.from_latest_throughput(synth_ratio=0.8)
        if router_model is not None:
            router_lines = format_router_cost_comparison(
                "NEXUS Router",
                n["avg_prompt_tokens"],
                n["avg_completion_tokens"],
                router_model,
            )
        else:
            router_lines = [
                "  WARNING: No throughput results found. Run",
                "  `python benchmarks/throughput_bench.py` first.",
            ]
            print(f"    {router_lines[0]}")
            print(f"    {router_lines[1]}")
            router_lines = []
        for line in router_lines:
            print(line)

        # Baseline with frontier for historical reference
        if b.get("avg_prompt_tokens"):
            print(f"\n  Historical reference — what frontier APIs would cost:")
            for model_name in ["gpt-4o-mini", "claude-haiku", "gemini-flash"]:
                cost = estimate_cost_per_1k(
                    b["avg_prompt_tokens"], b["avg_completion_tokens"],
                    model_backend=model_name, local=False,
                )
                print(f"    {'Baseline + ' + model_name:<38} ${cost:.2f}")
    
    # Per-hop accuracy breakdown
    if summary.get("accuracy_by_hops"):
        print()
        print("  -- Accuracy by Number of Hops --")
        print(f"  {'Hops':<8} {'Count':>8} {'Avg Accuracy':>14}")
        print(f"  {'-'*8} {'-'*8} {'-'*14}")
        for hop, data in summary["accuracy_by_hops"].items():
            print(f"  {hop:<8} {data['count']:>8} {data['avg_accuracy']:>13.2%}")
    
    # Per-question-type accuracy breakdown
    if summary.get("accuracy_by_type"):
        print()
        print("  -- Accuracy by Question Type --")
        print(f"  {'Type':<16} {'Count':>8} {'Avg Accuracy':>14}")
        print(f"  {'-'*16} {'-'*8} {'-'*14}")
        for qt, data in summary["accuracy_by_type"].items():
            print(f"  {qt:<16} {data['count']:>8} {data['avg_accuracy']:>13.2%}")
    
    print()
    print("  Fuzzy acc = numeric matching with 5% relative tolerance (primary metric).")
    print("  Exact acc = strict regex key-fact overlap (old metric, for comparison).")
    print("  NEXUS hallucination rate measures unsupported claims in generated answers.")
    print("  The evidence-blind baseline has no graph access - it can only use")
    print("  general knowledge extracted from the question text.")
    print("=" * 72)
    print()


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS QA Benchmark Harness -- compare NEXUS vs evidence-blind baseline"
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of questions to benchmark (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="WARNING: Do not use results.json — use timestamped filenames. "
             "Output file for results (e.g., benchmarks/results/my_run_20260709T1200Z.json)"
    )
    parser.add_argument(
        "--no-populate", action="store_true",
        help="Skip graph population (use existing populated graph -- for debugging)"
    )
    parser.add_argument(
        "--arm-rag", type=str, default="evidence_blind",
        choices=["evidence_blind", "rag_retrieval"],
        help="RAG arm mode: evidence_blind (no graph access) or rag_retrieval (keyword search)"
    )
    parser.add_argument(
        "--allow-experimental", action="store_true",
        help="Allow experimental flags (enable_cooccurrence_edges, enable_embedding_er) — "
             "disables config integrity guard"
    )
    args = parser.parse_args()

    # Resolve paths
    dataset_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
    output_path = _project_root / args.output

    if not dataset_path.exists():
        print(f"Error: QA dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading questions from: {dataset_path}")
    questions = load_questions(str(dataset_path), args.limit)
    total = len(questions)
    print(f"Loaded {total} questions (limit={args.limit})")

    # Configuration must be established before graph construction so graph
    # feature flags affect ingestion rather than only the result header.
    config = DEFAULT_CONFIG

    # Build graph deterministically
    if args.no_populate:
        print("\nSkipping graph population (--no-populate)")
        graph = InMemoryGraphStore()
        graph_provenance = {"node_count": 0, "edge_count": 0, "build_command": "skipped (--no-populate)"}
    else:
        print("\nBuilding benchmark graph (deterministic)...")
        graph, graph_provenance = build_benchmark_graph(config)
    print(f"Graph ready: {graph_provenance['node_count']} nodes, {graph_provenance['edge_count']} edges")

    # ── Embedding index for semantic entity resolution ──
    # Gated behind enable_embedding_er — Stage 1 candidate.
    embedding_index = None
    if config.enable_embedding_er:
        from nexus.query.embedding_resolver import NodeEmbeddingIndex
        print("\nBuilding embedding index (all-MiniLM-L6-v2)...")
        embedding_index = NodeEmbeddingIndex()
        embedding_index.build_index(graph)

    # Pinned for reproducibility — change only in controlled experiments.
    from nexus.reasoning.model_interface import OllamaModel
    primary_model = OllamaModel(model_name="qwen2.5:latest")
    model_name_nexus = primary_model._model_name  # e.g. "qwen2.5:latest"
    # Wrap in FallbackModel: uses LLM first, falls back to SynthesizingModel
    # when the LLM says "insufficient evidence" but evidence IS present
    nexus_model = FallbackModel(primary_model)
    verifier = Verifier(hallucination_threshold=0.2)

    # RAG arm: evidence-blind by default, configurable to rag_retrieval
    rag_arm_mode: str = args.arm_rag
    if rag_arm_mode == "rag_retrieval":
        # For RAG retrieval, use the same LLM (without Fallback) with keyword evidence
        rag_model = OllamaModel(model_name="qwen2.5:latest")
        model_name_rag = rag_model._model_name
    else:
        rag_model = EvidenceBlindModel()
        model_name_rag = model_name_nexus  # same underlying model

    print(f"\nArm config: NEXUS={model_name_nexus} (nexus), RAG={model_name_rag} ({rag_arm_mode})")
    print(f"\nRunning benchmark on {total} questions...\n")

    results: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        qtext = q["question"]
        qid = q.get("id", f"q{str(i).zfill(3)}")
        ground_truth = q.get("answer", "")
        
        # Progress
        marker = f"[{i}/{total}]"
        
        # Run NEXUS pipeline (honest entity resolution — no known_entity_ids bypass)
        nexus_result = run_nexus_pipeline(qtext, graph, nexus_model, verifier,
                                         embedding_index=embedding_index)
        
        # Entity resolution accuracy: check if parser found at least one correct entity.
        # Also accept sub-run prefixes: Exp_0_6_Validation_dense_openbook matches Exp_0_6_Validation.
        gt_entity_ids: list[str] = q.get("entities", [])
        nexus_parsed_ids: list[str] = nexus_result.get("parsed_entity_ids", [])
        
        # Resolution success: did we resolve ANY entities?
        entity_resolution_hit = bool(nexus_parsed_ids)
        # Entity accuracy: did we resolve the EXPECTED entities?
        entity_accuracy = bool(
            gt_entity_ids and any(
                gid == pid or pid.startswith(gid + "_")
                for gid in gt_entity_ids
                for pid in nexus_parsed_ids
            )
        ) if gt_entity_ids else None  # None when GT has no entities listed
        nexus_result["entity_resolution_hit"] = entity_resolution_hit
        nexus_result["entity_accuracy"] = entity_accuracy
        nexus_result["gt_entity_ids"] = gt_entity_ids
        
        # Compute accuracy for NEXUS (fuzzy + exact)
        nexus_scores = compute_key_fact_score(
            nexus_result["answer"], ground_truth
        )
        nexus_result["accuracy"] = nexus_scores["fuzzy_accuracy"]
        nexus_result["exact_accuracy"] = nexus_scores["exact_accuracy"]
        nexus_result["scoring_detail"] = nexus_scores["scoring_detail"]
        nexus_result["model"] = model_name_nexus

        # Compute NEXUS retrieval tokens from evidence_raw
        nexus_retrieval_tokens = _count_tokens(nexus_result.get("evidence_raw", ""))

        # Run RAG arm (evidence-blind or retrieval-based)
        if rag_arm_mode == "rag_retrieval":
            baseline_result = run_rag_retrieval(qtext, graph, rag_model)
        else:
            baseline_result = run_baseline(qtext, rag_model)
        
        # Compute accuracy for baseline/RAG (fuzzy + exact)
        baseline_scores = compute_key_fact_score(
            baseline_result["answer"], ground_truth
        )
        baseline_result["accuracy"] = baseline_scores["fuzzy_accuracy"]
        baseline_result["exact_accuracy"] = baseline_scores["exact_accuracy"]
        baseline_result["scoring_detail"] = baseline_scores["scoring_detail"]
        baseline_result["model"] = model_name_rag

        # Compute conciseness metric
        question_type = q.get("question_type", "factual")
        answer_tokens = _count_tokens(nexus_result["answer"])
        gt_tokens = _count_tokens(ground_truth)
        ratio = round(answer_tokens / gt_tokens, 2) if gt_tokens > 0 else 999.0
        # Factual questions: too_verbose if >3x; diagnostic/multi-hop: relaxed to 5x
        verbose_threshold = 5.0 if question_type in ("diagnostic", "multi-hop") else 3.0
        nexus_result["conciseness"] = {
            "answer_tokens": answer_tokens,
            "ground_truth_tokens": gt_tokens,
            "ratio": ratio,
            "too_verbose": ratio > verbose_threshold,
        }

        # Status indicator
        if nexus_result["error"]:
            status = "ERR"
        elif nexus_result["is_insufficient"]:
            status = "INS"
        elif nexus_result["passed"]:
            status = "PASS"
        else:
            status = f"HALL({nexus_result['hallucination_rate']:.0%})"
        
        nex_fuzzy = nexus_scores["fuzzy_accuracy"]
        bas_fuzzy = baseline_scores["fuzzy_accuracy"]
        er_hit = "HIT" if entity_resolution_hit else "MISS"
        er_method = nexus_result.get("entity_resolution_method", "?")
        gen_time = nexus_result.get("latency_breakdown", {}).get("generate_time", 0)
        print(f"  {marker} {qid}: {status} | ER={er_hit}({er_method}) | fuzzy={nex_fuzzy if nex_fuzzy is not None else 'N/A'} | paths={nexus_result['path_count']} | "
              f"nexus={nexus_result['latency_s']:.3f}s (gen={gen_time:.3f}s, {nexus_result['prompt_tokens']}->{nexus_result['completion_tokens']}tok) | "
              f"baseline={baseline_result['latency_s']:.3f}s (fuzzy={bas_fuzzy if bas_fuzzy is not None else 'N/A'})")
        
        # Determine rag retrieval tokens
        if rag_arm_mode == "rag_retrieval":
            rag_retrieval_tokens = baseline_result.get("retrieval_tokens", 0)
        else:
            rag_retrieval_tokens = 0

        # Emit two result rows per question: one for NEXUS arm, one for RAG arm.
        # Both carry the full nexus/baseline sub-dicts for reference, but each
        # has its own arm_mode + retrieval_tokens at the top level for validation.
        common_fields = {
            "question_id": qid,
            "question": qtext,
            "ground_truth": ground_truth,
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "hops": q.get("hops", 1),
            "nexus": nexus_result,
            "baseline": baseline_result,
        }

        # NEXUS arm row
        results.append({
            **common_fields,
            "arm_mode": "nexus",
            "retrieval_tokens": nexus_retrieval_tokens,
        })

        # RAG/baseline arm row
        results.append({
            **common_fields,
            "arm_mode": rag_arm_mode,
            "retrieval_tokens": rag_retrieval_tokens,
        })

    # Compute summary
    summary = compute_summary(results)

    # ── Stage 2.2: Distillation logging ──
    # Append verifier-passed (evidence→answer) pairs for Stage 4 training
    try:
        from benchmarks.distillation_logger import log_distillation_pairs, get_pair_count
        new_pairs = log_distillation_pairs(results)
        total_pairs = get_pair_count()
        if new_pairs > 0:
            print(f"\nDistillation: +{new_pairs} new pairs -> data/distillation/pairs.jsonl (total: {total_pairs})")
    except ImportError:
        pass  # Distillation logger not available — ok on old branches

    # ── Config header ──
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(_project_root)
        ).strip()[:8]
    except Exception:
        git_commit = "unknown"

    config_header = {
        "model_nexus": model_name_nexus,
        "model_rag": model_name_rag,
        "git_commit": git_commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arm_nexus": "nexus",
        "arm_rag": rag_arm_mode,
        "limit": args.limit,
        "verification_threshold": 0.2,
        "local_inference": True,
        "enable_cooccurrence_edges": config.enable_cooccurrence_edges,
        "enable_embedding_er": config.enable_embedding_er,
        "enable_associative_encoder": config.enable_associative_encoder,
        "enable_normalization": config.enable_normalization,
        "local_cost_model": {
            "type": "LocalCostModel",
            "watts_at_load": 65,
            "electricity_cost_per_kwh": 0.15,
            "target_per_1m_tokens": 0.01,
            "frontier_pricing": FRONTIER_PRICING,
        },
    }

    # ── Paired comparison (compute before validation) ──
    # Group results by question_id, extract NEXUS/RAG scores
    nexus_by_q: dict[str, float | None] = {}
    rag_by_q: dict[str, float | None] = {}
    for r in results:
        qid = r.get("question_id", "")
        arm = r.get("arm_mode", "")
        accuracy = r.get("nexus", {}).get("accuracy") if arm == "nexus" else r.get("baseline", {}).get("accuracy")
        if arm == "nexus":
            nexus_by_q[qid] = accuracy
        else:
            rag_by_q[qid] = accuracy

    # Align scores 1:1 by question_id for paired comparison
    all_qids = sorted(set(list(nexus_by_q.keys()) + list(rag_by_q.keys())))
    nexus_aligned = [nexus_by_q.get(qid) for qid in all_qids]
    rag_aligned = [rag_by_q.get(qid) for qid in all_qids]
    paired_comparison = compare_paired(nexus_aligned, rag_aligned, "NEXUS", "RAG")

    # ── Validation (all guards) ──
    validation_errors, validation_warnings = validate_benchmark_results(
        results, config_header,
        question_count=total,
        summary=summary,
        paired_comparison=paired_comparison,
        nexus_config_obj=config,
        allow_experimental=args.allow_experimental,
    )

    is_valid = len(validation_errors) == 0

    # Determine output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not is_valid:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
        original_stem = output_path.stem
        invalid_path = output_path.parent / f"{original_stem}_INVALID_{ts}.json"
        print(f"\n*** BENCHMARK VALIDATION FAILED ***", file=sys.stderr)
        for err in validation_errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        for warn in validation_warnings:
            print(f"  WARNING: {warn}", file=sys.stderr)
        print(f"  Writing results to: {invalid_path}", file=sys.stderr)
        final_output_path = invalid_path
    else:
        final_output_path = output_path
        if validation_warnings:
            print(f"\n*** WARNINGS ***", file=sys.stderr)
            for warn in validation_warnings:
                print(f"  WARNING: {warn}", file=sys.stderr)

    # Save results
    output_data = {
        "config": config_header,
        "graph_provenance": graph_provenance,
        "summary": summary,
        "paired_comparison": paired_comparison,
        "validation_warnings": validation_warnings,
        "results": results,
    }
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {final_output_path}")
    if not is_valid:
        print(f"  (original requested path: {output_path})")

    # Re-read and validate the exact artifact before publishing its status.
    artifact_errors, artifact_warnings = validate_benchmark_artifact(final_output_path)
    if artifact_errors:
        print("\n*** SERIALIZED ARTIFACT VALIDATION FAILED ***", file=sys.stderr)
        for err in artifact_errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Print comparison table
    print_comparison(summary)

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
