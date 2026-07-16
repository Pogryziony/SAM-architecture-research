"""Fail-closed runtime for the accepted comparison-plan Realizer checkpoint.

The neural model does not decide whether two values are equal.  NEXUS derives
that relation from immutable evidence, serializes it as a verified plan and
uses constrained candidate scoring only to confirm that the Realizer follows
the plan.  Exact sources, subjects and values are materialized afterwards.

PyTorch is imported lazily so the default non-neural NEXUS installation keeps
working without the optional ``train`` dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


BACKEND_NAME = "abstractive_plan_v3"
HYBRID_BACKEND_NAME = "grounded_v1"
MODEL_SCHEMA = "nexus-realizer-abstractive-checkpoint-v1"
MODEL_STATUS = "PILOT_CHECKPOINT_ACCEPTED"
DEFAULT_MODEL_DIR = "models/realizer/abstractive_v1_plan_v3"
DEFAULT_CONFIG_PATH = "training/nexus_realizer_abstractive_v1.json"
DEFAULT_WEIGHTS_SHA256 = (
    "bfa5855a57fba8db34e896d77848942733c5570049c927d4310646bea444e152"
)
_ROOT = Path(__file__).resolve().parents[2]


class ComparisonPlanError(ValueError):
    """Evidence, plan or artifact did not satisfy the runtime contract."""


@dataclass(frozen=True)
class ComparisonSlots:
    source_1: str
    value_1: str
    subject_1: str
    source_2: str
    value_2: str
    subject_2: str

    def as_dict(self) -> dict[str, str]:
        return {
            "SOURCE_1": self.source_1,
            "VALUE_1": self.value_1,
            "SUBJECT_1": self.subject_1,
            "SOURCE_2": self.source_2,
            "VALUE_2": self.value_2,
            "SUBJECT_2": self.subject_2,
        }


@dataclass(frozen=True)
class VerifiedComparisonPlan:
    slots: ComparisonSlots
    relation: str
    label: str


@dataclass(frozen=True)
class ComparisonRealization:
    answer: str
    strategy: str
    rejection_reason: str
    relation_plan: str
    predicted_relation: str
    checkpoint_sha256: str
    neural_used: bool
    diagnostics: dict[str, Any]
    slots: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "strategy": self.strategy,
            "rejection_reason": self.rejection_reason,
            "relation_plan": self.relation_plan,
            "predicted_relation": self.predicted_relation,
            "checkpoint_sha256": self.checkpoint_sha256,
            "neural_used": self.neural_used,
            "diagnostics": self.diagnostics,
            "slots": self.slots,
        }


LabelSelector = Callable[[str, tuple[str, str]], tuple[str, dict[str, Any]]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _ROOT / path


def normalize_comparison_value(value: Any) -> str:
    """Match the normalization used when the pilot relation was verified."""
    return " ".join(str(value).casefold().split())


def relation_for_values(value_1: Any, value_2: Any) -> tuple[str, str]:
    if normalize_comparison_value(value_1) == normalize_comparison_value(value_2):
        return "the same", "SAME"
    return "different", "DIFFERENT"


def _parse_fact(fact: dict[str, Any]) -> tuple[str, str, str] | None:
    """Extract source, subject and value from registered evidence forms.

    Structured fields take precedence.  The two textual forms are the exact
    forms emitted by the registered train-only acquisition pipeline.  Unknown
    prose is deliberately rejected instead of being heuristically guessed.
    """
    source = str(fact.get("source", "")).strip()
    subject = str(fact.get("subject", "")).strip()
    value = str(fact.get("value", "")).strip()
    if source and subject and value:
        return source, subject, value

    text = str(fact.get("text", "")).strip()
    if not source or not text:
        return None

    # build_zero_hop_pack may prefix a stable node id before the curated fact.
    in_marker = f"In {source}, "
    marker_index = text.find(in_marker)
    if marker_index >= 0:
        remainder = text[marker_index + len(in_marker):].strip()
        match = re.fullmatch(r"(.+?) is set to (.+)\.", remainder)
        if match:
            return source, match.group(1).strip(), match.group(2).strip()

    # Registered table evidence: ``For <row>, <field> is <value>.``.  The
    # subject slot is the row label; the field remains present in the question.
    table_index = text.find("For ")
    if table_index >= 0:
        remainder = text[table_index:]
        match = re.fullmatch(r"For (.+?), (.+?) is (.+)\.", remainder)
        if match:
            return source, match.group(1).strip(), match.group(3).strip()
    return None


def extract_comparison_slots(
    question: str,
    evidence_pack: dict[str, Any],
) -> ComparisonSlots:
    candidates: list[tuple[str, str, str]] = []
    for item in evidence_pack.get("node_facts", []):
        if not isinstance(item, dict):
            continue
        parsed = _parse_fact(item)
        if parsed is not None and parsed not in candidates:
            candidates.append(parsed)

    if len(candidates) > 2:
        mentioned = [item for item in candidates if item[0] in question]
        if len(mentioned) == 2:
            candidates = mentioned
    if len(candidates) != 2:
        raise ComparisonPlanError("comparison_requires_exactly_two_supported_facts")
    if candidates[0][0] == candidates[1][0]:
        raise ComparisonPlanError("comparison_requires_two_distinct_sources")

    def order_key(item: tuple[str, str, str]) -> tuple[int, str, str, str]:
        position = question.find(item[0])
        return (position if position >= 0 else len(question), *item)

    first, second = sorted(candidates, key=order_key)
    return ComparisonSlots(
        source_1=first[0], subject_1=first[1], value_1=first[2],
        source_2=second[0], subject_2=second[1], value_2=second[2],
    )


def build_verified_comparison_plan(
    question: str,
    evidence_pack: dict[str, Any],
) -> VerifiedComparisonPlan:
    slots = extract_comparison_slots(question, evidence_pack)
    relation, label = relation_for_values(slots.value_1, slots.value_2)
    return VerifiedComparisonPlan(slots=slots, relation=relation, label=label)


def serialize_verified_comparison_plan(
    plan: VerifiedComparisonPlan,
    max_bytes: int,
) -> str:
    relation, label = relation_for_values(plan.slots.value_1, plan.slots.value_2)
    if relation != plan.relation or label != plan.label:
        raise ComparisonPlanError("comparison_plan_contradicts_immutable_values")
    text = (
        "[TASK] Verbalize the verified comparison plan.\n"
        f"[VERIFIED_RELATION] {plan.label}\n"
        f"[VALUE_1] {plan.slots.value_1}\n"
        f"[VALUE_2] {plan.slots.value_2}"
    )
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def materialize_comparison(plan: VerifiedComparisonPlan) -> str:
    slots = plan.slots
    return (
        f"{slots.source_1} reports {slots.value_1} for {slots.subject_1}, while "
        f"{slots.source_2} reports {slots.value_2} for {slots.subject_2}; "
        f"the values are {plan.relation}."
    )


@lru_cache(maxsize=4)
def _load_checkpoint(
    model_dir_value: str,
    config_path_value: str,
    expected_weights_sha256: str,
):
    if not expected_weights_sha256:
        raise ComparisonPlanError("missing_expected_checkpoint_sha256")
    model_dir = _resolve_path(model_dir_value).resolve()
    config_path = _resolve_path(config_path_value).resolve()
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ComparisonPlanError("comparison_checkpoint_artifact_missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MODEL_SCHEMA:
        raise ComparisonPlanError("unsupported_comparison_checkpoint_schema")
    if manifest.get("status") != MODEL_STATUS:
        raise ComparisonPlanError("comparison_checkpoint_not_accepted")
    if config.get("data", {}).get("source_format") != "comparison_plan_v3":
        raise ComparisonPlanError("comparison_checkpoint_config_mismatch")
    if config.get("data", {}).get("target_format") != "relation_label_v2":
        raise ComparisonPlanError("comparison_checkpoint_target_mismatch")
    if _sha256_file(config_path) != manifest.get("config_sha256"):
        raise ComparisonPlanError("comparison_checkpoint_config_hash_mismatch")

    weights_name = manifest.get("weights", {}).get("path")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise ComparisonPlanError("unsafe_comparison_weights_path")
    weights_path = (model_dir / weights_name).resolve()
    if weights_path.parent != model_dir or not weights_path.is_file():
        raise ComparisonPlanError("comparison_weights_missing")
    actual_sha256 = _sha256_file(weights_path)
    if actual_sha256 != manifest.get("weights", {}).get("sha256"):
        raise ComparisonPlanError("comparison_manifest_weights_hash_mismatch")
    if actual_sha256 != expected_weights_sha256:
        raise ComparisonPlanError("comparison_expected_weights_hash_mismatch")

    try:
        import torch
    except ImportError as exc:
        raise ComparisonPlanError("pytorch_unavailable_for_comparison_realizer") from exc
    from nexus.realizer.model import build_model
    from nexus.realizer.tokenizer import ByteTokenizer

    model = build_model(config["model"])
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, ByteTokenizer(), config, actual_sha256


def _checkpoint_label_selector(
    model_dir: str,
    config_path: str,
    expected_weights_sha256: str,
) -> tuple[LabelSelector, dict[str, Any]]:
    model, tokenizer, config, actual_sha256 = _load_checkpoint(
        model_dir, config_path, expected_weights_sha256,
    )

    def select(source_text: str, candidates: tuple[str, str]):
        from nexus.realizer.decoder import score_candidate_texts

        source_ids = tokenizer.encode(
            source_text, int(config["model"]["max_input_tokens"]),
        )
        return score_candidate_texts(
            model,
            source_ids,
            list(candidates),
            tokenizer=tokenizer,
            max_length=int(config["model"]["max_output_tokens"]),
        )

    return select, {
        "checkpoint_sha256": actual_sha256,
        "max_input_tokens": int(config["model"]["max_input_tokens"]),
    }


def _rejected(reason: str, *, plan: VerifiedComparisonPlan | None = None) -> ComparisonRealization:
    return ComparisonRealization(
        answer="Insufficient evidence to answer.",
        strategy="insufficient_evidence",
        rejection_reason=reason,
        relation_plan=plan.label if plan else "",
        predicted_relation="",
        checkpoint_sha256="",
        neural_used=False,
        diagnostics={},
        slots=plan.slots.as_dict() if plan else {},
    )


def realize_comparison_plan(
    question: str,
    evidence_pack: dict[str, Any],
    *,
    model_dir: str = DEFAULT_MODEL_DIR,
    config_path: str = DEFAULT_CONFIG_PATH,
    expected_weights_sha256: str = DEFAULT_WEIGHTS_SHA256,
    label_selector: LabelSelector | None = None,
) -> ComparisonRealization:
    """Realize one verified comparison or return a fail-closed result."""
    try:
        plan = build_verified_comparison_plan(question, evidence_pack)
        max_bytes = 766
        identity: dict[str, Any] = {"checkpoint_sha256": "injected"}
        selector = label_selector
        if selector is None:
            selector, identity = _checkpoint_label_selector(
                model_dir, config_path, expected_weights_sha256,
            )
            max_bytes = identity["max_input_tokens"] - 2
        serialized = serialize_verified_comparison_plan(plan, max_bytes)
        selected, diagnostics = selector(serialized, ("SAME", "DIFFERENT"))
    except (ComparisonPlanError, OSError, ValueError, RuntimeError) as exc:
        return _rejected(str(exc), plan=locals().get("plan"))

    if selected not in {"SAME", "DIFFERENT"}:
        return _rejected("comparison_model_returned_unsupported_label", plan=plan)
    if selected != plan.label:
        return _rejected("comparison_model_contradicted_verified_plan", plan=plan)
    return ComparisonRealization(
        answer=materialize_comparison(plan),
        strategy=BACKEND_NAME,
        rejection_reason="",
        relation_plan=plan.label,
        predicted_relation=selected,
        checkpoint_sha256=str(identity["checkpoint_sha256"]),
        neural_used=True,
        diagnostics=diagnostics,
        slots=plan.slots.as_dict(),
    )


__all__ = [
    "BACKEND_NAME", "HYBRID_BACKEND_NAME", "ComparisonPlanError", "ComparisonRealization",
    "ComparisonSlots", "DEFAULT_CONFIG_PATH", "DEFAULT_MODEL_DIR",
    "DEFAULT_WEIGHTS_SHA256", "VerifiedComparisonPlan",
    "build_verified_comparison_plan", "extract_comparison_slots",
    "materialize_comparison", "normalize_comparison_value",
    "realize_comparison_plan", "relation_for_values",
    "serialize_verified_comparison_plan",
]
