"""Curated dual-side qualitative compare templates for L1 realization.

Renders paired compare_* node properties into a short contrast answer without
touching AnswerPlan weights. Templates are fail-closed: both sides required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nexus.graph.store import InMemoryGraphStore


@dataclass(frozen=True)
class QualitativeCompareTemplate:
    """A structured dual-side compare template."""

    name: str
    question_patterns: tuple[str, ...]
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    joiner: str = " "


# Prefer Decision_PivotToNEXUS as the curated compare carrier.
_DEFAULT_COMPARE_NODES = (
    "Decision_PivotToNEXUS",
    "Concept_NoiseTolerance",
    "Concept_MemoryCeiling",
)

TEMPLATES: tuple[QualitativeCompareTemplate, ...] = (
    QualitativeCompareTemplate(
        name="rag_vs_nexus_updates",
        question_patterns=(
            r"\brag\b.*\bnexus\b.*\b(updat|increment|re-?index|re-?embed)",
            r"\bnexus\b.*\brag\b.*\b(updat|increment)",
            r"how does rag handle knowledge updates",
            r"knowledge updates compared to nexus",
            r"compare rag vs nexus for knowledge updates",
        ),
        left_keys=("compare_rag_updates",),
        right_keys=("compare_nexus_updates",),
    ),
    QualitativeCompareTemplate(
        name="phase_1_4_vs_5",
        question_patterns=(
            r"phase\s*1.?4.*phase\s*5",
            r"phase\s*5.*phase\s*1.?4",
            r"sam phase 1.?4.*nexus phase 5",
            r"compare sam phase",
        ),
        left_keys=("compare_phase_1_4",),
        right_keys=("compare_phase_5",),
    ),
    QualitativeCompareTemplate(
        name="controlled_vs_realistic",
        question_patterns=(
            r"controlled.*realistic",
            r"realistic.*controlled",
            r"controlled distractors vs realistic",
        ),
        left_keys=("compare_controlled_distractors",),
        right_keys=("compare_realistic_distractors",),
    ),
    QualitativeCompareTemplate(
        name="oracle_filter_vs_learned_selector",
        question_patterns=(
            r"oracle.?filter.*selector",
            r"learned selector.*oracle",
            r"oracle-filter results with learned selector",
        ),
        left_keys=("compare_oracle_filter",),
        right_keys=("compare_learned_selector",),
    ),
    QualitativeCompareTemplate(
        name="dual_encoder_vs_chain_set",
        question_patterns=(
            r"dual.?encoder.*chain.?set",
            r"chain.?set.*dual.?encoder",
            r"compare dual encoder vs chain-set",
        ),
        left_keys=("compare_dual_encoder",),
        right_keys=("compare_chain_set",),
    ),
    QualitativeCompareTemplate(
        name="core_only_vs_oracle_memory",
        question_patterns=(
            r"core.?only.*oracle.?memory",
            r"oracle.?memory.*core.?only",
            r"compare sam core_only vs sam oracle_memory",
        ),
        left_keys=("compare_core_only",),
        right_keys=("compare_oracle_memory",),
    ),
    QualitativeCompareTemplate(
        name="exp0_vs_013a",
        question_patterns=(
            r"first sam experiment.*0\.13a|experiment \(0.*0\.13a",
            r"0\.13a.*diagnosis|diagnosis.*0\.13a",
            r"compare the first sam experiment",
            r"exp\s*0\b.*0\.13a|0\.13a.*exp\s*0",
        ),
        left_keys=("compare_exp0_diagnosis",),
        right_keys=("compare_exp013a",),
    ),
    QualitativeCompareTemplate(
        name="rag_vs_nexus_negation",
        question_patterns=(
            r"rag struggle.*negat",
            r"negation reasoning",
            r"why does rag struggle with questions requiring negation",
        ),
        left_keys=("compare_rag_negation",),
        right_keys=("compare_nexus_negation",),
    ),
    QualitativeCompareTemplate(
        name="rag_vs_nexus_representation",
        question_patterns=(
            r"compare rag vs nexus for knowledge representation",
            r"rag vs nexus for knowledge representation",
        ),
        left_keys=("compare_rag_representation",),
        right_keys=("compare_nexus_representation",),
    ),
    QualitativeCompareTemplate(
        name="rag_vs_nexus_multihop",
        question_patterns=(
            r"compare rag vs nexus for multi-hop",
            r"rag vs nexus for multi-hop reasoning",
        ),
        left_keys=("compare_rag_multihop",),
        right_keys=("compare_nexus_multihop",),
    ),
    QualitativeCompareTemplate(
        name="rag_vs_nexus_hallucination",
        question_patterns=(
            r"compare rag vs nexus for hallucination",
            r"rag vs nexus for hallucination risk",
        ),
        left_keys=("compare_rag_hallucination",),
        right_keys=("compare_nexus_hallucination",),
    ),
    QualitativeCompareTemplate(
        name="sam_vs_nexus_memory",
        question_patterns=(
            r"compare the role of memory in sam vs nexus",
            r"role of memory in sam vs nexus",
        ),
        left_keys=("compare_sam_memory",),
        right_keys=("compare_nexus_memory",),
    ),
    QualitativeCompareTemplate(
        name="sam_vs_nexus_training",
        question_patterns=(
            r"compare the training requirements of sam vs nexus",
            r"training requirements of sam vs nexus",
            r"training.*(sam|nexus).*(sam|nexus)",
        ),
        left_keys=("compare_sam_training",),
        right_keys=("compare_nexus_training",),
    ),
    QualitativeCompareTemplate(
        name="sam_vs_nexus_debuggability",
        question_patterns=(
            r"compare the debuggability of sam vs nexus",
            r"debuggability of sam vs nexus",
            r"\bdebuggability\b",
        ),
        left_keys=("compare_sam_debuggability",),
        right_keys=("compare_nexus_debuggability",),
    ),
    QualitativeCompareTemplate(
        name="dense_vs_nexus_compute",
        question_patterns=(
            r"compare the compute requirements of dense llms vs nexus",
            r"compute requirements of dense",
            r"dense llms vs nexus",
        ),
        left_keys=("compare_dense_compute",),
        right_keys=("compare_nexus_compute",),
    ),
    QualitativeCompareTemplate(
        name="when_rag_outperforms_nexus",
        question_patterns=(
            r"when would classic rag outperform nexus",
            r"when.*rag outperform",
            r"rag outperform nexus",
        ),
        left_keys=("compare_rag_outperforms_when",),
        right_keys=("compare_nexus_needs_structure",),
    ),
    QualitativeCompareTemplate(
        name="hardest_rag_easiest_nexus",
        question_patterns=(
            r"hardest for rag but easiest for nexus",
            r"type of question is hardest for rag",
            r"hardest for rag",
        ),
        left_keys=("compare_rag_hardest",),
        right_keys=("compare_nexus_easiest",),
    ),
    QualitativeCompareTemplate(
        name="nexus_vs_rag_context_size",
        question_patterns=(
            r"context size advantage of nexus over rag",
            r"context size advantage",
            r"context size.*nexus.*rag",
        ),
        left_keys=("compare_nexus_context_size",),
        right_keys=("compare_rag_context_size",),
    ),
)


def match_qualitative_compare_template(
    question: str,
) -> QualitativeCompareTemplate | None:
    """Return the first template whose patterns match the question."""
    q = (question or "").strip().lower()
    if not q:
        return None
    for template in TEMPLATES:
        for pattern in template.question_patterns:
            if re.search(pattern, q, flags=re.IGNORECASE):
                return template
    return None


def collect_compare_side_text(
    graph: InMemoryGraphStore,
    keys: tuple[str, ...],
    entity_ids: list[str] | None = None,
) -> str | None:
    """Collect the first non-empty property among keys from compare carrier nodes."""
    node_ids: list[str] = []
    if entity_ids:
        node_ids.extend(entity_ids)
    node_ids.extend(_DEFAULT_COMPARE_NODES)
    seen: set[str] = set()
    for node_id in node_ids:
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node = graph.get_node(node_id)
        if node is None:
            continue
        props = node.properties or {}
        for key in keys:
            value = props.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def render_qualitative_compare(
    template: QualitativeCompareTemplate,
    left_text: str,
    right_text: str,
) -> str:
    """Render a dual-side qualitative compare answer."""
    left = left_text.strip()
    right = right_text.strip()
    if not left or not right:
        return ""
    return f"{left}{template.joiner}{right}".strip()


def try_qualitative_compare(
    question: str,
    graph: InMemoryGraphStore,
    entity_ids: list[str] | None = None,
) -> tuple[str, QualitativeCompareTemplate] | None:
    """Match a template and return (answer, template) if both sides resolve."""
    template = match_qualitative_compare_template(question)
    if template is None:
        return None
    left = collect_compare_side_text(graph, template.left_keys, entity_ids)
    right = collect_compare_side_text(graph, template.right_keys, entity_ids)
    if not left or not right:
        return None
    answer = render_qualitative_compare(template, left, right)
    if not answer:
        return None
    return answer, template
