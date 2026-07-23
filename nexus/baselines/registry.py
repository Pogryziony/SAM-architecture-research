"""Registry of fair-comparison baseline arms.

Real provider runners are wired when credentials exist. Until then, arms
remain declared and return ``NOT_RUN``.
"""

from __future__ import annotations

from nexus.baselines.interface import BaselineArm


BASELINE_ARMS: dict[str, BaselineArm] = {
    "closed_book_llm": BaselineArm(
        arm_id="closed_book_llm",
        family="llm",
        description="Closed-book actual LLM (version-pinned).",
        requires_env=("NEXUS_LLM_API_KEY", "NEXUS_LLM_MODEL"),
    ),
    "long_context_llm": BaselineArm(
        arm_id="long_context_llm",
        family="llm",
        description="Long-context actual LLM with permitted corpus.",
        requires_env=("NEXUS_LLM_API_KEY", "NEXUS_LLM_MODEL"),
    ),
    "bm25_rag": BaselineArm(
        arm_id="bm25_rag",
        family="rag",
        description="BM25 lexical RAG with version-pinned answer model.",
        requires_env=("NEXUS_LLM_API_KEY", "NEXUS_LLM_MODEL"),
        requires_packages=("rank_bm25",),
        modern_rag=False,
    ),
    "dense_rag": BaselineArm(
        arm_id="dense_rag",
        family="rag",
        description="Dense-vector RAG with version-pinned embedder + LLM.",
        requires_env=("NEXUS_LLM_API_KEY", "NEXUS_LLM_MODEL"),
        requires_packages=("sentence_transformers",),
        modern_rag=False,
    ),
    "hybrid_rag": BaselineArm(
        arm_id="hybrid_rag",
        family="rag",
        description="Hybrid lexical+dense RAG.",
        requires_env=("NEXUS_LLM_API_KEY", "NEXUS_LLM_MODEL"),
        requires_packages=("rank_bm25", "sentence_transformers"),
        modern_rag=True,
    ),
    "hybrid_rag_rerank": BaselineArm(
        arm_id="hybrid_rag_rerank",
        family="rag",
        description="Hybrid RAG with competitive reranking.",
        requires_env=("NEXUS_LLM_API_KEY", "NEXUS_LLM_MODEL"),
        requires_packages=("rank_bm25", "sentence_transformers"),
        modern_rag=True,
    ),
    "graph_rag": BaselineArm(
        arm_id="graph_rag",
        family="rag",
        description="Graph-oriented RAG where feasible.",
        requires_env=("NEXUS_LLM_API_KEY", "NEXUS_LLM_MODEL"),
        modern_rag=True,
    ),
    "nexus_lexical_only": BaselineArm(
        arm_id="nexus_lexical_only",
        family="nexus",
        description="NEXUS lexical-only profile (local, no external LLM).",
    ),
    "nexus_lexical_er3": BaselineArm(
        arm_id="nexus_lexical_er3",
        family="nexus",
        description="NEXUS lexical∪ER3 profile.",
    ),
    "placeholder_synthesizing_model": BaselineArm(
        arm_id="placeholder_synthesizing_model",
        family="placeholder",
        description=(
            "Deterministic SynthesizingModel — internal diagnostic only; "
            "not a real LLM baseline."
        ),
        is_placeholder=True,
    ),
    "placeholder_evidence_blind": BaselineArm(
        arm_id="placeholder_evidence_blind",
        family="placeholder",
        description=(
            "Deterministic EvidenceBlindModel — internal diagnostic only; "
            "not a real closed-book LLM."
        ),
        is_placeholder=True,
    ),
}


def get_arm(arm_id: str) -> BaselineArm:
    try:
        return BASELINE_ARMS[arm_id]
    except KeyError as exc:
        raise KeyError(f"unknown baseline arm: {arm_id}") from exc


def list_arms(*, include_placeholders: bool = True) -> list[BaselineArm]:
    arms = list(BASELINE_ARMS.values())
    if not include_placeholders:
        arms = [a for a in arms if not a.is_placeholder]
    return arms
