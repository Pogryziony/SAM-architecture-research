"""
Configuration dataclass for NEXUS pipeline parameters.

Extracts all hardcoded constants into a single config object so that
sensitivity analysis and tuning can be performed without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NEXUSConfig:
    """Central configuration for NEXUS pipeline parameters.

    Model:
        model_name: Pinned for reproducibility — change only in controlled experiments.

    Entity resolution:
        fuzzy_cutoff: Minimum similarity for fuzzy matching (0.0–1.0)
        max_entry_nodes: Maximum number of entry nodes for traversal

    Ranking boosts:
        property_keyword_boost: Boost when question tokens appear in node key_finding
        curated_node_boost: Boost for nodes with key_finding property
        word_boundary_boost: Boost for exact word-boundary matches
        sub_run_penalty: Penalty for sub-experiment run nodes
        alias_match_boost: Boost when question literally contains an alias phrase

    Type priority:
        Lower = higher priority during entry-node ranking.

    Traversal:
        max_depth: Maximum path length in beam search
        beam_width: Number of paths to keep at each depth step

    Verification:
        hallucination_threshold: Maximum allowed hallucination rate

    Post-edit:
        post_edit_enabled: When True, corrects hallucinated numbers in answers.
            Defaults to False because post-edit masks the model's true accuracy.
            Enable only for explicit experiments studying post-edit behavior.
    """

    # Model — pinned for reproducibility; change only in controlled experiments
    model_name: str = "qwen2.5:latest"

    # Entity resolution
    fuzzy_cutoff: float = 0.5
    max_entry_nodes: int = 5

    # Ranking boosts
    property_keyword_boost: float = 0.25
    curated_node_boost: float = 0.10
    word_boundary_boost: float = 0.10
    sub_run_penalty: float = -0.15
    alias_match_boost: float = 50.0

    # Type priority (lower = higher priority)
    type_priority: dict[str, int] = field(default_factory=lambda: {
        "Experiment": 0,
        "Decision": 1,
        "Concept": 2,
        "Bug": 3,
        "Requirement": 4,
        "TestCase": 5,
        "Metric": 6,
        "Entity": 7,
        "Document": 8,
        "CodeFile": 9,
        "Function": 10,
    })

    # Traversal
    max_depth: int = 4
    beam_width: int = 5

    # Verification
    hallucination_threshold: float = 0.2

    # Post-edit — disabled by default; masks model's true accuracy
    post_edit_enabled: bool = False


DEFAULT_CONFIG = NEXUSConfig()
