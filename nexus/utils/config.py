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
        type_prior_boost: Additive boost for intent→type alignment (metric→Experiment/Metric, concept→Concept/Decision)

    Type priority:
        Lower = higher priority during entry-node ranking.

    Traversal:
        max_depth: Maximum path length in beam search
        beam_width: Number of paths to keep at each depth step.
            Raised from 5->20 to exploit co-occurrence edge density (47:1).
        edge_confidence_threshold: Skip edges below this confidence when
            higher-confidence typed edges exist for the same node pair.
        path_score_focus: Score/rank paths against the first N entry entities
            only (expansion still uses the full entry set). Prevents hub
            fillers from diluting entity-coverage scores. 0 = use all entries.
        max_paths: Maximum ranked paths kept for evidence/proof audit.

    Verification:
        hallucination_threshold: Maximum allowed hallucination rate

    Post-edit:
        post_edit_enabled: When True, corrects hallucinated numbers in answers.
            Defaults to False because post-edit masks the model's true accuracy.
            Enable only for explicit experiments studying post-edit behavior.

    Realizer:
        realizer_backend: ``synth`` preserves the historical template path;
            ``pointer_copy`` selects and copies evidence for factual queries;
            ``l1_acceptance`` is the zero-LLM L1 hybrid (path render + pointer).
        as_valid_at / as_known_at: optional ISO timestamps for bi-temporal
            traversal filtering (empty disables the filter).
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
    embedding_match_boost: float = 3.0
    type_prior_boost: float = 0.15

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
    beam_width: int = 25
    edge_confidence_threshold: float = 0.3
    # Explicit expansion budgets (auditability Stage 2). Exhaustion is reported
    # as truncation — never as a silent complete search.
    max_expanded_edges: int = 10_000
    max_expanded_nodes: int = 5_000
    # Optional monotonic wall-clock budget in milliseconds. 0 disables the check.
    max_traversal_ms: float = 0.0
    # Path ranking focus: score against top-N entries (0 = all entries).
    path_score_focus: int = 2
    # Proof/evidence path budget after ranking.
    max_paths: int = 12
    # Stage 6 point-in-time cutoffs for traversal (empty = no filter).
    as_valid_at: str = ""
    as_known_at: str = ""

    # Verification
    hallucination_threshold: float = 0.2

    # Deterministic reasoning-audit policy.  These thresholds only classify
    # the diagnostic recommendation; they do not rewrite model answers.
    readiness_answer_threshold: float = 0.70
    readiness_conditional_threshold: float = 0.40
    # When True, unconditional answers require SourceRecord locators on proof
    # nodes (legacy free-form sources alone are insufficient).
    require_structured_provenance: bool = False

    # Post-edit — disabled by default; masks model's true accuracy
    post_edit_enabled: bool = False

    # ``synth`` is the library default and preserves historical registered
    # semantics across all stages.  For production QA workloads, callers should
    # opt into ``ProductionNEXUSConfig.grounded()``, which routes factual
    # lookups to Pointer/Copy v3 and comparisons to the hash-verified
    # comparison-plan checkpoint.
    realizer_backend: str = "synth"

    # Accepted comparison-plan Realizer identity.  These fields are pre-bound
    # so ``ProductionNEXUSConfig.grounded()`` and the explicit comparison-plan
    # factory work without callers having to paste hashes.  They are ignored
    # unless ``realizer_backend`` selects a comparison-capable backend.
    # Keeping the expected hash in configuration makes a replaced checkpoint
    # fail closed.
    realizer_model_dir: str = "models/realizer/abstractive_v1_plan_v3"
    realizer_config_path: str = "training/nexus_realizer_abstractive_v1.json"
    realizer_checkpoint_sha256: str = (
        "bfa5855a57fba8db34e896d77848942733c5570049c927d4310646bea444e152"
    )

    # Tier-3 cascade backend — which answer generator to use for 0-hop evidence.
    #   "synth" (default): SynthesizingModel — template-based, never refuses.
    #   "llm_no_refusal": LLM with a prompt template that omits the
    #       insufficiency instruction (for A/B testing later).
    tier3_backend: str = "synth"

    # When False, grounded/L1/pointer/comparison/deterministic profiles must
    # abstain instead of cascading to SynthesizingModel / get_available_model().
    # Library default True preserves historical Stage-2 synth semantics.
    # Production factories set this False.
    allow_synth_fallback: bool = True

    # ── Stage 1 candidate gates (Stage 0 baseline: both False) ──
    # enable_embedding_er: When True, builds an all-MiniLM-L6-v2 embedding index
    #   for semantic entity resolution (Stage 1 experiment).
    # enable_cooccurrence_edges: When True, adds low-confidence related_to edges
    #   between all entity pairs co-occurring in the same document (Stage 1 experiment).
    enable_embedding_er: bool = False
    enable_cooccurrence_edges: bool = False

    # ── Stage 1 SAM+NEXUS associative stack ──
    # enable_normalization: When True, applies PL/EN lemmatization and stopword
    #   removal to question text before entity spotting (Stage 1 experiment).
    # enable_associative_encoder: When True, uses the trained associative encoder
    #   to predict entities + intent + category (Stage 1 experiment).
    enable_normalization: bool = False
    enable_associative_encoder: bool = False

    # ── Stage 3 dialogue state ──
    # dialogue_decay: Multiplicative decay factor applied each turn to entity
    #   activations in the dialogue state (0.0–1.0). Lower = faster forgetting.
    # dialogue_boost: Additive boost applied to entity ranking scores when the
    #   entity is active in the dialogue state. Helps resolve anaphora/ellipsis.
    dialogue_decay: float = 0.7
    dialogue_boost: float = 1.0


DEFAULT_CONFIG = NEXUSConfig()
