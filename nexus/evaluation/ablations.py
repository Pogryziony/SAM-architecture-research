"""Executable ablations and deterministic robustness transforms."""

from __future__ import annotations

import copy
import random
import re
from typing import Any, Callable

from nexus.pipeline.config import PipelineIdentity, ProductionNEXUSConfig


AblationFactory = Callable[[ProductionNEXUSConfig], ProductionNEXUSConfig]


def _with_backend(cfg: ProductionNEXUSConfig, backend: str) -> ProductionNEXUSConfig:
    return ProductionNEXUSConfig(
        pipeline_id=cfg.pipeline_id,
        realizer_backend=backend,
        require_structured_provenance=cfg.require_structured_provenance,
        max_depth=cfg.max_depth,
        max_entry_nodes=cfg.max_entry_nodes,
        path_score_focus=cfg.path_score_focus,
        as_valid_at=cfg.as_valid_at,
        as_known_at=cfg.as_known_at,
        enable_associative_encoder=cfg.enable_associative_encoder,
        enable_embedding_er=cfg.enable_embedding_er,
        enable_normalization=cfg.enable_normalization,
        realizer_model_dir=cfg.realizer_model_dir,
        realizer_config_path=cfg.realizer_config_path,
        realizer_checkpoint_sha256=cfg.realizer_checkpoint_sha256,
    )


ABLATIONS: dict[str, AblationFactory] = {
    "no_er3": lambda cfg: ProductionNEXUSConfig(
        pipeline_id=PipelineIdentity(
            lexical_fallback=True,
            entity_ranker_v3_enabled=False,
            domain_pack_id=cfg.pipeline_id.domain_pack_id,
            domain_pack_version=cfg.pipeline_id.domain_pack_version,
        ),
        realizer_backend=cfg.realizer_backend,
        require_structured_provenance=cfg.require_structured_provenance,
        realizer_model_dir=cfg.realizer_model_dir,
        realizer_config_path=cfg.realizer_config_path,
        realizer_checkpoint_sha256=cfg.realizer_checkpoint_sha256,
    ),
    "zero_hop_only": lambda cfg: ProductionNEXUSConfig(
        pipeline_id=cfg.pipeline_id,
        realizer_backend=cfg.realizer_backend,
        max_depth=1,
        max_paths=1,
        require_structured_provenance=cfg.require_structured_provenance,
        realizer_model_dir=cfg.realizer_model_dir,
        realizer_config_path=cfg.realizer_config_path,
        realizer_checkpoint_sha256=cfg.realizer_checkpoint_sha256,
    ),
    "no_multi_hop": lambda cfg: ProductionNEXUSConfig(
        pipeline_id=cfg.pipeline_id,
        realizer_backend=cfg.realizer_backend,
        max_depth=1,
        require_structured_provenance=cfg.require_structured_provenance,
        realizer_model_dir=cfg.realizer_model_dir,
        realizer_config_path=cfg.realizer_config_path,
        realizer_checkpoint_sha256=cfg.realizer_checkpoint_sha256,
    ),
    "no_path_scoring_focus": lambda cfg: ProductionNEXUSConfig(
        pipeline_id=cfg.pipeline_id,
        realizer_backend=cfg.realizer_backend,
        path_score_focus=0,
        require_structured_provenance=cfg.require_structured_provenance,
        realizer_model_dir=cfg.realizer_model_dir,
        realizer_config_path=cfg.realizer_config_path,
        realizer_checkpoint_sha256=cfg.realizer_checkpoint_sha256,
    ),
    "no_structured_provenance": lambda cfg: ProductionNEXUSConfig(
        pipeline_id=cfg.pipeline_id,
        realizer_backend=cfg.realizer_backend,
        require_structured_provenance=False,
        realizer_model_dir=cfg.realizer_model_dir,
        realizer_config_path=cfg.realizer_config_path,
        realizer_checkpoint_sha256=cfg.realizer_checkpoint_sha256,
    ),
    "no_temporal_filtering": lambda cfg: ProductionNEXUSConfig(
        pipeline_id=cfg.pipeline_id,
        realizer_backend=cfg.realizer_backend,
        as_valid_at="",
        as_known_at="",
        require_structured_provenance=cfg.require_structured_provenance,
        realizer_model_dir=cfg.realizer_model_dir,
        realizer_config_path=cfg.realizer_config_path,
        realizer_checkpoint_sha256=cfg.realizer_checkpoint_sha256,
    ),
    "deterministic_render": lambda cfg: _with_backend(cfg, "deterministic_render"),
    "l1_acceptance": lambda cfg: _with_backend(cfg, "l1_acceptance"),
    "lexical_synth": lambda cfg: _with_backend(cfg, "synth"),
}


def apply_ablation(name: str, base: ProductionNEXUSConfig) -> ProductionNEXUSConfig:
    try:
        factory = ABLATIONS[name]
    except KeyError as exc:
        raise KeyError(f"unknown ablation {name!r}; known={sorted(ABLATIONS)}") from exc
    return factory(base)


def list_ablations() -> list[str]:
    return sorted(ABLATIONS)


# ── Robustness transforms (development data only) ─────────────────────────


def _typo(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    i = rng.randrange(1, len(text) - 1)
    chars = list(text)
    chars[i] = rng.choice("aeioubcdfghjklmnprst")
    return "".join(chars)


def transform_question(
    record: dict[str, Any],
    transform: str,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a linked transformed copy of a development question.

    Never apply to sealed hidden tests unless the protocol explicitly allows it.
    """
    rng = random.Random(seed ^ hashlib_id(str(record.get("id", ""))))
    out = copy.deepcopy(record)
    original_id = str(record.get("id", ""))
    out["id"] = f"{original_id}__rob_{transform}"
    out["parent_question_id"] = original_id
    out["robustness_transform"] = transform
    out["robustness_seed"] = seed
    q = str(record.get("question") or "")

    if transform == "typo":
        out["question"] = _typo(q, rng)
    elif transform == "paraphrase_light":
        out["question"] = re.sub(r"\bWhat is\b", "What's", q, count=1) or q
        if out["question"] == q:
            out["question"] = q + " Please answer briefly."
    elif transform == "alias_variation":
        out["question"] = q.replace("Experiment", "Exp.").replace("experiment", "exp.")
    elif transform == "irrelevant_dialogue":
        out["question"] = (
            "By the way, I liked yesterday's weather. Anyway: " + q
        )
    elif transform == "case_noise":
        out["question"] = "".join(
            ch.upper() if rng.random() < 0.3 else ch for ch in q
        )
    else:
        raise KeyError(f"unknown robustness transform: {transform}")
    return out


def hashlib_id(text: str) -> int:
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


ROBUSTNESS_TRANSFORMS = (
    "typo",
    "paraphrase_light",
    "alias_variation",
    "irrelevant_dialogue",
    "case_noise",
)
