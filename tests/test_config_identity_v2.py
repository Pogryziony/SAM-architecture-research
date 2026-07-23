"""Regression tests for nexus-config-identity-v2."""

from __future__ import annotations

import pytest

from nexus.pipeline.config import (
    CONFIG_IDENTITY_SCHEMA,
    PipelineIdentity,
    ProductionNEXUSConfig,
    _portable_path,
    validate_config,
)


def test_identity_schema_is_v2():
    cfg = ProductionNEXUSConfig.lexical_only()
    assert cfg.identity_schema == CONFIG_IDENTITY_SCHEMA
    assert cfg.identity_schema == "nexus-config-identity-v2"
    assert cfg.to_dict()["identity_schema"] == "nexus-config-identity-v2"


def test_omitted_boost_fields_change_hash():
    base = ProductionNEXUSConfig.lexical_only()
    changed = ProductionNEXUSConfig.lexical_only(alias_match_boost=12.0)
    assert base.config_hash != changed.config_hash


def test_type_priority_change_changes_hash():
    base = ProductionNEXUSConfig.lexical_only()
    changed = ProductionNEXUSConfig.lexical_only(
        type_priority={**dict(base.type_priority), "Experiment": 99}
    )
    assert base.config_hash != changed.config_hash


def test_nested_type_priority_is_immutable():
    cfg = ProductionNEXUSConfig.lexical_only()
    with pytest.raises(TypeError):
        cfg.type_priority["Experiment"] = 99  # type: ignore[index]


def test_er3_dir_is_hashed():
    a = ProductionNEXUSConfig(
        pipeline_id=PipelineIdentity(
            entity_ranker_v3_enabled=True,
            entity_ranker_v3_dir="models/encoder/entity_ranker_v3_20260711T081545Z",
        )
    )
    b = ProductionNEXUSConfig(
        pipeline_id=PipelineIdentity(
            entity_ranker_v3_enabled=True,
            entity_ranker_v3_dir="models/encoder/other_ranker",
        )
    )
    assert a.config_hash != b.config_hash


def test_portable_path_strips_absolute_windows_prefix():
    abs_path = r"C:\Users\Pogry\Projects\SAM-architecture-research\models\encoder\foo"
    portable = _portable_path(abs_path)
    assert ":" not in portable
    assert portable.replace("\\", "/").endswith("models/encoder/foo") or portable.endswith(
        "encoder/foo"
    ) or "models/" in portable.replace("\\", "/")


def test_absolute_and_relative_model_dirs_hash_equally():
    rel = ProductionNEXUSConfig.comparison_plan(
        realizer_model_dir="models/realizer/abstractive_v1_plan_v3"
    )
    abs_dir = str(
        (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "models"
            / "realizer"
            / "abstractive_v1_plan_v3"
        )
    )
    abs_cfg = ProductionNEXUSConfig.comparison_plan(realizer_model_dir=abs_dir)
    assert rel.config_hash == abs_cfg.config_hash


def test_domain_pack_identity_changes_hash():
    a = ProductionNEXUSConfig(
        pipeline_id=PipelineIdentity(domain_pack_id="sam", domain_pack_version="sam-v1")
    )
    b = ProductionNEXUSConfig(
        pipeline_id=PipelineIdentity(domain_pack_id="mini", domain_pack_version="mini-v1")
    )
    assert a.config_hash != b.config_hash


def test_missing_er3_dir_with_sha_fails_closed():
    cfg = ProductionNEXUSConfig(
        pipeline_id=PipelineIdentity(
            entity_ranker_v3_enabled=True,
            entity_ranker_v3_dir="models/encoder/does_not_exist_xyz",
            entity_ranker_v3_checkpoint_sha256="a" * 64,
        )
    )
    errors = validate_config(cfg)
    assert any("entity_ranker_v3_dir not found" in e for e in errors)
