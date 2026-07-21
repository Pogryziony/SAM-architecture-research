"""Tests for the fail-closed Realizer architecture registry."""

from __future__ import annotations

import pytest

from training.architecture_registry import (
    ArchitectureBlockedError,
    SCHEMA_VERSION,
    accepted_ids,
    assert_training_allowed,
    describe_production_profiles,
    find_by_config,
    load_architecture_registry,
    rejected_ids,
)


def test_registry_loads_and_has_schema():
    registry = load_architecture_registry()
    assert registry["schema_version"] == SCHEMA_VERSION
    assert "realizer_transformer_v1" in rejected_ids(registry)
    assert "pointer_copy_v3" in accepted_ids(registry)


def test_find_by_config_matches_basename():
    entry = find_by_config("training/nexus_realizer_v1.json")
    assert entry is not None
    assert entry["id"] == "realizer_transformer_v1"
    assert entry["section"] == "rejected"


def test_rejected_full_training_is_blocked():
    with pytest.raises(ArchitectureBlockedError, match="REALIZER_PILOT_FAIL"):
        assert_training_allowed("training/nexus_realizer_v1.json", action="full_training")


def test_answer_plan_prepare_data_still_allowed():
    entry = assert_training_allowed(
        "training/realizer_answer_plan_v1.json",
        action="prepare_data",
    )
    assert entry is not None
    assert entry["id"] == "answer_plan_autoregressive_pointer_generator"


def test_autoregressive_answer_plan_entrypoint_is_blocked():
    """Registry policy blocks AR AnswerPlan training without importing torch."""
    with pytest.raises(ArchitectureBlockedError, match="FULL_TRAINING_BLOCKED"):
        assert_training_allowed(
            "training/realizer_answer_plan_v1.json",
            action="full_training",
        )


def test_autoregressive_answer_plan_script_main_is_blocked():
    """Blocked script ``main()`` must not require torch at import or call time."""
    from training.architecture_registry import ArchitectureBlockedError
    from benchmarks import train_answer_plan_pilots

    with pytest.raises(ArchitectureBlockedError, match="edit_transducer"):
        train_answer_plan_pilots.main()


def test_accepted_config_not_blocked():
    entry = assert_training_allowed(
        "training/pointer_copy_realizer_v3.json",
        action="full_training",
    )
    assert entry is not None
    assert entry["section"] == "accepted"


def test_unknown_config_returns_none():
    assert assert_training_allowed("training/does_not_exist.json") is None


def test_production_profile_descriptions():
    profiles = describe_production_profiles()
    assert "pointer_copy_v3" in profiles["ProductionNEXUSConfig.grounded()"]
    assert "synth" in profiles["library_default"]
