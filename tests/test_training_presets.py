"""Tests for training presets loader and CLI integration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stack.encoder.training_presets import (
    load_presets,
    get_preset,
    list_presets,
    apply_preset,
    get_model_defaults,
    DEFAULT_PRESETS_PATH,
)


class TestPresetsLoader:
    def test_load_presets_file_exists(self):
        presets = load_presets()
        assert presets["schema_version"] == "nexus-training-presets-v1"
        assert "presets" in presets
        assert "model_defaults" in presets

    def test_all_presets_present(self):
        presets = load_presets()
        names = set(presets["presets"].keys())
        assert names == {"smoke", "quick", "pilot", "standard", "full"}

    def test_each_preset_has_epochs(self):
        for name in list_presets():
            p = get_preset(name)
            assert "epochs" in p
            assert p["epochs"] > 0
            assert isinstance(p["epochs"], int)

    def test_smoke_is_minimal(self):
        p = get_preset("smoke")
        assert p["epochs"] == 1

    def test_full_is_maximal(self):
        p = get_preset("full")
        assert p["epochs"] == 50

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_preset("nonexistent")

    def test_model_defaults_exist(self):
        for model in ("realizer", "er3"):
            defaults = get_model_defaults(model)
            assert isinstance(defaults, dict)
            assert len(defaults) > 0



    def test_custom_path(self, tmp_path: Path):
        custom = tmp_path / "presets.json"
        custom.write_text(json.dumps({
            "schema_version": "nexus-training-presets-v1",
            "presets": {"test": {"epochs": 7, "patience": 1, "batch_size": 4}},
            "model_defaults": {},
        }))
        # Don't use the cached global — pass the path explicitly
        import stack.encoder.training_presets as tpm
        p = tpm.get_preset("test", str(custom))
        assert p["epochs"] == 7


class TestApplyPreset:
    def test_apply_overrides_current(self):
        params = apply_preset("smoke", {"epochs": 999, "lr": 0.01})
        assert params["epochs"] == 1  # smoke overrides
        assert params["learning_rate"] == 0.01  # normalized and preserved

    def test_cli_overrides_preset(self):
        params = apply_preset("smoke", {"epochs": 99}, cli_overrides={"epochs": 3})
        assert params["epochs"] == 3  # CLI wins

    def test_cli_none_does_not_override(self):
        params = apply_preset("quick", {"epochs": 99}, cli_overrides={"epochs": None})
        assert params["epochs"] == 5  # None = not provided, use preset

    def test_model_defaults_applied(self):
        params = apply_preset("smoke", {}, model_type="er3")
        assert "learning_rate" in params
        assert params["learning_rate"] == 0.001

    def test_pilot_with_model_defaults(self):
        params = apply_preset("pilot", {}, model_type="realizer")
        assert params["epochs"] == 12
        assert params["learning_rate"] == 0.0005

    def test_patience_from_preset(self):
        params = apply_preset("full", {})
        assert params["patience"] == 10

    def test_batch_size_from_preset(self):
        smoke = apply_preset("smoke", {})
        pilot = apply_preset("pilot", {})
        assert smoke["batch_size"] == 4
        assert pilot["batch_size"] == 16

    def test_priority_chain(self):
        """CLI > preset > model_defaults > current_params."""
        params = apply_preset(
            "quick",
            {"epochs": 999, "batch_size": 99},
            cli_overrides={"epochs": 7},
            model_type="realizer",
        )
        assert params["epochs"] == 7    # CLI
        assert params["batch_size"] == 8  # preset (quick)
        assert params["learning_rate"] == 0.0005  # model default


class TestPresetFilesExist:
    def test_presets_json_committed(self):
        assert DEFAULT_PRESETS_PATH.exists(), "training/presets.json must exist"

    def test_presets_json_is_valid(self):
        data = json.loads(DEFAULT_PRESETS_PATH.read_text(encoding="utf-8"))
        assert data["schema_version"] == "nexus-training-presets-v1"

    def test_run_er3_training_script_imports(self):
        """Preset module can be imported without torch."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "from stack.encoder.training_presets import load_presets; "
             "load_presets(); print('OK')"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
