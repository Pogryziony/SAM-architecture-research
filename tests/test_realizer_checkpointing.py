"""Tests for NEXUS Realizer v2 checkpointing, resume, early stopping, and config invariance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from benchmarks.train_nexus_realizer_v2 import (
    _generate_sample_predictions,
    _scheduler_total_epochs,
    _save_checkpoint,
    train_v2,
)
from benchmarks.train_nexus_realizer import (
    apply_training_overrides,
    effective_config_sha256,
)
from benchmarks.build_distillation_dataset import build_distillation_dataset
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.realizer.decoder import DecoderConfig
from nexus.realizer.model import build_model


_FAST_DECODER = DecoderConfig(
    strategy="greedy", repetition_penalty=1.2,
    no_repeat_ngram_size=3, max_length=16,
)


@pytest.fixture
def small_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal distillation dataset for testing."""
    graph = InMemoryGraphStore()
    questions = []
    for idx in range(30):
        source = f"Source{idx}"
        target = f"Target{idx}"
        graph.add_node(Node(
            source, "Concept",
            properties={"description": f"{source} achieved {90 + idx % 10}% accuracy."},
            sources=[f"docs/{source}.md"],
        ))
        graph.add_node(Node(
            target, "Concept",
            properties={"description": f"{target} supports the result."},
            sources=[f"docs/{target}.md"],
        ))
        graph.add_edge(Edge("related_to", source, target, evidence=f"docs/edge-{idx}.md"))
        questions.append({
            "id": f"q{idx}",
            "question": f"What accuracy did {source} achieve?",
            "answer": f"{source} achieved {90 + idx % 10}% accuracy.",
            "entities": [source],
            "source_split": "train",
        })
    dataset_root = tmp_path / "dataset"
    build_distillation_dataset(questions, graph, str(dataset_root), "a" * 40, min_pairs=30)
    manifest_path = dataset_root / "manifest.json"
    return manifest_path, Path("training/nexus_realizer_v1.json")


def test_save_checkpoint_saves_weights_and_manifest(tmp_path: Path):
    """Checkpoint saves model weights, manifest, and SHA-256 sidecar."""
    model = build_model({"d_model": 192, "decoder_layers": 3, "dim_feedforward": 512,
                         "dropout": 0.1, "encoder_layers": 3, "max_input_tokens": 1024,
                         "max_output_tokens": 256, "nhead": 6, "vocab_size": 259})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    decoder_cfg = DecoderConfig(strategy="greedy", repetition_penalty=1.2, no_repeat_ngram_size=3)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    ckpt = _save_checkpoint(
        model, optimizer, scheduler, output_dir, epoch=1,
        train_loss=100.0, validation_loss=110.0, learning_rate=0.0001,
        gen_metrics={"coherent_rate": 0.5, "eos_rate": 0.8, "rep_3gram_mean": 0.1},
        effective_config={"epochs": 5}, effective_config_hash="abc123",
        dataset_sha256="ds_sha", config_sha256="cfg_sha",
        parameter_count=1000000, elapsed_seconds=10.0, epoch_seconds=1.0,
        decoder_cfg=decoder_cfg,
        sample_predictions=[{"id": "q0", "generated_text": "test", "ground_truth": "test"}],
    )

    ckpt_dir = output_dir / "checkpoint_epoch_001"
    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "model.pt").is_file()
    assert (ckpt_dir / "manifest.json").is_file()
    assert (ckpt_dir / "manifest.json.sha256").is_file()
    assert (ckpt_dir / "sample_predictions.json").is_file()
    assert "weights" in ckpt
    assert ckpt["weights"]["sha256"] == hashlib.sha256(
        (ckpt_dir / "model.pt").read_bytes()
    ).hexdigest()
    assert ckpt["epoch"] == 1
    assert ckpt["files"]["model.pt"]["stored_in_git"] is False
    assert ckpt["optimizer"]["stored_in_git"] is False
    assert ckpt["scheduler"]["stored_in_git"] is False
    assert "\\" not in ckpt["weights"]["path"]


def test_one_epoch_run_keeps_three_epoch_scheduler_plan():
    config = {"training": {"scheduler_total_epochs": 3}}
    assert _scheduler_total_epochs(config, 1) == 3
    assert _scheduler_total_epochs(config, 5) == 5


def test_save_checkpoint_refuses_overwrite(tmp_path: Path):
    """Checkpoint save raises FileExistsError if weights already exist."""
    model = build_model({"d_model": 192, "decoder_layers": 3, "dim_feedforward": 512,
                         "dropout": 0.1, "encoder_layers": 3, "max_input_tokens": 1024,
                         "max_output_tokens": 256, "nhead": 6, "vocab_size": 259})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    decoder_cfg = DecoderConfig()

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    # First save succeeds
    _save_checkpoint(
        model, optimizer, scheduler, output_dir, epoch=1,
        train_loss=100.0, validation_loss=110.0, learning_rate=0.0001,
        gen_metrics={}, effective_config={}, effective_config_hash="",
        dataset_sha256="", config_sha256="", parameter_count=1000000,
        elapsed_seconds=1.0, epoch_seconds=1.0, decoder_cfg=decoder_cfg,
    )
    # Second save must fail
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _save_checkpoint(
            model, optimizer, scheduler, output_dir, epoch=1,
            train_loss=100.0, validation_loss=110.0, learning_rate=0.0001,
            gen_metrics={}, effective_config={}, effective_config_hash="",
            dataset_sha256="", config_sha256="", parameter_count=1000000,
            elapsed_seconds=1.0, epoch_seconds=1.0, decoder_cfg=decoder_cfg,
        )


def test_early_stopping_stops_within_patience(small_dataset, tmp_path: Path):
    """Training respects early_stopping_patience."""
    manifest_path, config_path = small_dataset
    output_dir = tmp_path / "run"
    result = train_v2(
        manifest_path, config_path, mode="pilot",
        output_dir=output_dir,
        training_overrides={"epochs": 10, "patience": 2},
        decoder_config=_FAST_DECODER,
        checkpoint_epochs=[],
    )
    # With patience=2 and 10 epochs, should stop early (not reach 10)
    assert result["epochs_completed"] < 10
    assert result["epochs_completed"] <= result["best_epoch"] + 2 + 1


def test_checkpoints_saved_at_specified_epochs(small_dataset, tmp_path: Path):
    """Checkpoints are saved at epochs 1, 3, 5."""
    manifest_path, config_path = small_dataset
    output_dir = tmp_path / "run"
    result = train_v2(
        manifest_path, config_path, mode="pilot",
        output_dir=output_dir,
        training_overrides={"epochs": 5, "patience": 3},
        decoder_config=_FAST_DECODER,
        checkpoint_epochs=[1, 3, 5],
    )
    saved = result.get("saved_checkpoints", [])
    epochs = [c["epoch"] for c in saved]
    # Should have checkpoints at requested epochs (may stop early)
    assert len(saved) >= 1
    for ckpt_epoch in [1, 3, 5]:
        if ckpt_epoch <= result["epochs_completed"]:
            assert ckpt_epoch in epochs, f"Missing checkpoint at epoch {ckpt_epoch}"

    # Verify each checkpoint has weights and manifest
    for ckpt in saved:
        ckpt_dir = output_dir / f"checkpoint_epoch_{ckpt['epoch']:03d}"
        assert ckpt_dir.is_dir()
        assert (ckpt_dir / "model.pt").is_file()
        assert (ckpt_dir / "manifest.json").is_file()
        assert ckpt["weights"]["sha256"] == hashlib.sha256(
            (ckpt_dir / "model.pt").read_bytes()
        ).hexdigest()


def test_config_invariance_across_runs(small_dataset, tmp_path: Path):
    """Training with the same config produces consistent effective_config_sha256."""
    manifest_path, config_path = small_dataset
    output_dir1 = tmp_path / "run1"
    output_dir2 = tmp_path / "run2"
    result1 = train_v2(
        manifest_path, config_path, mode="pilot",
        output_dir=output_dir1,
        training_overrides={"epochs": 2, "patience": 1},
        decoder_config=_FAST_DECODER,
        checkpoint_epochs=[],
    )
    result2 = train_v2(
        manifest_path, config_path, mode="pilot",
        output_dir=output_dir2,
        training_overrides={"epochs": 2, "patience": 1},
        decoder_config=_FAST_DECODER,
        checkpoint_epochs=[],
    )
    assert result1["effective_config_sha256"] == result2["effective_config_sha256"]


def test_custom_training_overrides_affect_effective_hash(small_dataset, tmp_path: Path):
    """Different training overrides produce different effective config hashes."""
    manifest_path, config_path = small_dataset
    output_dir1 = tmp_path / "run1"
    output_dir2 = tmp_path / "run2"
    result1 = train_v2(
        manifest_path, config_path, mode="pilot",
        output_dir=output_dir1,
        training_overrides={"epochs": 2, "learning_rate": 0.001},
        decoder_config=_FAST_DECODER,
        checkpoint_epochs=[],
    )
    result2 = train_v2(
        manifest_path, config_path, mode="pilot",
        output_dir=output_dir2,
        training_overrides={"epochs": 3, "learning_rate": 0.002},
        decoder_config=_FAST_DECODER,
        checkpoint_epochs=[],
    )
    assert result1["effective_config_sha256"] != result2["effective_config_sha256"]


def test_generate_sample_predictions_returns_valid_output(small_dataset):
    """Sample predictions include question, generated text, ground truth, and metrics."""
    manifest_path, config_path = small_dataset
    import json as _json
    manifest = _json.loads(manifest_path.read_text())
    train_path = manifest_path.parent / manifest["splits"]["train"]["path"]
    records = [_json.loads(line) for line in train_path.read_text().splitlines() if line][:5]
    model = build_model({"d_model": 192, "decoder_layers": 3, "dim_feedforward": 512,
                         "dropout": 0.1, "encoder_layers": 3, "max_input_tokens": 1024,
                         "max_output_tokens": 256, "nhead": 6, "vocab_size": 259})
    decoder_cfg = _FAST_DECODER
    preds = _generate_sample_predictions(
        model, records, {"model": {"max_input_tokens": 1024, "max_output_tokens": 256}},
        decoder_cfg, max_samples=3,
    )
    assert len(preds) <= 3
    for p in preds:
        assert "question" in p
        assert "generated_text" in p
        assert "ground_truth" in p
        assert "eos_reached" in p
        assert "rep_3gram" in p


def test_six_epochs_with_checkpoint_epochs(small_dataset, tmp_path: Path):
    """Checkpoint epochs filter: only epochs 1,3,5 saved if they complete."""
    manifest_path, config_path = small_dataset
    output_dir = tmp_path / "run"
    result = train_v2(
        manifest_path, config_path, mode="pilot",
        output_dir=output_dir,
        training_overrides={"epochs": 6, "patience": 5},
        decoder_config=_FAST_DECODER,
        checkpoint_epochs=[1, 3, 5],
    )
    saved = result.get("saved_checkpoints", [])
    epochs = {c["epoch"] for c in saved}
    completed = result["epochs_completed"]
    # At minimum epoch 1 should be saved
    assert 1 in epochs
    # Epoch 3 should be saved if training reached it
    if completed >= 3:
        assert 3 in epochs
    # Epoch 5 should be saved if training reached it (may stop early)
    if completed >= 5:
        assert 5 in epochs
    # Verify no checkpoints beyond completed epochs
    for ckpt in saved:
        assert ckpt["epoch"] <= completed
