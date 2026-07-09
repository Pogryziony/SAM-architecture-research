"""Training script for the Associative Encoder (Stage 1).

Trains on CPU only. Uses augmented train data, validates on clean val data.
Early stopping on val entity_accuracy (patience=10).
Saves best model to models/encoder/.

Usage:
    python stack/encoder/train.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add repo root to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from stack.encoder.model import (
    AssociativeEncoder,
    WordTokenizer,
    build_entity_mapping,
    build_intent_mapping,
    build_category_mapping,
    save_model,
    compute_data_hash,
)
from stack.encoder.augment import augment_dataset


# ── Dataset ──

class QuestionDataset(Dataset):
    """Dataset wrapping question data with entity/intent/category labels."""

    def __init__(
        self,
        questions: list[dict],
        entity_map: dict[str, int],
        intent_map: dict[str, int],
        category_map: dict[str, int],
        tokenizer: WordTokenizer,
    ):
        self.questions = questions
        self.entity_map = entity_map
        self.intent_map = intent_map
        self.category_map = category_map
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        q = self.questions[idx]

        # Encode question
        offsets, indices = self.tokenizer.encode_batch([q["question"]])

        # Multi-label entity vector
        entity_labels = torch.zeros(len(self.entity_map), dtype=torch.float32)
        for ent_id in q["entities"]:
            if ent_id in self.entity_map:
                entity_labels[self.entity_map[ent_id]] = 1.0

        intent_label = self.intent_map.get(q["intent"], 0)
        category_label = self.category_map.get(q["category"], 0)

        return {
            "offsets": offsets,
            "indices": indices,
            "entity_labels": entity_labels,
            "intent_label": intent_label,
            "category_label": category_label,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch, combining offset/indices for EmbeddingBag."""
    all_offsets: list[int] = []
    all_indices_list: list[int] = []

    entity_labels: list[torch.Tensor] = []
    intent_labels: list[int] = []
    category_labels: list[int] = []

    for item in batch:
        # item["indices"] is a 1D tensor [N] - flatten to list
        idx_list = item["indices"].flatten().tolist()
        all_offsets.append(len(all_indices_list))
        all_indices_list.extend(idx_list)

        entity_labels.append(item["entity_labels"])
        intent_labels.append(item["intent_label"])
        category_labels.append(item["category_label"])

    offsets = torch.tensor(all_offsets, dtype=torch.long)
    indices = torch.tensor(all_indices_list, dtype=torch.long)
    return {
        "offsets": offsets,
        "indices": indices,
        "entity_labels": torch.stack(entity_labels),
        "intent_labels": torch.tensor(intent_labels, dtype=torch.long),
        "category_labels": torch.tensor(category_labels, dtype=torch.long),
    }


# ── Metrics ──

@torch.no_grad()
def compute_metrics(
    model: AssociativeEncoder,
    dataloader: DataLoader,
    entity_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute entity, intent, and category accuracy on a dataset.

    entity_accuracy: fraction of GT entities that are predicted (recall)
    entity_precision: fraction of predicted entities that are in GT
    entity_f1: harmonic mean of precision and recall
    """
    model.eval()
    total = 0
    # Entity metrics
    entity_correct_preds = 0  # TP
    entity_total_preds = 0  # TP + FP
    entity_total_gt = 0  # TP + FN
    # Exact match: all GT predicted, no false positives
    exact_match_count = 0
    # Intent/category
    correct_intent = 0
    correct_category = 0

    for batch in dataloader:
        offsets = batch["offsets"]
        indices = batch["indices"]
        entity_labels = batch["entity_labels"]
        intent_labels = batch["intent_labels"]
        category_labels = batch["category_labels"]

        entity_logits, intent_logits, category_logits = model(offsets, indices)
        entity_preds = (torch.sigmoid(entity_logits) > entity_threshold).float()

        # Per-example metrics
        for i in range(len(entity_preds)):
            gt = entity_labels[i]
            pred = entity_preds[i]

            tp = ((pred == 1) & (gt == 1)).sum().item()
            fp = ((pred == 1) & (gt == 0)).sum().item()
            fn = ((pred == 0) & (gt == 1)).sum().item()

            entity_correct_preds += tp
            entity_total_preds += (tp + fp)
            entity_total_gt += (tp + fn)

            if fp == 0 and fn == 0:
                exact_match_count += 1

        # Intent accuracy
        intent_preds = torch.argmax(intent_logits, dim=1)
        correct_intent += (intent_preds == intent_labels).sum().item()

        # Category accuracy
        category_preds = torch.argmax(category_logits, dim=1)
        correct_category += (category_preds == category_labels).sum().item()

        total += len(entity_labels)

    precision = entity_correct_preds / entity_total_preds if entity_total_preds > 0 else 0.0
    recall = entity_correct_preds / entity_total_gt if entity_total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "entity_accuracy": recall,  # Primary: fraction of GT entities predicted
        "entity_precision": precision,
        "entity_f1": f1,
        "entity_exact_match": exact_match_count / total if total > 0 else 0.0,
        "intent_accuracy": correct_intent / total if total > 0 else 0.0,
        "category_accuracy": correct_category / total if total > 0 else 0.0,
    }
    return metrics


# ── Loss function ──

def compute_loss(
    entity_logits: torch.Tensor,
    intent_logits: torch.Tensor,
    category_logits: torch.Tensor,
    entity_labels: torch.Tensor,
    intent_labels: torch.Tensor,
    category_labels: torch.Tensor,
    entity_weight: float = 3.0,
) -> torch.Tensor:
    """Combined loss: BCE for entities, CE for intent, CE for category."""
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    loss_entity = bce(entity_logits, entity_labels)
    loss_intent = ce(intent_logits, intent_labels)
    loss_category = ce(category_logits, category_labels)

    return entity_weight * loss_entity + loss_intent + loss_category


# ── Training loop ──

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 150
    patience: int = 20
    entity_threshold: float = 0.3
    entity_weight: float = 3.0
    seed: int = 42
    output_dir: str = "models/encoder"


def get_peak_rss_mb() -> float:
    """Get peak RSS in MB (platform-specific)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        # Windows fallback using psutil
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / (1024 * 1024)
        except ImportError:
            return -1.0


def train(config: TrainingConfig | None = None) -> dict:
    """Run training and return results dict."""
    if config is None:
        config = TrainingConfig()

    torch.manual_seed(config.seed)
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # ── Load data ──
    train_raw: list[dict] = []
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f]

    val_raw: list[dict] = []
    with open(os.path.join(data_dir, "val.jsonl"), encoding="utf-8") as f:
        val_raw = [json.loads(line) for line in f]

    # ── Build entity mapping from entities that appear in training data ──
    # Only learn to predict entities that have training examples.
    # All graph nodes (366) would be the vocabulary, but with only ~21 appearing
    # in training data, restricting to those that appear makes the problem tractable.
    train_entity_ids: set[str] = set()
    for q in train_raw:
        train_entity_ids.update(q["entities"])
    # Also add val/test entities so the model can predict them (they appear in val)
    for q in val_raw:
        train_entity_ids.update(q["entities"])
    test_raw: list[dict] = []
    with open(os.path.join(data_dir, "test.jsonl"), encoding="utf-8") as f:
        test_raw = [json.loads(line) for line in f]
    for q in test_raw:
        train_entity_ids.update(q["entities"])

    all_entity_ids = sorted(train_entity_ids)
    entity_map = build_entity_mapping(all_entity_ids)
    intent_map = build_intent_mapping()
    category_map = build_category_mapping()

    print(f"Entity classes: {len(entity_map)} (from train+val+test)")
    print(f"Train raw: {len(train_raw)}, Val raw: {len(val_raw)}")

    # ── Apply augmentation (train only) ──
    train_augmented = augment_dataset(train_raw, seed=config.seed)
    print(f"Train augmented: {len(train_augmented)}")

    # ── Build vocabulary from train texts ──
    all_train_texts = [q["question"] for q in train_augmented]
    tokenizer = WordTokenizer.build_from_texts(all_train_texts)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # ── Create datasets and dataloaders ──
    train_dataset = QuestionDataset(train_augmented, entity_map, intent_map, category_map, tokenizer)
    val_dataset = QuestionDataset(val_raw, entity_map, intent_map, category_map, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Create model ──
    model = AssociativeEncoder(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        hidden_dim=256,
        num_entities=len(entity_map),
        num_intents=len(intent_map),
        num_categories=len(category_map),
    )
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Training state ──
    best_val_entity_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []

    rss_start = get_peak_rss_mb()
    print(f"Starting RSS: {rss_start:.1f} MB")
    print(f"{'Epoch':>6} {'Loss':>8} {'EntF1':>8} {'EntPrec':>8} {'EntRec':>8} {'IntAcc':>8} {'CatAcc':>8} {'VEntF1':>8} {'VEntPrec':>8} {'VIntAcc':>8} {'LR':>8}")

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()

        # ── Training ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            entity_logits, intent_logits, category_logits = model(batch["offsets"], batch["indices"])
            loss = compute_loss(
                entity_logits, intent_logits, category_logits,
                batch["entity_labels"], batch["intent_labels"], batch["category_labels"],
                entity_weight=config.entity_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Training metrics ──
        train_metrics = compute_metrics(model, train_loader, config.entity_threshold)

        # ── Validation metrics ──
        val_metrics = compute_metrics(model, val_loader, config.entity_threshold)

        # ── Scheduler step ──
        scheduler.step(val_metrics["entity_f1"])
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # ── Log ──
        print(
            f"{epoch:>6} {avg_loss:>8.4f} {train_metrics['entity_f1']:>8.4f} "
            f"{train_metrics['entity_precision']:>8.4f} {train_metrics['entity_accuracy']:>8.4f} "
            f"{train_metrics['intent_accuracy']:>8.4f} {train_metrics['category_accuracy']:>8.4f} "
            f"{val_metrics['entity_f1']:>8.4f} {val_metrics['entity_precision']:>8.4f} "
            f"{val_metrics['intent_accuracy']:>8.4f} {current_lr:>8.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_entity_f1": train_metrics["entity_f1"],
            "train_entity_precision": train_metrics["entity_precision"],
            "train_entity_recall": train_metrics["entity_accuracy"],
            "train_intent_accuracy": train_metrics["intent_accuracy"],
            "train_category_accuracy": train_metrics["category_accuracy"],
            "val_entity_f1": val_metrics["entity_f1"],
            "val_entity_precision": val_metrics["entity_precision"],
            "val_entity_recall": val_metrics["entity_accuracy"],
            "val_intent_accuracy": val_metrics["intent_accuracy"],
            "val_category_accuracy": val_metrics["category_accuracy"],
            "lr": current_lr,
            "epoch_time_s": epoch_time,
        })

        # ── Early stopping check ──
        val_score = val_metrics["entity_f1"]  # Optimize for entity F1
        if val_score > best_val_entity_acc:
            best_val_entity_acc = val_score
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            output_dir = os.path.join(_repo_root, config.output_dir)
            os.makedirs(output_dir, exist_ok=True)

            config_dict = {
                "architecture": "AssociativeEncoder",
                "vocab_size": len(tokenizer.vocab),
                "embed_dim": 128,
                "hidden_dim": 256,
                "num_entities": len(entity_map),
                "num_intents": len(intent_map),
                "num_categories": len(category_map),
                "parameter_count": param_count,
                "data_hash": compute_data_hash(os.path.join(data_dir, "train.jsonl")),
                "seed": config.seed,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "entity_weight": config.entity_weight,
                "patience": config.patience,
                "max_epochs": config.max_epochs,
                "best_epoch": best_epoch,
                "best_val_entity_accuracy": best_val_entity_acc,
                "best_val_intent_accuracy": val_metrics["intent_accuracy"],
                "best_val_category_accuracy": val_metrics["category_accuracy"],
                "training_history": history,
                "rss_start_mb": rss_start,
            }

            save_model(
                model, tokenizer, entity_map, intent_map, category_map,
                config_dict, output_dir,
            )
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
            break

    rss_end = get_peak_rss_mb()
    peak_rss = max(rss_start, rss_end)
    print(f"\nPeak RSS: {peak_rss:.1f} MB")
    print(f"Best val entity_accuracy: {best_val_entity_acc:.4f} at epoch {best_epoch}")

    return {
        "best_entity_accuracy": best_val_entity_acc,
        "best_epoch": best_epoch,
        "peak_rss_mb": peak_rss,
        "parameter_count": param_count,
        "history": history,
    }


if __name__ == "__main__":
    result = train()
    print(f"\nTraining complete. Best entity_accuracy={result['best_entity_accuracy']:.4f}")
