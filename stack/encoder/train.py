"""Training script for the Associative Encoder v2 (Stage 1b).

Trains on CPU only. Uses augmented train data, validates on clean val data.
Early stopping on val intent_accuracy (patience=10).
Saves best model to models/encoder_v2/.

Architecture changes from Stage 1:
  - Char n-gram tokenizer (OOV robustness)
  - Sequential component (1-layer bidirectional GRU)
  - Entity re-ranker head (scores candidates, not open-set classification)
  - Focal loss on entity scoring and intent classification

Usage:
    python stack/encoder/train.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add repo root to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from stack.encoder.model import (
    AssociativeEncoderV2,
    build_entity_mapping,
    build_intent_mapping,
    build_category_mapping,
    save_model_v2,
    compute_data_hash,
)
from stack.encoder.char_tokenizer import CharNgramTokenizer
from stack.encoder.augment import augment_dataset


# ── Focal Loss ──

class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification (binary or multi-class).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    For binary: input logits [B] or [B, 1], targets [B]
    For multi-class: input logits [B, C], targets [B]
    """

    def __init__(self, gamma: float = 2.0, alpha: float | torch.Tensor = 1.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(alpha) if isinstance(alpha, float) else alpha)

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.size(1) == 1):
            # Binary focal loss
            if inputs.dim() == 2:
                inputs = inputs.squeeze(-1)
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets.float(), reduction="none",
            )
            p_t = torch.where(targets == 1, torch.sigmoid(inputs), 1 - torch.sigmoid(inputs))
            modulating = (1 - p_t) ** self.gamma
            return (modulating * bce_loss).mean()

        # Multi-class focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        modulating = (1 - p_t) ** self.gamma
        return (modulating * ce_loss).mean()


# ── Dataset ──

class QuestionDatasetV2(Dataset):
    """Dataset for v2 encoder: question + candidate entity set.

    For each question, produces:
    - Feature IDs (char n-gram tokenizer)
    - Intent label, category label
    - Entity candidates (top-K from lexical mock + GT for training)
    - Entity candidate labels (positive/negative)
    """

    def __init__(
        self,
        questions: list[dict],
        entity_map: dict[str, int],
        intent_map: dict[str, int],
        category_map: dict[str, int],
        tokenizer: CharNgramTokenizer,
        all_entity_ids: list[str],
        all_entity_descriptions: list[str],
        max_candidates: int = 20,
        seed: int = 42,
    ):
        self.questions = questions
        self.entity_map = entity_map
        self.intent_map = intent_map
        self.category_map = category_map
        self.tokenizer = tokenizer
        self.all_entity_ids = all_entity_ids
        self.all_entity_descriptions = all_entity_descriptions
        self.max_candidates = max_candidates
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        q = self.questions[idx]
        gt_entities = set(q["entities"])

        # Encode question features
        offsets_list, indices_list = self.tokenizer.tokenize_batch([q["question"]])

        # Build candidate entity set: include all GT + random distractors
        gt_candidates = [e for e in self.all_entity_ids if e in gt_entities]
        # Sample distractors from non-GT entities
        distractors = [e for e in self.all_entity_ids if e not in gt_entities]
        n_needed = self.max_candidates - len(gt_candidates)
        if n_needed > 0 and distractors:
            sampled = self._rng.sample(distractors, min(n_needed, len(distractors)))
        else:
            sampled = []
        candidates = gt_candidates + sampled

        # Entity labels: 1 for GT, 0 for distractors
        entity_labels = [1.0 if e in gt_entities else 0.0 for e in candidates]

        intent_label = self.intent_map.get(q["intent"], 0)
        category_label = self.category_map.get(q.get("category", q.get("question_type", "")), 0)

        return {
            "offsets": offsets_list,
            "indices": indices_list,
            "candidate_ids": candidates,
            "entity_labels": entity_labels,
            "intent_label": intent_label,
            "category_label": category_label,
        }


def collate_fn_v2(batch: list[dict]) -> dict:
    """Collate v2 batch for EmbeddingBag + candidate entity re-ranking."""
    all_indices: list[int] = []
    all_offsets: list[int] = [0]
    all_entity_labels: list[torch.Tensor] = []
    all_intent_labels: list[int] = []
    all_category_labels: list[int] = []
    all_candidate_descriptions: list[list[str]] = []

    for item in batch:
        all_offsets.append(all_offsets[-1] + len(item["indices"]))
        all_indices.extend(item["indices"])
        all_entity_labels.append(torch.tensor(item["entity_labels"], dtype=torch.float32))
        all_intent_labels.append(item["intent_label"])
        all_category_labels.append(item["category_label"])
        # Resolve candidate descriptions from tokenizer vocab
        all_candidate_descriptions.append(item["candidate_ids"])

    offsets = torch.tensor(all_offsets[:-1], dtype=torch.long)
    indices = torch.tensor(all_indices, dtype=torch.long)

    return {
        "offsets": offsets,
        "indices": indices,
        "entity_labels": all_entity_labels,
        "intent_labels": torch.tensor(all_intent_labels, dtype=torch.long),
        "category_labels": torch.tensor(all_category_labels, dtype=torch.long),
        "candidate_descriptions": all_candidate_descriptions,
    }


# ── Metrics ──

@torch.no_grad()
def compute_metrics_v2(
    model: AssociativeEncoderV2,
    dataloader: DataLoader,
    tokenizer: CharNgramTokenizer,
    entity_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute entity, intent, and category accuracy on a dataset."""
    model.eval()
    total = 0
    entity_correct_preds = 0
    entity_total_preds = 0
    entity_total_gt = 0
    correct_intent = 0
    correct_category = 0

    for batch in dataloader:
        offsets = batch["offsets"]
        indices = batch["indices"]
        intent_labels = batch["intent_labels"]
        category_labels = batch["category_labels"]
        candidate_desc_batches = batch["candidate_descriptions"]

        # Embed candidates
        all_cand_feats = []
        for descs in candidate_desc_batches:
            if descs:
                feats = model.embed_entities(descs, tokenizer)  # [1, K, E]
                all_cand_feats.append(feats.squeeze(0))
            else:
                all_cand_feats.append(torch.empty(0, model.embedding.embedding_dim))

        # Pad to same K
        max_k = max(f.shape[0] for f in all_cand_feats) if all_cand_feats else 0
        if max_k > 0:
            padded = []
            for f in all_cand_feats:
                if f.shape[0] < max_k:
                    pad = torch.zeros(max_k - f.shape[0], f.shape[1])
                    padded.append(torch.cat([f, pad]))
                elif f.shape[0] == max_k:
                    padded.append(f)
                else:
                    padded.append(f[:max_k])
            cand_feats = torch.stack(padded)  # [B, K, E]
        else:
            cand_feats = None

        intent_logits, cat_logits, entity_scores = model(
            indices, offsets, cand_feats,
        )

        # Intent accuracy
        intent_preds = torch.argmax(intent_logits, dim=1)
        correct_intent += (intent_preds == intent_labels).sum().item()

        # Category accuracy
        cat_preds = torch.argmax(cat_logits, dim=1)
        correct_category += (cat_preds == category_labels).sum().item()

        # Entity metrics
        if entity_scores is not None:
            entity_labels = batch["entity_labels"]
            for i in range(len(entity_labels)):
                gt = entity_labels[i]
                scores = torch.sigmoid(entity_scores[i])
                preds = (scores > entity_threshold).float()
                # Only consider non-padded positions
                valid_len = len(candidate_desc_batches[i])
                if valid_len > 0:
                    gt_v = gt[:valid_len]
                    pred_v = preds[:valid_len]
                    tp = ((pred_v == 1) & (gt_v == 1)).sum().item()
                    fp = ((pred_v == 1) & (gt_v == 0)).sum().item()
                    fn = ((pred_v == 0) & (gt_v == 1)).sum().item()
                    entity_correct_preds += tp
                    entity_total_preds += tp + fp
                    entity_total_gt += tp + fn

        total += len(intent_labels)

    recall = entity_correct_preds / entity_total_gt if entity_total_gt > 0 else 0.0
    precision = entity_correct_preds / entity_total_preds if entity_total_preds > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return {
        "entity_accuracy": recall,
        "entity_precision": precision,
        "entity_f1": f1,
        "intent_accuracy": correct_intent / total if total > 0 else 0.0,
        "category_accuracy": correct_category / total if total > 0 else 0.0,
    }


# ── Loss function ──

def compute_loss_v2(
    intent_logits: torch.Tensor,
    category_logits: torch.Tensor,
    entity_scores: torch.Tensor | None,
    intent_labels: torch.Tensor,
    category_labels: torch.Tensor,
    entity_labels_batch: list[torch.Tensor],
    intent_focal: FocalLoss,
    entity_focal: FocalLoss | None = None,
    entity_weight: float = 2.0,
) -> torch.Tensor:
    """Combined loss: focal on intent, CE on category, focal on entity re-ranker."""
    loss_intent = intent_focal(intent_logits, intent_labels)
    loss_category = F.cross_entropy(category_logits, category_labels)

    total_loss = loss_intent + loss_category

    if entity_scores is not None and entity_labels_batch:
        # Entity re-ranker loss: per-example binary focal loss on candidates
        entity_losses = []
        for i in range(len(entity_labels_batch)):
            gt = entity_labels_batch[i]
            valid_len = len(gt)
            if valid_len > 0:
                scores_i = entity_scores[i, :valid_len]
                if entity_focal is not None:
                    loss_e = entity_focal(scores_i, gt)
                else:
                    loss_e = F.binary_cross_entropy_with_logits(scores_i, gt)
                entity_losses.append(loss_e)
        if entity_losses:
            loss_entity = torch.stack(entity_losses).mean()
            total_loss = total_loss + entity_weight * loss_entity

    return total_loss


# ── Training config ──

@dataclass
class TrainingConfigV2:
    """Training hyperparameters for v2."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 150
    patience: int = 10
    entity_threshold: float = 0.5
    entity_weight: float = 2.0
    focal_gamma: float = 2.0
    seed: int = 42
    output_dir: str = "models/encoder_v2"
    max_candidates: int = 20


def get_peak_rss_mb() -> float:
    """Get peak RSS in MB (platform-specific)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / (1024 * 1024)
        except ImportError:
            return -1.0


def train_v2(config: TrainingConfigV2 | None = None) -> dict:
    """Run training and return results dict."""
    if config is None:
        config = TrainingConfigV2()

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # ── Load data ──
    train_raw: list[dict] = []
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f]

    val_raw: list[dict] = []
    with open(os.path.join(data_dir, "val.jsonl"), encoding="utf-8") as f:
        val_raw = [json.loads(line) for line in f]

    test_raw: list[dict] = []
    with open(os.path.join(data_dir, "test.jsonl"), encoding="utf-8") as f:
        test_raw = [json.loads(line) for line in f]

    # ── Build entity pool (all entities appearing in train+val+test) ──
    all_entity_ids_set: set[str] = set()
    for q in train_raw:
        all_entity_ids_set.update(q["entities"])
    for q in val_raw:
        all_entity_ids_set.update(q["entities"])
    for q in test_raw:
        all_entity_ids_set.update(q["entities"])
    all_entity_ids = sorted(all_entity_ids_set)

    # Entity descriptions: use entity ID as description for now
    # In production, these would come from the graph
    all_entity_descriptions = [
        eid.replace("_", " ") for eid in all_entity_ids
    ]

    entity_map = build_entity_mapping(all_entity_ids)
    intent_map = build_intent_mapping()
    category_map = build_category_mapping()

    print(f"Entity pool: {len(all_entity_ids)}")
    print(f"Train raw: {len(train_raw)}, Val raw: {len(val_raw)}")

    # ── Apply augmentation (train only) ──
    train_augmented = augment_dataset(train_raw, seed=config.seed)
    print(f"Train augmented: {len(train_augmented)}")

    # ── Build char n-gram tokenizer ──
    all_texts = [q["question"] for q in train_augmented]
    # Also add entity descriptions to vocab
    all_texts.extend(all_entity_descriptions)
    tokenizer = CharNgramTokenizer(tri_buckets=2000, penta_buckets=1000)
    tokenizer.add_words(all_texts)
    tokenizer.freeze()
    print(f"Feature dim: {tokenizer.feature_dim} (words={tokenizer.word_vocab_size})")

    # ── Create datasets ──
    train_dataset = QuestionDatasetV2(
        train_augmented, entity_map, intent_map, category_map,
        tokenizer, all_entity_ids, all_entity_descriptions,
        max_candidates=config.max_candidates, seed=config.seed,
    )
    val_dataset = QuestionDatasetV2(
        val_raw, entity_map, intent_map, category_map,
        tokenizer, all_entity_ids, all_entity_descriptions,
        max_candidates=config.max_candidates, seed=config.seed + 100,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn_v2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn_v2,
    )

    # ── Create model ──
    model = AssociativeEncoderV2(
        feature_dim=tokenizer.feature_dim,
        embed_dim=128,
        hidden_dim=256,
        num_intents=len(intent_map),
        num_categories=len(category_map),
    )
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")

    # ── Loss functions ──
    # Compute class weights for intent
    intent_counts = {}
    for q in train_augmented:
        il = q.get("intent", "")
        intent_counts[il] = intent_counts.get(il, 0) + 1
    total_intents = sum(intent_counts.values())
    max_count = max(intent_counts.values())
    intent_weights = torch.tensor([
        max_count / intent_counts.get(k, 1) if intent_counts.get(k, 0) > 0 else 1.0
        for k in ["factual_lookup", "comparison", "multi_hop", "diagnostic"]
    ], dtype=torch.float32)
    # Normalize weights
    intent_weights = intent_weights / intent_weights.sum() * len(intent_weights)
    print(f"Intent class weights: {intent_weights.tolist()}")

    intent_focal = FocalLoss(gamma=config.focal_gamma, alpha=intent_weights)
    entity_focal = FocalLoss(gamma=config.focal_gamma)

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # ── Training state ──
    best_val_intent_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    rss_start = get_peak_rss_mb()

    print(f"Starting RSS: {rss_start:.1f} MB")
    header = f"{'Epoch':>6} {'Loss':>8} {'EntF1':>8} {'EntPr':>8} {'EntRe':>8} {'IntAcc':>8} {'CatAcc':>8} {'VIntAcc':>8} {'VEntF1':>8} {'LR':>8}"
    print(header)

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # Embed candidate entities
            candidate_desc_batches = batch["candidate_descriptions"]
            all_cand_feats = []
            for descs in candidate_desc_batches:
                if descs:
                    feats = model.embed_entities(descs, tokenizer)
                    all_cand_feats.append(feats.squeeze(0))
                else:
                    all_cand_feats.append(
                        torch.empty(0, model.embedding.embedding_dim),
                    )

            max_k = max(f.shape[0] for f in all_cand_feats) if all_cand_feats else 0
            if max_k > 0:
                padded = []
                for f in all_cand_feats:
                    if f.shape[0] < max_k:
                        pad = torch.zeros(max_k - f.shape[0], f.shape[1])
                        padded.append(torch.cat([f, pad]))
                    else:
                        padded.append(f[:max_k])
                cand_feats = torch.stack(padded)
            else:
                cand_feats = None

            intent_logits, cat_logits, entity_scores = model(
                batch["indices"], batch["offsets"], cand_feats,
            )

            loss = compute_loss_v2(
                intent_logits, cat_logits, entity_scores,
                batch["intent_labels"], batch["category_labels"],
                batch["entity_labels"],
                intent_focal, entity_focal, config.entity_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Metrics ──
        train_metrics = compute_metrics_v2(
            model, train_loader, tokenizer, config.entity_threshold,
        )
        val_metrics = compute_metrics_v2(
            model, val_loader, tokenizer, config.entity_threshold,
        )

        scheduler.step(val_metrics["intent_accuracy"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"{epoch:>6} {avg_loss:>8.4f} {train_metrics['entity_f1']:>8.4f} "
            f"{train_metrics['entity_precision']:>8.4f} {train_metrics['entity_accuracy']:>8.4f} "
            f"{train_metrics['intent_accuracy']:>8.4f} {train_metrics['category_accuracy']:>8.4f} "
            f"{val_metrics['intent_accuracy']:>8.4f} {val_metrics['entity_f1']:>8.4f} "
            f"{current_lr:>8.6f}"
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

        # ── Early stopping on val intent_accuracy ──
        val_score = val_metrics["intent_accuracy"]
        if val_score > best_val_intent_acc:
            best_val_intent_acc = val_score
            best_epoch = epoch
            patience_counter = 0

            output_dir = os.path.join(_repo_root, config.output_dir)
            os.makedirs(output_dir, exist_ok=True)

            config_dict = {
                "architecture": "AssociativeEncoderV2",
                "feature_dim": tokenizer.feature_dim,
                "word_vocab_size": tokenizer.word_vocab_size,
                "tri_buckets": tokenizer.tri_buckets,
                "penta_buckets": tokenizer.penta_buckets,
                "embed_dim": 128,
                "hidden_dim": 256,
                "num_intents": len(intent_map),
                "num_categories": len(category_map),
                "max_candidates": config.max_candidates,
                "parameter_count": param_count,
                "data_hash": compute_data_hash(os.path.join(data_dir, "train.jsonl")),
                "seed": config.seed,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "focal_gamma": config.focal_gamma,
                "entity_weight": config.entity_weight,
                "patience": config.patience,
                "max_epochs": config.max_epochs,
                "best_epoch": best_epoch,
                "best_val_intent_accuracy": best_val_intent_acc,
                "best_val_entity_accuracy": val_metrics["entity_accuracy"],
                "training_history": history,
                "rss_start_mb": rss_start,
            }

            save_model_v2(
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
    print(f"Best val intent_accuracy: {best_val_intent_acc:.4f} at epoch {best_epoch}")

    return {
        "best_intent_accuracy": best_val_intent_acc,
        "best_epoch": best_epoch,
        "peak_rss_mb": peak_rss,
        "parameter_count": param_count,
        "history": history,
    }


if __name__ == "__main__":
    result = train_v2()
    print(f"\nTraining complete. Best intent_accuracy={result['best_intent_accuracy']:.4f}")
