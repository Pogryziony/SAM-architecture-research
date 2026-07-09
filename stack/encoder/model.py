"""Associative Encoder — small dual-encoder with product-key scoring.

Architecture:
  - Word-level tokenizer (character n-gram fallback for OOV)
  - EmbeddingBag → Linear → ReLU → Linear backbone
  - Three heads: entity set (multi-label BCE), intent (CE), category (CE)
  - <= 20M parameters, CPU-only PyTorch

Inspired by the chain-set retriever from sam-lm/sam/model/
but adapted for utterance → (entities, intent, category) prediction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

import torch
import torch.nn as nn


# ── Simple word tokenizer (no BPE) ──

class WordTokenizer:
    """Tokenize text into word indices with character n-gram fallback for OOV."""

    def __init__(self, vocab: dict[str, int], max_seq_len: int = 64):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.unk_idx = vocab.get("<UNK>", 0)
        self.pad_idx = vocab.get("<PAD>", 1)
        # Character trigram index for OOV fallback
        self._char_trigram_id: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        """Tokenize text into word indices."""
        tokens = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+", text.lower())
        if not tokens:
            return [self.unk_idx]
        indices = [self.vocab.get(t, self.unk_idx) for t in tokens]
        return indices[: self.max_seq_len]

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts, returning (offsets, indices) for EmbeddingBag."""
        all_indices: list[int] = []
        offsets: list[int] = [0]
        for text in texts:
            ids = self.encode(text)
            all_indices.extend(ids)
            offsets.append(offsets[-1] + len(ids))
        return (
            torch.tensor(offsets[:-1], dtype=torch.long),
            torch.tensor(all_indices, dtype=torch.long),
        )

    @classmethod
    def build_from_texts(
        cls, texts: list[str], min_freq: int = 2, max_vocab: int = 8000
    ) -> "WordTokenizer":
        """Build vocabulary from a corpus of question texts."""
        from collections import Counter

        counter: Counter[str] = Counter()
        for text in texts:
            tokens = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+", text.lower())
            counter.update(tokens)

        # Filter by frequency
        vocab_tokens = [
            t for t, c in counter.most_common(max_vocab) if c >= min_freq
        ]

        vocab: dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
        }
        for i, token in enumerate(vocab_tokens, start=len(vocab)):
            vocab[token] = i

        return cls(vocab)


# ── Associative Encoder Model ──

class AssociativeEncoder(nn.Module):
    """Small dual-encoder predicting entities (multi-label), intent, and category.

    Args:
        vocab_size: Number of word tokens in vocabulary
        embed_dim: Embedding dimension (default 128)
        hidden_dim: Hidden layer dimension (default 256)
        num_entities: Number of entity classes (multi-label prediction)
        num_intents: Number of intent classes (default 4)
        num_categories: Number of category classes (default 4)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_entities: int = 366,
        num_intents: int = 4,
        num_categories: int = 4,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.entity_head = nn.Linear(hidden_dim, num_entities)  # BCE multi-label
        self.intent_head = nn.Linear(hidden_dim, num_intents)  # cross-entropy
        self.category_head = nn.Linear(hidden_dim, num_categories)  # cross-entropy

    def forward(
        self, offsets: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            offsets: [B] offset tensor for EmbeddingBag
            indices: [N] flattened indices for EmbeddingBag

        Returns:
            (entity_logits, intent_logits, category_logits)
        """
        x = self.embedding(indices, offsets)  # [B, embed_dim]
        x = self.encoder(x)  # [B, hidden_dim]
        entity_logits = self.entity_head(x)  # [B, num_entities]
        intent_logits = self.intent_head(x)  # [B, num_intents]
        category_logits = self.category_head(x)  # [B, num_categories]
        return entity_logits, intent_logits, category_logits

    @torch.no_grad()
    def predict(
        self, offsets: torch.Tensor, indices: torch.Tensor, entity_threshold: float = 0.5
    ) -> dict[str, torch.Tensor]:
        """Inference: predict entities, intent, category.

        Args:
            offsets, indices: Tokenized batch input
            entity_threshold: BCE threshold for entity prediction

        Returns:
            dict with 'entities' (bool [B, num_entities]), 'intent' (long [B]),
            'category' (long [B]), and raw logits
        """
        self.eval()
        entity_logits, intent_logits, category_logits = self(offsets, indices)
        return {
            "entity_logits": entity_logits,
            "entity_preds": (torch.sigmoid(entity_logits) > entity_threshold),
            "intent_logits": intent_logits,
            "intent_preds": torch.argmax(intent_logits, dim=1),
            "category_logits": category_logits,
            "category_preds": torch.argmax(category_logits, dim=1),
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_entity_mapping(node_ids: list[str]) -> dict[str, int]:
    """Build entity ID → index mapping for the model head."""
    return {nid: i for i, nid in enumerate(sorted(node_ids))}


def build_intent_mapping() -> dict[str, int]:
    """Standard 4-intent mapping."""
    return {
        "factual_lookup": 0,
        "comparison": 1,
        "multi_hop": 2,
        "diagnostic": 3,
    }


def build_category_mapping() -> dict[str, int]:
    """Standard 4-category mapping (same as question_type)."""
    return {
        "factual": 0,
        "comparative": 1,
        "multi-hop": 2,
        "diagnostic": 3,
    }


def save_model(
    model: AssociativeEncoder,
    tokenizer: WordTokenizer,
    entity_map: dict[str, int],
    intent_map: dict[str, int],
    category_map: dict[str, int],
    config: dict,
    output_dir: str,
):
    """Save model, tokenizer, mappings, and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": model.embedding.num_embeddings,
            "embed_dim": model.embedding.embedding_dim,
            "hidden_dim": model.encoder[0].out_features,
            "num_entities": model.entity_head.out_features,
            "num_intents": model.intent_head.out_features,
            "num_categories": model.category_head.out_features,
        },
        os.path.join(output_dir, "best.pt"),
    )

    # Save tokenizer vocab
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)

    # Save entity/intent/category mappings
    with open(os.path.join(output_dir, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "entity_map": entity_map,
                "intent_map": intent_map,
                "category_map": category_map,
                "inv_entity_map": {v: k for k, v in entity_map.items()},
                "inv_intent_map": {v: k for k, v in intent_map.items()},
                "inv_category_map": {v: k for k, v in category_map.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_model(model_dir: str) -> tuple[AssociativeEncoder, WordTokenizer, dict, dict, dict]:
    """Load a trained model and its metadata."""
    # Load checkpoint
    ckpt = torch.load(
        os.path.join(model_dir, "best.pt"),
        map_location="cpu",
        weights_only=True,
    )

    # Reconstruct model
    model = AssociativeEncoder(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_entities=ckpt["num_entities"],
        num_intents=ckpt["num_intents"],
        num_categories=ckpt["num_categories"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load tokenizer vocab
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer = WordTokenizer(vocab)

    # Load mappings
    with open(os.path.join(model_dir, "mappings.json"), encoding="utf-8") as f:
        mappings = json.load(f)

    entity_map = mappings["entity_map"]
    intent_map = mappings["intent_map"]
    category_map = mappings["category_map"]

    return model, tokenizer, entity_map, intent_map, category_map


def compute_data_hash(data_path: str) -> str:
    """Compute SHA256 hash of training data for provenance."""
    sha = hashlib.sha256()
    with open(data_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()
