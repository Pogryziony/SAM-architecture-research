"""Associative Encoder v2 — Stage 1b.

Architecture changes from Stage 1:
  1. Char n-gram tokenizer (fastText-style tri/penta-gram hashing) for OOV robustness
  2. Pseudo-sequential component (1-layer bidirectional GRU on reshaped embedding)
  3. Entity re-ranker head (scores candidates instead of open-set classification)
  4. Rule-based intent classifier as first pass, model handles remainder

Inference: rule-based intent → encoder for remaining heads.
Training: focal loss on entity scoring, focal loss on intent (class-weighted), CE on category.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from stack.encoder.char_tokenizer import CharNgramTokenizer
from stack.encoder.intent_rules import get_rule_classifier


# ── BackCompat: WordTokenizer kept for v1 model loading ──

class WordTokenizer:
    """Tokenize text into word indices with character n-gram fallback for OOV.

    Kept for backward compatibility with Stage 1 models.
    """

    def __init__(self, vocab: dict[str, int], max_seq_len: int = 64):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.unk_idx = vocab.get("<UNK>", 0)
        self.pad_idx = vocab.get("<PAD>", 1)

    def encode(self, text: str) -> list[int]:
        tokens = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+", text.lower())
        if not tokens:
            return [self.unk_idx]
        indices = [self.vocab.get(t, self.unk_idx) for t in tokens]
        return indices[: self.max_seq_len]

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
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
        from collections import Counter

        counter: Counter[str] = Counter()
        for text in texts:
            tokens = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+", text.lower())
            counter.update(tokens)
        vocab_tokens = [t for t, c in counter.most_common(max_vocab) if c >= min_freq]
        vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for i, token in enumerate(vocab_tokens, start=len(vocab)):
            vocab[token] = i
        return cls(vocab)


# ── Associative Encoder v2: char n-grams + sequential + re-ranker ──

class AssociativeEncoderV2(nn.Module):
    """Stage 1b encoder — char n-grams + sequential + re-ranker head.

    Args:
        feature_dim: Total feature space size (word vocab + tri-buckets + penta-buckets)
        embed_dim: Embedding dimension (default 128)
        hidden_dim: Hidden dimension for sequential component
        num_intents: Number of intent classes (default 4)
        num_categories: Number of category classes (default 4)
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_intents: int = 4,
        num_categories: int = 4,
    ):
        super().__init__()
        # EmbeddingBag for efficiency (handles variable-length feature lists)
        self.embedding = nn.EmbeddingBag(feature_dim, embed_dim, mode="mean")
        # Sequential component: 1-layer bidirectional GRU
        # Reshape embed_dim into 4 chunks of (embed_dim // 4) each
        self._seq_chunk_size = embed_dim // 4
        self._seq_chunks = 4
        gru_input = self._seq_chunk_size  # 32
        gru_hidden = hidden_dim // 4  # 64
        self.gru = nn.GRU(
            gru_input, gru_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        gru_output = gru_hidden * 2  # 128 (bidirectional)
        # Combined representation
        combined_dim = embed_dim + gru_output  # 128 + 128 = 256
        # Intent head (classifier)
        self.intent_head = nn.Linear(combined_dim, num_intents)
        # Category head
        self.category_head = nn.Linear(combined_dim, num_categories)
        # Entity re-ranker: score each candidate entity independently
        self.entity_scorer = nn.Linear(combined_dim + embed_dim, 1)

    def _encode_question(
        self, feature_ids: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Encode a question into a combined representation (without entity head).

        Returns:
            combined: [B, combined_dim] tensor
        """
        emb = self.embedding(feature_ids, offsets)  # [B, embed_dim]
        # Reshape to pseudo-sequence: [B, embed_dim] → [B, seq_chunks, chunk_size]
        seq = emb.view(-1, self._seq_chunks, self._seq_chunk_size)
        gru_out, _ = self.gru(seq)  # [B, seq_chunks, gru_output]
        gru_pooled = gru_out.mean(dim=1)  # [B, gru_output]
        combined = torch.cat([emb, gru_pooled], dim=1)  # [B, combined_dim]
        return combined

    def forward(
        self,
        feature_ids: torch.Tensor,
        offsets: torch.Tensor,
        candidate_entity_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            feature_ids: [N] flattened feature IDs for EmbeddingBag
            offsets: [B] offset tensor
            candidate_entity_feats: [B, K, embed_dim] pre-embedded candidates

        Returns:
            (intent_logits, category_logits, entity_scores)
        """
        combined = self._encode_question(feature_ids, offsets)  # [B, combined_dim]
        intent_logits = self.intent_head(combined)
        cat_logits = self.category_head(combined)
        entity_scores: torch.Tensor | None = None
        if candidate_entity_feats is not None:
            B, K, E = candidate_entity_feats.shape
            # Expand combined: [B, combined_dim] → [B, K, combined_dim]
            combined_expanded = combined.unsqueeze(1).expand(-1, K, -1)
            # Concatenate question + candidate features → [B, K, combined_dim + E]
            pair_feats = torch.cat([combined_expanded, candidate_entity_feats], dim=-1)
            entity_scores = self.entity_scorer(pair_feats).squeeze(-1)  # [B, K]
        return intent_logits, cat_logits, entity_scores

    @torch.no_grad()
    def predict(
        self,
        feature_ids: torch.Tensor,
        offsets: torch.Tensor,
        candidate_entity_feats: torch.Tensor | None = None,
        entity_score_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Inference: predict intent, category, entity scores.

        Returns:
            dict with 'intent_logits', 'intent_preds', 'category_logits',
            'category_preds', 'entity_scores' (if candidates provided)
        """
        self.eval()
        intent_logits, cat_logits, entity_scores = self(
            feature_ids, offsets, candidate_entity_feats,
        )
        result: dict[str, torch.Tensor] = {
            "intent_logits": intent_logits,
            "intent_preds": torch.argmax(intent_logits, dim=1),
            "category_logits": cat_logits,
            "category_preds": torch.argmax(cat_logits, dim=1),
        }
        if entity_scores is not None:
            result["entity_scores"] = torch.sigmoid(entity_scores)
            result["entity_preds"] = (
                result["entity_scores"] > entity_score_threshold
            )
        return result

    def embed_entities(
        self, entity_descriptions: list[str], tokenizer: CharNgramTokenizer
    ) -> torch.Tensor:
        """Embed entity descriptions using the same EmbeddingBag.

        Args:
            entity_descriptions: List of entity description strings
            tokenizer: CharNgramTokenizer instance

        Returns:
            [1, K, embed_dim] tensor of entity embeddings
        """
        if not entity_descriptions:
            return torch.empty(1, 0, self.embedding.embedding_dim)
        offsets_list, indices_list = tokenizer.tokenize_batch(entity_descriptions)
        offsets_t = torch.tensor(offsets_list, dtype=torch.long)
        indices_t = torch.tensor(indices_list, dtype=torch.long)
        with torch.no_grad():
            emb = self.embedding(indices_t, offsets_t)  # [K, embed_dim]
        return emb.unsqueeze(0)  # [1, K, embed_dim]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── BackCompat: AssociativeEncoder alias for v1 API ──

class AssociativeEncoder(AssociativeEncoderV2):
    """Backward-compatible alias for Stage 1 API consumers.

    Forward method signature matches v1: (offsets, indices) → (entity, intent, cat).
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
        super().__init__(
            feature_dim=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_intents=num_intents,
            num_categories=num_categories,
        )
        # Legacy entity head for backward compat (unused in v2 re-ranker mode)
        self._legacy_entity_head = nn.Linear(embed_dim + hidden_dim // 4 * 2, num_entities)

    def forward(
        self, offsets: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy forward: returns (entity_logits, intent_logits, category_logits)."""
        combined = self._encode_question(indices, offsets)
        entity_logits = self._legacy_entity_head(combined)
        intent_logits = self.intent_head(combined)
        cat_logits = self.category_head(combined)
        return entity_logits, intent_logits, cat_logits

    @torch.no_grad()
    def predict(
        self, offsets: torch.Tensor, indices: torch.Tensor, entity_threshold: float = 0.5
    ) -> dict[str, torch.Tensor]:
        """Legacy predict for backward compatibility with Stage 1 eval."""
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


# ── Mappings and save/load ──

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
    """Standard 4-category mapping."""
    return {
        "factual": 0,
        "comparative": 1,
        "multi-hop": 2,
        "diagnostic": 3,
    }


def save_model_v2(
    model: AssociativeEncoderV2,
    tokenizer: CharNgramTokenizer,
    entity_map: dict[str, int],
    intent_map: dict[str, int],
    category_map: dict[str, int],
    config: dict,
    output_dir: str,
):
    """Save model, tokenizer, mappings, and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": model.embedding.num_embeddings,
            "embed_dim": model.embedding.embedding_dim,
            "hidden_dim": model.gru.hidden_size * 4,
            "num_intents": model.intent_head.out_features,
            "num_categories": model.category_head.out_features,
        },
        os.path.join(output_dir, "best.pt"),
    )

    tokenizer.save_vocab(os.path.join(output_dir, "vocab.json"))

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

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_model_v2(
    model_dir: str,
) -> tuple[AssociativeEncoderV2, CharNgramTokenizer, dict, dict, dict]:
    """Load a trained v2 model and its metadata."""
    ckpt = torch.load(
        os.path.join(model_dir, "best.pt"),
        map_location="cpu",
        weights_only=True,
    )

    model = AssociativeEncoderV2(
        feature_dim=ckpt["feature_dim"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_intents=ckpt["num_intents"],
        num_categories=ckpt["num_categories"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = CharNgramTokenizer.load_vocab(os.path.join(model_dir, "vocab.json"))

    with open(os.path.join(model_dir, "mappings.json"), encoding="utf-8") as f:
        mappings = json.load(f)

    return (
        model,
        tokenizer,
        mappings["entity_map"],
        mappings["intent_map"],
        mappings["category_map"],
    )


# Legacy save/load kept for backward compat

def save_model(
    model, tokenizer, entity_map, intent_map, category_map, config, output_dir,
):
    """Legacy save_model — delegates to save_model_v2 for v2 models."""
    if isinstance(model, AssociativeEncoderV2):
        save_model_v2(model, tokenizer, entity_map, intent_map, category_map, config, output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
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
        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)
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
                f, ensure_ascii=False, indent=2,
            )
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def load_model(model_dir: str):
    """Legacy load_model — auto-detects v1 vs v2."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    arch = config.get("architecture", "")

    if "V2" in arch or "v2" in arch:
        return load_model_v2(model_dir)
    # v1 path
    ckpt = torch.load(
        os.path.join(model_dir, "best.pt"),
        map_location="cpu",
        weights_only=True,
    )
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
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer = WordTokenizer(vocab)
    with open(os.path.join(model_dir, "mappings.json"), encoding="utf-8") as f:
        mappings = json.load(f)
    return model, tokenizer, mappings["entity_map"], mappings["intent_map"], mappings["category_map"]


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

