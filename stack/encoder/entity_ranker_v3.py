"""Entity Ranker V3 — genuinely question-conditioned CPU-only encoder/ranker.

Replaces the defective linear-concat scorer from model.py with bilinear or
dot-product interaction that makes entity rankings depend on the question.

Architecture choices (preregistered):
  Option A: Bilinear — score(q, e) = q^T @ W @ e
  Option B: Dot-product projection — score(q, e) = dot(proj_q(q), proj_e(e))

We implement Option B (dot-product projection) as the primary choice because
it is simpler, has fewer parameters, and generalizes better with small data.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from stack.encoder.char_tokenizer import CharNgramTokenizer


SEED = 20260710


class QuestionConditionedEntityRanker(nn.Module):
    """Entity ranker with genuine question-conditioned scoring.

    Architecture:
      1. Shared EmbeddingBag encodes both questions and entities
      2. Question encoder: EmbeddingBag → GRU → combined representation
      3. Entity encoder: EmbeddingBag → pooled representation
      4. Projection layers map question and entity to a common space
      5. Score = cosine similarity or dot product of projected vectors

    Unlike the defective linear-concat scorer (score = W_q·q + W_e·e + b),
    this model's entity ranking genuinely depends on the question because
    q and e interact multiplicatively: score(q, e) = proj_q(q) · proj_e(e).
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        dropout: float = 0.2,
        use_bilinear: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj_dim = proj_dim
        self.use_bilinear = use_bilinear

        # Shared embedding
        self.embedding = nn.EmbeddingBag(feature_dim, embed_dim, mode="mean")

        # Pseudo-sequential GRU (same as V2 encoder)
        self._seq_chunk_size = embed_dim // 4
        self._seq_chunks = 4
        gru_input = self._seq_chunk_size
        gru_hidden = hidden_dim // 4
        self.gru = nn.GRU(
            gru_input, gru_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        gru_output = gru_hidden * 2
        combined_dim = embed_dim + gru_output

        # Question projection
        self.q_proj = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, proj_dim),
        )

        self.e_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, proj_dim),
        )

        if use_bilinear:
            # Bilinear: q @ W @ e^T — W is [proj_dim, proj_dim]
            self.bilinear = nn.Bilinear(proj_dim, proj_dim, 1)

    def encode_question(
        self, feature_ids: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Encode question text into a combined representation.

        Returns:
            combined: [B, combined_dim]
        """
        emb = self.embedding(feature_ids, offsets)  # [B, embed_dim]
        seq = emb.view(-1, self._seq_chunks, self._seq_chunk_size)
        gru_out, _ = self.gru(seq)  # [B, seq_chunks, gru_output]
        gru_pooled = gru_out.mean(dim=1)  # [B, gru_output]
        combined = torch.cat([emb, gru_pooled], dim=1)  # [B, combined_dim]
        return combined

    def project_question(self, combined: torch.Tensor) -> torch.Tensor:
        """Project question encoding into common space.

        Returns:
            q_proj: [B, proj_dim]
        """
        return F.normalize(self.q_proj(combined), p=2, dim=-1)

    def encode_entities(
        self, entity_descriptions: list[str], tokenizer: CharNgramTokenizer,
    ) -> torch.Tensor:
        """Encode entity descriptions. Returns [K, embed_dim]."""
        if not entity_descriptions:
            return torch.empty(
                0, self.embedding.embedding_dim,
                device=next(self.parameters()).device,
            )
        offsets_list, indices_list = tokenizer.tokenize_batch(entity_descriptions)
        offsets_t = torch.tensor(offsets_list[:-1], dtype=torch.long)
        indices_t = torch.tensor(indices_list, dtype=torch.long)
        emb = self.embedding(indices_t, offsets_t)  # [K, embed_dim]
        return emb

    def project_entities(self, entity_emb: torch.Tensor) -> torch.Tensor:
        """Project entity embeddings into common space.

        Returns:
            e_proj: [K, proj_dim]
        """
        return F.normalize(self.e_proj(entity_emb), p=2, dim=-1)

    def score(self, q_proj: torch.Tensor, e_proj: torch.Tensor) -> torch.Tensor:
        """Compute question-conditioned entity scores.

        Args:
            q_proj: [B, proj_dim] projected question vectors
            e_proj: [B, K, proj_dim] projected entity vectors

        Returns:
            scores: [B, K] — one score per candidate entity per question
        """
        if self.use_bilinear:
            # Bilinear: for each candidate, compute q @ W @ e
            B, K, D = e_proj.shape
            q_expanded = q_proj.unsqueeze(1).expand(-1, K, -1).reshape(B * K, D)
            e_flat = e_proj.reshape(B * K, D)
            scores = self.bilinear(q_expanded, e_flat).view(B, K)
        else:
            # Dot product: q_proj · e_proj for each candidate
            # q_proj: [B, proj_dim], e_proj: [B, K, proj_dim]
            scores = (q_proj.unsqueeze(1) * e_proj).sum(dim=-1)  # [B, K]

        return scores

    def forward(
        self,
        q_feature_ids: torch.Tensor,
        q_offsets: torch.Tensor,
        entity_descriptions: list[str],
        tokenizer: CharNgramTokenizer,
    ) -> torch.Tensor:
        """Full forward pass: encode question, encode entities, score.

        Args:
            q_feature_ids: Flattened question feature IDs
            q_offsets: Question offsets
            entity_descriptions: List of entity text strings
            tokenizer: CharNgramTokenizer

        Returns:
            scores: [1, K] entity relevance scores
        """
        combined = self.encode_question(q_feature_ids, q_offsets)
        q_proj = self.project_question(combined)  # [1, proj_dim]
        e_emb = self.encode_entities(entity_descriptions, tokenizer)  # [K, embed_dim]
        e_proj = self.project_entities(e_emb)  # [K, proj_dim]
        e_proj = e_proj.unsqueeze(0)  # [1, K, proj_dim]
        scores = self.score(q_proj, e_proj)  # [1, K]
        return scores

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed_and_project_entities(
        self, entity_descriptions: list[str], tokenizer: CharNgramTokenizer,
    ) -> torch.Tensor:
        """Encode + project entities for use as pre-computed candidates.

        Returns [1, K, proj_dim] — batch dimension for compatibility.
        """
        e_emb = self.encode_entities(entity_descriptions, tokenizer)
        e_proj = self.project_entities(e_emb)
        return e_proj.unsqueeze(0)


def save_ranker_v3(
    model: QuestionConditionedEntityRanker,
    tokenizer: CharNgramTokenizer,
    config: dict,
    output_dir: str,
) -> None:
    """Save ranker model, tokenizer, and config to an immutable directory.

    The directory must already exist.  Fails if any output file already exists.
    """
    out = os.path.abspath(output_dir)
    if not os.path.isdir(out):
        raise NotADirectoryError(f"Output directory does not exist: {out}")

    def _save(path: str, write_fn):
        full = os.path.join(out, path)
        if os.path.exists(full):
            raise FileExistsError(
                f"Refusing to overwrite existing artifact: {full}"
            )
        write_fn(full)

    _save("weights.pt", lambda p: torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": model.embedding.num_embeddings,
            "embed_dim": model.embedding.embedding_dim,
            "hidden_dim": model.gru.hidden_size * 4,
            "proj_dim": model.proj_dim,
            "use_bilinear": model.use_bilinear,
        },
        p,
    ))

    _save("vocab.json", lambda p: tokenizer.save_vocab(p))

    _save("config.json", lambda p: open(p, "w", encoding="utf-8").write(
        json.dumps(config, ensure_ascii=False, indent=2)
    ))


def load_ranker_v3(
    model_dir: str,
) -> tuple[QuestionConditionedEntityRanker, CharNgramTokenizer, dict]:
    """Load a trained V3 ranker and its metadata."""
    ckpt = torch.load(
        os.path.join(model_dir, "weights.pt"),
        map_location="cpu",
        weights_only=True,
    )

    model = QuestionConditionedEntityRanker(
        feature_dim=ckpt["feature_dim"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        proj_dim=ckpt["proj_dim"],
        use_bilinear=ckpt.get("use_bilinear", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = CharNgramTokenizer.load_vocab(os.path.join(model_dir, "vocab.json"))

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    return model, tokenizer, config


def compute_data_hash(data_path: str) -> str:
    """Compute SHA256 hash of a data file for provenance."""
    sha = hashlib.sha256()
    with open(data_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()
