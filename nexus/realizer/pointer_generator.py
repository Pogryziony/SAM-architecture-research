"""Compact pointer-generator for grounded AnswerPlan surface realization."""

from __future__ import annotations

import math
from typing import Any


def build_pointer_generator(config: dict[str, Any]):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for pointer-generator pilots") from exc

    class PointerGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vocab = int(config["vocab_size"])
            hidden = int(config["hidden_size"])
            self.vocab_size = vocab
            self.hidden_size = hidden
            self.embedding = nn.Embedding(vocab, hidden, padding_idx=0)
            self.encoder = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
            self.decoder = nn.GRU(hidden, hidden, batch_first=True)
            self.encoder_key = nn.Linear(hidden, hidden, bias=False)
            self.decoder_query = nn.Linear(hidden, hidden, bias=False)
            self.vocab_projection = nn.Linear(hidden * 2, vocab)
            self.copy_gate = nn.Linear(hidden * 3, 1)
            self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))

        def forward(self, source, target, copy_mask=None):
            source_embedding = self.dropout(self.embedding(source))
            target_embedding = self.dropout(self.embedding(target))
            encoded, hidden_bi = self.encoder(source_embedding)
            hidden = torch.cat((hidden_bi[-2], hidden_bi[-1]), dim=-1).unsqueeze(0)
            decoded, _ = self.decoder(target_embedding, hidden)
            scores = torch.bmm(
                self.decoder_query(decoded), self.encoder_key(encoded).transpose(1, 2)
            ) / math.sqrt(self.hidden_size)
            scores = scores.masked_fill(source.eq(0).unsqueeze(1), float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            context = torch.bmm(attention, encoded)
            vocabulary = torch.softmax(
                self.vocab_projection(torch.cat((decoded, context), dim=-1)), dim=-1
            )
            gate = torch.sigmoid(
                self.copy_gate(torch.cat((decoded, context, target_embedding), dim=-1))
            )
            copy_attention = attention
            if copy_mask is not None:
                copy_attention = attention * copy_mask.unsqueeze(1).to(attention.dtype)
                copy_attention = copy_attention / copy_attention.sum(-1, keepdim=True).clamp_min(1e-9)
            copy = torch.zeros(
                source.shape[0], target.shape[1], self.vocab_size,
                dtype=vocabulary.dtype, device=source.device,
            )
            copy.scatter_add_(
                2, source.unsqueeze(1).expand(-1, target.shape[1], -1), copy_attention
            )
            probabilities = (1.0 - gate) * vocabulary + gate * copy
            return torch.log(probabilities.clamp_min(1e-9))

    return PointerGenerator()


__all__ = ["build_pointer_generator"]
