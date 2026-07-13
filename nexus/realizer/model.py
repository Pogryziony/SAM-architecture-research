"""Lazy PyTorch construction for the preregistered Realizer v1."""

from __future__ import annotations

import math
from typing import Any


def validate_model_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("vocab_size", "d_model", "nhead", "encoder_layers", "decoder_layers", "dim_feedforward", "max_input_tokens", "max_output_tokens"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"{key} must be a positive integer")
    if not errors and config["d_model"] % config["nhead"] != 0:
        errors.append("d_model must be divisible by nhead")
    dropout = config.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0.0 <= dropout < 1.0:
        errors.append("dropout must be within [0, 1)")
    return errors


def build_model(config: dict[str, Any]):
    errors = validate_model_config(config)
    if errors:
        raise ValueError("invalid model config: " + "; ".join(errors))
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install the project 'train' extra") from exc

    class NEXUSRealizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vocab_size = config["vocab_size"]
            d_model = config["d_model"]
            max_positions = max(config["max_input_tokens"], config["max_output_tokens"])
            self.scale = math.sqrt(d_model)
            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.position_embedding = nn.Embedding(max_positions, d_model)
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=config["nhead"],
                num_encoder_layers=config["encoder_layers"],
                num_decoder_layers=config["decoder_layers"],
                dim_feedforward=config["dim_feedforward"],
                dropout=float(config["dropout"]),
                batch_first=True,
                norm_first=True,
            )
            self.output = nn.Linear(d_model, vocab_size, bias=False)
            self.output.weight = self.token_embedding.weight

        def _embed(self, tokens):
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            return self.token_embedding(tokens) * self.scale + self.position_embedding(positions)

        def forward(self, source, target):
            target_length = target.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                target_length, device=target.device
            )
            hidden = self.transformer(
                self._embed(source),
                self._embed(target),
                tgt_mask=causal_mask,
                src_key_padding_mask=source.eq(0),
                tgt_key_padding_mask=target.eq(0),
                memory_key_padding_mask=source.eq(0),
            )
            return self.output(hidden)

    return NEXUSRealizer()


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["build_model", "parameter_count", "validate_model_config"]
