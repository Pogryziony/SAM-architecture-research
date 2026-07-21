"""Non-autoregressive copy/edit transducer for AnswerPlan surface realization.

Unlike the autoregressive pointer-generator, this model predicts every
output token simultaneously from the encoder hidden states.  Each canonical-
answer position receives exactly one prediction (keep, replace or delete).

PyTorch is imported lazily so the default non-neural installation keeps
working.
"""

from __future__ import annotations

import math
import re
from typing import Any

from nexus.realizer.edit_script import DELETE_ID, KEEP_ID, OUTPUT_OFFSET


_PIECE_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
CONFIG_SCHEMA = "nexus-copy-edit-transducer-config-v1"


def build_copy_edit_transducer(config: dict[str, Any]):
    """Build a non-autoregressive copy/edit transducer from *config*.

    Required config keys:

    * ``vocab_size`` — total source vocabulary size.
    * ``hidden_size`` — embedding and GRU hidden dimension.
    * ``output_vocab_size`` — number of output token labels (including
      ``[DELETE]``, ``[KEEP]`` and the special-token offset).
    * ``dropout`` — optional dropout rate (default 0.1).
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the copy/edit transducer"
        ) from exc

    class CopyEditTransducer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vocab = int(config["vocab_size"])
            hidden = int(config["hidden_size"])
            output_vocab = int(config["output_vocab_size"])
            self.vocab_size = vocab
            self.hidden_size = hidden
            self.output_vocab_size = output_vocab

            self.embedding = nn.Embedding(vocab, hidden, padding_idx=0)
            self.encoder = nn.GRU(
                hidden, hidden // 2, batch_first=True, bidirectional=True,
            )
            self.output_projection = nn.Linear(hidden, output_vocab)
            self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))

        def forward(
            self,
            source_ids: "torch.Tensor",
            fact_positions: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """Return output logits for every source position.

            *source_ids*: ``(batch, seq_len)`` tensor of input token IDs.
            *fact_positions*: ``(batch, num_fact_positions)`` tensor of
            source positions belonging to the canonical answer span.  When
            provided the returned logits are restricted to those positions.
            When ``None`` logits are returned for all positions.

            Returns ``(batch, positions, output_vocab_size)``.
            """
            embedded = self.dropout(self.embedding(source_ids))
            encoded, _ = self.encoder(embedded)
            logits = self.output_projection(encoded)  # (B, T, V_out)
            if fact_positions is not None:
                batch_indices = torch.arange(
                    source_ids.shape[0], device=source_ids.device,
                ).unsqueeze(1).expand_as(fact_positions)
                logits = logits[batch_indices, fact_positions]
            return logits

        def predict(
            self,
            source_ids: "torch.Tensor",
            fact_positions: "torch.Tensor",
            tokenizer: Any,
        ) -> list[list[str]]:
            """Non-autoregressive inference.

            Returns one list of output token strings per batch element.
            Each list has the same length as the canonical answer.
            """
            import torch
            self.eval()
            with torch.no_grad():
                logits = self.forward(source_ids, fact_positions)
                predictions = logits.argmax(dim=-1)  # (B, F)
            results: list[list[str]] = []
            for batch_idx in range(predictions.shape[0]):
                tokens: list[str] = []
                for pos in range(predictions.shape[1]):
                    token_id = int(predictions[batch_idx, pos])
                    if token_id == DELETE_ID:
                        tokens.append("[DELETE]")
                    elif token_id < OUTPUT_OFFSET:
                        # KEEP or special — treat as KEEP (identity).
                        src_id = int(
                            source_ids[batch_idx, int(fact_positions[batch_idx, pos])]
                        )
                        tokens.append(_safe_decode_token(tokenizer, src_id))
                    else:
                        # Untrained heads may emit IDs outside the tokenizer
                        # vocabulary; fall back to KEEP rather than crashing.
                        tokens.append(
                            _safe_decode_token(
                                tokenizer,
                                token_id,
                                fallback_id=int(
                                    source_ids[
                                        batch_idx,
                                        int(fact_positions[batch_idx, pos]),
                                    ]
                                ),
                            )
                        )
                results.append(tokens)
            return results

    return CopyEditTransducer()


def tokenize_for_transducer(text: str) -> list[str]:
    """Split *text* into tokens matching the transducer's granularity."""
    return _PIECE_RE.findall(text)


def find_fact_positions(
    source_ids: list[int],
    tokenizer: Any,
    fact_text: str,
) -> list[int]:
    """Return the source positions belonging to the canonical fact span.

    Searches for the exact token-ID sequence of *fact_text* within
    *source_ids* and returns the index of each matching position.
    Raises ``ValueError`` when the fact is not found.
    """
    fact_ids = tokenizer.encode(fact_text, add_special_tokens=False)
    if not fact_ids:
        return []
    for start in range(len(source_ids) - len(fact_ids) + 1):
        if source_ids[start:start + len(fact_ids)] == fact_ids:
            return list(range(start, start + len(fact_ids)))
    raise ValueError(f"fact span not found in source: {fact_text!r}")


def build_label_ids(
    labels: list[str],
    tokenizer: Any,
) -> list[int]:
    """Convert string edit labels to output token IDs.

    ``"[DELETE]"`` → ``DELETE_ID``.
    Any other label → the tokenizer ID for that label string.
    """
    result: list[int] = []
    for label in labels:
        if label == "[DELETE]":
            result.append(DELETE_ID)
        else:
            result.append(_token_id(tokenizer, label))
    return result


def _token_id(tokenizer: Any, token: str) -> int:
    """Encode a single output token to its vocabulary ID.

    Falls back to the OUTPUT_OFFSET byte range for unknown tokens so the
    model can emit characters it has never seen as labelled tokens.
    """
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded and encoded[0] >= OUTPUT_OFFSET:
        return encoded[0]
    # Fallback: use the first byte of the UTF-8 encoding.
    raw = token.encode("utf-8")
    if raw:
        return OUTPUT_OFFSET + raw[0]
    return DELETE_ID


def _safe_decode_token(
    tokenizer: Any,
    token_id: int,
    *,
    fallback_id: int | None = None,
) -> str:
    """Decode *token_id*, falling back to KEEP semantics on unknown IDs."""
    try:
        return str(tokenizer.decode_token(token_id))
    except (ValueError, KeyError, IndexError):
        if fallback_id is None:
            return "[KEEP]"
        try:
            return str(tokenizer.decode_token(fallback_id))
        except (ValueError, KeyError, IndexError):
            return "[KEEP]"


__all__ = [
    "CONFIG_SCHEMA",
    "build_copy_edit_transducer",
    "build_label_ids",
    "find_fact_positions",
    "tokenize_for_transducer",
]
