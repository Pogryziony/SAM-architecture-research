"""Deterministic train-only frequency subword tokenizer with byte fallback.

Frequent lexical/whitespace pieces are learned from the training split only.
Every other string is encoded as UTF-8 bytes, so encoding is lossless and has
no unknown-token path.  This replaces the historical fixed 256-byte window;
it does not claim to implement BPE or SentencePiece.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Iterable


TOKENIZER_VERSION = "nexus-frequency-subword-v1"
_PIECE_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)


class TrainOnlySubwordTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    BYTE_OFFSET = 3
    BYTE_VOCAB = 256

    def __init__(self, pieces: list[str] | None = None):
        self.pieces = list(pieces or [])
        self._piece_to_id = {
            piece: self.BYTE_OFFSET + self.BYTE_VOCAB + index
            for index, piece in enumerate(self.pieces)
        }

    @property
    def vocab_size(self) -> int:
        return self.BYTE_OFFSET + self.BYTE_VOCAB + len(self.pieces)

    @classmethod
    def train(cls, texts: Iterable[str], max_pieces: int = 4096) -> "TrainOnlySubwordTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(_PIECE_RE.findall(text))
        ranked = sorted(
            (piece for piece, count in counts.items() if count >= 2 and len(piece) > 1),
            key=lambda piece: (-counts[piece], piece),
        )
        return cls(ranked[:max_pieces])

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = [self.BOS] if add_special_tokens else []
        for piece in _PIECE_RE.findall(text):
            known = self._piece_to_id.get(piece)
            if known is not None:
                ids.append(known)
            else:
                ids.extend(self.BYTE_OFFSET + byte for byte in piece.encode("utf-8"))
        if add_special_tokens:
            ids.append(self.EOS)
        return ids

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its string representation."""
        if token_id == self.PAD:
            return ""
        if token_id == self.BOS:
            return ""
        if token_id == self.EOS:
            return ""
        if self.BYTE_OFFSET <= token_id < self.BYTE_OFFSET + self.BYTE_VOCAB:
            return bytes([token_id - self.BYTE_OFFSET]).decode("utf-8", errors="replace")
        index = token_id - self.BYTE_OFFSET - self.BYTE_VOCAB
        if 0 <= index < len(self.pieces):
            return self.pieces[index]
        raise ValueError(f"unknown token id: {token_id}")

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        output: list[str] = []
        byte_buffer = bytearray()

        def flush() -> None:
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8", errors="strict"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id in {self.PAD, self.BOS, self.EOS}:
                if skip_special_tokens:
                    continue
                flush()
                continue
            if self.BYTE_OFFSET <= token_id < self.BYTE_OFFSET + self.BYTE_VOCAB:
                byte_buffer.append(token_id - self.BYTE_OFFSET)
                continue
            flush()
            index = token_id - self.BYTE_OFFSET - self.BYTE_VOCAB
            if not 0 <= index < len(self.pieces):
                raise ValueError(f"unknown token id: {token_id}")
            output.append(self.pieces[index])
        flush()
        return "".join(output)

    def to_dict(self) -> dict:
        body = {
            "schema_version": TOKENIZER_VERSION,
            "training_scope": "train_only",
            "algorithm": "frequency_lexical_pieces_with_utf8_byte_fallback",
            "special_tokens": {"pad": self.PAD, "bos": self.BOS, "eos": self.EOS},
            "pieces": self.pieces,
            "vocab_size": self.vocab_size,
            "unknown_token": None,
        }
        body["canonical_sha256"] = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return body

    @classmethod
    def from_dict(cls, value: dict) -> "TrainOnlySubwordTokenizer":
        if value.get("schema_version") != TOKENIZER_VERSION:
            raise ValueError("unsupported tokenizer schema")
        tokenizer = cls(list(value.get("pieces", [])))
        if tokenizer.to_dict() != value:
            raise ValueError("tokenizer artifact hash or metadata mismatch")
        return tokenizer
