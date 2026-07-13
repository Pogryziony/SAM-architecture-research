"""Deterministic UTF-8 byte tokenizer with no learned vocabulary."""

from __future__ import annotations


class ByteTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    BYTE_OFFSET = 3
    vocab_size = 259

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        payload = list(text.encode("utf-8"))[: max_length - 2]
        return [self.BOS, *(value + self.BYTE_OFFSET for value in payload), self.EOS]

    def decode(self, token_ids: list[int]) -> str:
        values = bytes(
            token - self.BYTE_OFFSET
            for token in token_ids
            if self.BYTE_OFFSET <= token < self.vocab_size
        )
        return values.decode("utf-8", errors="replace")
