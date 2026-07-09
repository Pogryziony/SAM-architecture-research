"""Character n-gram tokenizer for OOV-robust feature extraction.

Maps words to hashed n-gram features, capturing subword patterns
that help with domain terms like "verifier", "BCE", "PKM", "InfoNCE",
and Polish-inflected forms like "eksperymentu", "dokladnosc".

Inspired by fastText subword hashing.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional


def _stable_hash(s: str, buckets: int) -> int:
    """Deterministic hash of a string into [0, buckets)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % buckets


class CharNgramTokenizer:
    """Character tri-gram and penta-gram hashing for OOV robustness.

    Maps text to feature IDs: word IDs + hashed n-gram IDs.
    Feature space: word_vocab_size + tri_buckets + penta_buckets.

    Args:
        tri_buckets: Number of hash buckets for character tri-grams (default 2000).
        penta_buckets: Number of hash buckets for character penta-grams (default 1000).
    """

    def __init__(
        self,
        tri_buckets: int = 2000,
        penta_buckets: int = 1000,
    ):
        self.tri_buckets = tri_buckets
        self.penta_buckets = penta_buckets
        self._word_vocab: dict[str, int] = {}
        self._next_word_id: int = 0
        self._frozen: bool = False

    @property
    def word_vocab_size(self) -> int:
        return self._next_word_id

    @property
    def feature_dim(self) -> int:
        """Total number of feature IDs: words + tri-grams + penta-grams."""
        return self._next_word_id + self.tri_buckets + self.penta_buckets

    def _tokenize_text(self, text: str) -> list[str]:
        """Split text into word tokens, supporting English and Polish."""
        return re.findall(
            r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+(?:@\d+)?", text.lower()
        )

    def add_words(self, texts: list[str]) -> None:
        """Build vocabulary from a corpus of texts. Call before training."""
        if self._frozen:
            return
        for text in texts:
            for word in self._tokenize_text(text):
                if word not in self._word_vocab:
                    self._word_vocab[word] = self._next_word_id
                    self._next_word_id += 1

    def freeze(self) -> None:
        """Freeze vocabulary; unknown words will only get n-gram features."""
        self._frozen = True

    def tokenize(self, text: str) -> list[int]:
        """Convert text to feature IDs: word IDs + hashed n-gram IDs.

        For frozen tokenizers, unknown words contribute only n-gram features.
        """
        features: list[int] = []
        words = self._tokenize_text(text)

        for word in words:
            # Word-level feature
            if word in self._word_vocab:
                features.append(self._word_vocab[word])
            elif not self._frozen:
                # Add to vocab dynamically (training mode)
                self._word_vocab[word] = self._next_word_id
                features.append(self._next_word_id)
                self._next_word_id += 1
            # Unknown word in frozen mode: skip word-level, rely on n-grams

            # Character tri-gram hashing
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigram = word[i : i + 3]
                    h = _stable_hash(trigram, self.tri_buckets)
                    features.append(self._next_word_id + h)

            # Character penta-gram hashing
            if len(word) >= 5:
                for i in range(len(word) - 4):
                    pentagram = word[i : i + 5]
                    h = _stable_hash(pentagram, self.penta_buckets)
                    features.append(
                        self._next_word_id + self.tri_buckets + h
                    )

        if not features:
            # Empty text fallback: use a zero-length representation
            features = [0]

        return features

    def tokenize_batch(
        self, texts: list[str]
    ) -> tuple[list[int], list[int]]:
        """Tokenize a batch of texts, returning (offsets, indices) for EmbeddingBag.

        Returns:
            offsets: [B] list of start offsets in indices.
            indices: [N] flat list of feature IDs.
        """
        all_indices: list[int] = []
        offsets: list[int] = [0]
        for text in texts:
            ids = self.tokenize(text)
            all_indices.extend(ids)
            offsets.append(offsets[-1] + len(ids))
        return offsets, all_indices

    def save_vocab(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "word_vocab": self._word_vocab,
                    "next_word_id": self._next_word_id,
                    "tri_buckets": self.tri_buckets,
                    "penta_buckets": self.penta_buckets,
                    "frozen": self._frozen,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load_vocab(cls, path: str) -> "CharNgramTokenizer":
        """Load vocabulary from a JSON file."""
        import json
        import os

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(
            tri_buckets=data["tri_buckets"],
            penta_buckets=data["penta_buckets"],
        )
        tok._word_vocab = data["word_vocab"]
        tok._next_word_id = data["next_word_id"]
        tok._frozen = data.get("frozen", True)
        return tok
