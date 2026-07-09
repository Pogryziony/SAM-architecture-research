"""Encoder loader — loads trained associative encoder and reports RSS delta.

Provides a lazy-loading interface for the AssociativeEncoder model.
Tracks RSS delta on first load for gate evaluation.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import torch

# Cache the loaded model to avoid reloading
_encoder_cache: dict[str, object] = {}


def get_peak_rss_mb() -> float:
    """Get peak RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return -1.0


class EncoderLoader:
    """Lazy-loader for the AssociativeEncoder model.

    Tracks:
    - parameter count
    - RSS delta on first load (MB)
    - inference time per question
    """

    def __init__(self, model_dir: str = "models/encoder"):
        self.model_dir = model_dir
        self.model: Optional[object] = None
        self.tokenizer: Optional[object] = None
        self.entity_map: dict[str, int] = {}
        self.inv_entity_map: dict[int, str] = {}
        self.intent_map: dict[str, int] = {}
        self.inv_intent_map: dict[int, str] = {}
        self.category_map: dict[str, int] = {}
        self.inv_category_map: dict[int, str] = {}
        self.param_count: int = 0
        self.rss_delta_mb: float = 0.0
        self._loaded: bool = False

    def load(self) -> bool:
        """Load the model. Returns True if successful, False otherwise."""
        if self._loaded:
            return True

        from stack.encoder.model import load_model

        rss_before = get_peak_rss_mb()
        t0 = time.time()

        try:
            model, tokenizer, entity_map, intent_map, category_map = load_model(
                os.path.join(os.path.dirname(__file__), "..", "..", self.model_dir)
            )
        except FileNotFoundError:
            print(f"[EncoderLoader] Model not found at {self.model_dir}")
            return False
        except Exception as e:
            print(f"[EncoderLoader] Failed to load model: {e}")
            return False

        rss_after = get_peak_rss_mb()
        load_time = time.time() - t0

        self.model = model
        self.tokenizer = tokenizer
        self.entity_map = entity_map
        self.inv_entity_map = {v: k for k, v in entity_map.items()}
        self.intent_map = intent_map
        self.inv_intent_map = {v: k for k, v in intent_map.items()}
        self.category_map = category_map
        self.inv_category_map = {v: k for k, v in category_map.items()}
        self.param_count = model.count_parameters()
        self.rss_delta_mb = rss_after - rss_before
        self._loaded = True

        print(
            f"[EncoderLoader] Loaded in {load_time:.2f}s, "
            f"{self.param_count:,} params, "
            f"RSS delta: {self.rss_delta_mb:.1f} MB"
        )
        return True

    def predict(
        self, question: str, entity_threshold: float = 0.5
    ) -> dict:
        """Predict entities, intent, and category for a question.

        Args:
            question: Natural language question text
            entity_threshold: BCE threshold for entity prediction

        Returns:
            dict with:
                'entity_ids': list of predicted entity node IDs
                'entity_scores': list of (entity_id, confidence) tuples
                'intent': predicted intent string
                'intent_confidence': confidence score
                'category': predicted category string
                'category_confidence': confidence score
        """
        if not self._loaded and not self.load():
            return {
                "entity_ids": [],
                "entity_scores": [],
                "intent": "factual_lookup",
                "intent_confidence": 0.0,
                "category": "factual",
                "category_confidence": 0.0,
            }

        offsets, indices = self.tokenizer.encode_batch([question])
        with torch.no_grad():
            result = self.model.predict(offsets, indices, entity_threshold)

        # Entity predictions
        entity_preds = result["entity_preds"][0]  # [num_entities]
        entity_logits = result["entity_logits"][0]
        entity_scores_raw = torch.sigmoid(entity_logits)

        entity_ids: list[str] = []
        entity_scores: list[tuple[str, float]] = []
        for idx in range(len(entity_preds)):
            if entity_preds[idx]:
                eid = self.inv_entity_map.get(idx, f"entity_{idx}")
                score = float(entity_scores_raw[idx])
                entity_ids.append(eid)
                entity_scores.append((eid, score))

        # Sort by confidence
        entity_scores.sort(key=lambda x: x[1], reverse=True)

        # Intent prediction
        intent_idx = result["intent_preds"][0].item()
        intent = self.inv_intent_map.get(intent_idx, "factual_lookup")
        intent_conf = float(
            torch.softmax(result["intent_logits"][0], dim=0)[intent_idx]
        )

        # Category prediction
        cat_idx = result["category_preds"][0].item()
        category = self.inv_category_map.get(cat_idx, "factual")
        cat_conf = float(
            torch.softmax(result["category_logits"][0], dim=0)[cat_idx]
        )

        return {
            "entity_ids": entity_ids,
            "entity_scores": entity_scores,
            "intent": intent,
            "intent_confidence": intent_conf,
            "category": category,
            "category_confidence": cat_conf,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton instance for reuse across calls
_global_loader: Optional[EncoderLoader] = None


def get_encoder(model_dir: str = "models/encoder") -> EncoderLoader:
    """Get or create the global encoder loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = EncoderLoader(model_dir)
    return _global_loader
