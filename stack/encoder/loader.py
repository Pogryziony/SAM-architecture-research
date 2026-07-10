"""Encoder loader — loads trained associative encoder and reports RSS delta.

Provides a lazy-loading interface for the AssociativeEncoderV2 model.
Tracks RSS delta on first load for gate evaluation.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from stack.encoder.intent_rules import get_rule_classifier

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


def select_entity_candidates(
    candidate_ids: list[str],
    scores: list[float],
    threshold: float,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Select reranker candidates with explicit, strict threshold semantics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("entity threshold must be within [0, 1]")
    scored = [
        (candidate_ids[index], float(score))
        for index, score in enumerate(scores)
        if index < len(candidate_ids)
    ]
    selected = [(eid, score) for eid, score in scored if score > threshold]
    ranked = sorted(selected, key=lambda item: (-item[1], candidate_ids.index(item[0])))
    # Preserve candidate order for the legacy capped baseline; expose ranked
    # scores separately for diagnostics and top-1 evaluation.
    return [eid for eid, _score in selected], ranked


class EncoderLoader:
    """Lazy-loader for the AssociativeEncoderV2 model.

    Tracks:
    - parameter count
    - RSS delta on first load (MB)
    - inference time per question
    - rule-based intent classifier for first-pass prediction
    """

    def __init__(self, model_dir: str = "models/encoder_v2"):
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
        self._rule_classifier = get_rule_classifier()
        # Property cache for embed_entities
        self._entity_descriptions: list[str] = []
        self._entity_node_ids: list[str] = []

    def load(self) -> bool:
        """Load the model. Returns True if successful, False otherwise."""
        if self._loaded:
            return True

        from stack.encoder.model import load_model_v2

        rss_before = get_peak_rss_mb()
        t0 = time.time()

        try:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", self.model_dir,
            )
            model, tokenizer, entity_map, intent_map, category_map = (
                load_model_v2(model_path)
            )
        except FileNotFoundError:
            # Fall back to v1 path
            try:
                from stack.encoder.model import load_model
                model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "models/encoder",
                )
                model, tokenizer, entity_map, intent_map, category_map = (
                    load_model(model_path)
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

    def set_entity_candidates(
        self, node_ids: list[str], descriptions: list[str],
    ):
        """Pre-set entity descriptions for re-ranker scoring.

        Args:
            node_ids: Graph node IDs to consider as candidates.
            descriptions: Human-readable descriptions for each node.
        """
        self._entity_node_ids = node_ids
        self._entity_descriptions = descriptions

    def predict(
        self,
        question: str,
        entity_threshold: float = 0.5,
        entity_candidates: list[str] | None = None,
        entity_descriptions: list[str] | None = None,
    ) -> dict:
        """Predict entities, intent, and category for a question.

        Uses rule-based intent when possible, encoder for entity re-ranking.

        Args:
            question: Natural language question text
            entity_threshold: Sigmoid threshold for entity scoring
            entity_candidates: Node IDs to score (re-ranker mode)
            entity_descriptions: Descriptions for each candidate node

        Returns:
            dict with entity_ids, entity_scores, intent, category, etc.
        """
        import torch

        # ── Rule-based intent first ──
        rule_intent, rule_conf = self._rule_classifier.classify_with_confidence(
            question,
        )

        if not self._loaded and not self.load():
            return {
                "entity_ids": [],
                "entity_scores": [],
                "intent": rule_intent or "factual_lookup",
                "intent_confidence": rule_conf if rule_intent else 0.0,
                "category": "factual",
                "category_confidence": 0.0,
            }

        # ── Tokenize ──
        from stack.encoder.char_tokenizer import CharNgramTokenizer
        from stack.encoder.model import WordTokenizer

        if isinstance(self.tokenizer, CharNgramTokenizer):
            offsets_list, indices_list = self.tokenizer.tokenize_batch([question])
            offsets = torch.tensor(offsets_list[:-1], dtype=torch.long)
            indices = torch.tensor(indices_list, dtype=torch.long)
        elif isinstance(self.tokenizer, WordTokenizer):
            offsets, indices = self.tokenizer.encode_batch([question])
        else:
            return {
                "entity_ids": [],
                "entity_scores": [],
                "intent": rule_intent or "factual_lookup",
                "intent_confidence": rule_conf if rule_intent else 0.0,
                "category": "factual",
                "category_confidence": 0.0,
            }

        # ── Entity re-ranking ──
        entity_cands = entity_candidates or self._entity_node_ids
        entity_descs = entity_descriptions or self._entity_descriptions

        entity_ids: list[str] = []
        entity_scores: list[tuple[str, float]] = []
        candidate_scores: dict[str, float] = {}

        if entity_cands and entity_descs and len(entity_cands) == len(entity_descs):
            from stack.encoder.model import AssociativeEncoderV2
            if isinstance(self.model, AssociativeEncoderV2):
                cand_feats = self.model.embed_entities(entity_descs, self.tokenizer)
                with torch.no_grad():
                    result = self.model.predict(
                        indices, offsets, cand_feats, entity_threshold,
                    )
                entity_score_tensor = result.get("entity_scores")
                if entity_score_tensor is not None:
                    scores = entity_score_tensor[0].tolist()
                    entity_ids, entity_scores = select_entity_candidates(
                        entity_cands, scores, entity_threshold,
                    )
                    candidate_scores = {
                        entity_cands[i]: float(score)
                        for i, score in enumerate(scores)
                        if i < len(entity_cands)
                    }
                entity_scores.sort(key=lambda x: (-x[1], entity_cands.index(x[0])))
                # Collect also the intent/category from model
                model_intent_idx = result["intent_preds"][0].item()
                model_intent = self.inv_intent_map.get(model_intent_idx, "factual_lookup")
                model_intent_conf = float(
                    torch.softmax(result["intent_logits"][0], dim=0)[model_intent_idx],
                )
                cat_idx = result["category_preds"][0].item()
                category = self.inv_category_map.get(cat_idx, "factual")
                cat_conf = float(
                    torch.softmax(result["category_logits"][0], dim=0)[cat_idx],
                )
                # Intent: rule-first, model fallback
                if rule_intent is not None:
                    intent = rule_intent
                    intent_conf = rule_conf
                else:
                    intent = model_intent
                    intent_conf = model_intent_conf
                return {
                    "entity_ids": entity_ids,
                    "entity_scores": entity_scores,
                    "candidate_scores": candidate_scores,
                    "entity_threshold": entity_threshold,
                    "intent": intent,
                    "intent_confidence": intent_conf,
                    "category": category,
                    "category_confidence": cat_conf,
                }

        # Fallback: no entity candidates — just do intent + category
        with torch.no_grad():
            result = self.model.predict(indices, offsets)

        model_intent_idx = result["intent_preds"][0].item()
        model_intent = self.inv_intent_map.get(model_intent_idx, "factual_lookup")
        model_intent_conf = float(
            torch.softmax(result["intent_logits"][0], dim=0)[model_intent_idx],
        )
        cat_idx = result["category_preds"][0].item()
        category = self.inv_category_map.get(cat_idx, "factual")
        cat_conf = float(
            torch.softmax(result["category_logits"][0], dim=0)[cat_idx],
        )

        if rule_intent is not None:
            intent = rule_intent
            intent_conf = rule_conf
        else:
            intent = model_intent
            intent_conf = model_intent_conf

        return {
            "entity_ids": [],
            "entity_scores": [],
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


def get_encoder(model_dir: str = "models/encoder_v2") -> EncoderLoader:
    """Get or create the global encoder loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = EncoderLoader(model_dir)
    return _global_loader

