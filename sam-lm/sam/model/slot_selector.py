"""Learned slot selector for SAM retrieved-memory candidate filtering.

Given topK retrieved candidate slots, predicts which are required for QA.
Used in aggregation_mode "learned_selector".

Architecture:
  - Input features per slot: query embedding, slot embedding, value vector,
    retrieval score, rank position, score margin from top.
  - Output: logit per slot (probability that slot is required).
  - Loss: BCEWithLogitsLoss with positive class weighting.
  - Aggregation: select slots with p >= threshold or topN by selector score.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotSelector(nn.Module):
    """Learned binary classifier for retrieved candidate slots.

    Predicts whether each candidate slot is required for QA given the
    query and slot features from the retriever.
    """

    def __init__(
        self,
        query_dim: int = 256,
        value_dim: int = 128,
        hidden_dim: int = 256,
        num_score_features: int = 4,  # raw score, rank, margin, normalized score
        use_hop_count: bool = True,
        num_hops: int = 4,  # 0 (unknown), 1, 2, 3
        hop_emb_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.use_hop_count = use_hop_count

        total_in = query_dim + query_dim + value_dim + num_score_features  # q + s + val + score features
        if use_hop_count:
            self.hop_emb = nn.Embedding(num_hops, hop_emb_dim)
            total_in += hop_emb_dim

        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.use_hop_count:
            nn.init.normal_(self.hop_emb.weight, std=0.02)

    def _score_features(
        self,
        scores: torch.Tensor,  # [B, K]
        top_scores: torch.Tensor,  # [B, 1]
        rank_positions: torch.Tensor,  # [B, K] — 0-indexed ranks
    ) -> torch.Tensor:
        """Compute per-slot score-based features.

        Returns [B, K, 4]: raw score, rank, score margin from top, normalized score.
        """
        B, K = scores.shape
        # Score margin from top score
        margin = top_scores - scores  # [B, K] — non-negative
        # Normalized score (min-max within batch)
        score_min = scores.min(dim=-1, keepdim=True).values
        score_max = scores.max(dim=-1, keepdim=True).values
        norm_score = (scores - score_min) / (score_max - score_min).clamp(min=1e-8)

        feats = torch.stack([
            scores,
            rank_positions.float(),
            margin,
            norm_score,
        ], dim=-1)
        return feats

    def forward(
        self,
        query_emb: torch.Tensor,  # [B, query_dim] — query embedding from retriever
        slot_embs: torch.Tensor,  # [B, K, query_dim] — slot embeddings from retriever
        slot_vals: torch.Tensor,  # [B, K, value_dim] — slot value vectors
        scores: torch.Tensor,  # [B, K] — retrieval scores
        hops: Optional[torch.Tensor] = None,  # [B] — hop counts (for embedding)
    ) -> torch.Tensor:
        """Compute selector logits for each candidate slot.

        Returns [B, K] — logits (before sigmoid).
        """
        B, K = scores.shape

        # Score features
        top_scores = scores.max(dim=-1, keepdim=True).values
        rank_positions = torch.arange(K, device=scores.device).unsqueeze(0).expand(B, -1)
        score_feats = self._score_features(scores, top_scores, rank_positions)  # [B, K, 4]

        # Query embedding: expand to [B, K, query_dim]
        q = query_emb.unsqueeze(1).expand(B, K, -1)

        # Concatenate all features
        features = [q, slot_embs, slot_vals, score_feats]

        if self.use_hop_count and hops is not None:
            h_emb = self.hop_emb(hops.clamp(0, 3))  # [B, hop_emb_dim]
            h_emb = h_emb.unsqueeze(1).expand(B, K, -1)
            features.append(h_emb)

        x = torch.cat(features, dim=-1)  # [B, K, total_in]

        # Run through MLP
        logits = self.net(x).squeeze(-1)  # [B, K]
        return logits

    @torch.no_grad()
    def select_slots(
        self,
        logits: torch.Tensor,  # [B, K]
        slot_ids: torch.Tensor,  # [B, K]
        threshold: float = 0.5,
        top_n: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select slots based on selector probabilities.

        Returns (selected_mask [B, K], selected_probs [B, K]).
        """
        probs = torch.sigmoid(logits)  # [B, K]
        if top_n is not None:
            # Select top_n by selector probability
            _, top_idx = probs.topk(min(top_n, probs.size(-1)), dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, top_idx, 1.0)
        else:
            mask = (probs >= threshold).float()

        # Ensure at least 1 slot selected per example
        for i in range(mask.size(0)):
            if mask[i].sum() == 0:
                mask[i, 0] = 1.0  # fallback to top-ranked

        return mask, probs

    @torch.no_grad()
    def compute_diagnostics(
        self,
        logits: torch.Tensor,  # [B, K]
        slot_ids: torch.Tensor,  # [B, K]
        required_slots: torch.Tensor,  # [B, max_req]
        selected_mask: torch.Tensor,  # [B, K]
        hops: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Compute selector diagnostic metrics.

        Returns dict with keys:
          selector_precision, selector_recall, selector_f1,
          selected_slot_count_mean, selected_required_count_mean,
          selected_distractor_count_mean, selected_all_required_present_rate,
          selected_required_coverage, selected_distractor_rate.
        """
        B, K = slot_ids.shape
        probs = torch.sigmoid(logits)
        # Build required set per example
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        n_sel_total = 0.0
        n_req_sel_total = 0.0
        n_dist_sel_total = 0.0
        all_req_present_count = 0
        total_required = 0
        total_selected_required = 0
        valid_examples = 0

        for i in range(B):
            req_i = set(int(s) for s in required_slots[i] if int(s) >= 0)
            if not req_i:
                continue
            valid_examples += 1
            sel_idx = selected_mask[i].nonzero(as_tuple=False).flatten()
            sel_slots = set(int(slot_ids[i, j].item()) for j in sel_idx)

            tp = len(sel_slots & req_i)
            fp = len(sel_slots - req_i)
            fn = len(req_i - sel_slots)

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            n_sel_total += len(sel_slots)
            n_req_sel_total += tp
            n_dist_sel_total += fp
            total_required += len(req_i)
            total_selected_required += tp

            if req_i.issubset(sel_slots):
                all_req_present_count += 1

        n = max(valid_examples, 1)
        return {
            "selector_precision": total_precision / n,
            "selector_recall": total_recall / n,
            "selector_f1": total_f1 / n,
            "selected_slot_count_mean": n_sel_total / n,
            "selected_required_count_mean": n_req_sel_total / n,
            "selected_distractor_count_mean": n_dist_sel_total / n,
            "selected_all_required_present_rate": all_req_present_count / n,
            "selected_required_coverage": total_selected_required / max(total_required, 1),
            "selected_distractor_rate": n_dist_sel_total / max(n_sel_total, 1),
        }
