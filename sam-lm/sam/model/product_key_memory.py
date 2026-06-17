"""Product-key associative memory (Lample et al. 2019 style).

Addressing (per query):
  1. split query -> q1, q2  (each of dim key_dim)
  2. score q1 against sub-key table K1, q2 against K2     (each [num_subkeys, key_dim])
  3. take top_a from scores1 and top_b from scores2
  4. candidate slots = cartesian product;  slot_id = k1 * num_subkeys + k2
  5. candidate score = score1[k1] + score2[k2]            (additive -> no extra key reads)
  6. final top_k slots; softmax over their scores for weighting
  7. value(slot) = value_emb[ object_token[slot] ]        (see note below)
  8. memory_output = sum_i softmax(score)_i * value_i

total_slots = num_subkeys ** 2.

Value content (v0): each slot stores the *object token* of its fact. The actual
value vector is looked up from a (trainable, small) embedding table owned by the
caller and passed in as ``value_emb_weight`` of shape [vocab, value_dim]. This
keeps the value store at vocab x value_dim instead of total_slots x value_dim and
guarantees the readout is well-posed. Per-slot trainable / int4 values are Phase 5
and intentionally out of scope here.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyMemory(nn.Module):
    def __init__(
        self,
        num_subkeys: int,
        key_dim: int,
        value_dim: int,
        top_a: int,
        top_b: int,
        top_k: int,
        soft_candidates: bool = False,
        use_cosine: bool = False,
        slot_key_alpha: float = 0.0,
    ):
        super().__init__()
        self.num_subkeys = num_subkeys
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.total_slots = num_subkeys * num_subkeys
        self.top_a = min(top_a, num_subkeys)
        self.top_b = min(top_b, num_subkeys)
        self.top_k = min(top_k, self.top_a * self.top_b)
        self.soft_candidates = soft_candidates
        self.use_cosine = use_cosine
        self.slot_key_alpha = slot_key_alpha

        # two sub-key codebooks
        self.K1 = nn.Parameter(torch.randn(num_subkeys, key_dim) * 0.02)
        self.K2 = nn.Parameter(torch.randn(num_subkeys, key_dim) * 0.02)
        # per-slot key embeddings for fine-grained discrimination
        self.slot_keys = nn.Parameter(torch.randn(self.total_slots, key_dim) * 0.02)
        # per-slot object token (value content pointer); -1 == dead slot
        self.register_buffer("slot_value_token",
                             torch.full((self.total_slots,), -1, dtype=torch.long))

        # Compact mode: maps from compact slot index -> original slot ID
        # Built by set_slot_value_tokens when the input has more slots than total_slots
        self.register_buffer("compact_to_original", torch.empty(0, dtype=torch.long))
        self.register_buffer("original_to_compact",
                             torch.full((1,), -1, dtype=torch.long))  # resized lazily

    # -- setup ---------------------------------------------------------------
    def set_slot_value_tokens(self, slot_value_token: torch.Tensor) -> None:
        """Set slot value tokens. If input has more slots than total_slots,
        compact live slots into the available PKM address space."""
        incoming = slot_value_token.to(self.slot_value_token.device)

        if incoming.numel() <= self.total_slots:
            # Standard mode: one-to-one mapping
            self.slot_value_token.copy_(incoming)
            # Build identity compact mapping
            live_mask = incoming >= 0
            compact_ids = live_mask.nonzero(as_tuple=False).flatten()
            self.compact_to_original = compact_ids
            self.original_to_compact = torch.full((incoming.numel(),), -1, dtype=torch.long, device=incoming.device)
            self.original_to_compact[compact_ids] = torch.arange(len(compact_ids), device=incoming.device)
        else:
            # Compact mode: incoming has more slots than PKM can hold.
            # Remap live slots into compact [0, total_slots) range.
            # Only the first total_slots live slots are used; excess is silently dropped.
            live_mask = incoming >= 0
            live_indices = live_mask.nonzero(as_tuple=False).flatten()
            num_compact = min(live_indices.numel(), self.total_slots)

            # Reset all slots to dead
            self.slot_value_token.fill_(-1)
            # Fill compact slots with live values
            for i in range(num_compact):
                orig_id = live_indices[i].item()
                self.slot_value_token[i] = incoming[orig_id].item()

            # Build bidict
            self.compact_to_original = torch.full((self.total_slots,), -1, dtype=torch.long, device=incoming.device)
            self.original_to_compact = torch.full((incoming.numel(),), -1, dtype=torch.long, device=incoming.device)
            for i in range(num_compact):
                orig_id = live_indices[i].item()
                self.compact_to_original[i] = orig_id
                self.original_to_compact[orig_id] = i

    def map_original_to_compact(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Map original slot IDs to compact indices. -1 for dead/out-of-range."""
        if self.original_to_compact.numel() <= slot_ids.max().item():
            return torch.full_like(slot_ids, -1)
        return self.original_to_compact[slot_ids.clamp(min=0)]

    def map_compact_to_original(self, compact_ids: torch.Tensor) -> torch.Tensor:
        """Map compact indices back to original slot IDs."""
        if self.compact_to_original.numel() == 0:
            return compact_ids
        valid = (compact_ids >= 0) & (compact_ids < self.compact_to_original.numel())
        result = torch.full_like(compact_ids, -1)
        result[valid] = self.compact_to_original[compact_ids[valid]]
        return result

    # -- addressing primitives ----------------------------------------------
    def _subscores(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1, q2 = query[..., : self.key_dim], query[..., self.key_dim:]
        if self.use_cosine:
            q1 = F.normalize(q1, dim=-1)
            q2 = F.normalize(q2, dim=-1)
            K1n = F.normalize(self.K1, dim=-1)
            K2n = F.normalize(self.K2, dim=-1)
            s1 = q1 @ K1n.t()
            s2 = q2 @ K2n.t()
        else:
            s1 = q1 @ self.K1.t()
            s2 = q2 @ self.K2.t()
        return s1, s2

    def _candidates(self, s1: torch.Tensor, s2: torch.Tensor,
                    query_full: torch.Tensor = None):
        v1, i1 = s1.topk(self.top_a, dim=-1)         # [N, top_a]
        v2, i2 = s2.topk(self.top_b, dim=-1)         # [N, top_b]
        cand_scores = v1[:, :, None] + v2[:, None, :]            # [N, A, B]
        cand_ids = i1[:, :, None] * self.num_subkeys + i2[:, None, :]
        N = s1.size(0)

        if self.slot_key_alpha > 0 and query_full is not None:
            if self.use_cosine:
                qn = F.normalize(query_full, dim=-1)
                kn = F.normalize(self.slot_keys, dim=-1)
                slot_all = qn @ kn.t()
            else:
                slot_all = query_full @ self.slot_keys.t()
            flat_ids = cand_ids.reshape(N, -1)
            slot_k_scores = torch.gather(slot_all, 1, flat_ids.clamp(0, self.total_slots - 1))
            cand_scores = cand_scores + self.slot_key_alpha * slot_k_scores.reshape_as(cand_scores)

        return cand_scores.reshape(N, -1), cand_ids.reshape(N, -1)

    # -- public API ----------------------------------------------------------
    def forward(self, query: torch.Tensor,
                value_emb_weight: Optional[torch.Tensor] = None,
                return_slot_values: bool = False):
        """query: [..., 2*key_dim].

        Returns (memory_output [..., value_dim] or None, slot_ids [..., top_k],
        weights [..., top_k][, slot_values [..., top_k, value_dim]]).
        """
        lead = query.shape[:-1]
        q = query.reshape(-1, query.shape[-1])
        s1, s2 = self._subscores(q)
        q_full = q[:, :self.key_dim] if self.slot_key_alpha > 0 else None
        cand_scores, cand_ids = self._candidates(s1, s2, q_full)         # [N, A*B]
        sv, si = cand_scores.topk(self.top_k, dim=-1)            # [N, top_k]
        slot_ids = torch.gather(cand_ids, 1, si)                 # [N, top_k]
        weights = F.softmax(sv, dim=-1)

        mem = None
        slot_vals = None
        if value_emb_weight is not None:
            summed, vals = self._read_values(slot_ids, weights, value_emb_weight)
            mem = summed.reshape(*lead, self.value_dim)
            if return_slot_values:
                slot_vals = vals.reshape(*lead, self.top_k, self.value_dim)
        out = (mem, slot_ids.reshape(*lead, self.top_k),
               weights.reshape(*lead, self.top_k))
        if return_slot_values:
            return out + (slot_vals,)
        return out

    def _read_values(self, slot_ids: torch.Tensor, weights: torch.Tensor,
                     value_emb_weight: torch.Tensor):
        # slot_ids, weights: [N, K]
        # If in compact mode, map compact indices to original slot IDs for value lookup
        if self.compact_to_original.numel() > 0 and slot_ids.max() < self.compact_to_original.numel():
            orig_ids = self.compact_to_original[slot_ids.clamp(min=0)]
        else:
            orig_ids = slot_ids

        obj = self.slot_value_token[orig_ids.clamp(min=0)]           # [N, K]
        valid = (obj >= 0).float()
        vals = F.embedding(obj.clamp(min=0), value_emb_weight)       # [N, K, value_dim]
        vals = vals * valid[..., None]
        summed = (weights[..., None] * vals).sum(dim=1)              # [N, value_dim]
        return summed, vals

    @torch.no_grad()
    def retrieve_topk(self, query: torch.Tensor, k: int):
        """Return (slot_ids [..., k], scores [..., k]) for recall evaluation."""
        lead = query.shape[:-1]
        q = query.reshape(-1, query.shape[-1])
        s1, s2 = self._subscores(q)
        q_full = q[:, :self.key_dim] if self.slot_key_alpha > 0 else None
        cand_scores, cand_ids = self._candidates(s1, s2, q_full)
        k = min(k, cand_scores.size(-1))
        sv, si = cand_scores.topk(k, dim=-1)
        slot_ids = torch.gather(cand_ids, 1, si)
        return slot_ids.reshape(*lead, k), sv.reshape(*lead, k)

    def score_slots(self, query: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        """Additive product-key score + optional slot-key score.

        query: [N, 2*key_dim]; slot_ids: [N, M] -> scores [N, M].
        """
        s1, s2 = self._subscores(query)
        k1 = slot_ids // self.num_subkeys
        k2 = slot_ids % self.num_subkeys
        pk_score = torch.gather(s1, 1, k1) + torch.gather(s2, 1, k2)

        if self.slot_key_alpha > 0:
            q_full = query[..., :self.key_dim]
            if self.use_cosine:
                q_full = F.normalize(q_full, dim=-1)
                slot_k = F.normalize(self.slot_keys, dim=-1)
                slot_scores = q_full @ slot_k.t()
            else:
                slot_scores = q_full @ self.slot_keys.t()
            slot_key_score = torch.gather(slot_scores, 1, slot_ids.clamp(0, self.total_slots - 1))
            return pk_score + self.slot_key_alpha * slot_key_score
        return pk_score

    def read_slot_values(self, slot_ids: torch.Tensor, value_emb_weight: torch.Tensor,
                         uniform: bool = True, scores: Optional[torch.Tensor] = None,
                         aggregation_mode: str = "uniform_mean",
                         temperature: float = 0.1,
                         required_slots: Optional[torch.Tensor] = None,
                         threshold: Optional[float] = None,
                         top_n: Optional[int] = None,
                         delta: Optional[float] = None,
                         mass_p: Optional[float] = None) -> torch.Tensor:
        """Read and aggregate slot values with configurable aggregation.

        slot_ids: [N, M] in ORIGINAL slot ID space. Maps to compact if needed.
        scores: [N, M] optional retrieval scores for weighted aggregation.
        aggregation_mode:
          Original modes:
          - "uniform_mean" (default): equal-weight average over all slots.
          - "top1": use only the highest-scoring slot.
          - "top3": use only the top 3 scoring slots (uniform).
          - "score_weighted_softmax": softmax(scores / temperature) weighting.
          - "score_weighted_top3": softmax over top 3 only.
          - "oracle_filter_diagnostic": filter to required_slots only (diagnostic).

          Experiment 0.10 — Threshold/margin selection modes:
          - "score_threshold_absolute": select slots where score >= threshold.
          - "score_threshold_relative_to_top": select slots where
            score >= top_score - delta.
          - "softmax_mass_threshold": select smallest prefix of ranked slots
            such that cumulative softmax mass >= mass_p.
          - "score_gap_cutoff": select ranked slots until the score gap
            between adjacent slots exceeds threshold.
          - "fixed_topN": select top N slots by score (uniform average).

        Extra params for threshold modes:
          threshold: float (required for score_threshold_absolute, score_gap_cutoff)
          delta: float (required for score_threshold_relative_to_top)
          mass_p: float (required for softmax_mass_threshold)
          top_n: int (required for fixed_topN)
        Returns [N, value_dim].
        """
        # Map original slot IDs to compact if in compact mode
        if self.original_to_compact.numel() > slot_ids.max().item():
            compact_ids = self.original_to_compact[slot_ids.clamp(min=0)]
        else:
            compact_ids = slot_ids

        valid = (compact_ids >= 0).float()                          # [N, M]
        # Use compact_ids to index into slot_value_token
        obj = self.slot_value_token[compact_ids.clamp(min=0)]       # [N, M]
        obj_valid = (obj >= 0).float() * valid
        vals = F.embedding(obj.clamp(min=0), value_emb_weight)      # [N, M, value_dim]
        vals = vals * obj_valid[..., None]

        N, M, V = vals.shape

        # Compute aggregation weights
        if aggregation_mode == "oracle_filter_diagnostic" and required_slots is not None:
            # Filter to only required slots: build mask from required_slots per batch item
            mask = torch.zeros(N, M, device=vals.device)
            for i in range(N):
                req_i = set(int(s) for s in required_slots[i] if int(s) >= 0)
                if not req_i:
                    mask[i, :] = obj_valid[i]
                else:
                    for j in range(M):
                        sid = int(slot_ids[i, j].item())
                        if sid in req_i:
                            mask[i, j] = 1.0
            mask = mask * obj_valid
            weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode == "score_threshold_absolute":
            assert threshold is not None, "score_threshold_absolute requires threshold"
            assert scores is not None, "score_threshold_absolute requires scores"
            # Select slots where score >= threshold
            mask = (scores >= threshold).float() * obj_valid
            weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode == "score_threshold_relative_to_top":
            assert delta is not None, "score_threshold_relative_to_top requires delta"
            assert scores is not None, "score_threshold_relative_to_top requires scores"
            # Select slots where score >= top_score - delta
            top_scores = scores.max(dim=-1, keepdim=True).values  # [N, 1]
            mask = (scores >= top_scores - delta).float() * obj_valid
            weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode == "softmax_mass_threshold":
            assert mass_p is not None, "softmax_mass_threshold requires mass_p"
            assert scores is not None, "softmax_mass_threshold requires scores"
            # Select smallest prefix such that cumulative softmax mass >= mass_p
            # Slots are already ranked. Compute softmax weights, then cumulative sum.
            w_soft = F.softmax(scores / temperature, dim=-1)  # [N, M]
            cumsum = w_soft.cumsum(dim=-1)  # [N, M]
            # Find first index where cumsum >= mass_p
            exceed = (cumsum >= mass_p).float()  # [N, M]
            # For each row, mask up to and including the first exceeding index
            # Use argmax to find the first 1
            first_idx = exceed.argmax(dim=-1, keepdim=True)  # [N, 1] — 0 if never exceeds
            # Build mask: include all positions <= first_idx
            range_t = torch.arange(M, device=vals.device).unsqueeze(0)  # [1, M]
            mask = (range_t <= first_idx).float() * obj_valid
            weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode == "score_gap_cutoff":
            assert threshold is not None, "score_gap_cutoff requires threshold"
            assert scores is not None, "score_gap_cutoff requires scores"
            # Select ranked slots until score gap between adjacent slots exceeds threshold.
            # Compute gap: scores[:, :-1] - scores[:, 1:]  ([N, M-1])
            # Find first gap that exceeds threshold.
            gaps = scores[:, :-1] - scores[:, 1:]  # [N, M-1]
            gap_mask = (gaps > threshold).float()  # [N, M-1]
            # Find first position where gap exceeds threshold
            first_gap_idx = gap_mask.argmax(dim=-1, keepdim=True)  # [N, 1]
            # Build mask: include positions up to and including first_gap_idx
            range_t = torch.arange(M, device=vals.device).unsqueeze(0)  # [1, M]
            mask = (range_t <= first_gap_idx).float() * obj_valid
            weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode == "top1":
            # Use only the top-scoring slot
            weights = torch.zeros(N, M, device=vals.device)
            if scores is not None:
                best_idx = scores.argmax(dim=-1)  # [N]
                weights[torch.arange(N, device=vals.device), best_idx] = 1.0
            else:
                weights[:, 0] = 1.0
            weights = weights * obj_valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode == "top3":
            # Use top 3 scoring slots (uniform among them)
            weights = torch.zeros(N, M, device=vals.device)
            if scores is not None and M >= 3:
                _, top3_idx = scores.topk(min(3, M), dim=-1)  # [N, 3]
                for i in range(N):
                    for j in top3_idx[i]:
                        weights[i, j] = 1.0
            else:
                k = min(3, M)
                weights[:, :k] = 1.0
            weights = weights * obj_valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode == "fixed_topN":
            assert top_n is not None, "fixed_topN requires top_n"
            n = min(top_n, M)
            weights = torch.zeros(N, M, device=vals.device)
            if scores is not None and M >= n:
                _, top_idx = scores.topk(n, dim=-1)  # [N, n]
                for i in range(N):
                    for j in top_idx[i]:
                        weights[i, j] = 1.0
            else:
                weights[:, :n] = 1.0
            weights = weights * obj_valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        elif aggregation_mode in ("score_weighted_softmax", "score_weighted_top3"):
            if scores is not None:
                if aggregation_mode == "score_weighted_top3" and M > 3:
                    # Only keep top 3, mask others
                    top3_vals, top3_idx = scores.topk(min(3, M), dim=-1)
                    masked_scores = torch.full_like(scores, float('-inf'))
                    masked_scores.scatter_(1, top3_idx, top3_vals)
                    w = F.softmax(masked_scores / temperature, dim=-1)
                else:
                    w = F.softmax(scores / temperature, dim=-1)
                weights = w * obj_valid
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            else:
                weights = obj_valid / obj_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:  # uniform_mean (default)
            weights = obj_valid / obj_valid.sum(dim=1, keepdim=True).clamp(min=1.0)

        result = (vals * weights.unsqueeze(-1)).sum(dim=1)  # [N, V]
        return result

    def num_live_slots(self) -> int:
        return int((self.slot_value_token >= 0).sum().item())
