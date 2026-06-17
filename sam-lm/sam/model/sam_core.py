"""SAM-Tiny: a small causal reasoning core with optional product-key memory.

Memory modes (the experimental knobs):
  core_only        - no memory at all (capacity floor; must be weak on knowledge)
  oracle_memory    - inject the values of the *correct* required slots (bypass retrieval)
  retrieved_memory - inject values from learned product-key retrieval (the real SAM)
  random_memory    - inject values of random live slots (A2 placebo)

Memory is injected at every ``memory_every``-th block through a gated residual
(spec default) or an optional cross-attention over retrieved slots.

Value content (v0): a learnable embedding of each slot's stored object token,
shared across slots and memory layers (see product_key_memory.py).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm, TransformerBlock
from .product_key_memory import ProductKeyMemory

MEMORY_MODES = ("core_only", "oracle_memory", "retrieved_memory", "random_memory", "oracle_text_memory",
                "retrieved_memory_external_text_query", "retrieved_memory_hidden_adapter",
                "train_memory_adapter", "retrieved_oracle_slots",
                "retrieved_multi_query_union")


class DualEncoderWrapper:
    """Wraps a trained dual encoder for use as SAM's retrieval backend."""
    def __init__(self, ckpt_path: str, tokenizer, device: str = "cpu"):
        from .transformer import RMSNorm
        from ..training.train_retrieval import DualEncoderRetriever, QueryEncoder

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        ms = state.get("model_state", state)

        # Reconstruct encoder
        enc = QueryEncoder(
            vocab_size=tokenizer.vocab_size, d_model=256,
            n_layers=3, n_heads=4, d_ff=1024, query_dim=256,
            max_seq_len=64, pad_id=tokenizer.pad
        )
        num_slots = ms["slot_emb.weight"].shape[0]
        enc.load_state_dict({k.replace("encoder.", ""): v for k, v in ms.items() if k.startswith("encoder.")}, strict=False)

        self.dual = DualEncoderRetriever(enc, ms["slot_emb.weight"].shape[1], num_slots)
        self.dual.load_state_dict(ms, strict=False)
        self.dual.to(device)
        self.dual.eval()
        self._slot_emb = self.dual.slot_emb.weight.clone()
        self.device = device

    @torch.no_grad()
    def retrieve(self, query_vectors: torch.Tensor, k: int = 8):
        """query_vectors: [B, D] — from SAM's memory head query projection.
        Returns (slot_ids [B, k], scores [B, k])."""
        # Project query to slot space
        q = F.normalize(self.dual.query_proj(query_vectors[:, :256].to(self.device)), dim=-1)
        s = F.normalize(self._slot_emb.to(self.device), dim=-1)
        scores = q @ s.t()
        sv, si = scores.topk(k, dim=-1)
        return si, sv

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
        """Encode raw question text through the dual encoder's query encoder.
        Returns normalized query vectors [B, query_dim]."""
        q, _ = self.dual(input_ids.to(self.device), prompt_lens.to(self.device))
        return q  # already normalized by dual.forward


class ChainSetRetrieverWrapper:
    """Wraps a trained chain-set retriever for use as SAM's retrieval backend."""
    def __init__(self, ckpt_path: str, tokenizer, device: str = "cpu"):
        from ..training.train_retrieval import ChainSetRetriever, QueryEncoder

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        ms = state.get("model_state", state)

        enc = QueryEncoder(
            vocab_size=tokenizer.vocab_size, d_model=256,
            n_layers=3, n_heads=4, d_ff=1024, query_dim=256,
            max_seq_len=64, pad_id=tokenizer.pad
        )
        num_slots = ms["slot_emb.weight"].shape[0]
        slot_dim = ms["slot_emb.weight"].shape[1]

        self.chain = ChainSetRetriever(enc, slot_dim, num_slots)
        self.chain.load_state_dict(ms, strict=False)
        self.chain.to(device)
        self.chain.eval()
        self._slot_emb = self.chain.slot_emb.weight.clone()
        self.device = device

    @torch.no_grad()
    def retrieve(self, query_vectors: torch.Tensor, k: int = 8):
        """query_vectors: [B, D] — from SAM's memory head query projection.
        Returns (slot_ids [B, k], scores [B, k])."""
        q = F.normalize(self.chain.query_proj(query_vectors[:, :256].to(self.device)), dim=-1)
        s = F.normalize(self._slot_emb.to(self.device), dim=-1)
        scores = q @ s.t()
        sv, si = scores.topk(k, dim=-1)
        return si, sv

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
        """Encode raw question text through the chain-set retriever's query encoder.
        Returns normalized query vectors [B, query_dim]."""
        q, _ = self.chain(input_ids.to(self.device), prompt_lens.to(self.device))
        return q  # already normalized by chain.forward

    @property
    def dual(self):
        """Compatibility shim for SAM's set_kb which accesses retriever.dual."""
        return self.chain


class MemoryQueryAdapter(nn.Module):
    """Adapter from SAM hidden state to dual encoder query embedding space.

    Architecture: Linear or 2-layer MLP with LayerNorm, output L2-normalized.
    Used by retrieved_memory_hidden_adapter mode.
    """
    def __init__(self, hidden_dim: int, query_dim: int, hidden_mult: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        mid_dim = hidden_dim * hidden_mult
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, query_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        """h_last: [B, hidden_dim] — SAM hidden state at last prompt token.
        Returns normalized query [B, query_dim]."""
        h = self.norm(h_last)
        h = F.relu(self.fc1(h))
        q = self.fc2(h)
        return F.normalize(q, dim=-1)


class MemoryHead(nn.Module):
    """Per-memory-layer projections. The bank (keys+values) is shared at model level."""

    def __init__(self, d_model: int, key_dim: int, value_dim: int,
                 integration: str = "gated_sum"):
        super().__init__()
        self.integration = integration
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, 2 * key_dim, bias=False)
        self.mem_proj = nn.Linear(value_dim, d_model, bias=False)
        self.Wg = nn.Linear(2 * d_model, d_model)
        if integration == "cross_attention":
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(value_dim, d_model, bias=False)
            self.v_proj = nn.Linear(value_dim, d_model, bias=False)

    def integrate_gated(self, x: torch.Tensor, mem_val: torch.Tensor) -> torch.Tensor:
        mem_d = self.mem_proj(mem_val)                      # [B,T,d]
        gate = torch.sigmoid(self.Wg(torch.cat([x, mem_d], dim=-1)))
        return x + gate * mem_d

    def integrate_xattn(self, x: torch.Tensor, slot_vals: torch.Tensor) -> torch.Tensor:
        # slot_vals: [B,T,K,value_dim]
        qh = self.q_proj(x)                                 # [B,T,d]
        kh = self.k_proj(slot_vals)                         # [B,T,K,d]
        vh = self.v_proj(slot_vals)
        att = (qh[:, :, None, :] * kh).sum(-1) / math.sqrt(self.d_model)
        att = att.softmax(dim=2)                            # [B,T,K]
        ctx = (att[..., None] * vh).sum(2)                  # [B,T,d]
        gate = torch.sigmoid(self.Wg(torch.cat([x, ctx], dim=-1)))
        return x + gate * ctx


class SamModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
        max_seq_len: int = 128,
        memory_every: int = 3,
        memory_query: str = "tokenwise",
        memory_integration: str = "gated_sum",
        memory_cfg: Optional[Dict] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        memory_cfg = memory_cfg or {}
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.memory_query = memory_query
        self.memory_integration = memory_integration
        self.memory_mode = "retrieved_memory"  # default; overridden per-forward / by trainer

        self.key_dim = int(memory_cfg.get("key_dim", 128))
        self.value_dim = int(memory_cfg.get("value_dim", d_model))

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # which blocks get a memory layer
        self.memory_at = [i for i in range(n_layers) if (i + 1) % memory_every == 0]
        self.mem_index = {layer: j for j, layer in enumerate(self.memory_at)}

        # shared memory bank
        self.pkm = ProductKeyMemory(
            num_subkeys=int(memory_cfg.get("num_subkeys", 1024)),
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            top_a=int(memory_cfg.get("top_a", 16)),
            top_b=int(memory_cfg.get("top_b", 16)),
            top_k=int(memory_cfg.get("top_k", 4)),
            soft_candidates=bool(memory_cfg.get("soft_candidates", False)),
            use_cosine=bool(memory_cfg.get("use_cosine", False)),
            slot_key_alpha=float(memory_cfg.get("slot_key_alpha", 0.0)),
        )
        # value content: learnable embedding of the stored object token
        self.value_emb = nn.Embedding(vocab_size, self.value_dim)

        self.memory_heads = nn.ModuleList(
            [MemoryHead(d_model, self.key_dim, self.value_dim, memory_integration)
             for _ in self.memory_at]
        )

        # Adapter for hidden-state → dual encoder query (retrieved_memory_hidden_adapter)
        adapter_query_dim = int(memory_cfg.get("adapter_query_dim", 256))
        self.memory_query_adapter = MemoryQueryAdapter(
            hidden_dim=d_model, query_dim=adapter_query_dim,
        )
        self._adapter_frozen = False

        self.register_buffer("live_slot_ids", torch.zeros(1, dtype=torch.long))
        self.rand_m = int(memory_cfg.get("top_k", 4))
        self._aggregation_mode = memory_cfg.get("aggregation_mode", "uniform_mean")
        self._aggregation_temperature = float(memory_cfg.get("aggregation_temperature", 0.1))
        # Experiment 0.10: Threshold/margin selection parameters
        self._aggregation_threshold = memory_cfg.get("aggregation_threshold", None)
        self._aggregation_top_n = memory_cfg.get("aggregation_top_n", None)
        self._aggregation_delta = memory_cfg.get("aggregation_delta", None)
        self._aggregation_mass_p = memory_cfg.get("aggregation_mass_p", None)
        self._multi_query_k = memory_cfg.get("multi_query_k", 8)  # topK per query in multi-query mode
        self._multi_query_count = memory_cfg.get("multi_query_count", 4)  # number of query variants
        self._tokenizer = None  # Set via set_tokenizer() for multi-query
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # -- KB wiring -----------------------------------------------------------
    def set_kb(self, slot_value_token: torch.Tensor, retriever: Optional[object] = None) -> None:
        self.pkm.set_slot_value_tokens(slot_value_token)
        live = torch.nonzero(self.pkm.slot_value_token >= 0, as_tuple=False).flatten()
        self.live_slot_ids = live.to(self.pkm.slot_value_token.device)
        self._retriever = retriever  # optional dual-encoder for retrieved_memory

        # Cache frozen slot embeddings for adapter retrieval
        if retriever is not None:
            self._slot_emb_frozen = retriever.dual.slot_emb.weight.clone().detach().to(
                self.pkm.slot_value_token.device
            )
        else:
            self._slot_emb_frozen = None

    def set_tokenizer(self, tokenizer) -> None:
        """Store tokenizer reference for multi-query mode."""
        self._tokenizer = tokenizer

    def _sample_random_slots(self, B: int, M: int, device) -> torch.Tensor:
        n = max(1, self.live_slot_ids.numel())
        idx = torch.randint(0, n, (B, M), device=device)
        return self.live_slot_ids.to(device)[idx]

    @torch.no_grad()
    def _compute_multi_query_retrieval(
        self,
        input_ids: torch.Tensor,
        prompt_lens: torch.Tensor,
        task_types: List[str],
        hops: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-query union retrieval for multi-hop diagnostic.

        Generates multiple query variants per example:
          1. Original question text
          2. Question + task_type context
          3. Question + hops context
          4. Task type + hops combined context

        Retrieves topK_per_query for each variant, unions, deduplicates,
        and reranks by max score across queries.
        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            # Fallback: just use original question
            q_text = self._retriever.encode_text(input_ids, prompt_lens.to(device))
            s_frozen = F.normalize(self._slot_emb_frozen.to(device), dim=-1)
            scores = q_text @ s_frozen.t()
            return scores.topk(getattr(self, '_retrieval_k', 8), dim=-1)

        retriever = self._retriever
        slot_emb = F.normalize(self._slot_emb_frozen.to(device), dim=-1)
        k_per_query = getattr(self, '_multi_query_k', 8)
        ret_k = getattr(self, '_retrieval_k', 32)
        n_queries = getattr(self, '_multi_query_count', 4)
        B = input_ids.size(0)

        # Collect all query texts per example
        # Each example contributes up to n_queries query variants
        all_query_texts: List[List[str]] = [[] for _ in range(B)]
        for i in range(B):
            # Query 1: Original question
            p_len = int(prompt_lens[i].item())
            orig_text = tokenizer.decode(input_ids[i, :p_len].tolist()).strip()
            all_query_texts[i].append(orig_text)

            # Query 2: Question + task_type
            tt = task_types[i] if i < len(task_types) else "unknown"
            all_query_texts[i].append(f"{orig_text} [task: {tt}]")

            # Query 3: Question + hops
            h = hops[i] if i < len(hops) else 0
            all_query_texts[i].append(f"{orig_text} [hops: {h}]")

            # Query 4: Task + hops combined
            all_query_texts[i].append(f"task:{tt} hops:{h}")

            # Trim to requested count
            all_query_texts[i] = all_query_texts[i][:n_queries]

        # Flatten: encode all queries at once
        flat_texts: List[str] = []
        example_query_counts: List[int] = []
        for i in range(B):
            texts = all_query_texts[i]
            flat_texts.extend(texts)
            example_query_counts.append(len(texts))

        # Tokenize and encode all queries
        max_q_len = 64  # Match dual encoder config
        pad_id = tokenizer.pad
        flat_input_ids = torch.full((len(flat_texts), max_q_len), pad_id, dtype=torch.long, device=device)
        flat_prompt_lens = torch.zeros(len(flat_texts), dtype=torch.long, device=device)
        for j, t in enumerate(flat_texts):
            ids = tokenizer.encode(t)[:max_q_len]
            flat_input_ids[j, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            flat_prompt_lens[j] = min(len(ids), max_q_len)

        # Encode all queries through dual encoder
        q_vectors = retriever.encode_text(flat_input_ids, flat_prompt_lens)  # [total_queries, D]
        scores_all = q_vectors @ slot_emb.t()  # [total_queries, num_slots]

        # Collect per-example results: union topK from each query variant
        all_slots: List[List[int]] = []
        all_scores: List[List[float]] = []
        offset = 0
        for i in range(B):
            nq = example_query_counts[i]
            # Get topK per query for this example's variants
            seen_slots: Dict[int, float] = {}  # slot_id -> max score
            for q_idx in range(nq):
                q_scores = scores_all[offset + q_idx]  # [num_slots]
                top_scores, top_slots = q_scores.topk(k_per_query, dim=-1)
                for s in range(top_slots.numel()):
                    sid = int(top_slots[s].item())
                    sc = float(top_scores[s].item())
                    if sid not in seen_slots or sc > seen_slots[sid]:
                        seen_slots[sid] = sc

            # Sort by score and take top ret_k
            sorted_slots = sorted(seen_slots.items(), key=lambda x: x[1], reverse=True)[:ret_k]
            slots = [s for s, _ in sorted_slots]
            scores = [sc for _, sc in sorted_slots]
            # Pad if fewer than ret_k
            while len(slots) < ret_k:
                slots.append(-1)
                scores.append(float('-inf'))
            all_slots.append(slots)
            all_scores.append(scores)
            offset += nq

        # Convert to tensors
        ret_slots = torch.tensor(all_slots, dtype=torch.long, device=device)
        ret_scores = torch.tensor(all_scores, dtype=torch.float, device=device)

        return ret_slots, ret_scores

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        required_slots: Optional[torch.Tensor] = None,
        prompt_lens: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        mode = mode or self.memory_mode
        assert mode in MEMORY_MODES, mode
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]
        device = input_ids.device
        pos = torch.arange(T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None]

        # last-prompt-token index per example (for sequence query + diagnostics)
        if prompt_lens is not None:
            last_idx = (prompt_lens.to(device) - 1).clamp(0, T - 1)
        else:
            last_idx = torch.full((B,), T - 1, device=device, dtype=torch.long)
        arangeB = torch.arange(B, device=device)

        # precompute oracle / random memory vector (same at all positions/layers)
        broadcast_vec = None
        if mode in ("oracle_memory", "retrieved_oracle_slots"):
            assert required_slots is not None, f"{mode} needs required_slots"
            broadcast_vec = self.pkm.read_slot_values(
                required_slots.to(device), self.value_emb.weight)        # [B,vd]
        elif mode == "random_memory":
            rand_slots = self._sample_random_slots(B, self.rand_m, device)
            broadcast_vec = self.pkm.read_slot_values(rand_slots, self.value_emb.weight)

        # Precompute retrieval for external text query mode (uses raw input_ids)
        external_text_slots = None
        external_text_scores = None
        if mode == "retrieved_memory_external_text_query":
            if hasattr(self, '_retriever') and self._retriever is not None:
                ret_k = getattr(self, '_retrieval_k', 8)
                q_text = self._retriever.encode_text(input_ids, prompt_lens.to(device))
                s_frozen = F.normalize(self._slot_emb_frozen.to(device), dim=-1)
                scores = q_text @ s_frozen.t()
                external_text_scores, external_text_slots = scores.topk(ret_k, dim=-1)
        elif mode == "retrieved_multi_query_union":
            if hasattr(self, '_retriever') and self._retriever is not None:
                # Need task_types and hops from batch
                # These are passed via extra kwargs or can be inferred
                task_types_list = getattr(self, '_batch_task_types', ["unknown"] * B)
                hops_list = getattr(self, '_batch_hops', [0] * B)
                external_text_slots, external_text_scores = self._compute_multi_query_retrieval(
                    input_ids, prompt_lens, task_types_list, hops_list, device,
                )

        aux: Dict = {}
        first_done = False
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.memory_at and mode != "core_only":
                head = self.memory_heads[self.mem_index[i]]
                if mode in ("oracle_memory", "random_memory", "retrieved_oracle_slots"):
                    mem_val = broadcast_vec[:, None, :].expand(B, T, self.value_dim)
                    x = head.integrate_gated(x, mem_val)
                elif mode in ("retrieved_memory_external_text_query", "retrieved_multi_query_union"):
                    if external_text_slots is not None:
                        extra_kw = {}
                        if self._aggregation_mode == "oracle_filter_diagnostic":
                            extra_kw["required_slots"] = required_slots
                        # Pass threshold/margin parameters
                        if self._aggregation_threshold is not None:
                            extra_kw["threshold"] = self._aggregation_threshold
                        if self._aggregation_top_n is not None:
                            extra_kw["top_n"] = self._aggregation_top_n
                        if self._aggregation_delta is not None:
                            extra_kw["delta"] = self._aggregation_delta
                        if self._aggregation_mass_p is not None:
                            extra_kw["mass_p"] = self._aggregation_mass_p
                        mem_val = self.pkm.read_slot_values(
                            external_text_slots, self.value_emb.weight,
                            scores=external_text_scores,
                            aggregation_mode=self._aggregation_mode,
                            temperature=self._aggregation_temperature,
                            **extra_kw,
                        )
                        mem_val = mem_val[:, None, :].expand(B, T, self.value_dim)
                        x = head.integrate_gated(x, mem_val)
                elif mode in ("retrieved_memory_hidden_adapter", "train_memory_adapter"):
                    # Use adapter on h_last to query frozen slot embeddings
                    h_last = x[arangeB, last_idx]
                    q_adapter = self.memory_query_adapter(h_last)
                    s_frozen = F.normalize(self._slot_emb_frozen.to(device), dim=-1)
                    scores = q_adapter @ s_frozen.t()
                    ret_k = getattr(self, '_retrieval_k', 8)
                    top_scores, slot_ids = scores.topk(ret_k, dim=-1)
                    mem_val = self.pkm.read_slot_values(
                        slot_ids, self.value_emb.weight,
                        scores=top_scores,
                        aggregation_mode=self._aggregation_mode,
                        temperature=self._aggregation_temperature,
                        threshold=self._aggregation_threshold,
                        top_n=self._aggregation_top_n,
                        delta=self._aggregation_delta,
                        mass_p=self._aggregation_mass_p,
                    )
                    mem_val = mem_val[:, None, :].expand(B, T, self.value_dim)
                    x = head.integrate_gated(x, mem_val)
                    if not first_done:
                        aux["adapter_query"] = q_adapter.detach()
                        aux["adapter_retrieved_slots"] = slot_ids.detach()
                        aux["adapter_scores"] = top_scores.detach()
                else:  # retrieved_memory (original)
                    if hasattr(self, '_retriever') and self._retriever is not None:
                        # Use external dual-encoder retriever
                        h_last = x[arangeB, last_idx]
                        slots, scores = self._retriever.retrieve(h_last, k=4)
                        mem_val = self.pkm.read_slot_values(slots, self.value_emb.weight)
                        mem_val = mem_val[:, None, :].expand(B, T, self.value_dim)
                        x = head.integrate_gated(x, mem_val)
                    elif self.memory_query == "sequence":
                        h_last = x[arangeB, last_idx]                    # [B,d]
                        q = head.Wq(h_last)                              # [B,2k]
                        mem, sids, w = self.pkm(q, self.value_emb.weight)
                        mem_val = mem[:, None, :].expand(B, T, self.value_dim)
                        x = head.integrate_gated(x, mem_val)
                    else:  # tokenwise
                        q = head.Wq(x)                                   # [B,T,2k]
                        if self.memory_integration == "cross_attention":
                            mem, sids, w, slot_vals = self.pkm(
                                q, self.value_emb.weight, return_slot_values=True)
                            x = head.integrate_xattn(x, slot_vals)
                        else:
                            mem, sids, w = self.pkm(q, self.value_emb.weight)
                            x = head.integrate_gated(x, mem)
                    if not first_done:
                        # primary retrieval diagnostic from last prompt token
                        h_last = x[arangeB, last_idx]
                        q_prim = head.Wq(h_last)
                        rs, _ = self.pkm.retrieve_topk(q_prim, self.pkm.top_k)
                        aux["primary_query"] = q_prim.detach()
                        aux["retrieved_slots"] = rs.detach()
                first_done = True

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            labels = labels[:, :T]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, aux

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 6,
                 eos_id: Optional[int] = None,
                 required_slots: Optional[torch.Tensor] = None,
                 mode: Optional[str] = None) -> torch.Tensor:
        self.eval()
        ids = prompt_ids.clone()
        if ids.dim() == 1:
            ids = ids[None]
        req = required_slots[None] if (required_slots is not None and required_slots.dim() == 1) \
            else required_slots
        for _ in range(max_new_tokens):
            cur = ids[:, -self.max_seq_len:]
            plens = torch.tensor([cur.size(1)], device=ids.device)
            logits, _, _ = self.forward(cur, required_slots=req, prompt_lens=plens, mode=mode)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
        return ids[0, prompt_ids.shape[-1]:]

    @torch.no_grad()
    def retrieve(self, input_ids: torch.Tensor, prompt_lens: Optional[torch.Tensor],
                 k: int, mode: Optional[str] = None) -> Optional[torch.Tensor]:
        """Diagnostic retrieval: top-k slots. Returns slot_ids [B, k] (or None).

        For external_text_query / hidden_adapter: uses the respective retrieval path.
        For other modes: uses PKM from first memory layer."""
        mode = mode or self.memory_mode

        if mode in ("retrieved_memory_external_text_query", "retrieved_multi_query_union"):
            if hasattr(self, '_retriever') and self._retriever is not None:
                q_text = self._retriever.encode_text(input_ids, prompt_lens)
                s_frozen = F.normalize(self._slot_emb_frozen.to(q_text.device), dim=-1)
                scores = q_text @ s_frozen.t()
                _, slots = scores.topk(k, dim=-1)
                return slots
            return None

        if mode == "retrieved_memory_hidden_adapter":
            if not self.memory_at or self._slot_emb_frozen is None:
                return None
            self.eval()
            B, T = input_ids.shape
            T = min(T, self.max_seq_len)
            input_ids = input_ids[:, :T]
            device = input_ids.device
            pos = torch.arange(T, device=device)
            x = self.token_emb(input_ids) + self.pos_emb(pos)[None]
            if prompt_lens is not None:
                last_idx = (prompt_lens.to(device) - 1).clamp(0, T - 1)
            else:
                last_idx = torch.full((B,), T - 1, device=device, dtype=torch.long)
            arangeB = torch.arange(B, device=device)
            first = self.memory_at[0]
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i == first:
                    h_last = x[arangeB, last_idx]
                    q_adapter = self.memory_query_adapter(h_last)
                    s_frozen = F.normalize(self._slot_emb_frozen.to(device), dim=-1)
                    scores = q_adapter @ s_frozen.t()
                    _, slots = scores.topk(k, dim=-1)
                    return slots
            return None

        # Original PKM retrieval path
        if not self.memory_at:
            return None
        self.eval()
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]
        device = input_ids.device
        pos = torch.arange(T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None]
        if prompt_lens is not None:
            last_idx = (prompt_lens.to(device) - 1).clamp(0, T - 1)
        else:
            last_idx = torch.full((B,), T - 1, device=device, dtype=torch.long)
        arangeB = torch.arange(B, device=device)
        first = self.memory_at[0]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == first:
                q = self.memory_heads[0].Wq(x[arangeB, last_idx])
                slots, _ = self.pkm.retrieve_topk(q, k)
                return slots
        return None

    # -- bookkeeping ---------------------------------------------------------
    def _unique_params(self):
        seen = set()
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                yield p

    def param_count(self) -> int:
        return sum(p.numel() for p in self._unique_params())

    def memory_param_count(self) -> int:
        return (self.pkm.K1.numel() + self.pkm.K2.numel() + self.value_emb.weight.numel())

    def core_active_param_count(self) -> int:
        return self.param_count() - self.memory_param_count()
