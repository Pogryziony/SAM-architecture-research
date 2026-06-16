"""Product-key memory retrieval pretraining (Gate 1 diagnostic).

Trains a small query encoder + product keys to retrieve required_slots
using InfoNCE contrastive loss. Reports Recall@1, Recall@8, Recall@32.

Usage:
    python -m sam.training.train_retrieval --config configs/retrieval_1m.yaml
    python -m sam.training.train_retrieval --config configs/retrieval_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..data.dataset import QADataset, Tokenizer, collate_qa, build_kb_tensors
from ..model.product_key_memory import ProductKeyMemory
from ..model.transformer import RMSNorm
from ..utils.config import load_config, Config
from ..utils.seed import seed_everything
from ..utils.logging import MetricLogger


def _pick_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def _cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class QueryEncoder(nn.Module):
    """Small transformer encoder that produces a query vector from the question."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, query_dim: int,
                 max_seq_len: int = 64, dropout: float = 0.0, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.query_dim = query_dim
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            _EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, query_dim, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]
        pos = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Pool: last prompt token
        last_idx = (prompt_lens - 1).clamp(0, T - 1)
        arangeB = torch.arange(B, device=input_ids.device)
        h = x[arangeB, last_idx]              # [B, d_model]
        return self.proj(h)                   # [B, query_dim]


class _EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x):
        # Causal self-attention
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class RetrievalModel(nn.Module):
    """Query encoder + product-key memory for retrieval-only training."""

    def __init__(self, encoder: QueryEncoder, memory: ProductKeyMemory):
        super().__init__()
        self.encoder = encoder
        self.memory = memory

    def forward(self, input_ids, prompt_lens):
        q = self.encoder(input_ids, prompt_lens)
        # Return query + top slots for loss and recall
        slots, scores = self.memory.retrieve_topk(q, k=self.memory.top_k)
        return q, slots, scores

    def param_count(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    @property
    def max_seq_len(self) -> int:
        return self.encoder.max_seq_len


class ClassifierRetriever(nn.Module):
    """Direct classifier over live slots — determines if retrieval is learnable."""
    def __init__(self, encoder: QueryEncoder, num_live: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.query_dim, num_live, bias=False)

    def forward(self, input_ids, prompt_lens):
        q = self.encoder(input_ids, prompt_lens)
        return self.classifier(q)  # [B, num_live]

    def retrieve_topk(self, logits, k):
        scores, indices = logits.topk(min(k, logits.size(-1)), dim=-1)
        return indices, scores

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def max_seq_len(self):
        return self.encoder.max_seq_len


class CosineRetriever(nn.Module):
    """Dense cosine retriever."""
    def __init__(self, encoder: QueryEncoder, num_live: int, slot_dim: int):
        super().__init__()
        self.encoder = encoder
        self.slot_emb = nn.Embedding(num_live, slot_dim)
        nn.init.normal_(self.slot_emb.weight, std=0.02)

    def forward(self, input_ids, prompt_lens):
        q = self.encoder(input_ids, prompt_lens)
        q_n = F.normalize(q, dim=-1)
        s_n = F.normalize(self.slot_emb.weight, dim=-1)
        return q_n @ s_n.t()

    def retrieve_topk(self, scores, k):
        sv, si = scores.topk(min(k, scores.size(-1)), dim=-1)
        return si, sv

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def max_seq_len(self):
        return self.encoder.max_seq_len


class ContrastiveRetriever(nn.Module):
    """Contrastive retriever: InfoNCE pulls same-slot questions together."""
    def __init__(self, encoder: QueryEncoder, temperature: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(encoder.query_dim, encoder.query_dim),
            nn.ReLU(),
            nn.Linear(encoder.query_dim, 128),
        )

    def forward(self, input_ids, prompt_lens):
        q = self.encoder(input_ids, prompt_lens)
        return F.normalize(self.proj(q), dim=-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def max_seq_len(self):
        return self.encoder.max_seq_len


def contrastive_loss_fn(z: torch.Tensor, slot_ids: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE: positives = questions about the same slot."""
    B = z.size(0)
    device = z.device
    sim = z @ z.t() / temperature
    pos_mask = (slot_ids.unsqueeze(0) == slot_ids.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)
    exp_sim = sim.exp() * (1 - torch.eye(B, device=device))
    num = (exp_sim * pos_mask).sum(dim=1)
    denom = exp_sim.sum(dim=1)
    valid = pos_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return -torch.log((num[valid] + 1e-10) / (denom[valid] + 1e-10)).mean()


def train_contrastive(cfg: Config):
    """Train contrastive retriever with k-NN evaluation."""
    seed_everything(cfg.get("seed", 42))
    device = _pick_device(cfg.train.get("device", "auto"))
    data_dir = cfg.get("data_dir", "data/synthetic")
    output_dir = cfg.get("output_dir", "experiments/contrastive")
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_dir(data_dir)
    run_name = cfg.get("run_name", "contrastive")
    mlogger = MetricLogger(output_dir, run_name)
    mlogger.logger.info("Contrastive training — device=%s data=%s", device, data_dir)

    ec = cfg.model.query_encoder if hasattr(cfg.model, 'query_encoder') else cfg.model.get("query_encoder", {})
    d_model = ec.get("d_model", 256) if isinstance(ec, dict) else getattr(ec, 'd_model', 256)
    n_l = ec.get("n_layers", 3) if isinstance(ec, dict) else getattr(ec, 'n_layers', 3)
    n_h = ec.get("n_heads", 4) if isinstance(ec, dict) else getattr(ec, 'n_heads', 4)
    d_ff = ec.get("d_ff", 1024) if isinstance(ec, dict) else getattr(ec, 'd_ff', 1024)
    ms = ec.get("max_seq_len", 64) if isinstance(ec, dict) else getattr(ec, 'max_seq_len', 64)

    encoder = QueryEncoder(vocab_size=tokenizer.vocab_size, d_model=d_model,
                           n_layers=n_l, n_heads=n_h, d_ff=d_ff, query_dim=256,
                           max_seq_len=ms, pad_id=tokenizer.pad)
    temp = cfg.train.get("temperature", 0.07)
    model = ContrastiveRetriever(encoder, temp).to(device)
    mlogger.logger.info("Model: %d params", model.param_count())

    t_cfg = cfg.train
    bs = t_cfg.get("batch_size", 128)
    train_ds = QADataset(data_dir, "train", tokenizer, kind="qa", open_book=False, max_seq_len=ms)
    val_ds = QADataset(data_dir, "val", tokenizer, kind="qa", open_book=False, max_seq_len=ms)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=lambda b: collate_qa(b, tokenizer.pad))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=lambda b: collate_qa(b, tokenizer.pad))
    mlogger.logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    optimizer = AdamW(model.parameters(), lr=t_cfg.get("lr", 3e-4))
    epochs = t_cfg.get("epochs", 10)
    total_steps = epochs * len(train_loader)
    scheduler = _cosine_warmup_schedule(optimizer, t_cfg.get("warmup_steps", 100), total_steps)

    global_step = 0
    best_recall = 0.0
    log_every = t_cfg.get("log_every", 50)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            prompt_lens = torch.tensor(batch["prompt_len"], device=device)
            slot_ids = batch["required_slots"][:, 0].clamp(min=0).to(device)
            z = model(input_ids, prompt_lens)
            loss = contrastive_loss_fn(z, slot_ids, model.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % log_every == 0:
                val_acc = knn_eval(model, val_loader, train_loader, device, k=8)
                mlogger.log(global_step, {"loss": loss.item(), "knn_rec@8": val_acc,
                                          "lr": scheduler.get_last_lr()[0]})
                if val_acc > best_recall:
                    best_recall = val_acc
                    mlogger.logger.info("New best knn rec@8=%.4f (step %d)", val_acc, global_step)
        mlogger.logger.info("Epoch %d/%d — best knn rec@8=%.4f", epoch + 1, epochs, best_recall)

    summary = {"run_name": run_name, "best_knn_recall@8": best_recall,
               "train_examples": len(train_ds), "val_examples": len(val_ds)}
    mlogger.save_summary(summary)
    return summary


@torch.no_grad()
def knn_eval(model, val_loader, train_loader, device, k=8):
    model.eval()
    train_z, train_slots = [], []
    for batch in train_loader:
        z = model(batch["input_ids"].to(device), torch.tensor(batch["prompt_len"], device=device))
        train_z.append(z)
        train_slots.append(batch["required_slots"][:, 0].clamp(min=0).to(device))
    train_z = torch.cat(train_z); train_slots = torch.cat(train_slots)
    hits = total = 0
    for batch in val_loader:
        z = model(batch["input_ids"].to(device), torch.tensor(batch["prompt_len"], device=device))
        _, indices = (z @ train_z.t()).topk(k, dim=1)
        for i in range(z.size(0)):
            target = batch["required_slots"][i, 0].clamp(min=0).item()
            hits += int((train_slots[indices[i]] == target).any())
            total += 1
    model.train()
    return hits / max(total, 1)


@torch.no_grad()
def compute_recall(slots: torch.Tensor, required: torch.Tensor,
                   k_values: Tuple[int, ...] = (1, 8, 32)) -> Dict[str, float]:
    """Compute recall@k from retrieved slots.

    slots: [B, top_k] retrieved slot IDs.
    required: [B, max_req] with -1 padding.
    """
    results = {}
    B = slots.size(0)
    for k in k_values:
        hits = 0
        total = 0
        for i in range(B):
            req = set(int(s) for s in required[i] if int(s) >= 0)
            if not req:
                continue
            total += 1
            ret = set(int(s) for s in slots[i, :k])
            if req & ret:
                hits += 1
        results[f"recall_at_{k}"] = hits / max(total, 1)
    return results


def info_nce_loss(query: torch.Tensor, positive_idx: torch.Tensor,
                  memory: ProductKeyMemory, num_negatives: int) -> torch.Tensor:
    """InfoNCE contrastive loss for retrieval training.

    query: [B, 2*key_dim] encoder output.
    positive_idx: [B] positive slot IDs (first required slot).
    memory: product-key memory (provides key tables for scoring and live slot info).
    num_negatives: number of negative slots to sample.

    CRITICAL: Negatives are sampled from LIVE slots only, not the full PKM address
    space. Otherwise >99% of negatives are dead slots and the model never learns
    to distinguish between live candidates.
    """
    B = query.size(0)
    device = query.device

    # Get live slot IDs (slot_value_token >= 0 means slot is populated)
    live_mask = memory.slot_value_token >= 0
    live_slots = live_mask.nonzero(as_tuple=False).flatten()
    num_live = live_slots.numel()

    if num_live < 2:
        # Fallback: not enough live slots
        return torch.tensor(0.0, device=device, requires_grad=True)

    N = min(num_negatives, num_live - 1)
    N = max(1, N)

    # Sample negatives from live slots, avoiding the positive
    neg_indices = torch.randint(0, num_live, (B, N + B), device=device)
    neg_slots = live_slots[neg_indices]  # [B, N + B]

    # Remove accidental positives (replace with another live slot)
    for i in range(B):
        bad = (neg_slots[i] == positive_idx[i].clamp(min=0)).nonzero(as_tuple=True)[0]
        if bad.numel() > 0:
            replacements = torch.randint(0, num_live, (bad.numel(),), device=device)
            neg_slots[i, bad] = live_slots[replacements]

    neg_slots = neg_slots[:, :N]  # [B, N]

    # Build candidate matrix: positives + negatives
    pos = positive_idx.clamp(min=0)
    candidates = torch.cat([pos.unsqueeze(1), neg_slots], dim=1)  # [B, 1+N]

    # Score all candidates using PKM additive scoring
    scores = memory.score_slots(query, candidates)  # [B, 1+N]

    # InfoNCE: positive is index 0
    labels = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(scores, labels)


def subkey_loss(query: torch.Tensor, positive_idx: torch.Tensor,
                memory) -> torch.Tensor:
    """Auxiliary loss: predict k1 and k2 subkey indices directly.

    slot_id = k1 * num_subkeys + k2
    This loss teaches the PKM to assign the correct subkey halves,
    which improves candidate generation quality.
    """
    B = query.size(0)
    key_dim = memory.key_dim
    num_subkeys = memory.num_subkeys
    q1 = query[:, :key_dim]
    q2 = query[:, key_dim:]

    pos = positive_idx.clamp(min=0)
    k1_target = pos // num_subkeys
    k2_target = pos % num_subkeys

    s1 = q1 @ memory.K1.t()  # [B, num_subkeys]
    s2 = q2 @ memory.K2.t()

    loss_k1 = F.cross_entropy(s1, k1_target)
    loss_k2 = F.cross_entropy(s2, k2_target)
    return loss_k1 + loss_k2


def margin_loss(query: torch.Tensor, positive_idx: torch.Tensor,
                memory, margin: float = 0.2, num_hard: int = 16) -> torch.Tensor:
    """Margin loss: push positive score above hard negative scores.

    Selects top-k wrong candidates and applies:
    L = max(0, margin - score_pos + score_neg)
    """
    B = query.size(0)
    device = query.device
    num_subkeys = memory.num_subkeys

    # Get live slots for hard negative mining
    live_mask = memory.slot_value_token >= 0
    live_slots = live_mask.nonzero(as_tuple=False).flatten()
    num_live = live_slots.numel()

    if num_live < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pos = positive_idx.clamp(min=0)

    # Sample candidates from live slots
    N = min(num_hard * 3, num_live)
    neg_indices = torch.randint(0, num_live, (B, N), device=device)
    neg_slots = live_slots[neg_indices]

    # Remove positives
    for i in range(B):
        bad = (neg_slots[i] == pos[i]).nonzero(as_tuple=True)[0]
        if bad.numel() > 0:
            rand = torch.randint(0, num_live, (bad.numel(),), device=device)
            neg_slots[i, bad] = live_slots[rand]

    # Score all candidates
    candidates = torch.cat([pos.unsqueeze(1), neg_slots], dim=1)
    scores = memory.score_slots(query, candidates)  # [B, 1+N]

    pos_score = scores[:, 0]  # [B]
    neg_scores = scores[:, 1:]  # [B, N]

    # Hard negative mining: take top-k highest-scoring negatives
    hard_neg, _ = neg_scores.topk(min(num_hard, N), dim=1)  # [B, num_hard]

    # Margin loss: max(0, margin - pos_score + neg_score)
    losses = torch.clamp(margin - pos_score.unsqueeze(1) + hard_neg, min=0)
    return losses.mean()


def retrieval_diagnostics(query: torch.Tensor, positive_idx: torch.Tensor,
                          memory, k: int = 32) -> dict:
    """Compute detailed retrieval diagnostics for a batch."""
    B = query.size(0)
    num_subkeys = memory.num_subkeys
    key_dim = memory.key_dim
    device = query.device

    q1 = query[:, :key_dim]
    q2 = query[:, key_dim:]

    pos = positive_idx.clamp(min=0)
    k1_target = pos // num_subkeys
    k2_target = pos % num_subkeys

    # Subkey scores
    s1 = q1 @ memory.K1.t()
    s2 = q2 @ memory.K2.t()

    # Check if required subkeys are in top-a/top-b
    top_a = memory.top_a
    top_b = memory.top_b

    _, top_a_idx = s1.topk(top_a, dim=1)
    _, top_b_idx = s2.topk(top_b, dim=1)

    k1_in_top = (top_a_idx == k1_target.unsqueeze(1)).any(dim=1).float()
    k2_in_top = (top_b_idx == k2_target.unsqueeze(1)).any(dim=1).float()

    # Candidate generation
    cand_scores, cand_ids = memory._candidates(s1, s2)
    pos_in_cand = (cand_ids == pos.unsqueeze(1)).any(dim=1).float()

    # Full retrieval
    slots, scores_out = memory.retrieve_topk(query, k)
    pos_rank = (slots == pos.unsqueeze(1)).float().argmax(dim=1)
    pos_rank[~(slots == pos.unsqueeze(1)).any(dim=1)] = -1

    pos_score = memory.score_slots(query, pos.unsqueeze(1)).squeeze(1)

    # Live candidate rate
    live_mask = memory.slot_value_token >= 0
    num_live = live_mask.sum().item()
    cand_live = live_mask[cand_ids.clamp(min=0, max=memory.total_slots - 1)].float()
    cand_live_rate = cand_live.mean(dim=1)

    # Score statistics
    score_std = scores_out.std(dim=1)
    score_probs = F.softmax(scores_out, dim=1)
    score_entropy = -(score_probs * score_probs.clamp(min=1e-10).log()).sum(dim=1)

    # Distance from negative closest score to positive
    all_scores = memory.score_slots(query, cand_ids)
    top_neg = all_scores.topk(2, dim=1)[0][:, -1]  # second-highest

    return {
        "k1_in_topA": k1_in_top.mean().item(),
        "k2_in_topB": k2_in_top.mean().item(),
        "pos_in_candidates": pos_in_cand.mean().item(),
        "pos_rank_mean": pos_rank[pos_rank >= 0].float().mean().item() if (pos_rank >= 0).any() else -1.0,
        "pos_score_mean": pos_score.mean().item(),
        "top_neg_score_mean": top_neg.mean().item(),
        "score_margin": (pos_score - top_neg).mean().item(),
        "candidate_live_rate": cand_live_rate.mean().item(),
        "score_std_mean": score_std.mean().item(),
        "score_entropy_mean": score_entropy.mean().item(),
    }


def train_retrieval(cfg: Config):
    # --- setup ---
    seed_everything(cfg.get("seed", 42))
    device = _pick_device(cfg.train.get("device", "auto"))
    data_dir = cfg.get("data_dir", "data/synthetic")
    output_dir = cfg.get("output_dir", "experiments/exp_001_pkm_retrieval")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer.from_dir(data_dir)
    mlogger = MetricLogger(output_dir, cfg.get("run_name", "retrieval_1m"))
    mlogger.logger.info("Retrieval training — device=%s data=%s", device, data_dir)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg.to_dict(), f)

    # --- memory ---
    mem_cfg = cfg.model.memory
    memory = ProductKeyMemory(
        num_subkeys=mem_cfg.get("num_subkeys", 1024),
        key_dim=mem_cfg.get("key_dim", 128),
        value_dim=mem_cfg.get("value_dim", 128),
        top_a=mem_cfg.get("top_a", 32),
        top_b=mem_cfg.get("top_b", 32),
        top_k=mem_cfg.get("top_k", 32),
        soft_candidates=mem_cfg.get("soft_candidates", False),
    )
    memory.query_dim = 2 * mem_cfg.key_dim

    # Wire KB
    total_slots = mem_cfg.num_subkeys ** 2
    slot_value_token, num_live = build_kb_tensors(data_dir, total_slots, tokenizer)
    memory.set_slot_value_tokens(slot_value_token)
    mlogger.logger.info("Memory: %d total, %d live slots", total_slots, num_live)

    # --- encoder ---
    enc_cfg = cfg.model.query_encoder
    encoder = QueryEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=enc_cfg.get("d_model", 256),
        n_layers=enc_cfg.get("n_layers", 2),
        n_heads=enc_cfg.get("n_heads", 4),
        d_ff=enc_cfg.get("d_ff", 1024),
        query_dim=2 * mem_cfg.get("key_dim", 128),
        max_seq_len=enc_cfg.get("max_seq_len", 64),
        dropout=enc_cfg.get("dropout", 0.0),
        pad_id=tokenizer.pad,
    )

    model = RetrievalModel(encoder, memory).to(device)
    mlogger.logger.info("Retrieval model: %d params", model.param_count())

    # --- data ---
    t_cfg = cfg.train
    batch_size = t_cfg.get("batch_size", 128)
    train_ds = QADataset(data_dir, "train", tokenizer, kind="qa",
                         open_book=False,
                         max_seq_len=enc_cfg.get("max_seq_len", 64))
    val_ds = QADataset(data_dir, "val", tokenizer, kind="qa",
                       open_book=False,
                       max_seq_len=enc_cfg.get("max_seq_len", 64))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_qa(b, tokenizer.pad),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval.get("batch_size", 128) if hasattr(cfg, 'eval') else 128,
        shuffle=False,
        collate_fn=lambda b: collate_qa(b, tokenizer.pad),
    )

    # --- optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=t_cfg.get("lr", 1e-3),
        weight_decay=t_cfg.get("weight_decay", 0.0),
    )
    epochs = t_cfg.get("epochs", 10)
    max_steps = t_cfg.get("max_steps", None)
    steps_per_epoch = len(train_loader)
    total_steps = max_steps or (epochs * steps_per_epoch)
    warmup = t_cfg.get("warmup_steps", 200)
    scheduler = _cosine_warmup_schedule(optimizer, warmup, total_steps)
    grad_clip = t_cfg.get("grad_clip", 1.0)
    num_negatives = t_cfg.get("num_negatives", 64)
    lambda_subkey = t_cfg.get("lambda_subkey", 0.0)
    lambda_margin = t_cfg.get("lambda_margin", 0.0)
    margin_val = t_cfg.get("margin_val", 0.2)

    # --- loop ---
    log_every = t_cfg.get("log_every", 50)
    eval_every = t_cfg.get("eval_every", 500)
    global_step = 0
    best_recall = 0.0

    mlogger.logger.info("Training: epochs=%d steps=%d batch=%d negatives=%d",
                        epochs, total_steps, batch_size, num_negatives)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            if max_steps and global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            required = batch["required_slots"].to(device)
            prompt_lens = torch.tensor(batch["prompt_len"], device=device)

            q, slots, scores = model(input_ids, prompt_lens)

            # InfoNCE loss: use first required slot as positive
            positive_idx = required[:, 0].clamp(min=0)
            loss = info_nce_loss(q, positive_idx, memory, num_negatives)
            total_loss = loss

            # Optional subkey loss
            subkey_l = torch.tensor(0.0, device=device)
            if lambda_subkey > 0:
                subkey_l = subkey_loss(q, positive_idx, memory)
                total_loss = total_loss + lambda_subkey * subkey_l

            # Optional margin loss
            margin_l = torch.tensor(0.0, device=device)
            if lambda_margin > 0:
                margin_l = margin_loss(q, positive_idx, memory,
                                       margin=margin_val, num_hard=16)
                total_loss = total_loss + lambda_margin * margin_l

            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                recall = compute_recall(slots, required, k_values=(1, 8, 32))
                log_metrics = {
                    "train_loss": loss.item(),
                    "total_loss": total_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    **recall,
                }
                if lambda_subkey > 0:
                    log_metrics["subkey_loss"] = subkey_l.item()
                if lambda_margin > 0:
                    log_metrics["margin_loss"] = margin_l.item()
                # Add diagnostics every 10th log
                if global_step % (log_every * 5) == 0:
                    diag = retrieval_diagnostics(q, positive_idx, memory, k=32)
                    log_metrics.update({f"diag_{k}": v for k, v in diag.items()})
                mlogger.log(global_step, log_metrics)

            if global_step % eval_every == 0:
                val_recall = _evaluate_recall(model, val_loader, device)
                avg_recall = val_recall.get("recall_at_8", 0.0)
                mlogger.log(global_step, {f"val_{k}": v for k, v in val_recall.items()})
                if avg_recall > best_recall:
                    best_recall = avg_recall
                    _save_checkpoint(model, optimizer, epoch, global_step,
                                     os.path.join(output_dir, "checkpoint_best.pt"),
                                     extra={"val_recall": val_recall})
                    mlogger.logger.info("New best: recall@8=%.4f", avg_recall)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        mlogger.logger.info("Epoch %d/%d — avg_loss=%.4f lr=%.6f",
                            epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0])

        # Always evaluate recall at end of epoch
        val_recall = _evaluate_recall(model, val_loader, device)
        avg_recall = val_recall.get("recall_at_8", 0.0)
        log_rec = {f"val_{k}": v for k, v in val_recall.items()}
        log_rec["epoch"] = epoch + 1
        mlogger.log(global_step, log_rec)
        if avg_recall > best_recall:
            best_recall = avg_recall
            _save_checkpoint(model, optimizer, epoch, global_step,
                             os.path.join(output_dir, "checkpoint_best.pt"),
                             extra={"val_recall": val_recall})
            mlogger.logger.info("New best (epoch end): recall@8=%.4f", avg_recall)

    # --- final ---
    _save_checkpoint(model, optimizer, epoch, global_step,
                     os.path.join(output_dir, "checkpoint_last.pt"))

    final_recall = _evaluate_recall(model, val_loader, device)
    summary = {
        "run_name": cfg.get("run_name", "retrieval_1m"),
        "data_dir": data_dir,
        "param_count": model.param_count(),
        "num_live_slots": num_live,
        "best_recall_at_8": best_recall,
        "final_recall_at_8": final_recall.get("recall_at_8", 0.0),
        "final_recall_at_32": final_recall.get("recall_at_32", 0.0),
        "final_epoch": epoch + 1,
        "total_steps": global_step,
    }
    mlogger.save_summary(summary)
    mlogger.logger.info("Training complete. Best recall@8=%.4f", best_recall)

    # Gate 1 check
    if best_recall >= 0.80:
        mlogger.logger.info("GATE 1 PASSED: Recall@8 >= 0.80")
    else:
        mlogger.logger.info("GATE 1 FAILED: Recall@8=%.4f < 0.80. Improve retrieval before SAM end-to-end.",
                            best_recall)

    return model, summary


@torch.no_grad()
def _evaluate_recall(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_hits: Dict[int, int] = {k: 0 for k in (1, 8, 32)}
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        required = batch["required_slots"].to(device)
        prompt_lens = torch.tensor(batch["prompt_len"], device=device)

        q, slots, scores = model(input_ids, prompt_lens)
        B = required.size(0)
        for i in range(B):
            req = set(int(s) for s in required[i] if int(s) >= 0)
            if not req:
                continue
            total += 1
            ret = set(int(s) for s in slots[i])
            for k in (1, 8, 32):
                if req & set(list(ret)[:k]):
                    all_hits[k] += 1
    model.train()
    return {f"recall_at_{k}": all_hits[k] / max(total, 1) for k in (1, 8, 32)}


def _save_checkpoint(model, optimizer, epoch, step, path, extra=None):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


class DualEncoderRetriever(nn.Module):
    """Dual encoder: query_encoder + slot_emb, cosine similarity."""
    def __init__(self, query_encoder: QueryEncoder, slot_dim: int, num_slots: int):
        super().__init__()
        self.query_encoder = query_encoder
        self.slot_emb = nn.Embedding(num_slots, slot_dim)
        nn.init.normal_(self.slot_emb.weight, std=0.02)
        self.query_proj = nn.Linear(query_encoder.query_dim, slot_dim, bias=False)

    def forward(self, input_ids, prompt_lens):
        q = self.query_encoder(input_ids, prompt_lens)
        q = F.normalize(self.query_proj(q), dim=-1)
        s = F.normalize(self.slot_emb.weight, dim=-1)
        return q, s

    def param_count(self): return sum(p.numel() for p in self.parameters())
    @property
    def max_seq_len(self): return self.query_encoder.max_seq_len


def dual_encoder_loss_fn(q, s, slot_ids, temp=0.07):
    """InfoNCE: q[i] scores against s[i] (positive) and s[j!=i] (in-batch negatives).
    q, s: [B, D]; slot_ids: [B]."""
    B = q.size(0); device = q.device
    sim = q @ s.t() / temp  # [B, B]
    pos_mask = (slot_ids.unsqueeze(0) == slot_ids.unsqueeze(1)).float()
    exp_sim = sim.exp()
    num = (exp_sim * pos_mask).sum(1) - pos_mask.diagonal()
    denom = exp_sim.sum(1) - exp_sim.diagonal()
    valid = num > 0
    if valid.sum() == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    return -torch.log((num[valid] + 1e-10) / (denom[valid] + 1e-10)).mean()


def train_dual_encoder(cfg):
    seed_everything(cfg.get("seed", 42)); device = _pick_device(cfg.train.get("device", "auto"))
    data_dir = cfg.get("data_dir", "data/synthetic")
    output_dir = cfg.get("output_dir", "experiments/exp_0_5/dual_encoder")
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_dir(data_dir)
    mlogger = MetricLogger(output_dir, cfg.get("run_name", "dual_enc"))
    _, num_live = build_kb_tensors(data_dir, 65536, tokenizer)
    mlogger.logger.info("Dual encoder — %d slots, device=%s", num_live, device)

    ec = cfg.model.query_encoder; sd = cfg.model.get("slot_dim", 256)
    encoder = QueryEncoder(vocab_size=tokenizer.vocab_size, d_model=ec.get("d_model", 256),
                           n_layers=ec.get("n_layers", 3), n_heads=ec.get("n_heads", 4),
                           d_ff=ec.get("d_ff", 1024), query_dim=256,
                           max_seq_len=ec.get("max_seq_len", 64), pad_id=tokenizer.pad)
    model = DualEncoderRetriever(encoder, sd, num_live).to(device)
    mlogger.logger.info("Model: %d params", model.param_count())

    tc = cfg.train; bs = tc.get("batch_size", 128); ms = ec.get("max_seq_len", 64)
    train_ds = QADataset(data_dir, "train", tokenizer, kind="qa", open_book=False, max_seq_len=ms)
    val_ds = QADataset(data_dir, "val", tokenizer, kind="qa", open_book=False, max_seq_len=ms)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=lambda b: collate_qa(b, tokenizer.pad))
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=lambda b: collate_qa(b, tokenizer.pad))
    mlogger.logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    opt = AdamW(model.parameters(), lr=tc.get("lr", 3e-4), weight_decay=tc.get("weight_decay", 1e-4))
    epochs = tc.get("epochs", 15); total_steps = epochs * len(train_ld)
    sched = _cosine_warmup_schedule(opt, tc.get("warmup_steps", 200), total_steps)
    temp = tc.get("temperature", 0.07); log_every = tc.get("log_every", 40)

    gs = 0; best = 0.0
    for ep in range(epochs):
        model.train()
        for batch in train_ld:
            ids = batch["input_ids"].to(device); pl = torch.tensor(batch["prompt_len"], device=device)
            slots = batch["required_slots"][:, 0].clamp(min=0, max=num_live - 1).to(device)
            q, all_s = model(ids, pl)
            # Gather slot embeddings for this batch's positive slots
            s = all_s[slots]  # [B, D]
            loss = dual_encoder_loss_fn(q, s, slots, temp)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); gs += 1
            if gs % log_every == 0:
                r = _evaluate_dual(model, val_ld, device)
                mlogger.log(gs, {"loss": loss.item(), **{f"val_{k}": v for k, v in r.items()}, "lr": sched.get_last_lr()[0]})
                if r.get("recall_at_8", 0) > best:
                    best = r["recall_at_8"]
                    mlogger.logger.info("New best Rec@8=%.4f (step %d)", best, gs)
        mlogger.logger.info("Epoch %d/%d — best Rec@8=%.4f", ep + 1, epochs, best)
    mlogger.save_summary({"run_name": cfg.get("run_name", "dual_enc"), "best_recall_at_8": best, "num_live": num_live})
    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint.pt"))
    mlogger.logger.info("Saved checkpoint")
    return best


@torch.no_grad()
def _evaluate_dual(model, dataloader, device, k_max=32):
    model.eval()
    dummy = torch.zeros(1, 64, dtype=torch.long, device=device)
    _, all_s = model(dummy, torch.tensor([1], device=device))
    hits = {1: 0, 8: 0, 32: 0}; total = 0
    for batch in dataloader:
        q, _ = model(batch["input_ids"].to(device), torch.tensor(batch["prompt_len"], device=device))
        req_slots = batch["required_slots"]
        scores = q @ all_s.t()  # [B, num_slots]
        for i in range(q.size(0)):
            req = set(int(s) for s in req_slots[i] if int(s) >= 0)
            if not req: continue
            total += 1
            _, si = scores[i].topk(k_max)
            for kv in (1, 8, 32):
                if req & set(int(s.item()) for s in si[:kv]): hits[kv] += 1
    model.train()
    return {f"recall_at_{kv}": hits[kv] / max(total, 1) for kv in (1, 8, 32)}


def train_baseline(cfg: Config, retriever_type: str):
    """Train a classifier or cosine baseline retriever."""
    seed_everything(cfg.get("seed", 42))
    device = _pick_device(cfg.train.get("device", "auto"))
    data_dir = cfg.get("data_dir", "data/synthetic")
    output_dir = cfg.get("output_dir", "experiments/baseline")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer.from_dir(data_dir)
    run_name = cfg.get("run_name", f"{retriever_type}_baseline")
    mlogger = MetricLogger(output_dir, run_name)
    mlogger.logger.info("%s training — device=%s data=%s", retriever_type, device, data_dir)

    # Get live slot count
    slot_value_token, num_live = build_kb_tensors(data_dir, 65536, tokenizer)
    mlogger.logger.info("Live slots: %d", num_live)

    # Encoder
    ec = cfg.model.query_encoder if hasattr(cfg.model, 'query_encoder') else cfg.model.get("query_encoder", {})
    query_dim = 2 * (cfg.model.get("memory", {}).get("key_dim", 128) if hasattr(cfg.model, 'memory') else 256)
    if retriever_type == "classifier":
        query_dim = 256
    elif retriever_type == "cosine":
        query_dim = 256

    encoder = QueryEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=ec.get("d_model", 256) if isinstance(ec, dict) else getattr(ec, 'd_model', 256),
        n_layers=ec.get("n_layers", 3) if isinstance(ec, dict) else getattr(ec, 'n_layers', 3),
        n_heads=ec.get("n_heads", 4) if isinstance(ec, dict) else getattr(ec, 'n_heads', 4),
        d_ff=ec.get("d_ff", 1024) if isinstance(ec, dict) else getattr(ec, 'd_ff', 1024),
        query_dim=query_dim,
        max_seq_len=ec.get("max_seq_len", 64) if isinstance(ec, dict) else getattr(ec, 'max_seq_len', 64),
        pad_id=tokenizer.pad,
    )

    if retriever_type == "classifier":
        model = ClassifierRetriever(encoder, num_live).to(device)
    elif retriever_type == "cosine":
        slot_dim = cfg.model.get("slot_dim", 256)
        model = CosineRetriever(encoder, num_live, slot_dim).to(device)
    else:
        raise ValueError(retriever_type)

    mlogger.logger.info("Model: %d params", model.param_count())

    # Data
    t_cfg = cfg.train
    batch_size = t_cfg.get("batch_size", 64)
    max_seq = ec.get("max_seq_len", 64) if isinstance(ec, dict) else getattr(ec, 'max_seq_len', 64)
    train_ds = QADataset(data_dir, "train", tokenizer, kind="qa", open_book=False, max_seq_len=max_seq)
    val_ds = QADataset(data_dir, "val", tokenizer, kind="qa", open_book=False, max_seq_len=max_seq)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_qa(b, tokenizer.pad))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_qa(b, tokenizer.pad))
    mlogger.logger.info("Train examples: %d, Val: %d", len(train_ds), len(val_ds))

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=t_cfg.get("lr", 1e-3), weight_decay=t_cfg.get("weight_decay", 0.0))
    epochs = t_cfg.get("epochs", 20)
    total_steps = epochs * len(train_loader)
    warmup = t_cfg.get("warmup_steps", 100)
    scheduler = _cosine_warmup_schedule(optimizer, warmup, total_steps)
    grad_clip = t_cfg.get("grad_clip", 1.0)

    # Training loop
    global_step = 0
    best_recall = 0.0
    log_every = t_cfg.get("log_every", 20)
    eval_every = t_cfg.get("eval_every", 50)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            required = batch["required_slots"].to(device)
            prompt_lens = torch.tensor(batch["prompt_len"], device=device)

            logits = model(input_ids, prompt_lens)  # [B, num_live]
            # Target: first required slot (mapped to compact index)
            # For classifier/cosine, the output indices are 0..num_live-1
            # We need to map the original required slot IDs to compact indices
            target = required[:, 0].clamp(min=0, max=num_live - 1)
            # If slots are remapped compactly, they should already be in 0..num_live-1
            # But dataset uses original slot IDs (0..4780). Just clamp.
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                slots, _ = model.retrieve_topk(logits, k=32)
                recall = compute_recall(slots, required, k_values=(1, 8, 32))
                mlogger.log(global_step, {"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0], **recall})

            if global_step % eval_every == 0 or global_step == total_steps:
                val_recall = _evaluate_baseline_recall(model, val_loader, device)
                avg = val_recall.get("recall_at_8", 0)
                mlogger.log(global_step, {f"val_{k}": v for k, v in val_recall.items()})
                if avg > best_recall:
                    best_recall = avg
                    mlogger.logger.info("New best: recall@8=%.4f (step %d)", avg, global_step)

        mlogger.logger.info("Epoch %d/%d — avg_loss=%.4f", epoch + 1, epochs,
                            epoch_loss / max(len(train_loader), 1))

    summary = {
        "run_name": run_name, "data_dir": data_dir, "retriever_type": retriever_type,
        "best_recall_at_8": best_recall, "num_live": num_live,
        "train_examples": len(train_ds), "val_examples": len(val_ds),
        "total_steps": global_step,
    }
    mlogger.save_summary(summary)
    mlogger.logger.info("Best recall@8=%.4f", best_recall)


@torch.no_grad()
def _evaluate_baseline_recall(model, dataloader, device):
    model.eval()
    hits = {1: 0, 8: 0, 32: 0}
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        required = batch["required_slots"].to(device)
        prompt_lens = torch.tensor(batch["prompt_len"], device=device)
        logits = model(input_ids, prompt_lens)
        slots, _ = model.retrieve_topk(logits, k=32)
        for i in range(required.size(0)):
            req = set(int(s) for s in required[i] if int(s) >= 0)
            if not req:
                continue
            total += 1
            ret = set(int(s) for s in slots[i])
            for k in (1, 8, 32):
                if req & set(list(ret)[:k]):
                    hits[k] += 1
    model.train()
    return {f"recall_at_{k}": hits[k] / max(total, 1) for k in (1, 8, 32)}


def main():
    ap = argparse.ArgumentParser(description="Train product-key memory retrieval.")
    ap.add_argument("--config", default="configs/retrieval_1m.yaml")
    ap.add_argument("--override", nargs="*", default=None)
    args = ap.parse_args()

    overrides = {}
    if args.override:
        for ov in args.override:
            k, v = ov.split("=")
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
            overrides[k] = v

    cfg = load_config(args.config, overrides)
    retriever_type = cfg.model.get("retriever_type", cfg.model.get("type", "retrieval"))
    if retriever_type in ("classifier", "cosine"):
        train_baseline(cfg, retriever_type)
    elif retriever_type == "contrastive":
        train_contrastive(cfg)
    elif retriever_type == "dual_encoder":
        train_dual_encoder(cfg)
    else:
        train_retrieval(cfg)


if __name__ == "__main__":
    main()
