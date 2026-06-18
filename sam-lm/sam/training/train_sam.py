"""SAM model training with memory modes.

Modes:
  core_only                          — no memory (capacity floor)
  oracle_memory                      — inject correct required slot values (upper bound)
  retrieved_memory                   — learned product-key retrieval (the real SAM)
  random_memory                      — inject random live slot values (placebo control)
  oracle_text_memory                 — inject oracle text into input (text upper bound)
  retrieved_memory_external_text_query — query dual encoder with raw question text
  retrieved_memory_hidden_adapter    — train adapter from hidden state to query space
  train_memory_adapter               — pretrain adapter to match dual encoder queries

Usage:
    python -m sam.training.train_sam --mode core_only --config configs/sam_tiny.yaml
    python -m sam.training.train_sam --mode retrieved_memory_external_text_query --config configs/sam_retrieved_external_text_dense.yaml
    python -m sam.training.train_sam --mode train_memory_adapter --config configs/sam_memory_adapter_dense.yaml
    python -m sam.training.train_sam --mode retrieved_memory_hidden_adapter --config configs/sam_retrieved_hidden_adapter_dense.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..data.dataset import QADataset, Tokenizer, collate_qa, build_kb_tensors
from ..model.sam_core import SamModel, MEMORY_MODES
from ..utils.config import load_config, Config
from ..utils.seed import seed_everything
from ..utils.logging import MetricLogger
from ..eval.metrics import accuracy_by_hop, recall_at_k


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


def _save_checkpoint(model, optimizer, epoch, step, path, extra=None):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "param_count": model.param_count(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def _load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state, strict=False)
    return model


def contrastive_retrieval_loss(
    aux: Dict,
    required_slots: torch.Tensor,
    num_subkeys: int,
    num_negatives: int = 64,
) -> Optional[torch.Tensor]:
    """Optional InfoNCE loss that pushes the memory query toward required slots.

    Uses the first memory layer's query and the PKM scoring function.
    Only active when aux contains 'primary_query'.
    """
    if "primary_query" not in aux or "retrieved_slots" not in aux:
        return None

    query = aux["primary_query"]  # [B, 2*key_dim]
    B = query.size(0)
    device = query.device

    # Positive: first required slot
    positive = required_slots[:, 0].clamp(min=0)  # [B]

    # Negative: random other live slots
    N = min(num_negatives, 255)  # reasonable cap
    neg = torch.randint(0, num_subkeys * num_subkeys, (B, N), device=device)

    candidates = torch.cat([positive.unsqueeze(1), neg], dim=1)  # [B, 1+N]

    # We need the PKM to score these slots. Access through model.pkm
    # This function receives num_subkeys to compute k1, k2
    k1 = candidates // num_subkeys
    k2 = candidates % num_subkeys

    # Decompose query into q1, q2
    key_dim = query.size(-1) // 2
    q1 = query[:, :key_dim]  # [B, key_dim]
    q2 = query[:, key_dim:]  # [B, key_dim]

    # Score using the key tables (passed via aux or model)
    # We assume the PKM key tables are accessible. For standalone use,
    # this is wired through a closure. For simplicity, we return None
    # if we can't access key tables — the main LM loss is sufficient.
    return None  # LM loss + retrieval handled jointly by SAM forward pass


def train_sam(cfg: Config, mode: str):
    """Train SAM model in the specified memory mode."""
    assert mode in MEMORY_MODES, f"Unknown mode: {mode}. Choose from {MEMORY_MODES}"

    # --- setup ---
    seed_everything(cfg.get("seed", 42))
    device = _pick_device(cfg.train.get("device", "auto"))
    data_dir = cfg.get("data_dir", "data/synthetic")

    # train_memory_adapter uses its own output_dir
    if mode == "train_memory_adapter":
        output_dir = cfg.get("output_dir", "experiments/exp_sam/memory_adapter")
    else:
        output_dir = os.path.join(cfg.get("output_dir", "experiments/exp_sam"), mode)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer.from_dir(data_dir)
    run_name = f"{cfg.get('run_name', 'sam_tiny')}_{mode}"
    mlogger = MetricLogger(output_dir, run_name)
    mlogger.logger.info("SAM training — mode=%s device=%s data=%s", mode, device, data_dir)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg.to_dict(), f)

    # --- model ---
    m_cfg = cfg.model
    memory_cfg = m_cfg.get("memory", {}).to_dict() if hasattr(m_cfg.get("memory", {}), 'to_dict') else dict(m_cfg.get("memory", {}))
    model = SamModel(
        vocab_size=tokenizer.vocab_size,
        d_model=m_cfg.get("d_model", 512),
        n_layers=m_cfg.get("n_layers", 6),
        n_heads=m_cfg.get("n_heads", 8),
        d_ff=m_cfg.get("d_ff", 2048),
        dropout=m_cfg.get("dropout", 0.0),
        max_seq_len=m_cfg.get("max_seq_len", 128),
        memory_every=m_cfg.get("memory_every", 3),
        memory_query=m_cfg.get("memory_query", "tokenwise"),
        memory_integration=m_cfg.get("memory_integration", "gated_sum"),
        memory_cfg=memory_cfg,
        pad_id=tokenizer.pad,
    )
    model.memory_mode = mode
    model._aggregation_mode = cfg.get("memory_aggregation_mode",
                                       memory_cfg.get("aggregation_mode", "uniform_mean"))
    model._aggregation_temperature = float(cfg.get("memory_score_temperature",
                                                    memory_cfg.get("aggregation_temperature", 0.1)))
    # Experiment 0.10: Threshold/margin selection parameters
    if cfg.get("aggregation_threshold") is not None:
        model._aggregation_threshold = float(cfg.get("aggregation_threshold"))
    if cfg.get("aggregation_top_n") is not None:
        model._aggregation_top_n = int(cfg.get("aggregation_top_n"))
    if cfg.get("aggregation_delta") is not None:
        model._aggregation_delta = float(cfg.get("aggregation_delta"))
    if cfg.get("aggregation_mass_p") is not None:
        model._aggregation_mass_p = float(cfg.get("aggregation_mass_p"))

    # Experiment 0.13A: Controlled noisy memory mode
    if cfg.get("memory_noise_mode") is not None:
        model._memory_noise_mode = cfg.get("memory_noise_mode")
    if cfg.get("num_distractors") is not None:
        model._num_distractors = int(cfg.get("num_distractors"))
    if cfg.get("distractor_score") is not None:
        model._distractor_score = float(cfg.get("distractor_score"))
    if cfg.get("memory_integration_mode") is not None:
        model._integration_mode = cfg.get("memory_integration_mode")

    # Set retrieval topK
    model._retrieval_k = cfg.get("topK", 8)

    # Wire KB and retriever
    n_subkeys = memory_cfg.get("num_subkeys", 1024)
    total_slots = n_subkeys * n_subkeys
    slot_value_token, num_live = build_kb_tensors(data_dir, total_slots, tokenizer)

    # Wire dual encoder retriever for all retrieved modes
    retrieved_modes = ("retrieved_memory", "retrieved_memory_external_text_query",
                       "retrieved_memory_hidden_adapter", "train_memory_adapter")
    retriever = None
    if mode in retrieved_modes and cfg.get("retriever_backend") == "dual_encoder":
        r_ckpt = cfg.get("retriever_checkpoint")
        if r_ckpt and os.path.exists(r_ckpt):
            from ..model.sam_core import DualEncoderWrapper
            retriever = DualEncoderWrapper(r_ckpt, tokenizer, device)
            mlogger.logger.info("Dual encoder retriever loaded from %s", r_ckpt)
        else:
            mlogger.logger.warning("Dual encoder checkpoint not found: %s", r_ckpt)
    elif mode in retrieved_modes and cfg.get("retriever_backend") == "chain_set":
        r_ckpt = cfg.get("retriever_checkpoint")
        if r_ckpt and os.path.exists(r_ckpt):
            from ..model.sam_core import ChainSetRetrieverWrapper
            retriever = ChainSetRetrieverWrapper(r_ckpt, tokenizer, device)
            mlogger.logger.info("Chain-set retriever loaded from %s", r_ckpt)
        else:
            mlogger.logger.warning("Chain-set checkpoint not found: %s", r_ckpt)

    model.set_kb(slot_value_token, retriever=retriever)
    # Set tokenizer for multi-query mode
    if mode == "retrieved_multi_query_union":
        model.set_tokenizer(tokenizer)

    # Load pretrained SAM core for adapter-only training (after KB setup)
    if mode == "train_memory_adapter":
        core_ckpt = cfg.get("core_checkpoint")
        if core_ckpt and os.path.exists(core_ckpt):
            ckpt_state = torch.load(core_ckpt, map_location=device, weights_only=False)
            ms = ckpt_state.get("model_state", ckpt_state)
            ms.pop("live_slot_ids", None)
            ms.pop("pkm.slot_value_token", None)
            model.load_state_dict(ms, strict=False)
            mlogger.logger.info("Loaded SAM core from %s", core_ckpt)

        # Freeze SAM core, only train adapter
        for name, param in model.named_parameters():
            if "memory_query_adapter" not in name:
                param.requires_grad = False
        model._adapter_frozen = True
        mlogger.logger.info("Froze SAM core; only adapter is trainable")

    model.to(device)

    pc = model.param_count()
    core = model.core_active_param_count()
    mem = model.memory_param_count()
    mlogger.logger.info("Model: %d total (%d core + %d memory) params, %d live slots",
                        pc, core, mem, num_live)

    # Experiment 0.12: Curriculum training
    curriculum_cfg = cfg.get("memory_curriculum", None)
    curriculum_stages = []
    if curriculum_cfg is not None and curriculum_cfg.get("enabled", False):
        curriculum_stages = curriculum_cfg.get("stages", [])
        mlogger.logger.info("Curriculum training enabled: %d stages", len(curriculum_stages))
        for stage in curriculum_stages:
            mlogger.logger.info("  Stage: %s (%d epochs)", stage.get("name", "unknown"), stage.get("epochs", 0))

    # --- data ---
    t_cfg = cfg.train
    batch_size = t_cfg.get("batch_size", 64)
    oracle_text = (mode == "oracle_text_memory")
    # For new retrieved modes, forward mode is the mode itself (not core_only)
    if mode in ("retrieved_memory_external_text_query", "retrieved_memory_hidden_adapter",
                "train_memory_adapter", "retrieved_oracle_slots",
                "retrieved_multi_query_union"):
        forward_mode = mode
    else:
        forward_mode = "core_only" if oracle_text else mode
    train_ds = QADataset(data_dir, "train", tokenizer, kind="qa",
                         open_book=False, oracle_text=oracle_text,
                         max_seq_len=m_cfg.get("max_seq_len", 128))
    val_ds = QADataset(data_dir, "val", tokenizer, kind="qa",
                       open_book=False, oracle_text=oracle_text,
                       max_seq_len=m_cfg.get("max_seq_len", 128))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_qa(b, tokenizer.pad),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval.get("batch_size", 128), shuffle=False,
        collate_fn=lambda b: collate_qa(b, tokenizer.pad),
    )

    # --- optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=t_cfg.get("lr", 3e-4),
        weight_decay=t_cfg.get("weight_decay", 0.01),
    )
    epochs = t_cfg.get("epochs", 8)
    max_steps = t_cfg.get("max_steps", None)
    steps_per_epoch = len(train_loader)
    total_steps = max_steps or (epochs * steps_per_epoch)
    warmup = t_cfg.get("warmup_steps", 200)
    scheduler = _cosine_warmup_schedule(optimizer, warmup, total_steps)
    grad_clip = t_cfg.get("grad_clip", 1.0)
    lambda_contrastive = t_cfg.get("lambda_contrastive", 0.5)

    # --- loop ---
    log_every = t_cfg.get("log_every", 50)
    eval_every = t_cfg.get("eval_every", 500)
    global_step = 0
    best_val_loss = float("inf")
    epoch_offset = 0  # for curriculum tracking

    # Experiment 0.12: Curriculum stages
    if curriculum_stages:
        total_curriculum_epochs = sum(s.get("epochs", 0) for s in curriculum_stages)
        mlogger.logger.info("Training: curriculum=%d stages total_epochs=%d steps_per_epoch=%d warmup=%d batch=%d",
                            len(curriculum_stages), total_curriculum_epochs, steps_per_epoch, warmup, batch_size)
    else:
        mlogger.logger.info("Training: epochs=%d steps=%d warmup=%d batch=%d",
                            epochs, total_steps, warmup, batch_size)

    # Compute total curriculum steps
    if curriculum_stages:
        total_curriculum_steps = sum(s.get("epochs", 0) * steps_per_epoch for s in curriculum_stages)
    else:
        total_curriculum_steps = total_steps

    for epoch in range(epochs):
        # Apply curriculum stage
        current_curriculum_stage = None
        if curriculum_stages:
            cumulative = 0
            for stage in curriculum_stages:
                cumulative += stage.get("epochs", 0)
                if epoch < cumulative:
                    current_curriculum_stage = stage.get("name", "unknown")
                    break
            if current_curriculum_stage is None:
                current_curriculum_stage = curriculum_stages[-1].get("name", "unknown")
            model._curriculum_stage = current_curriculum_stage
            # Update retrieval_k for curriculum stages that use chain_set
            if current_curriculum_stage == "chain_set_top32":
                model._retrieval_k = 32
            elif current_curriculum_stage == "chain_set_top64":
                model._retrieval_k = 64
            else:
                # For oracle stages, use enough slots for required + distractors
                model._retrieval_k = 32  # generous default
            
            if epoch == 0 or (curriculum_stages and current_curriculum_stage != prev_stage):
                mlogger.logger.info("Curriculum stage: %s (epoch %d)", 
                                    current_curriculum_stage, epoch + 1)
            prev_stage = getattr(model, '_curriculum_stage', None) if epoch > 0 else current_curriculum_stage

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            if max_steps and global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            required_slots = batch["required_slots"].to(device)
            prompt_lens = torch.tensor(batch["prompt_len"], device=device)

            # Set batch metadata for multi-query mode and fixed_top_by_hop aggregation
            if mode in ("retrieved_multi_query_union", "retrieved_memory_external_text_query"):
                model._batch_task_types = batch["task_type"]
                model._batch_hops = batch["hops"]

            _, loss, aux = model(
                input_ids, labels=labels,
                required_slots=required_slots,
                prompt_lens=prompt_lens,
                mode=forward_mode,
            )

            if loss is None:
                continue

            # Optional InfoNCE contrastive loss on retrieved slots
            total_loss = loss
            aux_loss_value = 0.0
            adapter_cos_value = 0.0
            selector_loss_value = 0.0

            # Experiment 0.12: Selector loss
            sel_diag_metrics = {}
            if mode == "retrieved_memory_external_text_query" and "selector_logits" in aux:
                sel_logits = aux["selector_logits"]
                sel_target = aux.get("selector_target")
                if sel_target is not None:
                    pos_w = float(cfg.get("selector_positive_weight", 8.0))
                    sel_loss = F.binary_cross_entropy_with_logits(
                        sel_logits, sel_target,
                        pos_weight=torch.tensor([pos_w], device=device),
                    )
                    sel_weight = float(cfg.get("selector_loss_weight", 1.0))
                    total_loss = total_loss + sel_weight * sel_loss
                    selector_loss_value = sel_loss.item()
                    # Compute diagnostics (no_grad)
                    with torch.no_grad():
                        sel_probs = aux.get("selector_probs",
                            torch.sigmoid(sel_logits))
                        sel_mask = aux.get("selector_mask",
                            (sel_probs >= float(cfg.get("selector_threshold", 0.5))).float())
                        # Simple precision/recall from target
                        pred_pos = (sel_probs >= 0.5).float()
                        tp = ((pred_pos == 1) & (sel_target == 1)).float().sum()
                        fp = ((pred_pos == 1) & (sel_target == 0)).float().sum()
                        fn = ((pred_pos == 0) & (sel_target == 1)).float().sum()
                        precision = tp / (tp + fp).clamp(min=1)
                        recall = tp / (tp + fn).clamp(min=1)
                        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
                        sel_diag_metrics = {
                            "sel_precision": precision.item(),
                            "sel_recall": recall.item(),
                            "sel_f1": f1.item(),
                            "sel_selected": (pred_pos.sum() / pred_pos.size(0)).item(),
                        }

            # Adapter alignment loss for hidden_adapter / train_memory_adapter modes
            lambda_adapter = t_cfg.get("lambda_adapter", 1.0)
            lambda_retrieval = t_cfg.get("lambda_retrieval", 0.5)
            adapter_modes = ("retrieved_memory_hidden_adapter", "train_memory_adapter")
            if mode in adapter_modes and "adapter_query" in aux:
                # Cosine alignment: push adapter output toward teacher query
                if hasattr(model, '_retriever') and model._retriever is not None:
                    teacher_q = model._retriever.encode_text(input_ids, prompt_lens)
                    student_q = aux["adapter_query"]
                    cos_sim = F.cosine_similarity(student_q, teacher_q, dim=-1)
                    adapter_loss = (1.0 - cos_sim).mean()
                    adapter_cos_value = cos_sim.mean().item()

                    # For train_memory_adapter, only use adapter loss
                    if mode == "train_memory_adapter":
                        total_loss = lambda_adapter * adapter_loss
                    else:
                        total_loss = loss + lambda_adapter * adapter_loss

                    # Optional retrieval InfoNCE: push adapter query toward correct slots
                    if lambda_retrieval > 0 and "adapter_query" in aux:
                        student_q = aux["adapter_query"]
                        s_frozen = F.normalize(model._slot_emb_frozen.to(device), dim=-1)
                        full_scores = student_q @ s_frozen.t()  # [B, num_slots]
                        pos_slot = required_slots[:, 0].clamp(min=0)
                        retrieval_loss = F.cross_entropy(full_scores, pos_slot)
                        total_loss = total_loss + lambda_retrieval * retrieval_loss

            if mode == "retrieved_memory" and lambda_contrastive > 0:
                if "primary_query" in aux and "retrieved_slots" in aux:
                    # InfoNCE with live-slot negatives
                    query = aux["primary_query"]
                    B = query.size(0)
                    key_dim = model.key_dim
                    q1 = query[:, :key_dim]
                    q2 = query[:, key_dim:]

                    # Get live slot IDs for negative sampling
                    live_mask = model.pkm.slot_value_token >= 0
                    live_slots = live_mask.nonzero(as_tuple=False).flatten()
                    num_live = live_slots.numel()

                    pos = required_slots[:, 0].clamp(min=0)

                    if num_live > 1:
                        neg_count = min(64, num_live - 1)
                        neg_count = max(1, neg_count)
                        neg_indices = torch.randint(0, num_live, (B, neg_count + B), device=device)
                        neg_slots = live_slots[neg_indices]
                        # Remove accidental positives
                        for i in range(B):
                            bad = (neg_slots[i] == pos[i]).nonzero(as_tuple=True)[0]
                            if bad.numel() > 0:
                                replacements = torch.randint(0, num_live, (bad.numel(),), device=device)
                                neg_slots[i, bad] = live_slots[replacements]
                        neg_slots = neg_slots[:, :neg_count]

                        candidates = torch.cat([pos.unsqueeze(1), neg_slots], dim=1)
                        k1_cand = candidates // n_subkeys
                        k2_cand = candidates % n_subkeys

                        cand_scores = torch.gather(
                            q1 @ model.pkm.K1.t(), 1, k1_cand) + torch.gather(
                            q2 @ model.pkm.K2.t(), 1, k2_cand)

                        labels_contrast = torch.zeros(B, dtype=torch.long, device=device)
                        aux_loss = F.cross_entropy(cand_scores, labels_contrast)
                        total_loss = loss + lambda_contrastive * aux_loss
                        aux_loss_value = aux_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            log_metrics = {
                "train_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }
            if aux_loss_value > 0:
                log_metrics["contrastive_loss"] = aux_loss_value
            if adapter_cos_value > 0:
                log_metrics["adapter_cos"] = adapter_cos_value
            if selector_loss_value > 0:
                log_metrics["selector_loss"] = selector_loss_value
            if sel_diag_metrics:
                log_metrics.update(sel_diag_metrics)

            if global_step % log_every == 0:
                mlogger.log(global_step, log_metrics)

            if global_step % eval_every == 0:
                val_loss = _evaluate_loss(model, val_loader, device, forward_mode)
                mlogger.log(global_step, {"val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(model, optimizer, epoch, global_step,
                                     os.path.join(output_dir, "checkpoint_best.pt"),
                                     extra={"val_loss": val_loss, "mode": mode})
                    mlogger.logger.info("New best: val_loss=%.4f", val_loss)

        # Always evaluate at end of epoch
        val_loss = _evaluate_loss(model, val_loader, device, forward_mode)
        mlogger.log(global_step, {"val_loss": val_loss, "epoch": epoch + 1})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, global_step,
                             os.path.join(output_dir, "checkpoint_best.pt"),
                             extra={"val_loss": val_loss, "mode": mode})
            mlogger.logger.info("New best (epoch end): val_loss=%.4f", val_loss)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        mlogger.logger.info("Epoch %d/%d — avg_loss=%.4f lr=%.6f",
                            epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0])

    # --- final ---
    _save_checkpoint(model, optimizer, epoch, global_step,
                     os.path.join(output_dir, "checkpoint_last.pt"))

    # Final validation eval
    model.eval()
    acc = accuracy_by_hop(model, val_loader, tokenizer,
                          max_new_tokens=cfg.eval.get("max_new_tokens", 6),
                          mode=forward_mode, device=device)
    recall = {}
    eval_mode = mode if mode in ("retrieved_memory", "retrieved_memory_external_text_query",
                                  "retrieved_memory_hidden_adapter") else None
    if eval_mode:
        recall = recall_at_k(model, val_loader, tokenizer,
                            k_values=(1, 8, 32), device=device, mode=eval_mode)

    # Compute adapter cosine alignment on val set
    adapter_cos_val = 0.0
    if mode in ("retrieved_memory_hidden_adapter", "train_memory_adapter"):
        if hasattr(model, '_retriever') and model._retriever is not None:
            cos_vals = []
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                pl = torch.tensor(batch["prompt_len"], device=device)
                teacher_q = model._retriever.encode_text(ids, pl)
                _, _, aux_batch = model(ids, prompt_lens=pl, mode=forward_mode)
                if "adapter_query" in aux_batch:
                    cos = F.cosine_similarity(aux_batch["adapter_query"], teacher_q, dim=-1)
                    cos_vals.append(cos.mean().item())
            if cos_vals:
                adapter_cos_val = sum(cos_vals) / len(cos_vals)

    summary = {
        "run_name": run_name,
        "mode": mode,
        "data_dir": data_dir,
        "best_val_loss": best_val_loss,
        "param_count": model.param_count(),
        "core_active_params": model.core_active_param_count(),
        "memory_params": model.memory_param_count(),
        "num_live_slots": num_live,
        "final_epoch": epoch + 1,
        "total_steps": global_step,
        **{f"val_{k}": v for k, v in acc.items() if not isinstance(v, dict)},
        **{f"val_{k}": v for k, v in recall.items()},
    }
    if adapter_cos_val > 0:
        summary["val_adapter_cosine"] = adapter_cos_val
    mlogger.save_summary(summary)

    mlogger.logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    mlogger.logger.info("Val accuracy: overall=%.4f single=%.4f two=%.4f three=%.4f",
                        acc.get("accuracy_overall", 0),
                        acc.get("accuracy_single_hop", 0),
                        acc.get("accuracy_two_hop", 0),
                        acc.get("accuracy_three_hop", 0))
    if recall:
        mlogger.logger.info("Recall: @1=%.4f @8=%.4f @32=%.4f",
                            recall.get("recall_at_1", 0),
                            recall.get("recall_at_8", 0),
                            recall.get("recall_at_32", 0))

    # Experiment 0.13A: Detailed evaluation with gate/noise diagnostics
    if model._memory_noise_mode == "oracle_plus_distractors" or model._integration_mode is not None:
        mlogger.logger.info("Running detailed evaluation with gate/noise diagnostics...")
        detailed_metrics, _ = _detailed_evaluate_sam(
            model, val_loader, tokenizer, forward_mode, device, output_dir,
            max_new_tokens=cfg.eval.get("max_new_tokens", 6),
        )
        for k, v in detailed_metrics.items():
            summary[f"detailed_{k}"] = v
        mlogger.logger.info("Detailed acc: overall=%.4f 1-hop=%.4f 2-hop=%.4f 3-hop=%.4f",
                            detailed_metrics.get("accuracy_overall", 0),
                            detailed_metrics.get("accuracy_1_hop", 0),
                            detailed_metrics.get("accuracy_2_hop", 0),
                            detailed_metrics.get("accuracy_3_hop", 0))

    return model, summary


@torch.no_grad()
def _detailed_evaluate_sam(
    model,
    dataloader: DataLoader,
    tokenizer,
    mode: str,
    device: str,
    output_dir: str,
    max_new_tokens: int = 6,
):
    """Enhanced evaluation with gate diagnostics and prediction-level logging.

    Saves:
      experiments/debug/noisy_memory_0_13_metrics.json
      experiments/debug/noisy_memory_0_13_predictions.jsonl
    """
    import json as _json
    model.eval()
    model.to(device)

    correct: Dict[str, int] = {"overall": 0, "1-hop": 0, "2-hop": 0, "3-hop": 0}
    total: Dict[str, int] = {"overall": 0, "1-hop": 0, "2-hop": 0, "3-hop": 0}
    gate_means = []
    gate_mins = []
    gate_maxs = []
    memory_norms = []
    residual_norms = []
    mem_res_ratios = []
    predictions = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        hops = batch["hops"]
        task_types = batch["task_type"]
        prompt_lens = batch["prompt_len"]
        required = batch.get("required_slots", None)
        if required is not None:
            required = required.to(device)

        # Set batch metadata
        if mode in ("retrieved_multi_query_union", "retrieved_memory_external_text_query"):
            if hasattr(model, '_batch_hops'):
                model._batch_task_types = batch["task_type"]
                model._batch_hops = batch["hops"]

        B = input_ids.size(0)
        for i in range(B):
            p_len = prompt_lens[i]
            prompt = input_ids[i, :p_len]
            target_ids = labels[i]
            target_mask = target_ids != -100
            expected = target_ids[target_mask].tolist()

            # Set per-example hops
            if hasattr(model, '_batch_hops'):
                model._batch_hops = [hops[i]]
                if hasattr(model, '_batch_task_types'):
                    model._batch_task_types = [task_types[i]]

            # Forward pass to get aux diagnostics
            pl_tensor = torch.tensor([p_len], device=device)
            req_i = required[i:i+1] if required is not None else None
            _, _, aux = model(
                prompt.unsqueeze(0), labels=None,
                required_slots=req_i, prompt_lens=pl_tensor, mode=mode,
            )

            # Generate prediction
            if mode is not None and mode != "core_only":
                generated = model.generate(
                    prompt, max_new_tokens=max_new_tokens,
                    eos_id=tokenizer.eos, required_slots=req_i, mode=mode,
                )
            else:
                generated = model.generate(
                    prompt, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos,
                )

            pred_text = tokenizer.decode(generated.tolist()).strip().split()[0] if generated.numel() > 0 else ""
            expected_text = tokenizer.decode(expected).strip().split()[0] if expected else ""
            is_correct = (pred_text == expected_text)

            h = int(hops[i])
            hop_key = "1-hop" if h == 1 else "2-hop" if h == 2 else "3-hop"
            correct[hop_key] = correct.get(hop_key, 0) + int(is_correct)
            total[hop_key] = total.get(hop_key, 0) + 1
            correct["overall"] += int(is_correct)
            total["overall"] += 1

            # Collect gate diagnostics
            if "gate_mean" in aux:
                gate_means.append(aux["gate_mean"])
                gate_mins.append(aux["gate_min"])
                gate_maxs.append(aux["gate_max"])
            if "memory_norm" in aux:
                memory_norms.append(aux["memory_norm"])
                residual_norms.append(aux["residual_norm"])
                mem_res_ratios.append(aux["memory_residual_ratio"])

            # Build prediction row
            noisy_info = aux.get("_noisy_memory_info", {})
            ex_info = noisy_info.get(i, {}) if isinstance(noisy_info, dict) else {}
            req_slots = [int(s) for s in required[i] if int(s) >= 0] if required is not None else []
            pred_row = {
                "question": tokenizer.decode(prompt.tolist()),
                "reasoning_hops": h,
                "required_slots": req_slots,
                "distractor_slots": ex_info.get("distractor_slots", []),
                "all_slots_injected": ex_info.get("all_slots_injected", []),
                "num_distractors": ex_info.get("num_distractors", 0),
                "aggregation_mode": getattr(model, '_aggregation_mode', 'unknown'),
                "integration_mode": getattr(model, '_integration_mode', 'gated_sum'),
                "gate_mean": aux.get("gate_mean", -1.0),
                "memory_norm": aux.get("memory_norm", -1.0),
                "memory_residual_ratio": aux.get("memory_residual_ratio", -1.0),
                "prediction": pred_text,
                "gold_answer": expected_text,
                "correct": is_correct,
            }
            predictions.append(pred_row)

    # Compute accuracy metrics
    metrics = {}
    for key in ["overall", "1-hop", "2-hop", "3-hop"]:
        metrics[f"accuracy_{key.replace('-', '_')}"] = correct.get(key, 0) / max(total.get(key, 1), 1)
    if gate_means:
        import statistics
        metrics["gate_mean_avg"] = statistics.mean(gate_means)
        metrics["gate_mean_min"] = min(gate_means)
        metrics["gate_mean_max"] = max(gate_means)
    if memory_norms:
        import statistics
        metrics["memory_norm_avg"] = statistics.mean(memory_norms)
        metrics["residual_norm_avg"] = statistics.mean(residual_norms)
        metrics["memory_residual_ratio_avg"] = statistics.mean(mem_res_ratios)

    # Save diagnostics
    debug_dir = os.path.join("experiments", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    metrics_path = os.path.join(debug_dir, "noisy_memory_0_13_metrics.json")
    with open(metrics_path, "w") as f:
        _json.dump(metrics, f, indent=2)

    preds_path = os.path.join(debug_dir, "noisy_memory_0_13_predictions.jsonl")
    with open(preds_path, "w") as f:
        for pred in predictions:
            f.write(_json.dumps(pred) + "\n")

    # Also save to output_dir for per-run tracking
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "detailed_metrics.json"), "w") as f:
        _json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "detailed_predictions.jsonl"), "w") as f:
        for pred in predictions:
            f.write(_json.dumps(pred) + "\n")

    return metrics, predictions


@torch.no_grad()
def _evaluate_loss(model, dataloader, device, mode):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        required_slots = batch["required_slots"].to(device)
        prompt_lens = torch.tensor(batch["prompt_len"], device=device)

        # Set batch metadata for aggregation modes that need it
        if mode in ("retrieved_multi_query_union", "retrieved_memory_external_text_query"):
            model._batch_task_types = batch["task_type"]
            model._batch_hops = batch["hops"]

        _, loss, _ = model(
            input_ids, labels=labels,
            required_slots=required_slots,
            prompt_lens=prompt_lens,
            mode=mode,
        )
        if loss is not None:
            total_loss += loss.item() * input_ids.size(0)
            n += input_ids.size(0)
    model.train()
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Train SAM model.")
    ap.add_argument("--mode", required=True,
                    choices=list(MEMORY_MODES),
                    help="Memory mode: core_only, oracle_memory, retrieved_memory, random_memory, oracle_text_memory, retrieved_memory_external_text_query, retrieved_memory_hidden_adapter, train_memory_adapter")
    ap.add_argument("--config", default="configs/sam_tiny.yaml",
                    help="Path to YAML config file.")
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
    train_sam(cfg, args.mode)


if __name__ == "__main__":
    main()
