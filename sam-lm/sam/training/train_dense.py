"""Train the dense Transformer baseline (Experiment 0, variant A0).

The dense model is trained on QA sequences with optional fact injection
so it has the opportunity to memorise the knowledge base in its weights.

Usage:
    python -m sam.training.train_dense --config configs/dense_tiny.yaml
    python -m sam.training.train_dense --config configs/dense_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, ConcatDataset

from ..data.dataset import QADataset, Tokenizer, collate_qa
from ..model.transformer import DenseTransformer
from ..utils.config import load_config, Config
from ..utils.seed import seed_everything
from ..utils.logging import MetricLogger
from ..eval.metrics import accuracy_by_hop


def _pick_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def _cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def _save_checkpoint(model, optimizer, epoch: int, step: int, path: str, extra: Dict = None):
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


def train_dense(cfg: Config):
    # --- setup ---
    seed_everything(cfg.get("seed", 42))
    device = _pick_device(cfg.train.get("device", "auto"))
    data_dir = cfg.get("data_dir", "data/synthetic")
    output_dir = cfg.get("output_dir", "experiments/exp_000_dense_baseline")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer.from_dir(data_dir)
    mlogger = MetricLogger(output_dir, cfg.get("run_name", "dense_tiny"))
    mlogger.logger.info("Dense baseline training — device=%s data=%s", device, data_dir)

    # Save config copy
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg.to_dict(), f)

    # --- model ---
    m_cfg = cfg.model
    model = DenseTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=m_cfg.get("d_model", 512),
        n_layers=m_cfg.get("n_layers", 8),
        n_heads=m_cfg.get("n_heads", 8),
        d_ff=m_cfg.get("d_ff", 2048),
        dropout=m_cfg.get("dropout", 0.0),
        max_seq_len=m_cfg.get("max_seq_len", 256),
        pad_id=tokenizer.pad,
    )
    model.to(device)
    mlogger.logger.info("Model: %d params", model.param_count())

    # --- data ---
    open_book = m_cfg.get("open_book", False)
    train_qa = QADataset(data_dir, "train", tokenizer, kind="qa",
                         open_book=open_book, max_seq_len=m_cfg.get("max_seq_len", 256))
    val_qa = QADataset(data_dir, "val", tokenizer, kind="qa",
                       open_book=open_book, max_seq_len=m_cfg.get("max_seq_len", 256))

    # Fact injection: mix in raw fact sequences so the model can memorise
    fact_ratio = cfg.train.get("fact_injection_ratio", 0.5)
    if fact_ratio > 0 and not open_book:
        train_fact = QADataset(data_dir, "train", tokenizer, kind="fact",
                               max_seq_len=m_cfg.get("max_seq_len", 256))
        # Pad the fact dataset to match QA length
        mlogger.logger.info("Fact injection: ratio=%.2f, fact_examples=%d, qa_examples=%d",
                            fact_ratio, len(train_fact), len(train_qa))
        # Simple interleaved concat (approximate ratio)
        from torch.utils.data import ConcatDataset
        # Weighted sampler approach: duplicate facts to reach ratio
        n_fact = int(len(train_qa) * fact_ratio / (1 - fact_ratio))
        n_fact = max(1, min(n_fact, len(train_fact)))
        # Take a random subset of facts each epoch via a wrapper
        train_dataset = _MixedFactDataset(train_qa, train_fact, fact_ratio)
    else:
        train_dataset = train_qa

    t_cfg = cfg.train
    batch_size = t_cfg.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_qa(b, tokenizer.pad),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_qa, batch_size=cfg.eval.get("batch_size", 128), shuffle=False,
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

    # --- training loop ---
    log_every = t_cfg.get("log_every", 50)
    eval_every = t_cfg.get("eval_every", 500)
    global_step = 0
    best_val_loss = float("inf")
    val_loss = float("inf")  # track latest

    mlogger.logger.info("Training: epochs=%d steps=%d warmup=%d batch=%d",
                        epochs, total_steps, warmup, batch_size)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            if max_steps and global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(input_ids, labels=labels)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                mlogger.log(global_step, {
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                })

            if global_step % eval_every == 0:
                val_loss = _evaluate_loss(model, val_loader, device)
                mlogger.log(global_step, {"val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(model, optimizer, epoch, global_step,
                                     os.path.join(output_dir, "checkpoint_best.pt"),
                                     extra={"val_loss": val_loss})
                    mlogger.logger.info("New best checkpoint: val_loss=%.4f", val_loss)

        # Always evaluate at end of epoch
        val_loss = _evaluate_loss(model, val_loader, device)
        mlogger.log(global_step, {"val_loss": val_loss, "epoch": epoch + 1})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, global_step,
                             os.path.join(output_dir, "checkpoint_best.pt"),
                             extra={"val_loss": val_loss})
            mlogger.logger.info("New best checkpoint (epoch end): val_loss=%.4f", val_loss)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        mlogger.logger.info("Epoch %d/%d — avg_loss=%.4f lr=%.6f",
                            epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0])

    # --- final ---
    _save_checkpoint(model, optimizer, epoch, global_step,
                     os.path.join(output_dir, "checkpoint_last.pt"))

    # Final validation accuracy
    model.eval()
    acc = accuracy_by_hop(model, val_loader, tokenizer,
                          max_new_tokens=cfg.eval.get("max_new_tokens", 6),
                          device=device)

    summary = {
        "run_name": cfg.get("run_name", "dense_tiny"),
        "data_dir": data_dir,
        "best_val_loss": best_val_loss,
        "param_count": model.param_count(),
        "final_epoch": epoch + 1,
        "total_steps": global_step,
        **{f"val_{k}": v for k, v in acc.items() if not isinstance(v, dict)},
    }
    mlogger.save_summary(summary)

    mlogger.logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    mlogger.logger.info("Val accuracy: overall=%.4f single=%.4f two=%.4f three=%.4f",
                        acc.get("accuracy_overall", 0),
                        acc.get("accuracy_single_hop", 0),
                        acc.get("accuracy_two_hop", 0),
                        acc.get("accuracy_three_hop", 0))

    return model, summary


class _MixedFactDataset(torch.utils.data.Dataset):
    """Mix fact-presentation sequences into QA training at a given ratio."""

    def __init__(self, qa_ds, fact_ds, fact_ratio: float):
        self.qa_ds = qa_ds
        self.fact_ds = fact_ds
        self.fact_ratio = fact_ratio
        self.qa_len = len(qa_ds)
        self.fact_len = len(fact_ds)
        # Total "virtual" size: qa_len + fact_ratio * qa_len facts
        self.total_facts = int(self.qa_len * fact_ratio / (1 - fact_ratio)) if fact_ratio < 1 else self.qa_len
        self.n = self.qa_len + min(self.total_facts, self.fact_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if idx < self.qa_len:
            return self.qa_ds[idx]
        else:
            fact_idx = (idx - self.qa_len) % self.fact_len
            return self.fact_ds[min(fact_idx, self.fact_len - 1)]


@torch.no_grad()
def _evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels=labels)
        if loss is not None:
            total_loss += loss.item() * input_ids.size(0)
            n += input_ids.size(0)
    model.train()
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Train dense Transformer baseline.")
    ap.add_argument("--config", default="configs/dense_tiny.yaml",
                    help="Path to YAML config file.")
    ap.add_argument("--override", nargs="*", default=None,
                    help="Override config values, e.g. train.epochs=2")
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
    train_dense(cfg)


if __name__ == "__main__":
    main()
