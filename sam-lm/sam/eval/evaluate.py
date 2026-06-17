"""Main evaluation script that produces comparison tables across all model variants.

Usage:
    python -m sam.eval.evaluate --runs experiments/
    python -m sam.eval.evaluate --runs experiments/ --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..data.dataset import Tokenizer
from ..model.transformer import DenseTransformer
from ..model.sam_core import SamModel
from ..utils.config import load_config
from ..utils.seed import seed_everything
from ..eval.metrics import (
    accuracy_by_hop,
    recall_at_k,
    compute_derived_metrics,
    evaluate_gates,
)
from ..data.dataset import QADataset, collate_qa
from torch.utils.data import DataLoader


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_checkpoint(model, ckpt_path: str, device: str, vocab_size: int) -> Optional[object]:
    """Load checkpoint, returning None if vocab sizes are incompatible."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state.get("model_state", state)

    # Check vocab compatibility via token_emb.weight shape
    emb_key = "token_emb.weight"
    if emb_key in model_state:
        ckpt_vocab = model_state[emb_key].shape[0]
        if ckpt_vocab != vocab_size:
            print(f"  WARNING: checkpoint vocab={ckpt_vocab} != dataset vocab={vocab_size}. Skipping.")
            return None

    # Pop live_slot_ids buffer to avoid shape mismatch (it gets reconfigured by set_kb)
    model_state.pop("live_slot_ids", None)
    model_state.pop("pkm.slot_value_token", None)
    model_state.pop("pkm.compact_to_original", None)
    model_state.pop("pkm.original_to_compact", None)
    model.load_state_dict(model_state, strict=False)
    return model


def _find_checkpoint(run_dir: str) -> Optional[str]:
    """Find the best checkpoint in a run directory."""
    patterns = ["checkpoint_best.pt", "checkpoint_last.pt", "checkpoint.pt"]
    for p in patterns:
        path = os.path.join(run_dir, p)
        if os.path.exists(path):
            return path
    # Also check inside subdirectories for any .pt files
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(root, f)
    return None


def _find_config(run_dir: str) -> Optional[str]:
    """Find config.yaml in a run directory."""
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith(".yaml") and "config" in f:
                return os.path.join(root, f)
    return None


def _find_data_dir(run_dir: str) -> Optional[str]:
    """Find meta.json to determine data_dir."""
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f == "meta.json" or f == "summary.json":
                with open(os.path.join(root, f), "r") as fh:
                    meta = json.load(fh)
                    if "data_dir" in meta:
                        return meta["data_dir"]
    return None


def evaluate_model(
    model_name: str,
    model,
    data_dir: str,
    tokenizer: Tokenizer,
    device: str,
    mode: Optional[str] = None,
    compute_recall: bool = False,
    eval_cfg: Optional[Dict] = None,
    open_book: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single model variant."""
    eval_cfg = eval_cfg or {}
    max_new = eval_cfg.get("max_new_tokens", 6)
    batch_size = eval_cfg.get("batch_size", 128)

    model.eval()
    model.to(device)

    test_dataset = QADataset(data_dir, "test", tokenizer, kind="qa",
                             open_book=open_book, max_seq_len=model.max_seq_len)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_qa(b, tokenizer.pad),
    )

    acc = accuracy_by_hop(model, test_loader, tokenizer,
                          max_new_tokens=max_new, mode=mode, device=device)
    acc["parameter_count"] = model.param_count()
    acc["model_name"] = model_name

    if compute_recall:
        recall = recall_at_k(model, test_loader, tokenizer,
                             k_values=(1, 8, 32), device=device, mode=mode)
        acc.update(recall)

    return acc


def build_comparison_table(all_results: List[Dict[str, Any]]) -> str:
    """Build a human-readable comparison table from results list."""
    headers = ["model", "params", "single_hop", "two_hop", "three_hop",
               "overall", "recall@8", "oracle_gap", "dense_gap"]
    rows = []
    for r in all_results:
        rows.append([
            r.get("model_name", "?"),
            r.get("parameter_count", 0),
            f"{r.get('accuracy_single_hop', 0):.3f}",
            f"{r.get('accuracy_two_hop', 0):.3f}",
            f"{r.get('accuracy_three_hop', 0):.3f}",
            f"{r.get('accuracy_overall', 0):.3f}",
            f"{r.get('recall_at_8', 0):.3f}",
            f"{r.get('oracle_gap', 0):.3f}",
            f"{r.get('dense_gap', 0):.3f}",
        ])

    # Build table string
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                  for i in range(len(headers))]
    lines = []
    # Header
    lines.append(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * col_widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate all SAM experiment models on the test set.")
    ap.add_argument("--runs", default="experiments/",
                    help="Root directory containing experiment subdirectories.")
    ap.add_argument("--data", default=None,
                    help="Override data directory (auto-detected from run metadata).")
    ap.add_argument("--output", default=None,
                    help="Write full JSON results to this path.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = _pick_device() if args.device == "auto" else args.device

    runs_dir = os.path.abspath(args.runs)
    if not os.path.isdir(runs_dir):
        print(f"[evaluate] ERROR: runs directory not found: {runs_dir}")
        sys.exit(1)

    # Auto-discover run directories (recurse into subdirectories)
    run_dirs = []
    # Skip these directories
    skip_dirs = {"smoke", "debug", ".", "_"}
    for root, dirs, files in os.walk(runs_dir):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".") and not d.startswith("_")]
        # Check for checkpoints in this directory
        for p in ["checkpoint_best.pt", "checkpoint_last.pt", "checkpoint.pt"]:
            ckpt_path = os.path.join(root, p)
            if os.path.exists(ckpt_path):
                # Use relative path as name
                rel = os.path.relpath(root, runs_dir).replace(os.sep, "/")
                run_dirs.append((rel, root, ckpt_path, "sam" if "oracle" in rel or "retrieved" in rel or "core_only" in rel else None))
                break

    # Deduplicate: keep only one entry per directory (prefer best then last)
    seen_dirs = set()
    deduped = []
    for name, rd, ckpt, hint in run_dirs:
        if rd not in seen_dirs:
            seen_dirs.add(rd)
            deduped.append((name, rd, ckpt))
    run_dirs = deduped

    if not run_dirs:
        print("[evaluate] No checkpoints found under", runs_dir)
        sys.exit(1)

    print(f"[evaluate] Found {len(run_dirs)} run(s) with checkpoints:")
    for name, _, ckpt in run_dirs:
        print(f"  {name}: {ckpt}")

    # Determine data_dir
    data_dir = args.data
    if data_dir is None:
        for _, rd, _ in run_dirs:
            data_dir = _find_data_dir(rd)
            if data_dir:
                break
        if data_dir is None:
            data_dir = "data/synthetic"

    if not os.path.isdir(data_dir):
        print(f"[evaluate] ERROR: data directory not found: {data_dir}")
        print("  Run: python -m sam.data.synthetic_facts --output data/synthetic")
        sys.exit(1)

    print(f"[evaluate] Using data: {data_dir}")
    tokenizer = Tokenizer.from_dir(data_dir)
    print(f"[evaluate] Vocab size: {tokenizer.vocab_size}")

    all_results: List[Dict[str, Any]] = []

    # Heuristic: identify model types from run dir names and configs
    dense_runs = {"exp_000_dense_baseline", "dense", "dense_tiny",
                  "dense_openbook", "dense_smoke"}
    retrieval_runs = {"exp_001_pkm_retrieval", "retrieval", "retrieval_1m",
                      "retrieval_smoke"}

    for name, rd, ckpt in run_dirs:
        print(f"\n--- Evaluating: {name} ---")

        # Try to load config
        cfg_path = _find_config(rd)
        cfg = None
        if cfg_path and os.path.exists(cfg_path):
            cfg = load_config(cfg_path)

        model_type = None
        if cfg:
            model_type = cfg.model.get("type", None)
        if model_type is None:
            # Heuristic
            base_name = name.lower().replace("_", " ")
            if any(k in base_name for k in ["dense", "exp_000"]):
                model_type = "dense"
            elif any(k in base_name for k in ["retriev", "exp_001"]):
                model_type = "retrieval"
            else:
                model_type = "sam"

        if model_type == "dense" and cfg:
            model = DenseTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=cfg.model.get("d_model", 512),
                n_layers=cfg.model.get("n_layers", 8),
                n_heads=cfg.model.get("n_heads", 8),
                d_ff=cfg.model.get("d_ff", 2048),
                dropout=cfg.model.get("dropout", 0.0),
                max_seq_len=cfg.model.get("max_seq_len", 256),
                pad_id=tokenizer.pad,
            )
            model.max_seq_len = cfg.model.get("max_seq_len", 256)
            result = _load_checkpoint(model, ckpt, device, tokenizer.vocab_size)
            if result is None:
                print(f"  SKIPPED: vocab mismatch")
                continue
            model = result
            open_book = cfg.model.get("open_book", False)
            res = evaluate_model(f"dense_{name}", model, data_dir, tokenizer,
                                 device, eval_cfg=cfg.eval.to_dict() if cfg and hasattr(cfg, 'eval') else None,
                                 open_book=open_book)
            all_results.append(res)
            print(f"  acc_overall={res['accuracy_overall']:.4f} "
                  f"single={res['accuracy_single_hop']:.4f} "
                  f"two={res['accuracy_two_hop']:.4f} "
                  f"three={res['accuracy_three_hop']:.4f}")

        elif model_type == "retrieval" and cfg:
            # For retrieval-only models, we just report recall
            # They don't have a generation capability, so skip QA eval
            print("  [retrieval-only model — recall metrics only]")
            # Would need a separate retrieval-model class for standalone eval
            # For now, mark as evaluated
            all_results.append({
                "model_name": f"retrieval_{name}",
                "parameter_count": 0,
                "accuracy_overall": 0.0,
                "accuracy_single_hop": 0.0,
                "accuracy_two_hop": 0.0,
                "accuracy_three_hop": 0.0,
            })

        elif model_type == "sam" and cfg:
            mem_dict = {}
            if hasattr(cfg.model, 'memory') and cfg.model.memory is not None:
                if hasattr(cfg.model.memory, 'to_dict'):
                    mem_dict = cfg.model.memory.to_dict()
                elif isinstance(cfg.model.memory, dict):
                    mem_dict = cfg.model.memory
            else:
                mem_dict = {"num_subkeys": 64, "key_dim": 32, "value_dim": 48,
                           "top_a": 8, "top_b": 8, "top_k": 4}

            model = SamModel(
                vocab_size=tokenizer.vocab_size,
                d_model=cfg.model.get("d_model", 64),
                n_layers=cfg.model.get("n_layers", 3),
                n_heads=cfg.model.get("n_heads", 2),
                d_ff=cfg.model.get("d_ff", 128),
                dropout=cfg.model.get("dropout", 0.0),
                max_seq_len=cfg.model.get("max_seq_len", 64),
                memory_every=cfg.model.get("memory_every", 2),
                memory_query=cfg.model.get("memory_query", "tokenwise"),
                memory_integration=cfg.model.get("memory_integration", "gated_sum"),
                memory_cfg=mem_dict,
                pad_id=tokenizer.pad,
            )
            model.max_seq_len = cfg.model.get("max_seq_len", 64)
            result = _load_checkpoint(model, ckpt, device, tokenizer.vocab_size)
            if result is None:
                print(f"  SKIPPED: vocab mismatch")
                continue
            model = result

            # Set up KB
            from ..data.dataset import build_kb_tensors
            if hasattr(cfg.model, 'memory') and cfg.model.memory is not None:
                n_subkeys = cfg.model.memory.get("num_subkeys", 64) if hasattr(cfg.model.memory, 'get') else 64
            else:
                n_subkeys = 64
            total_slots = n_subkeys * n_subkeys
            slot_value_token, num_live = build_kb_tensors(data_dir, total_slots, tokenizer)

            # Wire dual encoder retriever if config specifies one
            retriever = None
            if cfg and cfg.get("retriever_backend") == "dual_encoder":
                r_ckpt = cfg.get("retriever_checkpoint")
                if r_ckpt and os.path.exists(r_ckpt):
                    from ..model.sam_core import DualEncoderWrapper
                    retriever = DualEncoderWrapper(r_ckpt, tokenizer, device)
                    print(f"  Dual encoder retriever loaded from {r_ckpt}")
                else:
                    print(f"  WARNING: Dual encoder checkpoint not found: {r_ckpt}")
            elif cfg and cfg.get("retriever_backend") == "chain_set":
                r_ckpt = cfg.get("retriever_checkpoint")
                if r_ckpt and os.path.exists(r_ckpt):
                    from ..model.sam_core import ChainSetRetrieverWrapper
                    retriever = ChainSetRetrieverWrapper(r_ckpt, tokenizer, device)
                    print(f"  Chain-set retriever loaded from {r_ckpt}")
                else:
                    print(f"  WARNING: Chain-set checkpoint not found: {r_ckpt}")

            # Set retrieval topK if configured
            if cfg and cfg.get("topK"):
                model._retrieval_k = cfg.get("topK")
            if cfg and cfg.get("memory_aggregation_mode"):
                model._aggregation_mode = cfg.get("memory_aggregation_mode")
            if cfg and cfg.get("memory_score_temperature"):
                model._aggregation_temperature = float(cfg.get("memory_score_temperature"))

            model.set_kb(slot_value_token, retriever=retriever)
            model.to(device)

            eval_cfg_dict = {}
            if cfg and hasattr(cfg, 'eval'):
                if hasattr(cfg.eval, 'to_dict'):
                    eval_cfg_dict = cfg.eval.to_dict()
                elif isinstance(cfg.eval, dict):
                    eval_cfg_dict = cfg.eval

            # Read trained mode from checkpoint (saved in extra['mode'])
            ckpt_mode = None
            ckpt_state = torch.load(ckpt, map_location=device, weights_only=False)
            ckpt_mode = ckpt_state.get("mode", None)

            # Detect mode from directory name (check more specific patterns first)
            mode = None
            name_lower = name.lower()
            if "oracle_text" in name_lower:
                mode = "oracle_text_memory"
            elif "retrieved_oracle" in name_lower:
                mode = "retrieved_oracle_slots"
            elif "external_text" in name_lower or "multi_query" in name_lower:
                mode = "retrieved_memory_external_text_query"
            elif "hidden_adapter" in name_lower:
                mode = "retrieved_memory_hidden_adapter"
            elif "memory_adapter" in name_lower and "pretrain" not in name_lower:
                mode = "train_memory_adapter"
            elif "oracle" in name_lower:
                mode = "oracle_memory"
            elif "retrieved" in name_lower:
                mode = "retrieved_memory"
            elif "random" in name_lower:
                mode = "random_memory"
            elif "core_only" in name_lower or "core" in name_lower:
                mode = "core_only"

            # Fall back to checkpoint mode if directory detection fails
            if mode is None and ckpt_mode is not None:
                mode = ckpt_mode

            # If mode detected, evaluate only that mode
            default_modes = ["core_only", "oracle_memory", "retrieved_memory"]
            modes_to_eval = [mode] if mode else default_modes
            for m in modes_to_eval:
                # Set memory_mode on model so core_only/random_memory forward correctly
                model.memory_mode = m
                print(f"  mode={m}...")
                compute_recall = m in ("retrieved_memory", "retrieved_memory_external_text_query",
                                        "retrieved_memory_hidden_adapter")
                res = evaluate_model(
                    f"sam_{name}_{m}", model, data_dir, tokenizer,
                    device, mode=m, compute_recall=compute_recall,
                    eval_cfg=eval_cfg_dict,
                )
                all_results.append(res)
                print(f"    acc_overall={res['accuracy_overall']:.4f} "
                      f"single={res['accuracy_single_hop']:.4f} "
                      f"two={res['accuracy_two_hop']:.4f} "
                      f"three={res['accuracy_three_hop']:.4f}")

    # Compute derived metrics
    dense_acc = next((r for r in all_results if "dense" in r.get("model_name", "")), {})
    core_acc = next((r for r in all_results if "core_only" in r.get("model_name", "")), {})
    oracle_acc = next((r for r in all_results if "oracle_memory" in r.get("model_name", "")), {})
    retrieved_acc = next((r for r in all_results if "retrieved_memory" in r.get("model_name", "")), {})
    recall = next((r for r in all_results if "recall_at_8" in r), {})

    if core_acc and oracle_acc and retrieved_acc and dense_acc:
        derived = compute_derived_metrics(dense_acc, core_acc, oracle_acc, retrieved_acc, recall)
        gates = evaluate_gates(recall, core_acc, oracle_acc, retrieved_acc, dense_acc)
        all_results.append({"model_name": "_derived_metrics", **derived})
        all_results.append({"model_name": "_decision_gates", **{f"gate_{k}": v for k, v in gates.items()}})

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print(build_comparison_table(all_results))

    # Print gates
    if 'gates' in dir():
        print("\n" + "=" * 80)
        print("DECISION GATES")
        for key, gate in gates.items():
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"  {status} {key}: {gate['message']}")

    # Save results
    if args.output:
        serializable = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    sr[k] = v
                elif isinstance(v, dict):
                    sr[k] = {str(kk): vv for kk, vv in v.items()
                            if isinstance(vv, (int, float, str, bool, type(None)))}
                elif isinstance(v, torch.Tensor):
                    sr[k] = v.tolist()
                else:
                    sr[k] = str(v)
            serializable.append(sr)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
