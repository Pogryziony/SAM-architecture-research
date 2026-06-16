"""Evaluation metrics and harness for the SAM POC.

Implements every metric specified in docs/metrics.md:
  accuracy_single_hop, accuracy_two_hop, accuracy_three_hop,
  accuracy_by_task_type, memory_recall_at_k, oracle_memory_accuracy,
  retrieved_memory_accuracy, oracle_gap, memory_gain, dense_gap,
  training_loss, validation_loss, parameter_count.

Also implements the five decision gates from docs/experiment-0.md.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import QADataset, Tokenizer, collate_qa


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy_by_hop(
    model,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    max_new_tokens: int = 6,
    mode: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute per-hop and overall QA accuracy.

    Returns dict with keys:
      accuracy_overall,
      accuracy_single_hop (hops==1),
      accuracy_two_hop (hops==2),
      accuracy_three_hop (hops==3),
      accuracy_by_task_type (nested dict),
      num_examples.
    """
    model.eval()
    model.to(device)
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    correct_by_type: Dict[str, int] = defaultdict(int)
    total_by_type: Dict[str, int] = defaultdict(int)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        hops = batch["hops"]
        task_types = batch["task_type"]
        prompt_lens = batch["prompt_len"]

        # For SAM, pass required_slots for oracle/retrieved modes
        required = batch.get("required_slots", None)
        if required is not None:
            required = required.to(device)

        B = input_ids.size(0)
        for i in range(B):
            p_len = prompt_lens[i]
            prompt = input_ids[i, :p_len]
            target_ids = labels[i]
            # Extract answer tokens (where labels != -100)
            target_mask = target_ids != -100
            expected = target_ids[target_mask].tolist()

            # Generate
            if mode is not None and mode != "core_only":
                # SAM model with memory
                req_i = required[i:i+1] if required is not None else None
                generated = model.generate(
                    prompt, max_new_tokens=max_new_tokens,
                    eos_id=tokenizer.eos, required_slots=req_i, mode=mode,
                )
            else:
                generated = model.generate(
                    prompt, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos,
                )

            pred_text_full = tokenizer.decode(generated.tolist()).strip()
            pred_text = pred_text_full.split()[0] if pred_text_full else ""
            expected_text_full = tokenizer.decode(expected).strip()
            expected_text = expected_text_full.split()[0] if expected_text_full else ""

            is_correct = (pred_text == expected_text)

            h = int(hops[i])
            hop_key = f"hop_{h}"
            if h == 1:
                hop_label = "single_hop"
            elif h == 2:
                hop_label = "two_hop"
            elif h == 3:
                hop_label = "three_hop"
            else:
                hop_label = hop_key

            correct[hop_label] += int(is_correct)
            total[hop_label] += 1
            correct["overall"] += int(is_correct)
            total["overall"] += 1

            tt = task_types[i]
            correct_by_type[tt] += int(is_correct)
            total_by_type[tt] += 1

    results: Dict[str, Any] = {"num_examples": total["overall"]}
    for key in ["overall", "single_hop", "two_hop", "three_hop"]:
        if total.get(key, 0) > 0:
            results[f"accuracy_{key}"] = correct[key] / total[key]
        else:
            results[f"accuracy_{key}"] = 0.0

    acc_by_type = {}
    for tt in sorted(correct_by_type.keys()):
        if total_by_type[tt] > 0:
            acc_by_type[tt] = correct_by_type[tt] / total_by_type[tt]
    results["accuracy_by_task_type"] = acc_by_type

    return results


@torch.no_grad()
def recall_at_k(
    model,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    k_values: Tuple[int, ...] = (1, 8, 32),
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute retrieval Recall@k for product-key memory.

    Uses model.retrieve() to get top-k slot IDs and checks whether
    any of them match a required_slot.

    Returns dict with recall_at_1, recall_at_8, recall_at_32.
    """
    model.eval()
    model.to(device)
    hits: Dict[int, int] = {k: 0 for k in k_values}
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        required_slots = batch["required_slots"].to(device)
        prompt_lens = torch.tensor(batch["prompt_len"], device=device)

        retrieved = model.retrieve(input_ids, prompt_lens, k=max(k_values))
        if retrieved is None:
            continue

        B = required_slots.size(0)
        for i in range(B):
            req = set(int(s) for s in required_slots[i] if int(s) >= 0)
            if not req:
                continue
            total += 1
            ret = set(int(s) for s in retrieved[i])
            for k in k_values:
                if req & set(list(ret)[:k]):
                    hits[k] += 1

    results = {}
    for k in k_values:
        results[f"recall_at_{k}"] = hits[k] / max(total, 1)
    return results


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def compute_derived_metrics(
    dense_acc: Dict[str, float],
    core_only_acc: Dict[str, float],
    oracle_acc: Dict[str, float],
    retrieved_acc: Dict[str, float],
    recall: Dict[str, float],
) -> Dict[str, float]:
    """Compute derived metrics from raw evaluation results.

    Args:
        dense_acc: accuracy dict from dense baseline eval
        core_only_acc: accuracy dict from SAM core-only eval
        oracle_acc: accuracy dict from SAM oracle-memory eval
        retrieved_acc: accuracy dict from SAM retrieved-memory eval
        recall: recall@k dict from retrieval eval

    Returns dict with oracle_gap, memory_gain, dense_gap, etc.
    """
    derived: Dict[str, float] = {}

    def _get(d, key, default=0.0):
        return float(d.get(key, default))

    # Batch compute all gaps
    for key in ["accuracy_overall", "accuracy_single_hop", "accuracy_two_hop",
                "accuracy_three_hop"]:
        o = _get(oracle_acc, key)
        r = _get(retrieved_acc, key)
        c = _get(core_only_acc, key)
        d = _get(dense_acc, key)
        suffix = "" if key == "accuracy_overall" else "_" + key.replace("accuracy_", "")

        derived[f"oracle_gap{suffix}"] = o - r
        derived[f"memory_gain{suffix}"] = r - c
        derived[f"dense_gap{suffix}"] = r - d

    derived["recall_at_8"] = recall.get("recall_at_8", 0.0)
    derived["recall_at_32"] = recall.get("recall_at_32", 0.0)

    return derived


# ---------------------------------------------------------------------------
# Decision gates
# ---------------------------------------------------------------------------

def evaluate_gates(
    recall: Dict[str, float],
    core_only_acc: Dict[str, float],
    oracle_acc: Dict[str, float],
    retrieved_acc: Dict[str, float],
    dense_acc: Dict[str, float],
    threshold_recall: float = 0.80,
    threshold_gap: float = 0.20,
) -> Dict[str, Any]:
    """Evaluate the five decision gates and return pass/fail status.

    Returns dict with gate_1..gate_5 statuses and diagnostic messages.
    """
    gates: Dict[str, Any] = {}

    # Gate 1: Retrieval quality
    r8 = recall.get("recall_at_8", 0.0)
    gates["gate_1_retrieval"] = {
        "passed": r8 >= threshold_recall,
        "recall_at_8": r8,
        "threshold": threshold_recall,
        "message": "PASS" if r8 >= threshold_recall
        else f"FAIL: Recall@8={r8:.3f} < {threshold_recall}. Improve retrieval before training SAM end-to-end.",
    }

    # Gate 2: Memory usefulness (oracle vs core-only)
    o_overall = oracle_acc.get("accuracy_overall", 0.0)
    c_overall = core_only_acc.get("accuracy_overall", 0.0)
    gates["gate_2_memory_usefulness"] = {
        "passed": o_overall > c_overall,
        "oracle_accuracy": o_overall,
        "core_only_accuracy": c_overall,
        "message": "PASS" if o_overall > c_overall
        else f"FAIL: Oracle memory ({o_overall:.3f}) does not beat core-only ({c_overall:.3f}). Model not using memory correctly.",
    }

    # Gate 3: Retrieval gap
    gap = o_overall - retrieved_acc.get("accuracy_overall", 0.0)
    gates["gate_3_retrieval_gap"] = {
        "passed": gap <= threshold_gap,
        "oracle_gap": gap,
        "threshold": threshold_gap,
        "message": "PASS" if gap <= threshold_gap
        else f"FAIL: Oracle-gap={gap:.3f} > {threshold_gap}. Retrieval is the bottleneck.",
    }

    # Gate 4: Multi-hop reasoning
    s_oracle = oracle_acc.get("accuracy_single_hop", 0.0)
    t_oracle = oracle_acc.get("accuracy_two_hop", 0.0)
    th_oracle = oracle_acc.get("accuracy_three_hop", 0.0)
    gates["gate_4_reasoning"] = {
        "passed": t_oracle > s_oracle * 0.5 and th_oracle > s_oracle * 0.15,
        "oracle_single_hop": s_oracle,
        "oracle_two_hop": t_oracle,
        "oracle_three_hop": th_oracle,
        "message": "PASS" if (t_oracle > s_oracle * 0.5 and th_oracle > s_oracle * 0.15)
        else f"FAIL: Oracle two-hop ({t_oracle:.3f}) / three-hop ({th_oracle:.3f}) vs single-hop ({s_oracle:.3f}). Model is recall-only, not reasoning.",
    }

    # Gate 5: Dense baseline
    r_overall = retrieved_acc.get("accuracy_overall", 0.0)
    d_overall = dense_acc.get("accuracy_overall", 0.0)
    gates["gate_5_dense_baseline"] = {
        "passed": r_overall > d_overall,
        "sam_retrieved_accuracy": r_overall,
        "dense_baseline_accuracy": d_overall,
        "message": "PASS" if r_overall > d_overall
        else f"FAIL: SAM retrieved ({r_overall:.3f}) does not beat dense baseline ({d_overall:.3f}). Do not scale.",
    }

    return gates


# ---------------------------------------------------------------------------
# Full evaluation harness
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_qa_eval(
    model,
    data_dir: str,
    tokenizer: Tokenizer,
    device: str = "cpu",
    max_new_tokens: int = 6,
    eval_batch_size: int = 128,
    mode: Optional[str] = None,
    compute_recall: bool = False,
) -> Dict[str, Any]:
    """Run a full evaluation of a model on the test set.

    Args:
        model: The model to evaluate.
        data_dir: Path to the synthetic data directory.
        tokenizer: Tokenizer instance.
        device: Device string.
        max_new_tokens: Max tokens to generate for answer.
        eval_batch_size: Batch size for eval DataLoader.
        mode: Memory mode for SAM models (core_only, oracle_memory, etc.).
        compute_recall: Whether to also compute recall@k (for retrieval models).

    Returns:
        Dict with accuracy metrics, recall metrics (if requested), and param_count.
    """
    test_dataset = QADataset(data_dir, "test", tokenizer, kind="qa",
                             open_book=False, max_seq_len=model.max_seq_len)
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        collate_fn=lambda b: collate_qa(b, tokenizer.pad),
    )

    results = accuracy_by_hop(model, test_loader, tokenizer,
                              max_new_tokens=max_new_tokens, mode=mode, device=device)
    results["parameter_count"] = model.param_count()

    if compute_recall:
        recall = recall_at_k(model, test_loader, tokenizer,
                             k_values=(1, 8, 32), device=device)
        results.update(recall)

    return results
