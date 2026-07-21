"""Run bounded AnswerPlan neural pilots; never launches full training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from torch.nn import functional as F

from benchmarks.realizer_corpus_v2_contracts import iter_jsonl, normalized_text, sha256_file, sha256_json
from nexus.realizer.pointer_generator import build_pointer_generator
from nexus.realizer.plan_serializer import serialize_answer_plan_for_model
from nexus.realizer.subword_tokenizer import TrainOnlySubwordTokenizer


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?)(?!\w)")


def _token_f1(prediction: str, target: str) -> float:
    pred = Counter(_WORD_RE.findall(normalized_text(prediction)))
    gold = Counter(_WORD_RE.findall(normalized_text(target)))
    common = sum((pred & gold).values())
    if not pred or not gold:
        return float(pred == gold)
    precision, recall = common / sum(pred.values()), common / sum(gold.values())
    return 2 * precision * recall / (precision + recall) if common else 0.0


def _eligible(row: dict[str, Any], tokenizer: TrainOnlySubwordTokenizer) -> tuple[list[int], list[int], list[bool]] | None:
    # Abstention is an upstream decision with a deterministic PL/EN surface.
    # Training it neurally only creates a high-frequency collapse target.
    if row["operator"] == "abstain":
        return None
    source = serialize_answer_plan_for_model(row["answer_plan"])
    if normalized_text(row["target"]) in normalized_text(source):
        return None
    source_ids = tokenizer.encode(source)
    target_ids = tokenizer.encode(row["target"])
    if len(source_ids) > 512 or len(target_ids) > 128:
        return None
    fact_ids = tokenizer.encode(
        row["answer_plan"]["resolved_answer"]["canonical_text"],
        add_special_tokens=False,
    )
    start = -1
    for index in range(len(source_ids) - len(fact_ids) + 1):
        if source_ids[index:index + len(fact_ids)] == fact_ids:
            start = index
    if start < 0:
        raise ValueError(f"fact span missing from serialized plan: {row['id']}")
    copy_mask = [False] * len(source_ids)
    for index in range(start, start + len(fact_ids)):
        copy_mask[index] = True
    return source_ids, target_ids, copy_mask


def _load_stratified(
    paths: list[Path], tokenizer: TrainOnlySubwordTokenizer, limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in iter_jsonl(path):
            encoded = _eligible(row, tokenizer)
            if encoded is None:
                continue
            row = dict(
                row, _source_ids=encoded[0], _target_ids=encoded[1],
                _copy_mask=encoded[2],
            )
            groups[(row["language"], row["operator"])].append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    available = sum(len(rows) for rows in groups.values())
    if limit is None or limit >= available:
        selected = [row for key in sorted(groups) for row in groups[key]]
    else:
        # Prevent the fixed Polish abstention sentence from dominating pilots.
        equal_share = limit // len(groups)
        quotas = {key: min(equal_share, len(rows)) for key, rows in groups.items()}
        while sum(quotas.values()) < limit:
            key = max(groups, key=lambda item: (len(groups[item]) - quotas[item], item))
            quotas[key] += 1
        while sum(quotas.values()) > limit:
            key = max((key for key in groups if quotas[key] > 1), key=lambda item: (quotas[item], item))
            quotas[key] -= 1
        selected = [row for key in sorted(groups) for row in groups[key][:quotas[key]]]
    random.Random(seed).shuffle(selected)
    return selected


def _pad(sequences: list[list[int]], pad: int = 0) -> torch.Tensor:
    width = max(map(len, sequences))
    result = torch.full((len(sequences), width), pad, dtype=torch.long)
    for index, values in enumerate(sequences):
        result[index, :len(values)] = torch.tensor(values, dtype=torch.long)
    return result


def _batches(rows: list[dict[str, Any]], batch_size: int, seed: int):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda row: len(row["_source_ids"]) // 32)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        source = _pad([row["_source_ids"] for row in batch])
        copy_mask = _pad([
            [int(value) for value in row["_copy_mask"]] for row in batch
        ]).bool()
        targets = _pad([row["_target_ids"] for row in batch])
        yield batch, source, copy_mask, targets


def _teacher_metrics(model: Any, rows: list[dict[str, Any]], batch_size: int) -> dict[str, float]:
    model.eval()
    loss_sum = correct = tokens = 0
    with torch.no_grad():
        for _, source, copy_mask, targets in _batches(rows, batch_size, 0):
            logits = model(source, targets[:, :-1], copy_mask)
            labels = targets[:, 1:]
            loss_sum += F.nll_loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=0, reduction="sum").item()
            mask = labels.ne(0)
            correct += (logits.argmax(-1).eq(labels) & mask).sum().item()
            tokens += mask.sum().item()
    return {"loss": loss_sum / max(1, tokens), "token_accuracy": correct / max(1, tokens)}


def _generate(
    model: Any, source_ids: list[int], copy_mask_values: list[bool],
    tokenizer: TrainOnlySubwordTokenizer, max_tokens: int,
) -> tuple[str, bool]:
    model.eval()
    source = torch.tensor([source_ids], dtype=torch.long)
    copy_mask = torch.tensor([copy_mask_values], dtype=torch.bool)
    beams: list[tuple[list[int], float, bool]] = [([], 0.0, False)]
    beam_width = 3
    with torch.no_grad():
        for _ in range(max_tokens):
            candidates: list[tuple[list[int], float, bool]] = []
            for generated, score, ended in beams:
                if ended:
                    candidates.append((generated, score, True))
                    continue
                target = torch.tensor([[tokenizer.BOS, *generated]], dtype=torch.long)
                log_probabilities = model(source, target, copy_mask)[0, -1].clone()
                log_probabilities[tokenizer.PAD] = float("-inf")
                if len(generated) >= 2:
                    prefix = tuple(generated[-2:])
                    blocked = {
                        generated[index + 2]
                        for index in range(len(generated) - 2)
                        if tuple(generated[index:index + 2]) == prefix
                    }
                    for token_id in blocked:
                        log_probabilities[token_id] = float("-inf")
                top_scores, top_ids = torch.topk(log_probabilities, beam_width)
                for token_score, token_id in zip(top_scores, top_ids):
                    value = int(token_id)
                    candidates.append((
                        generated if value == tokenizer.EOS else [*generated, value],
                        score + float(token_score), value == tokenizer.EOS,
                    ))
            beams = sorted(
                candidates,
                key=lambda item: item[1] / max(1.0, len(item[0]) ** 0.7),
                reverse=True,
            )[:beam_width]
            if all(item[2] for item in beams):
                break
    generated, _, eos = max(
        beams, key=lambda item: item[1] / max(1.0, len(item[0]) ** 0.7)
    )
    try:
        return tokenizer.decode(generated), eos
    except UnicodeDecodeError:
        return "", eos


def _generation_metrics(model: Any, rows: list[dict[str, Any]], tokenizer: TrainOnlySubwordTokenizer) -> dict[str, Any]:
    totals = Counter()
    f1 = 0.0
    outputs: set[str] = set()
    samples = []
    for row in rows:
        prediction, eos = _generate(
            model, row["_source_ids"], row["_copy_mask"], tokenizer,
            min(128, len(row["_target_ids"]) + 16),
        )
        target = row["target"]
        immutable = row["answer_plan"]["resolved_answer"]["immutable_values"]
        preserve = all(normalized_text(value) in normalized_text(prediction) for value in immutable)
        allowed_numbers = set(_NUMBER_RE.findall(serialize_answer_plan_for_model(row["answer_plan"])))
        unsupported = bool(set(_NUMBER_RE.findall(prediction)) - allowed_numbers)
        exact = normalized_text(prediction) == normalized_text(target)
        totals["count"] += 1
        totals["exact"] += exact
        totals["eos"] += eos
        totals["empty"] += not prediction.strip()
        totals["immutable"] += preserve
        totals["unsupported"] += unsupported
        f1 += _token_f1(prediction, target)
        outputs.add(normalized_text(prediction))
        if len(samples) < 12:
            samples.append({"id": row["id"], "language": row["language"], "operator": row["operator"], "target": target, "prediction": prediction, "eos": eos})
    count = totals["count"]
    return {
        "count": count, "exact_match": totals["exact"] / count,
        "token_f1": f1 / count, "eos_rate": totals["eos"] / count,
        "empty_rate": totals["empty"] / count,
        "immutable_preservation": totals["immutable"] / count,
        "unsupported_number_rate": totals["unsupported"] / count,
        "unique_output_rate": len(outputs) / count,
        "samples": samples,
    }


def _validation_subset(rows: list[dict[str, Any]], per_stratum: int = 16) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["operator"])].append(row)
    return [row for key in sorted(groups) for row in groups[key][:per_stratum]]


def run_stage(
    name: str, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]],
    tokenizer: TrainOnlySubwordTokenizer, output: Path, epochs: int,
    batch_size: int, seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = {
        "architecture": "pointer_generator_v1", "vocab_size": tokenizer.vocab_size,
        "hidden_size": 96, "dropout": 0.1,
        "max_input_tokens": 512, "max_output_tokens": 128,
    }
    model = build_pointer_generator(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    initial = _teacher_metrics(model, val_rows, batch_size)
    history = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum = tokens = 0
        for _, source, copy_mask, targets in _batches(train_rows, batch_size, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, targets[:, :-1], copy_mask)
            labels = targets[:, 1:]
            loss = F.nll_loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=0)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pilot loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            active = labels.ne(0).sum().item()
            loss_sum += loss.item() * active
            tokens += active
        validation = _teacher_metrics(model, val_rows, batch_size)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / max(1, tokens), "validation": validation})
    train_generated = _generation_metrics(model, _validation_subset(train_rows), tokenizer)
    generated = _generation_metrics(model, _validation_subset(val_rows), tokenizer)
    checkpoint = output / f"{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": config, "tokenizer_sha256": tokenizer.to_dict()["canonical_sha256"]}, checkpoint)
    return {
        "name": name, "status": "PILOT_COMPLETE", "records": len(train_rows),
        "epochs": epochs, "batch_size": batch_size, "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "duration_seconds": time.time() - started,
        "initial_validation": initial, "history": history,
        "train_generation": train_generated, "generation": generated,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint), "model_config": config,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point retained for discoverability; AR pilots are registry-blocked."""
    parser = argparse.ArgumentParser(
        description=(
            "Blocked: autoregressive AnswerPlan pilots are rejected. "
            "Use benchmarks/train_answer_plan_edit_transducer.py instead."
        )
    )
    parser.add_argument("--prepared-root", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.parse_args([] if argv is None else argv)
    from training.architecture_registry import ArchitectureBlockedError

    raise ArchitectureBlockedError(
        "answer_plan_autoregressive_pointer_generator is FULL_TRAINING_BLOCKED: "
        "use benchmarks/train_answer_plan_edit_transducer.py for AnswerPlan pilots "
        "(see training/REJECTED_ARCHITECTURES.json)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
