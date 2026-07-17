"""Run bounded AnswerPlan neural pilots with the non-autoregressive copy/edit transducer.

Mirrors the protocol from ``train_answer_plan_pilots.py`` but replaces the
autoregressive pointer-generator with the parallel copy/edit transducer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from benchmarks.realizer_corpus_v2_contracts import (
    iter_jsonl, normalized_text, sha256_file, sha256_json,
)
from nexus.realizer.copy_edit_transducer import (
    build_copy_edit_transducer, build_label_ids, find_fact_positions,
)
from nexus.realizer.edit_script import (
    apply_edit_target, compute_edit_target, edit_accuracy,
)
from nexus.realizer.plan_serializer import serialize_answer_plan_for_model
from nexus.realizer.subword_tokenizer import TrainOnlySubwordTokenizer


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?)(?!\w)")
_ABSTENTION_TARGETS = {
    "en": "The provided evidence is insufficient to answer the question.",
    "pl": "Podane dowody nie wystarczają do udzielenia odpowiedzi.",
}


def _token_f1(prediction: str, target: str) -> float:
    pred = Counter(_WORD_RE.findall(normalized_text(prediction)))
    gold = Counter(_WORD_RE.findall(normalized_text(target)))
    common = sum((pred & gold).values())
    if not pred or not gold:
        return float(pred == gold)
    precision, recall = common / sum(pred.values()), common / sum(gold.values())
    return 2 * precision * recall / (precision + recall) if common else 0.0


def _eligible(
    row: dict[str, Any], tokenizer: TrainOnlySubwordTokenizer,
) -> dict[str, Any] | None:
    """Filter and encode a single training record."""
    if row["operator"] == "abstain":
        return None   # abstention is deterministic; never train neurally
    source = serialize_answer_plan_for_model(row["answer_plan"])
    canonical = row["answer_plan"]["resolved_answer"]["canonical_text"]
    if normalized_text(row["target"]) in normalized_text(source):
        return None   # target is already visible in the input
    source_ids = tokenizer.encode(source)
    if len(source_ids) > 512:
        return None
    try:
        fact_positions = find_fact_positions(source_ids, tokenizer, canonical)
    except ValueError:
        return None
    labels = compute_edit_target(canonical, row["target"])
    target_ids = build_label_ids(labels, tokenizer)
    return dict(
        row, _source_ids=source_ids, _fact_positions=fact_positions,
        _labels=labels, _target_ids=target_ids,
    )


def _load_stratified(
    paths: list[Path], tokenizer: TrainOnlySubwordTokenizer,
    limit: int | None, seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in iter_jsonl(path):
            encoded = _eligible(row, tokenizer)
            if encoded is None:
                continue
            groups[(row["language"], row["operator"])].append(encoded)
    for key, rows in groups.items():
        rows.sort(key=lambda r: hashlib.sha256(
            f"{seed}:{r['id']}".encode(),
        ).hexdigest())
    available = sum(len(rows) for rows in groups.values())
    if limit is None or limit >= available:
        selected = [row for key in sorted(groups) for row in groups[key]]
    else:
        equal_share = limit // len(groups)
        quotas = {key: min(equal_share, len(rows)) for key, rows in groups.items()}
        while sum(quotas.values()) < limit:
            key = max(
                groups, key=lambda item: (len(groups[item]) - quotas[item], item),
            )
            quotas[key] += 1
        while sum(quotas.values()) > limit:
            key = max(
                (key for key in groups if quotas[key] > 1),
                key=lambda item: (quotas[item], item),
            )
            quotas[key] -= 1
        selected = [
            row for key in sorted(groups) for row in groups[key][:quotas[key]]
        ]
    random.Random(seed).shuffle(selected)
    return selected


def _pad(sequences: list[list[int]], pad: int = 0) -> torch.Tensor:
    width = max(map(len, sequences))
    result = torch.full((len(sequences), width), pad, dtype=torch.long)
    for index, values in enumerate(sequences):
        result[index, :len(values)] = torch.tensor(values, dtype=torch.long)
    return result


def _pad_labels(
    labels: list[list[int]], pad: int = 0,
) -> torch.Tensor:
    width = max(map(len, labels))
    result = torch.full((len(labels), width), pad, dtype=torch.long)
    for index, values in enumerate(labels):
        result[index, :len(values)] = torch.tensor(values, dtype=torch.long)
    return result


def _teacher_metrics(
    model: Any, rows: list[dict[str, Any]],
) -> dict[str, float]:
    model.eval()
    loss_sum = tokens = 0
    with torch.no_grad():
        for row in rows:
            source = torch.tensor([row["_source_ids"]], dtype=torch.long)
            fact_pos = torch.tensor([row["_fact_positions"]], dtype=torch.long)
            logits = model(source, fact_pos)
            targets = torch.tensor([row["_target_ids"]], dtype=torch.long)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1), ignore_index=0,
                reduction="sum",
            ).item()
            active = targets.ne(0).sum().item()
            correct = (logits.argmax(-1).eq(targets) & targets.ne(0)).sum().item()
            tokens += active
    return {
        "loss": loss_sum / max(1, tokens),
        "token_accuracy": correct / max(1, tokens) if tokens else 0.0,
    }


def _generation_metrics(
    model: Any, rows: list[dict[str, Any]],
    tokenizer: TrainOnlySubwordTokenizer,
) -> dict[str, Any]:
    model.eval()
    totals = Counter()
    f1 = 0.0
    samples = []
    with torch.no_grad():
        for row in rows:
            source = torch.tensor([row["_source_ids"]], dtype=torch.long)
            fact_pos = torch.tensor([row["_fact_positions"]], dtype=torch.long)
            predicted_labels = model.predict(source, fact_pos, tokenizer)[0]
            canonical = row["answer_plan"]["resolved_answer"]["canonical_text"]
            target = row["target"]
            prediction = apply_edit_target(canonical, predicted_labels)
            immutable = (
                row["answer_plan"]["resolved_answer"]["immutable_values"]
            )
            preserve = all(
                normalized_text(value) in normalized_text(prediction)
                for value in immutable
            )
            allowed_numbers = set(
                _NUMBER_RE.findall(
                    serialize_answer_plan_for_model(row["answer_plan"]),
                )
            )
            unsupported = bool(
                set(_NUMBER_RE.findall(prediction)) - allowed_numbers,
            )
            exact = normalized_text(prediction) == normalized_text(target)
            eos = not prediction.endswith("[DELETE]") and prediction.strip()
            totals["count"] += 1
            totals["exact"] += exact
            totals["eos"] += eos
            totals["empty"] += not prediction.strip()
            totals["immutable"] += preserve
            totals["unsupported"] += unsupported
            f1 += _token_f1(prediction, target)
            if len(samples) < 12:
                samples.append({
                    "id": row["id"], "language": row["language"],
                    "operator": row["operator"], "target": target,
                    "prediction": prediction, "eos": eos,
                })
    count = totals["count"]
    return {
        "count": count,
        "exact_match": totals["exact"] / count if count else 0.0,
        "token_f1": f1 / count if count else 0.0,
        "eos_rate": totals["eos"] / count if count else 0.0,
        "empty_rate": totals["empty"] / count if count else 0.0,
        "immutable_preservation": (
            totals["immutable"] / count if count else 0.0
        ),
        "unsupported_number_rate": (
            totals["unsupported"] / count if count else 0.0
        ),
        "samples": samples,
    }


def _validation_subset(
    rows: list[dict[str, Any]], per_stratum: int = 16,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["operator"])].append(row)
    return [
        row for key in sorted(groups) for row in groups[key][:per_stratum]
    ]


def run_stage(
    name: str, train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    tokenizer: TrainOnlySubwordTokenizer, output: Path, epochs: int,
    batch_size: int, seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = {
        "architecture": "copy_edit_transducer_v1",
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": 96,
        "output_vocab_size": tokenizer.vocab_size + 5,
        "dropout": 0.1,
    }
    model = build_copy_edit_transducer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    initial = _teacher_metrics(model, val_rows)
    history = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum = tokens = 0
        random.Random(seed + epoch).shuffle(train_rows)
        for start in range(0, len(train_rows), batch_size):
            batch = train_rows[start:start + batch_size]
            source = _pad([row["_source_ids"] for row in batch])
            fact_pos = _pad([
                row["_fact_positions"] for row in batch
            ])
            targets = _pad_labels([row["_target_ids"] for row in batch])
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, fact_pos)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1), ignore_index=0,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pilot loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            active = targets.ne(0).sum().item()
            loss_sum += loss.item() * active
            tokens += active
        validation = _teacher_metrics(model, val_rows)
        history.append({
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, tokens),
            "validation": validation,
        })
    generated = _generation_metrics(
        model, _validation_subset(val_rows), tokenizer,
    )
    checkpoint = output / f"{name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "tokenizer_sha256": tokenizer.to_dict()["canonical_sha256"],
    }, checkpoint)
    return {
        "name": name, "status": "PILOT_COMPLETE",
        "records": len(train_rows), "epochs": epochs,
        "batch_size": batch_size, "seed": seed,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "duration_seconds": time.time() - started,
        "initial_validation": initial, "history": history,
        "generation": generated,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stages", nargs="+",
        choices=("overfit", "small", "representative"),
        default=["overfit", "small", "representative"],
    )
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    root, output = args.prepared_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    readiness = json.loads((root / "readiness.json").read_text())
    if (
        readiness["status"] != "READY_FOR_BOUNDED_PILOT"
        or readiness["pilot_protocol"]["full_training_authorized"]
    ):
        raise ValueError(
            "prepared data does not authorize bounded-only pilots",
        )
    prepared_tokenizer_dict = json.loads(
        (root / "tokenizer.json").read_text(),
    )
    tokenizer = TrainOnlySubwordTokenizer.from_dict(prepared_tokenizer_dict)
    validation = _load_stratified(
        [root / "validation.jsonl", root / "abstention_validation.jsonl"],
        tokenizer, None, args.seed,
    )
    stage_specs = {
        "overfit": (64, 60, 16),
        "small": (2048, 1, 24),
        "representative": (17000, 1, 24),
    }
    results = []
    for stage in args.stages:
        limit, epochs, batch = stage_specs[stage]
        rows = _load_stratified(
            [root / "train.jsonl"], tokenizer, limit, args.seed,
        )
        result = run_stage(
            stage, rows, validation, tokenizer, output, epochs, batch,
            args.seed,
        )
        results.append(result)
        report = {
            "schema_version": "nexus-answer-plan-transducer-pilot-v1",
            "status": "PILOTS_IN_PROGRESS",
            "full_training_launched": False,
            "prepared_manifest_sha256": json.loads(
                (root / "manifest.json").read_text(),
            )["artifact_sha256"],
            "readiness_sha256": readiness["canonical_sha256"],
            "stages": results,
        }
        report["canonical_sha256"] = sha256_json(report)
        (output / "pilot_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    report["status"] = "BOUNDED_PILOTS_COMPLETE"
    report["canonical_sha256"] = sha256_json({
        k: v for k, v in report.items() if k != "canonical_sha256"
    })
    (output / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "canonical_sha256": report["canonical_sha256"],
        "stages": [item["name"] for item in results],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
