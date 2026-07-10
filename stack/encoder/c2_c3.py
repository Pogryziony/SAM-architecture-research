"""C2/C3 candidate-pool training, validation selection, and frozen artifacts.

The module is deliberately validation-only: it accepts train/val records and a
caller-supplied candidate builder.  It never opens a split implicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from stack.encoder.stage1c import _tokens

SEED = 20260710
BASELINE_RECALL10 = 0.3571428571


class DatasetLeakError(ValueError):
    """Raised when generated supervision overlaps a held-out question."""


@dataclass(frozen=True)
class PairDataset:
    groups: list[dict[str, Any]]
    stats: dict[str, Any]


def normalized_question(question: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", question.casefold()))


def _validate_no_leakage(train: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]], test_path: str | Path | None) -> None:
    train_norm = {normalized_question(str(row["question"])) for row in train}
    val_norm = {normalized_question(str(row["question"])) for row in val}
    if train_norm & val_norm:
        raise DatasetLeakError("normalized train/validation question overlap")
    # The test path is intentionally metadata-only.  Opening it here would make
    # this experiment invalid, so callers receive a clear guard instead.
    if test_path is not None and Path(test_path).name.casefold() != "test.jsonl":
        raise DatasetLeakError("test_path must identify test.jsonl")


def _group(question_id: str, question: str, positives: Iterable[str], candidates: Iterable[str], source: str) -> dict[str, Any] | None:
    positive_ids = sorted(set(str(x) for x in positives))
    candidate_ids = list(dict.fromkeys(str(x) for x in candidates))
    if not positive_ids:
        return None
    if not set(positive_ids) <= set(candidate_ids):
        return None
    return {"question_id": question_id, "question": question, "candidate_ids": candidate_ids,
            "positive_ids": positive_ids, "source": source}


def build_training_groups(
    train_questions: Sequence[Mapping[str, Any]],
    val_questions: Sequence[Mapping[str, Any]],
    graph: Any,
    test_path: str | Path | None = None,
    candidate_builder: Callable[[str], Sequence[str]] | None = None,
    graph_pairs: Sequence[Mapping[str, Any]] | None = None,
    seed: int = SEED,
    hard_negative_k: int = 15,
    random_negative_fraction: float = 0.20,
) -> PairDataset:
    """Build deterministic groups; each question and graph pair gets its own pool."""
    _validate_no_leakage(train_questions, val_questions, test_path)
    if candidate_builder is None:
        if graph is None:
            raise ValueError("graph or candidate_builder is required")
        from stack.encoder.trivial_baseline import candidate_pool
        candidate_builder = lambda q: [str(item["node_id"]) for item in candidate_pool(q, graph)]
    rng = random.Random(seed)
    groups: list[dict[str, Any]] = []
    hard_count = random_count = negative_count = 0

    def add_record(record: Mapping[str, Any], source: str) -> None:
        nonlocal hard_count, random_count, negative_count
        positives = [str(x) for x in record.get("entities", record.get("positive_ids", []))]
        # Preserve the existing pipeline ordering, but guarantee every gold node
        # is represented so each train question contributes a supervised group.
        pool = list(candidate_builder(str(record["question"])))
        pool.extend(x for x in positives if x not in pool)
        group = _group(str(record.get("id", record.get("question_id", ""))), str(record["question"]), positives, pool, source)
        if group is None:
            return
        positive_set = set(group["positive_ids"])
        ordered_non_gt = [x for x in group["candidate_ids"] if x not in positive_set]
        hard = ordered_non_gt[: max(0, hard_negative_k)]
        remaining = ordered_non_gt[len(hard):]
        # Random negatives are a small stability sample in addition to the
        # approximately 15 hardest pipeline negatives, not the majority pool.
        random_n = int(round(len(hard) * random_negative_fraction / max(1e-9, 1.0 - random_negative_fraction)))
        random_n = min(random_n, len(remaining))
        random_part = sorted(rng.sample(remaining, random_n)) if random_n else []
        selected = group["positive_ids"] + hard + random_part
        group["candidate_ids"] = list(dict.fromkeys(selected))
        group["hard_negative_ids"] = hard
        group["random_negative_ids"] = random_part
        groups.append(group)
        hard_count += len(hard)
        random_count += len(random_part)
        negative_count += len(hard) + len(random_part)

    for record in train_questions:
        add_record(record, "train_candidate_pipeline")
    for record in graph_pairs or []:
        add_record(record, "stage1d_graph_mined")
    total = len(groups)
    stats = {
        "seed": seed, "groups": total, "positive_count": sum(len(g["positive_ids"]) for g in groups),
        "negative_count": negative_count, "pair_count": sum(len(g["candidate_ids"]) for g in groups),
        "positive_negative_ratio": (sum(len(g["positive_ids"]) for g in groups) / negative_count if negative_count else None),
        "hard_negative_count": hard_count, "random_negative_count": random_count,
        "hard_negative_share": hard_count / negative_count if negative_count else 0.0,
        "random_negative_share": random_count / negative_count if negative_count else 0.0,
        "sources": {"train_candidate_pipeline": sum(g["source"] == "train_candidate_pipeline" for g in groups),
                    "stage1d_graph_mined": sum(g["source"] == "stage1d_graph_mined" for g in groups)},
        "test_split_read": False,
    }
    return PairDataset(groups, stats)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = " " + re.sub(r"\s+", " ", text.casefold()) + " "
    return {normalized[i:i+n] for i in range(max(0, len(normalized) - n + 1))}


def _cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _node_features(question: str, node_id: str, graph: Any) -> list[float]:
    node = graph.get_node(node_id) if graph is not None else None
    node_type = str(getattr(node, "type", "")) if node is not None else ""
    properties = getattr(node, "properties", {}) if node is not None else {}
    aliases = [str(x) for x in getattr(node, "aliases", [])] if node is not None else []
    node_text = " ".join([node_id, *aliases, str(properties.get("key_finding", "")), str(properties.get("description", ""))])
    q_tokens = _tokens(question)
    n_tokens = _tokens(node_text)
    alias_tokens = _tokens(" ".join(aliases))
    degree = (len(graph.get_outgoing(node_id)) + len(graph.get_incoming(node_id))) if graph is not None and node is not None else 0
    q_intent = "diagnostic" if question.casefold().startswith(("why", "how would", "what is the significance")) else "factual_lookup"
    type_rule = float((q_intent == "diagnostic" and node_type in {"Concept", "Decision"}) or (q_intent != "diagnostic" and node_type in {"Experiment", "Metric"}))
    alias_match = float(bool(alias_tokens & q_tokens) or any(normalized_question(a) in normalized_question(question) for a in aliases))
    return [len(q_tokens & n_tokens) / max(1, len(q_tokens)), _cosine(_char_ngrams(question), _char_ngrams(node_text)), type_rule,
            math.log1p(degree), alias_match]


def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(weights, features))


def train_logistic(groups: Sequence[Mapping[str, Any]], graph: Any, epochs: int = 80, lr: float = 0.08) -> dict[str, Any]:
    weights = [0.0] * 6
    for _ in range(epochs):
        for group in groups:
            positives = set(group["positive_ids"])
            for node_id in group["candidate_ids"]:
                x = [1.0, *_node_features(group["question"], node_id, graph)]
                y = 1.0 if node_id in positives else 0.0
                probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, _dot(weights, x)))))
                for index, value in enumerate(x):
                    weights[index] += lr * (y - probability) * value
    return {"kind": "feature_logistic", "weights": weights, "features": ["bias", "lexical_overlap", "char_ngram_cosine", "type_rule_intent", "node_degree_log", "alias_match"]}


def train_encoder(groups: Sequence[Mapping[str, Any]], epochs: int = 2) -> dict[str, Any]:
    """Train a compact char-ngram pairwise model using existing PyTorch classes lazily."""
    import torch
    from stack.encoder.char_tokenizer import CharNgramTokenizer
    from stack.encoder.model import AssociativeEncoderV2
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    tokenizer = CharNgramTokenizer()
    tokenizer.add_words([str(g["question"]) for g in groups])
    tokenizer.add_words([str(node).replace("_", " ") for g in groups for node in g["candidate_ids"]])
    tokenizer.freeze()
    model = AssociativeEncoderV2(feature_dim=tokenizer.feature_dim, embed_dim=32, hidden_dim=64, num_intents=2, num_categories=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for group in groups:
            positives = set(group["positive_ids"])
            offsets, indices = tokenizer.tokenize_batch([group["question"]])
            q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
            q_indices = torch.tensor(indices, dtype=torch.long)
            pos = [x for x in group["candidate_ids"] if x in positives]
            neg = [x for x in group["candidate_ids"] if x not in positives]
            if not pos or not neg:
                continue
            p_feats = model.embed_entities([pos[0].replace("_", " ")], tokenizer)
            n_feats = model.embed_entities([neg[0].replace("_", " ")], tokenizer)
            _i, _c, p_score = model(q_indices, q_offsets, p_feats)
            _i, _c, n_score = model(q_indices, q_offsets, n_feats)
            loss = torch.nn.functional.softplus(-(p_score[0, 0] - n_score[0, 0]))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"kind": "encoder_char_ngram_pairwise", "model": model, "tokenizer": tokenizer}


def score_group(group: Mapping[str, Any], graph: Any, ranker: Mapping[str, Any]) -> list[str]:
    if ranker["kind"] == "feature_logistic":
        weights = ranker["weights"]
        scored = [(node, _dot(weights, [1.0, *_node_features(group["question"], node, graph)])) for node in group["candidate_ids"]]
    else:
        model = ranker["model"]; tokenizer = ranker["tokenizer"]
        import torch
        offsets, indices = tokenizer.tokenize_batch([group["question"]])
        q_offsets = torch.tensor(offsets[:-1], dtype=torch.long); q_indices = torch.tensor(indices, dtype=torch.long)
        desc = [x.replace("_", " ") for x in group["candidate_ids"]]
        feats = model.embed_entities(desc, tokenizer)
        with torch.no_grad():
            _i, _c, scores = model(q_indices, q_offsets, feats)
        scored = list(zip(group["candidate_ids"], scores[0].tolist()))
    return [node for node, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]


def evaluate_ranker(groups: Sequence[Mapping[str, Any]], graph: Any, ranker: Mapping[str, Any], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    result: dict[str, float] = {}
    total_gold = sum(len(g["positive_ids"]) for g in groups)
    for k in ks:
        hit = retrieved = 0
        for group in groups:
            ranked = score_group(group, graph, ranker)[:k]
            hit += len(set(ranked) & set(group["positive_ids"]))
            retrieved += len(ranked)
        result[f"recall@{k}"] = hit / total_gold if total_gold else 0.0
        result[f"precision@{k}"] = hit / retrieved if retrieved else 0.0
    return result


def evaluate_selection_gate(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    return [(g["question_id"], g["candidate_ids"], g["positive_ids"]) for g in left] == [(g["question_id"], g["candidate_ids"], g["positive_ids"]) for g in right]


def select_winner(metrics: Mapping[str, Mapping[str, float]]) -> str:
    return sorted(metrics, key=lambda name: (-metrics[name]["recall@10"], -metrics[name]["recall@5"], name))[0]


def freeze_selection(metrics: Mapping[str, float], baseline_recall10: float = BASELINE_RECALL10) -> dict[str, Any]:
    required = baseline_recall10 + 0.15
    return {"baseline_recall@10": baseline_recall10, "required_recall@10": required,
            "winner_recall@10": metrics["recall@10"], "winner_recall@5": metrics["recall@5"],
            "gate": metrics["recall@10"] >= required}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_experiment(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    def load(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = load(root / "stack/encoder/data/train.jsonl"); val = load(root / "stack/encoder/data/val.jsonl")
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, graph_meta = build_benchmark_graph()
    from stack.encoder.trivial_baseline import candidate_pool
    builder = lambda question: [str(x["node_id"]) for x in candidate_pool(question, graph)]
    from stack.encoder.stage1c import generate_stage1c_pairs
    generated = build_training_groups(train, val, graph, test_path=root / "stack/encoder/data/test.jsonl", candidate_builder=builder, graph_pairs=generate_stage1c_pairs(graph))
    out_data = root / "stack/encoder/data/stage1c_pairs_c2.jsonl"
    _write_jsonl(out_data, generated.groups)
    stats_path = root / "stack/encoder/data/stage1c_pairs_c2_stats.json"
    stats_path.write_text(json.dumps(generated.stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    val_groups = []
    for row in val:
        group = _group(str(row["id"]), str(row["question"]), row["entities"], builder(row["question"]), "validation_candidate_pipeline")
        if group is not None: val_groups.append(group)
    logistic = train_logistic(generated.groups, graph)
    encoder = train_encoder(generated.groups)
    metrics = {"feature_logistic": evaluate_ranker(val_groups, graph, logistic), "encoder": evaluate_ranker(val_groups, graph, encoder)}
    winner = select_winner(metrics)
    gate = freeze_selection(metrics[winner])
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    model_dir = root / "models/encoder/stage1c"; model_dir.mkdir(parents=True, exist_ok=True)
    if winner == "encoder":
        import torch
        torch.save(encoder["model"].state_dict(), model_dir / "weights.pt")
    else:
        (model_dir / "weights.json").write_text(json.dumps(logistic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {"winner": winner, "source_sha": sha, "seed": SEED, "validation_only": True, "candidate_groups": "identical_for_both_rankers", "gate": gate}
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_log = root / "benchmarks/results/stage1c_full_selection_log.json"
    log = {"source_sha": sha, "seed": SEED, "split": "stack/encoder/data/val.jsonl", "graph": graph_meta, "dataset_stats": generated.stats,
           "evaluations": [{"model": name, "candidate_groups": len(val_groups), "metrics": values} for name, values in metrics.items()],
           "selection": {"winner": winner, "decision_rule": "recall@10, recall@5 tie-break", **gate}, "test_split_read": False}
    selection_log.parent.mkdir(parents=True, exist_ok=True); selection_log.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not gate["gate"]:
        raise RuntimeError("validation-only negative: winner missed the required +15 percentage point gate")
    return {"winner": winner, "metrics": metrics, "gate": gate, "dataset": generated.stats, "selection_log": str(selection_log), "source_sha": sha}


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))
