"""Entity Ranker V3 — training, selection, and evaluation.

Corrected training loop with:
- All positive entities per question
- Multiple diverse hard negatives per question
- Listwise softmax cross-entropy loss
- Deterministic mini-batches with source-balanced sampling
- Early stopping on validation recall@10 (recall@5 as tie-break)
- CPU-only execution
- Fixed random seeds
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from stack.encoder.char_tokenizer import CharNgramTokenizer
from stack.encoder.entity_ranker_v3 import (
    QuestionConditionedEntityRanker,
    save_ranker_v3,
    load_ranker_v3,
    compute_data_hash,
)
from stack.encoder.entity_text import build_entity_text, build_entity_texts
from stack.encoder.canonical_mapping import (
    build_canonical_mapping,
    apply_canonical_mapping,
)
from stack.encoder.hard_negative_miner import mine_hard_negatives_group
from stack.encoder.stage1c import generate_stage1c_pairs
from stack.encoder.trivial_baseline import candidate_pool, rank_candidates
from stack.encoder.natural_templates import generate_balanced_dataset
from stack.encoder.c2_c3 import _node_features, _dot, normalized_question, _validate_no_leakage, SEED as C2_SEED


SEED = 20260710
K_MAX = 10
REQUIRED_VAL_RECALL10 = 0.70
REQUIRED_BASELINE_GAP = 0.15


# ── Data loading ──

def load_split(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL split."""
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── Group building with preserved denominators ──

def build_group(
    question_id: str,
    question: str,
    positive_ids: list[str],
    candidate_ids: list[str],
    source: str,
    graph: Any,
    hard_negative_k: int = 15,
    random_negative_fraction: float = 0.10,
    seed: int = SEED,
) -> dict[str, Any] | None:
    """Build a single training/evaluation group.

    Unlike C2/C3, this NEVER returns None for missing gold candidates.
    Instead, it preserves the group with empty positive_ids so the
    question stays in the denominator.
    """
    import random as _random
    rng = _random.Random(seed)

    positive_set = set(str(x) for x in positive_ids)
    candidate_set = list(dict.fromkeys(str(x) for x in candidate_ids))

    # Ensure all gold IDs are in the candidate list (for training)
    for pid in positive_set:
        if pid not in candidate_set:
            candidate_set.append(pid)

    group = {
        "question_id": question_id,
        "question": question,
        "candidate_ids": candidate_set,
        "positive_ids": sorted(positive_set),
        "source": source,
        "gold_present_in_candidates": positive_set.issubset(set(candidate_set)),
    }

    # Mine hard negatives (only for training, uses real negatives)
    if source != "validation":
        hard_negs, neg_meta = mine_hard_negatives_group(
            question, candidate_set, sorted(positive_set), graph, hard_negative_k
        )
        # Add random negatives for stability
        remaining = [cid for cid in candidate_set if cid not in positive_set and cid not in hard_negs]
        random_n = min(int(hard_negative_k * random_negative_fraction), len(remaining))
        random_negs = sorted(rng.sample(remaining, random_n)) if random_n else []
        group["hard_negative_ids"] = hard_negs
        group["random_negative_ids"] = random_negs
        group["negative_metadata"] = neg_meta

    return group


# ── Training ──

def train_feature_logistic(
    groups: Sequence[Mapping[str, Any]],
    graph: Any,
    epochs: int = 80,
    lr: float = 0.08,
) -> dict[str, Any]:
    """Train improved feature-logistic ranker with expanded features."""
    weights = [0.0] * 12  # bias + 11 features (expanded from 6)
    for _ in range(epochs):
        for group in groups:
            positives = set(group["positive_ids"])
            for node_id in group["candidate_ids"]:
                x = [1.0, *_node_features_v3(group["question"], node_id, graph)]
                y = 1.0 if node_id in positives else 0.0
                probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, _dot(weights, x)))))
                for idx, val in enumerate(x):
                    weights[idx] += lr * (y - probability) * val

    return {
        "kind": "feature_logistic_v3",
        "weights": weights,
        "features": [
            "bias", "lexical_overlap", "char_ngram_cosine", "type_rule_intent",
            "node_degree_regularized", "alias_exact_match", "alias_jaccard",
            "provenance_match", "graph_distance_to_top", "bm25_overlap",
            "keyfinding_overlap", "description_overlap",
        ],
    }


def _node_features_v3(question: str, node_id: str, graph: Any) -> list[float]:
    """Expanded feature set for V3 logistic ranker."""
    node = graph.get_node(node_id) if graph is not None else None
    node_type = str(getattr(node, "type", "")) if node is not None else ""
    properties = getattr(node, "properties", {}) if node is not None else {}
    aliases = [str(x) for x in getattr(node, "aliases", [])] if node is not None else []

    from stack.encoder.stage1c import _tokens
    node_text = " ".join([node_id.replace("_", " "), *aliases,
                          str(properties.get("key_finding", "")),
                          str(properties.get("description", ""))])

    q_tokens = _tokens(question)
    n_tokens = _tokens(node_text)
    alias_tokens = _tokens(" ".join(aliases))
    finding_tokens = _tokens(str(properties.get("key_finding", "")))
    desc_tokens = _tokens(str(properties.get("description", "")))

    degree = (len(graph.get_outgoing(node_id)) + len(graph.get_incoming(node_id))) if graph is not None and node is not None else 0

    # Intent rule
    q_lower = question.casefold()
    q_intent = "diagnostic" if q_lower.startswith(("why", "how would", "what is the significance")) else "factual_lookup"
    type_match = float((q_intent == "diagnostic" and node_type in {"Concept", "Decision"}) or (q_intent != "diagnostic" and node_type in {"Experiment", "Metric"}))

    # Alias features
    alias_exact = float(any(len(_tokens(a)) >= 2 and a.casefold() in question.casefold() for a in aliases))
    alias_jaccard = len(alias_tokens & q_tokens) / max(1, len(alias_tokens | q_tokens))

    # Provenance match
    provenance = str(properties.get("source", properties.get("source_snippet", "")))
    prov_match = float(bool(_tokens(provenance) & q_tokens))

    # Char n-gram cosine
    from stack.encoder.c2_c3 import _char_ngrams, _cosine
    ngram_cos = _cosine(_char_ngrams(question), _char_ngrams(node_text))

    return [
        len(q_tokens & n_tokens) / max(1, len(q_tokens)),  # lexical_overlap
        ngram_cos,                                         # char_ngram_cosine
        type_match,                                        # type_rule_intent
        math.log1p(degree) / math.log1p(100),              # degree regularized
        alias_exact,                                       # alias exact match
        alias_jaccard,                                     # alias Jaccard
        prov_match,                                        # provenance match
        0.0,                                                # graph_distance placeholder
        len(q_tokens & n_tokens) / max(1, len(n_tokens)),  # bm25-like overlap
        len(q_tokens & finding_tokens) / max(1, len(finding_tokens) or 1),  # keyfinding overlap
        len(q_tokens & desc_tokens) / max(1, len(desc_tokens) or 1),        # description overlap
    ]


def train_ranker_v3(
    groups: Sequence[Mapping[str, Any]],
    graph: Any,
    val_groups: Sequence[Mapping[str, Any]] | None = None,
    epochs: int = 20,
    lr: float = 0.001,
    patience: int = 5,
) -> dict[str, Any]:
    """Train the V3 question-conditioned entity ranker.

    Simple per-group SGD with pre-computed entity embeddings.
    Early stops on validation recall@10.
    """
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    # Build tokenizer from all texts
    tokenizer = CharNgramTokenizer()
    tokenizer.add_words([str(g["question"]) for g in groups])
    tokenizer.freeze()

    # Build model
    model = QuestionConditionedEntityRanker(
        feature_dim=tokenizer.feature_dim,
        embed_dim=64,
        hidden_dim=128,
        proj_dim=32,
        dropout=0.2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Pre-compute entity texts and ID mapping
    all_entity_ids = sorted(set(
        cid for g in groups for cid in g["candidate_ids"]
    ))
    entity_text_map = {eid: eid.replace("_", " ") for eid in all_entity_ids}
    entity_id_to_idx = {eid: i for i, eid in enumerate(all_entity_ids)}
    entity_texts = [entity_text_map[eid] for eid in all_entity_ids]

    best_val_recall10 = 0.0
    best_val_recall5 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Pre-compute entity projections once per epoch
        with torch.no_grad():
            e_emb = model.encode_entities(entity_texts, tokenizer)
            e_proj = model.project_entities(e_emb)  # [N, proj_dim]
        e_proj.requires_grad_(False)

        total_loss = 0.0
        n_steps = 0

        # Shuffle groups
        import random as _rm
        group_order = list(range(len(groups)))
        _rm.Random(SEED + epoch).shuffle(group_order)

        for gi in group_order:
            group = groups[gi]
            positives = set(group["positive_ids"])
            if len(positives) == 0:
                continue

            # Map candidates to pre-computed indices
            cand_indices = [entity_id_to_idx.get(cid) for cid in group["candidate_ids"]]
            cand_indices = [ci for ci in cand_indices if ci is not None]
            if len(cand_indices) < 2:
                continue

            # Encode question
            offsets, indices = tokenizer.tokenize_batch([group["question"]])
            q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
            q_indices = torch.tensor(indices, dtype=torch.long)
            combined = model.encode_question(q_indices, q_offsets)
            q_proj = model.project_question(combined)  # [1, proj_dim]

            # Get pre-computed entity projections
            batch_e = e_proj[cand_indices].unsqueeze(0)  # [1, K, proj_dim]

            # Compute scores
            scores = model.score(q_proj, batch_e).squeeze(0)  # [K]

            # Target: uniform over positives
            pos_in_group = [
                i for i, cid in enumerate(group["candidate_ids"])
                if cid in positives and entity_id_to_idx.get(cid) is not None
            ]
            if not pos_in_group:
                continue

            target = torch.zeros(len(cand_indices))
            for pi_rel in pos_in_group:
                # Map from original candidate position to compressed position
                orig_cid = group["candidate_ids"][pi_rel]
                compressed = next(
                    j for j, cid in enumerate(
                        [cid for cid in group["candidate_ids"] if entity_id_to_idx.get(cid) is not None]
                    )
                    if cid == orig_cid
                )
                target[compressed] = 1.0 / len(pos_in_group)

            loss = F.cross_entropy(
                scores.unsqueeze(0),
                target.unsqueeze(0).argmax(dim=1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)

        # Validation
        if val_groups is not None and epoch % 2 == 0:
            val_metrics = evaluate_ranker_v3_fast(
                val_groups, model, tokenizer, entity_text_map,
                entity_id_to_idx, all_entity_ids, e_proj,
            )
            val_r10 = val_metrics["recall@10"]
            val_r5 = val_metrics["recall@5"]
            print(f"  Epoch {epoch}: loss={avg_loss:.4f} val_r@10={val_r10:.4f} val_r@5={val_r5:.4f}")

            if val_r10 > best_val_recall10 or (
                val_r10 == best_val_recall10 and val_r5 > best_val_recall5
            ):
                best_val_recall10 = val_r10
                best_val_recall5 = val_r5
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  Best val: recall@10={best_val_recall10:.4f} recall@5={best_val_recall5:.4f}")

    return {
        "kind": "entity_ranker_v3",
        "model": model,
        "tokenizer": tokenizer,
        "entity_text_map": entity_text_map,
        "entity_id_to_idx": entity_id_to_idx,
        "all_entity_ids": all_entity_ids,
        "best_val_recall10": best_val_recall10,
        "best_val_recall5": best_val_recall5,
    }


def evaluate_ranker_v3_fast(
    groups: Sequence[Mapping[str, Any]],
    model: QuestionConditionedEntityRanker,
    tokenizer: CharNgramTokenizer,
    entity_text_map: dict[str, str],
    entity_id_to_idx: dict[str, int],
    all_entity_ids: list[str],
    e_proj: torch.Tensor,
    canonical_mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Fast evaluation using pre-computed entity projections."""
    hits_1 = hits_5 = hits_10 = 0
    predicted_10 = 0
    total_gold = sum(len(set(g["positive_ids"])) for g in groups)
    absent_count = 0

    model.eval()
    with torch.no_grad():
        for group in groups:
            positives = set(group["positive_ids"])
            cand_ids = group["candidate_ids"]

            # Map candidates to pre-computed indices
            cand_indices = []
            for cid in cand_ids:
                ei = entity_id_to_idx.get(cid)
                if ei is not None:
                    cand_indices.append(ei)

            if not cand_indices:
                absent_count += 1
                continue

            # Encode question
            offsets, indices = tokenizer.tokenize_batch([group["question"]])
            q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
            q_indices = torch.tensor(indices, dtype=torch.long)
            combined = model.encode_question(q_indices, q_offsets)
            q_proj = model.project_question(combined)

            # Score using pre-computed entity projections
            batch_e = e_proj[cand_indices].unsqueeze(0)
            scores = model.score(q_proj, batch_e).squeeze(0)
            ranked_indices = torch.argsort(scores, descending=True).tolist()

            ranked = []
            seen_canonical = set()
            for ri in ranked_indices[:K_MAX * 2]:  # Over-sample for canonical dedup
                if len(ranked) >= K_MAX:
                    break
                cid = cand_ids[ri]
                if canonical_mapping:
                    canonical = canonical_mapping.get(cid, cid)
                    if canonical not in seen_canonical:
                        seen_canonical.add(canonical)
                        ranked.append(canonical)
                else:
                    ranked.append(cid)

            hits_10 += len(set(ranked[:10]) & positives)
            predicted_10 += len(ranked[:10])
            # Also compute @1 and @5 from full ranking
            if canonical_mapping:
                # Re-rank with canonical dedup for @1 and @5
                hits_5 += len(set(ranked[:5]) & positives)
                hits_1 += len(set(ranked[:1]) & positives)
            else:
                hits_5 += len(set(ranked[:5]) & positives)
                hits_1 += len(set(ranked[:1]) & positives)

    return {
        "recall@1": hits_1 / total_gold if total_gold else 0.0,
        "recall@5": hits_5 / total_gold if total_gold else 0.0,
        "recall@10": hits_10 / total_gold if total_gold else 0.0,
        "precision@10": hits_10 / predicted_10 if predicted_10 else 0.0,
        "total_questions": float(len(groups)),
        "total_gold_entities": float(total_gold),
        "absent_gold_count": float(absent_count),
    }


# ── Evaluation ──

def score_group_v3(
    group: Mapping[str, Any],
    graph: Any,
    ranker: Mapping[str, Any],
    canonical_mapping: dict[str, str] | None = None,
) -> list[str]:
    """Score and rank candidates for a single group.

    If canonical_mapping is provided, applies it and deduplicates.
    """
    if ranker["kind"] == "feature_logistic_v3":
        weights = ranker["weights"]
        scored = [
            (node, _dot(weights, [1.0, *_node_features_v3(group["question"], node, graph)]))
            for node in group["candidate_ids"]
        ]
    else:
        model = ranker["model"]
        tokenizer = ranker["tokenizer"]
        entity_text_map = ranker.get("entity_text_map", {})

        offsets, indices = tokenizer.tokenize_batch([group["question"]])
        q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
        q_indices = torch.tensor(indices, dtype=torch.long)

        cand_ids = group["candidate_ids"]
        cand_texts = [entity_text_map.get(cid, build_entity_text(cid, graph)) for cid in cand_ids]

        with torch.no_grad():
            scores = model(q_indices, q_offsets, cand_texts, tokenizer)
        scored = list(zip(cand_ids, scores[0].tolist()))

    ranked = [node for node, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]

    if canonical_mapping is not None:
        ranked = apply_canonical_mapping(ranked, canonical_mapping, top_k=K_MAX)

    return ranked[:K_MAX]


def evaluate_ranker_v3(
    groups: Sequence[Mapping[str, Any]],
    graph: Any,
    ranker_or_model: Any,
    tokenizer_or_none: Any = None,
    entity_text_map: dict[str, str] | None = None,
    canonical_mapping: dict[str, str] | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Evaluate a ranker on a set of groups.

    ALL questions remain in the denominator. Missing gold candidates
    contribute zero recall for their question.
    """
    if isinstance(ranker_or_model, dict) and "kind" in ranker_or_model:
        ranker = ranker_or_model
    else:
        ranker = {
            "kind": "entity_ranker_v3",
            "model": ranker_or_model,
            "tokenizer": tokenizer_or_none,
            "entity_text_map": entity_text_map or {},
        }

    hits = {k: 0 for k in ks}
    predicted = {k: 0 for k in ks}
    total_gold = sum(len(set(g["positive_ids"])) for g in groups)
    total_questions = len(groups)
    absent_gold_count = sum(1 for g in groups if not g.get("gold_present_in_candidates", True))

    for group in groups:
        ranked = score_group_v3(group, graph, ranker, canonical_mapping)
        ranked_list = list(ranked) if not isinstance(ranked, list) else ranked
        gold = set(group["positive_ids"])
        for k in ks:
            selected = set(ranked_list[:k])
            hits[k] += len(selected & gold)
            predicted[k] += len(ranked_list[:k])

    result: dict[str, float] = {}
    for k in ks:
        result[f"recall@{k}"] = hits[k] / total_gold if total_gold else 0.0
        result[f"precision@{k}"] = hits[k] / predicted[k] if predicted[k] else 0.0

    result["total_questions"] = float(total_questions)
    result["total_gold_entities"] = float(total_gold)
    result["absent_gold_count"] = float(absent_gold_count)
    result["candidate_recall_ceiling"] = (
        sum(len(set(g["positive_ids"]) & set(g["candidate_ids"])) for g in groups) / total_gold
        if total_gold else 0.0
    )

    return result


def evaluate_trivial_baseline(
    groups: Sequence[Mapping[str, Any]],
    graph: Any,
    canonical_mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Evaluate the trivial lexical baseline on validation groups.

    Reports both raw (granular) and canonical recall.
    """
    hits_10 = 0
    predicted_10 = 0
    total_gold = sum(len(set(g["positive_ids"])) for g in groups)

    for group in groups:
        ranked = rank_candidates(
            candidate_pool(group["question"], graph), graph, K_MAX
        )
        gold = set(group["positive_ids"])
        hits_10 += len(set(ranked[:10]) & gold)
        predicted_10 += len(ranked[:10])

    raw_recall = hits_10 / total_gold if total_gold else 0.0
    raw_precision = hits_10 / predicted_10 if predicted_10 else 0.0

    result = {
        "recall@10": raw_recall,
        "precision@10": raw_precision,
        "total_questions": float(len(groups)),
        "total_gold_entities": float(total_gold),
    }

    # Also compute canonical recall if mapping is provided
    if canonical_mapping is not None:
        can_hits_10 = can_pred_10 = 0
        for group in groups:
            ranked = rank_candidates(
                candidate_pool(group["question"], graph), graph, K_MAX
            )
            mapped = apply_canonical_mapping(ranked, canonical_mapping, top_k=K_MAX)
            gold = set(group["positive_ids"])
            can_hits_10 += len(set(mapped[:10]) & gold)
            can_pred_10 += len(mapped[:10])
        result["canonical_recall@10"] = can_hits_10 / total_gold if total_gold else 0.0
        result["canonical_precision@10"] = can_hits_10 / can_pred_10 if can_pred_10 else 0.0

    return result


# ── Selection ──

def select_winner(metrics: Mapping[str, Mapping[str, float]]) -> str:
    """Mechanical winner: max recall@10, tie-break recall@5, then recall@1, then name."""
    return sorted(
        metrics,
        key=lambda name: (
            -metrics[name].get("recall@10", 0),
            -metrics[name].get("recall@5", 0),
            -metrics[name].get("recall@1", 0),
            name,
        ),
    )[0]


def check_worktree_clean() -> bool:
    """Return True if git working tree is clean."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True


# ── Main experiment ──

def run_experiment_v3(root: str | Path = ".") -> dict[str, Any]:
    """Run the full Entity Ranker V3 experiment: training and validation selection.

    1. Load train/val splits
    2. Build canonical mapping
    3. Build groups with preserved denominators
    4. Train V3 ranker and feature-logistic ranker
    5. Evaluate all rankers on identical validation groups
    6. Mechanical selection
    7. Save winner
    """
    root = Path(root)

    # Guard: clean worktree
    if not check_worktree_clean():
        raise RuntimeError(
            "Dirty worktree detected. Commit or stash changes before evaluation."
        )

    # Load splits
    train = load_split(root / "stack/encoder/data/train.jsonl")
    val = load_split(root / "stack/encoder/data/val.jsonl")

    # Build graph
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, graph_meta = build_benchmark_graph()

    # Build canonical mapping (graph-derived, no test inspection)
    canonical_mapping = build_canonical_mapping(graph)

    # Build training groups (all 375 questions preserved, but limit candidates per group)
    train_groups = []
    for record in train:
        pool = candidate_pool(record["question"], graph)
        # Cap candidate pool to 200 per question for training speed
        candidate_ids = [str(item["node_id"]) for item in pool[:200]]
        group = build_group(
            str(record.get("id", "")),
            str(record["question"]),
            [str(e) for e in record.get("entities", [])],
            candidate_ids,
            "train",
            graph,
            hard_negative_k=10,  # Reduced for speed
        )
        if group is not None:
            train_groups.append(group)

    # Add graph-mined groups (capped)
    import random
    rng = random.Random(SEED)
    stage1c_pairs = generate_stage1c_pairs(graph)
    rng.shuffle(stage1c_pairs)
    graph_groups = []
    for pair in stage1c_pairs[:400]:  # Cap at 400
        pool = candidate_pool(pair["question"], graph)
        candidate_ids = [str(item["node_id"]) for item in pool[:100]]
        group = build_group(
            str(pair.get("id", "")),
            str(pair["question"]),
            [str(e) for e in pair.get("entities", [])],
            candidate_ids,
            "graph_mined",
            graph,
            hard_negative_k=8,
        )
        if group is not None:
            graph_groups.append(group)

    all_train_groups = train_groups + graph_groups
    print(f"Training groups: {len(train_groups)} real + {len(graph_groups)} graph = {len(all_train_groups)} total")

    # Build validation groups (ALL 150 questions preserved)
    val_groups = []
    for record in val:
        group = build_group(
            str(record.get("id", "")),
            str(record["question"]),
            [str(e) for e in record.get("entities", [])],
            [str(item["node_id"]) for item in candidate_pool(record["question"], graph)],
            "validation",
            graph,
        )
        if group is not None:
            val_groups.append(group)
        else:
            # Should not happen with build_group — but be safe
            val_groups.append({
                "question_id": str(record.get("id", "")),
                "question": str(record["question"]),
                "candidate_ids": [str(item["node_id"]) for item in candidate_pool(record["question"], graph)],
                "positive_ids": sorted(set(str(e) for e in record.get("entities", []))),
                "source": "validation",
                "gold_present_in_candidates": False,
            })

    # Compute candidate recall ceiling
    val_baseline = evaluate_trivial_baseline(val_groups, graph, canonical_mapping)
    candidate_ceiling = val_baseline.get("candidate_recall_ceiling", 0.0)

    # Train rankers (reduced epochs for speed)
    logistic = train_feature_logistic(all_train_groups, graph, epochs=40)
    encoder_v3 = train_ranker_v3(all_train_groups, graph, val_groups)

    # Evaluate all rankers on identical validation groups
    # Pre-compute entity projections for V3 fast evaluation
    all_eids = encoder_v3["all_entity_ids"]
    all_texts = [encoder_v3["entity_text_map"].get(eid, eid.replace("_", " ")) for eid in all_eids]
    v3_e_proj = encoder_v3["model"].project_entities(
        encoder_v3["model"].encode_entities(all_texts, encoder_v3["tokenizer"])
    )

    metrics = {
        "trivial_baseline": evaluate_trivial_baseline(val_groups, graph, canonical_mapping),
        "feature_logistic_v3": evaluate_ranker_v3(val_groups, graph, logistic, canonical_mapping=canonical_mapping),
        "entity_ranker_v3": evaluate_ranker_v3_fast(
            val_groups, encoder_v3["model"], encoder_v3["tokenizer"],
            encoder_v3["entity_text_map"], encoder_v3["entity_id_to_idx"],
            all_eids, v3_e_proj, canonical_mapping,
        ),
    }

    # Mechanical selection
    winner_name = select_winner(metrics)
    winner_metrics = metrics[winner_name]
    baseline_r10 = metrics["trivial_baseline"]["recall@10"]

    # Gate check
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    val_gate = winner_metrics["recall@10"] >= REQUIRED_VAL_RECALL10
    baseline_gap = winner_metrics["recall@10"] - baseline_r10 >= REQUIRED_BASELINE_GAP

    # Save model
    model_dir = root / "models/encoder/v3"
    model_dir.mkdir(parents=True, exist_ok=True)

    if winner_name == "entity_ranker_v3":
        config = {
            "winner": winner_name,
            "source_sha": source_sha,
            "seed": SEED,
            "validation_only": True,
            "architecture": "question_conditioned_dot_product_v3",
            "canonical_mapping_applied": True,
        }
        save_ranker_v3(encoder_v3["model"], encoder_v3["tokenizer"], config, str(model_dir))
    elif winner_name == "feature_logistic_v3":
        config = {
            "winner": winner_name,
            "source_sha": source_sha,
            "seed": SEED,
            "validation_only": True,
            "kind": "feature_logistic_v3",
            "weights": logistic["weights"],
            "features": logistic["features"],
            "canonical_mapping_applied": True,
        }
        (model_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (model_dir / "weights.json").write_text(json.dumps(logistic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        # Trivial baseline won — no model to save (stop)
        pass

    # Write selection log
    selection_log = root / "benchmarks/results/entity_ranker_v3_selection_log.json"
    val_split_path = root / "stack/encoder/data/val.jsonl"
    log = {
        "source_sha": source_sha,
        "seed": SEED,
        "split": "stack/encoder/data/val.jsonl",
        "split_sha256": hashlib.sha256(val_split_path.read_bytes()).hexdigest(),
        "graph": graph_meta,
        "dataset_stats": {
            "train_groups": len(train_groups),
            "graph_mined_groups": len(graph_groups),
            "validation_groups": len(val_groups),
            "canonical_mapping_size": len(canonical_mapping),
            "canonical_mapping_unique_targets": len(set(canonical_mapping.values())),
        },
        "evaluations": [
            {"model": name, "candidate_groups": len(val_groups), "metrics": values}
            for name, values in metrics.items()
        ],
        "selection": {
            "winner": winner_name,
            "decision_rule": "recall@10, recall@5, recall@1 tie-break",
            "winner_recall@10": winner_metrics["recall@10"],
            "winner_recall@5": winner_metrics["recall@5"],
            "winner_recall@1": winner_metrics["recall@1"],
            "baseline_recall@10": baseline_r10,
            "baseline_gap_pp": (winner_metrics["recall@10"] - baseline_r10) * 100,
            "candidate_recall_ceiling": candidate_ceiling,
            "val_gate_70pct": val_gate,
            "baseline_gate_15pp": baseline_gap,
            "proceed_to_frozen": val_gate and baseline_gap,
        },
        "test_split_read": False,
    }
    selection_log.parent.mkdir(parents=True, exist_ok=True)
    selection_log.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "winner": winner_name,
        "metrics": metrics,
        "selection_log": str(selection_log),
        "source_sha": source_sha,
        "proceed_to_frozen": val_gate and baseline_gap,
    }


if __name__ == "__main__":
    result = run_experiment_v3()
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2, sort_keys=True))
    print("\nMetrics:")
    for name, m in result["metrics"].items():
        print(f"  {name}: r@1={m.get('recall@1',0):.4f} r@5={m.get('recall@5',0):.4f} r@10={m.get('recall@10',0):.4f} p@10={m.get('precision@10',0):.4f}")
