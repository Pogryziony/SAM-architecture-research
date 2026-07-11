"""Entity Ranker V3 — training, selection, and evaluation.

Corrected training loop with:
- All positive entities per question
- Multiple diverse hard negatives per question
- Listwise softmax cross-entropy loss
- Deterministic mini-batches with source-balanced sampling
- Early stopping on validation recall@10 (recall@5 as tie-break)
- CPU-only execution
- Fixed random seeds
- Immutable, timestamped output artifacts
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from stack.encoder.char_tokenizer import CharNgramTokenizer
from stack.encoder.entity_ranker_v3 import (
    QuestionConditionedEntityRanker,
    save_ranker_v3,
)
from stack.encoder.entity_text import build_entity_text
from stack.encoder.canonical_mapping import (
    build_canonical_mapping,
    apply_canonical_mapping,
    export_canonical_mapping_metadata,
)
from stack.encoder.hard_negative_miner import mine_hard_negatives_group
from stack.encoder.trivial_baseline import candidate_pool, rank_candidates
from stack.encoder.natural_templates import generate_balanced_dataset
from stack.encoder.experiment_guard import check_worktree_clean
from stack.encoder.c2_c3 import _dot


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
    inject_missing_gold: bool | None = None,
) -> dict[str, Any] | None:
    """Build a single training/evaluation group.

    Evaluation groups preserve the original candidate pool. Training groups
    may inject missing gold entities so the ranker has a supervised positive.
    The pre-injection coverage flag is always retained for diagnostics.
    """
    import random as _random
    rng = _random.Random(seed)

    positive_set = set(str(x) for x in positive_ids)
    candidate_set = list(dict.fromkeys(str(x) for x in candidate_ids))

    gold_present_before_injection = positive_set.issubset(set(candidate_set))
    if inject_missing_gold is None:
        inject_missing_gold = source != "validation"
    if inject_missing_gold:
        for pid in positive_set:
            if pid not in candidate_set:
                candidate_set.append(pid)

    group = {
        "question_id": question_id,
        "question": question,
        "candidate_ids": candidate_set,
        "positive_ids": sorted(positive_set),
        "source": source,
        "gold_present_in_candidates": gold_present_before_injection,
        "gold_injected_for_training": bool(
            inject_missing_gold and not gold_present_before_injection
        ),
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


def build_training_group(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    """Build a supervised group; missing gold may be injected for training."""
    kwargs["inject_missing_gold"] = True
    return build_group(*args, **kwargs)


def build_evaluation_group(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    """Build an evaluation group without ever modifying its candidate pool."""
    kwargs["inject_missing_gold"] = False
    return build_group(*args, **kwargs)


# ── Training ──

def multi_positive_listwise_loss(
    scores: torch.Tensor, positive_indices: Sequence[int]
) -> torch.Tensor:
    """Negative log probability assigned to any positive candidate."""
    if not positive_indices:
        raise ValueError("positive_indices must not be empty")
    positive_scores = scores[
        torch.tensor(list(positive_indices), dtype=torch.long, device=scores.device)
    ]
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(
        positive_scores, dim=0
    )

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
    canonical_mapping: dict[str, str] | None = None,
    epochs: int = 20,
    lr: float = 0.001,
    patience: int = 5,
) -> dict[str, Any]:
    """Train the V3 question-conditioned entity ranker.

    End-to-end per-group listwise training. Both the question and entity
    encoders receive gradients. Early stopping uses the same canonical metric
    as model selection when a canonical mapping is supplied.
    """
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    # Build a graph-wide entity vocabulary. Graph text is public model input,
    # not validation supervision, and prevents unseen validation candidates
    # from disappearing during scoring.
    all_entity_ids = sorted(str(node_id) for node_id in graph._nodes)
    entity_text_map = {
        entity_id: build_entity_text(entity_id, graph)
        for entity_id in all_entity_ids
    }
    entity_id_to_idx = {
        entity_id: index for index, entity_id in enumerate(all_entity_ids)
    }

    # Build tokenizer from both questions and the full entity representations.
    tokenizer = CharNgramTokenizer()
    tokenizer.add_words([str(g["question"]) for g in groups])
    tokenizer.add_words(list(entity_text_map.values()))
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

    best_val_recall10 = 0.0
    best_val_recall5 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

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

            indexed_candidates = [
                cid for cid in group["candidate_ids"]
                if cid in entity_id_to_idx
            ]
            if len(indexed_candidates) < 2:
                continue

            # Encode question
            offsets, indices = tokenizer.tokenize_batch([group["question"]])
            q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
            q_indices = torch.tensor(indices, dtype=torch.long)
            combined = model.encode_question(q_indices, q_offsets)
            q_proj = model.project_question(combined)  # [1, proj_dim]

            # Encode candidate entities inside the autograd graph. This is
            # essential: V3 must learn both sides of the interaction.
            candidate_texts = [entity_text_map[cid] for cid in indexed_candidates]
            e_emb = model.encode_entities(candidate_texts, tokenizer)
            batch_e = model.project_entities(e_emb).unsqueeze(0)
            scores = model.score(q_proj, batch_e).squeeze(0)  # [K]

            pos_in_group = [
                index for index, cid in enumerate(indexed_candidates)
                if cid in positives
            ]
            if not pos_in_group:
                continue

            # Multi-positive listwise loss. Probability mass assigned to any
            # gold entity is rewarded; no positive is discarded via argmax.
            loss = multi_positive_listwise_loss(scores, pos_in_group)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)

        # Validation
        if val_groups is not None and epoch % 2 == 0:
            val_metrics = evaluate_ranker_v3(
                val_groups,
                graph,
                model,
                tokenizer,
                entity_text_map,
                canonical_mapping,
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
            if not positives.issubset(set(cand_ids)):
                absent_count += len(positives - set(cand_ids))

            # Keep IDs aligned with the compressed projection indices.
            indexed_candidates = [
                (cid, entity_id_to_idx[cid])
                for cid in cand_ids
                if cid in entity_id_to_idx
            ]

            if not indexed_candidates:
                continue

            # Encode question
            offsets, indices = tokenizer.tokenize_batch([group["question"]])
            q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
            q_indices = torch.tensor(indices, dtype=torch.long)
            combined = model.encode_question(q_indices, q_offsets)
            q_proj = model.project_question(combined)

            # Score using pre-computed entity projections
            batch_e = e_proj[
                [entity_index for _cid, entity_index in indexed_candidates]
            ].unsqueeze(0)
            scores = model.score(q_proj, batch_e).squeeze(0)
            ranked_indices = torch.argsort(scores, descending=True).tolist()

            ranked = []
            seen_canonical = set()
            for ri in ranked_indices:
                if len(ranked) >= K_MAX:
                    break
                cid = indexed_candidates[ri][0]
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
        "candidate_recall_ceiling": (
            sum(
                len(set(group["positive_ids"]) & set(group["candidate_ids"]))
                for group in groups
            ) / total_gold
            if total_gold else 0.0
        ),
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
    ceiling_hits = 0
    for group in groups:
        candidates = list(group["candidate_ids"])
        if canonical_mapping is not None:
            candidates = apply_canonical_mapping(
                candidates, canonical_mapping, top_k=max(1, len(candidates))
            )
        ceiling_hits += len(set(group["positive_ids"]) & set(candidates))
    result["candidate_recall_ceiling"] = (
        ceiling_hits / total_gold if total_gold else 0.0
    )

    return result


def evaluate_ranker_raw_and_canonical(
    groups: Sequence[Mapping[str, Any]],
    graph: Any,
    ranker_or_model: Any,
    tokenizer_or_none: Any = None,
    entity_text_map: dict[str, str] | None = None,
    canonical_mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Report raw and canonical metrics while selecting on canonical recall."""
    raw = evaluate_ranker_v3(
        groups, graph, ranker_or_model, tokenizer_or_none, entity_text_map, None
    )
    canonical = evaluate_ranker_v3(
        groups,
        graph,
        ranker_or_model,
        tokenizer_or_none,
        entity_text_map,
        canonical_mapping,
    )
    result: dict[str, float] = {
        "total_questions": canonical["total_questions"],
        "total_gold_entities": canonical["total_gold_entities"],
        "absent_gold_count": raw["absent_gold_count"],
        "raw_candidate_recall_ceiling": raw["candidate_recall_ceiling"],
        "canonical_candidate_recall_ceiling": canonical["candidate_recall_ceiling"],
    }
    for k in (1, 5, 10):
        result[f"raw_recall@{k}"] = raw[f"recall@{k}"]
        result[f"raw_precision@{k}"] = raw[f"precision@{k}"]
        result[f"canonical_recall@{k}"] = canonical[f"recall@{k}"]
        result[f"canonical_precision@{k}"] = canonical[f"precision@{k}"]
        result[f"recall@{k}"] = canonical[f"recall@{k}"]
        result[f"precision@{k}"] = canonical[f"precision@{k}"]
    return result


def evaluate_trivial_baseline(
    groups: Sequence[Mapping[str, Any]],
    graph: Any,
    canonical_mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Evaluate the baseline on the exact candidate pools used by rankers."""
    raw_hits = {1: 0, 5: 0, 10: 0}
    canonical_hits = {1: 0, 5: 0, 10: 0}
    raw_predicted = {1: 0, 5: 0, 10: 0}
    canonical_predicted = {1: 0, 5: 0, 10: 0}
    total_gold = sum(len(set(g["positive_ids"])) for g in groups)
    raw_ceiling_hits = 0
    canonical_ceiling_hits = 0

    for group in groups:
        pool_scores = {
            str(item["node_id"]): float(item["lexical_score"])
            for item in candidate_pool(group["question"], graph)
        }
        candidate_records = [
            {"node_id": candidate_id, "lexical_score": pool_scores.get(candidate_id, 0.0)}
            for candidate_id in group["candidate_ids"]
            if graph.get_node(candidate_id) is not None
        ]
        ranked = rank_candidates(candidate_records, graph, len(candidate_records))
        gold = set(group["positive_ids"])
        raw_ceiling_hits += len(set(group["candidate_ids"]) & gold)
        mapped_all = (
            apply_canonical_mapping(ranked, canonical_mapping, top_k=max(1, len(ranked)))
            if canonical_mapping is not None else ranked
        )
        canonical_ceiling_hits += len(set(mapped_all) & gold)
        canonical_ranked = (
            apply_canonical_mapping(ranked, canonical_mapping, top_k=K_MAX)
            if canonical_mapping is not None else ranked[:K_MAX]
        )
        for k in (1, 5, 10):
            raw_hits[k] += len(set(ranked[:k]) & gold)
            raw_predicted[k] += len(ranked[:k])
            canonical_hits[k] += len(set(canonical_ranked[:k]) & gold)
            canonical_predicted[k] += len(canonical_ranked[:k])

    result = {
        "total_questions": float(len(groups)),
        "total_gold_entities": float(total_gold),
        "raw_candidate_recall_ceiling": (
            raw_ceiling_hits / total_gold if total_gold else 0.0
        ),
        "canonical_candidate_recall_ceiling": (
            canonical_ceiling_hits / total_gold if total_gold else 0.0
        ),
    }
    for k in (1, 5, 10):
        result[f"raw_recall@{k}"] = raw_hits[k] / total_gold if total_gold else 0.0
        result[f"raw_precision@{k}"] = (
            raw_hits[k] / raw_predicted[k] if raw_predicted[k] else 0.0
        )
        result[f"canonical_recall@{k}"] = (
            canonical_hits[k] / total_gold if total_gold else 0.0
        )
        result[f"canonical_precision@{k}"] = (
            canonical_hits[k] / canonical_predicted[k]
            if canonical_predicted[k] else 0.0
        )
        # Selection aliases always use the canonical metric when a mapping is
        # supplied, otherwise they use the raw metric.
        prefix = "canonical" if canonical_mapping is not None else "raw"
        result[f"recall@{k}"] = result[f"{prefix}_recall@{k}"]
        result[f"precision@{k}"] = result[f"{prefix}_precision@{k}"]

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


# ── Main experiment ──

def _require_new_path(path: Path) -> Path:
    """Require that a path does not yet exist; fail if it does."""
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {path}. "
            "Each run must produce a unique timestamped artifact."
        )
    return path


def _write_json_artifact(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON artifact. Fails if path already exists."""
    _require_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.rename(path)


def run_experiment_v3(root: str | Path = ".") -> dict[str, Any]:
    """Run the full Entity Ranker V3 experiment: training and validation selection.

    1. Load train/val splits
    2. Build canonical mapping
    3. Build groups with preserved denominators
    4. Train V3 ranker and feature-logistic ranker
    5. Evaluate all rankers on identical validation groups
    6. Mechanical selection
    7. Save winner to immutable timestamped paths
    """
    root = Path(root)

    # Guard: clean worktree
    if not check_worktree_clean(root):
        raise RuntimeError(
            "Dirty worktree detected. Commit or stash changes before evaluation."
        )

    # ── Timestamped run identity ──
    utc_now = datetime.now(timezone.utc)
    run_ts = utc_now.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"entity_ranker_v3_{run_ts}"
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    # Load splits
    train_path = root / "stack/encoder/data/train.jsonl"
    val_path = root / "stack/encoder/data/val.jsonl"
    train = load_split(train_path)
    val = load_split(val_path)

    # Build graph
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, graph_meta = build_benchmark_graph()

    # Build canonical mapping (graph-derived, no test inspection)
    canonical_mapping = build_canonical_mapping(graph)
    canonical_meta = export_canonical_mapping_metadata(canonical_mapping, graph)

    # Build the preregistered source-balanced training dataset. Natural V3
    # templates are used here instead of the legacy Stage 1C generator.
    balanced_records = generate_balanced_dataset([dict(row) for row in train], graph)
    train_groups = []
    for record in balanced_records:
        pool = candidate_pool(record["question"], graph)
        candidate_ids = [str(item["node_id"]) for item in pool[:200]]
        source = str(record.get("source", "real_train"))
        group = build_training_group(
            str(record.get("id", "")),
            str(record["question"]),
            [str(e) for e in record.get("entities", [])],
            candidate_ids,
            source,
            graph,
            hard_negative_k=10,
        )
        if group is not None:
            train_groups.append(group)
    all_train_groups = train_groups
    source_counts = Counter(str(group["source"]) for group in all_train_groups)
    print(f"Training groups: {len(all_train_groups)} total; sources={dict(source_counts)}")

    # Build validation groups (ALL 150 questions preserved)
    val_groups = []
    for record in val:
        group = build_evaluation_group(
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
    candidate_ceiling = val_baseline["canonical_candidate_recall_ceiling"]

    # ── Compute data hashes for provenance ──
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    val_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()

    # Train rankers (reduced epochs for speed)
    logistic = train_feature_logistic(all_train_groups, graph, epochs=40)
    encoder_v3 = train_ranker_v3(
        all_train_groups, graph, val_groups, canonical_mapping=canonical_mapping
    )

    metrics = {
        "trivial_baseline": evaluate_trivial_baseline(val_groups, graph, canonical_mapping),
        "feature_logistic_v3": evaluate_ranker_raw_and_canonical(
            val_groups, graph, logistic, canonical_mapping=canonical_mapping
        ),
        "entity_ranker_v3": evaluate_ranker_raw_and_canonical(
            val_groups,
            graph,
            encoder_v3["model"],
            encoder_v3["tokenizer"],
            encoder_v3["entity_text_map"],
            canonical_mapping,
        ),
    }

    # Mechanical selection
    winner_name = select_winner(metrics)
    winner_metrics = metrics[winner_name]
    baseline_r10 = metrics["trivial_baseline"]["recall@10"]

    # Gate check
    val_gate = winner_metrics["recall@10"] >= REQUIRED_VAL_RECALL10
    baseline_gap = winner_metrics["recall@10"] - baseline_r10 >= REQUIRED_BASELINE_GAP

    # ── Save model to immutable timestamped directory ──
    model_dir = root / "models" / "encoder" / run_id
    _require_new_path(model_dir)
    model_dir.mkdir(parents=True)

    if winner_name == "entity_ranker_v3":
        config = {
            "run_id": run_id,
            "run_timestamp_utc": utc_now.isoformat(),
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
            "run_id": run_id,
            "run_timestamp_utc": utc_now.isoformat(),
            "winner": winner_name,
            "source_sha": source_sha,
            "seed": SEED,
            "validation_only": True,
            "kind": "feature_logistic_v3",
            "weights": logistic["weights"],
            "features": logistic["features"],
            "canonical_mapping_applied": True,
        }
        _write_json_artifact(model_dir / "config.json", config)
        _write_json_artifact(model_dir / "weights.json", logistic)
    else:
        # Trivial baseline won — still record the result
        config = {
            "run_id": run_id,
            "run_timestamp_utc": utc_now.isoformat(),
            "winner": winner_name,
            "source_sha": source_sha,
            "seed": SEED,
            "validation_only": True,
        }
        _write_json_artifact(model_dir / "config.json", config)

    # ── Write immutable selection log ──
    selection_log = root / "benchmarks" / "results" / f"entity_ranker_v3_selection_{run_ts}.json"
    log = {
        "run_id": run_id,
        "run_timestamp_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "seed": SEED,
        "split": "stack/encoder/data/val.jsonl",
        "split_sha256": val_hash,
        "train_split_sha256": train_hash,
        "graph": graph_meta,
        "canonical_mapping": canonical_meta,
        "dataset_stats": {
            "training_groups": len(all_train_groups),
            "training_source_counts": dict(source_counts),
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
        "model_dir": str(model_dir.relative_to(root)),
    }
    _write_json_artifact(selection_log, log)

    return {
        "winner": winner_name,
        "metrics": metrics,
        "selection_log": str(selection_log),
        "model_dir": str(model_dir),
        "source_sha": source_sha,
        "run_id": run_id,
        "proceed_to_frozen": val_gate and baseline_gap,
    }


if __name__ == "__main__":
    result = run_experiment_v3()
    display = {
        "winner": result["winner"],
        "source_sha": result["source_sha"],
        "run_id": result["run_id"],
        "selection_log": result["selection_log"],
        "model_dir": result["model_dir"],
        "proceed_to_frozen": result["proceed_to_frozen"],
    }
    print(json.dumps(display, indent=2, sort_keys=True))
    print("\nMetrics:")
    for name, m in result["metrics"].items():
        print(f"  {name}: r@1={m.get('recall@1',0):.4f} r@5={m.get('recall@5',0):.4f} r@10={m.get('recall@10',0):.4f} p@10={m.get('precision@10',0):.4f}")
