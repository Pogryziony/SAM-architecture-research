"""
Entity extraction evaluation script for NEXUS Phase 1->2 gate.

Loads a labeled set of text snippets with ground-truth entities and relations,
runs the entity_extractor on each snippet, and computes precision/recall/F1.

Usage:
    python experiments/entity-extraction/evaluate_extraction.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from nexus.graph import Node, Edge
from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.entity_extractor import extract_from_markdown
from nexus.ingestion.relation_extractor import extract_relations
from nexus.ingestion.normalizer import canonicalize

# Import from hyphenated directory via exec (Python can't import "entity-extraction" as module name)
_corpus_quality_path = _project_root / "experiments" / "entity-extraction" / "corpus_quality.py"
_cq_ns: dict[str, Any] = {}
exec(_corpus_quality_path.read_text(encoding="utf-8"), _cq_ns)
corpus_health_report = _cq_ns["corpus_health_report"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELED_SET = _project_root / "experiments" / "entity-extraction" / "labeled_set.jsonl"
RESULTS_FILE = _project_root / "experiments" / "entity-extraction" / "evaluation_results.json"

# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_match(name_a: str, name_b: str, threshold: float = 0.85) -> bool:
    """Return True if two entity names are similar enough (case-insensitive)."""
    a_lower = name_a.strip().lower()
    b_lower = name_b.strip().lower()
    if a_lower == b_lower:
        return True
    if a_lower in b_lower or b_lower in a_lower:
        return True
    return SequenceMatcher(None, a_lower, b_lower).ratio() >= threshold


def _entity_in_set(entity: dict, candidate_set: list[dict], fuzzy: bool) -> bool:
    """Check if *entity* matches any entry in *candidate_set* by name."""
    name = entity["name"]
    for cand in candidate_set:
        if fuzzy:
            if _fuzzy_match(name, cand["name"]):
                return True
        else:
            if name.strip().lower() == cand["name"].strip().lower():
                return True
    return False


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_entity_metrics(
    predicted: list[dict], ground_truth: list[dict], fuzzy: bool
) -> dict[str, float]:
    """Compute precision, recall, F1 for a single example."""
    tp = sum(1 for gt in ground_truth if _entity_in_set(gt, predicted, fuzzy))
    n_pred = len(predicted)
    n_gt = len(ground_truth)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "pred_count": n_pred, "gt_count": n_gt}


def compute_relation_metrics(
    predicted_rels: list[dict], ground_truth_rels: list[dict],
    fuzzy: bool = False,
) -> dict[str, float]:
    """Compute precision, recall, F1 for relation triples (source, target, type)."""
    def _key(rel: dict) -> tuple[str, str, str]:
        return (
            rel["source"].strip().lower(),
            rel["target"].strip().lower(),
            rel["type"].strip().lower(),
        )

    gt_keys = {_key(r) for r in ground_truth_rels}
    pred_keys = {_key(r) for r in predicted_rels}

    if fuzzy and len(gt_keys) > 0 and len(pred_keys) > 0:
        # Use fuzzy matching for source/target names (type must still match exactly)
        tp = 0
        matched_preds: set[int] = set()
        for gt_idx, (gs, gt, gtype) in enumerate(gt_keys):
            for pred_idx, (ps, pt, ptype) in enumerate(pred_keys):
                if pred_idx in matched_preds:
                    continue
                if gtype != ptype:
                    continue
                if _fuzzy_match(gs, ps) and _fuzzy_match(gt, pt):
                    tp += 1
                    matched_preds.add(pred_idx)
                    break
    else:
        tp = len(gt_keys & pred_keys)

    n_pred = len(pred_keys)
    n_gt = len(gt_keys)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "pred_count": n_pred, "gt_count": n_gt}


def macro_average(metrics_list: list[dict]) -> dict[str, float]:
    """Average metric values across examples."""
    if not metrics_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        k: sum(m[k] for m in metrics_list) / len(metrics_list)
        for k in ("precision", "recall", "f1")
    }


def micro_average(entity_metrics: list[dict]) -> dict[str, float]:
    """Compute micro-averaged metrics from per-example TP/FP/FN counts."""
    total_tp = sum(m["tp"] for m in entity_metrics)
    total_pred = sum(m["pred_count"] for m in entity_metrics)
    total_gt = sum(m["gt_count"] for m in entity_metrics)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate() -> None:
    # 1. Load labeled data
    labeled = []
    with open(LABELED_SET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labeled.append(json.loads(line))

    print(f"Loaded {len(labeled)} labeled examples from {LABELED_SET}")

    # 2. Run extraction
    results_exact = []  # per-example results with exact matching
    results_fuzzy = []  # per-example results with fuzzy matching

    for entry in labeled:
        text = entry["text"]
        source = entry["source"]
        gt_entities = entry.get("entities", [])
        gt_relations = entry.get("relations", [])

        # Run the extractors
        predicted = extract_from_markdown(text, source)
        raw_relations, _ = extract_relations(text, source, predicted)

        # Map extractor field names (source_name/target_name/edge_type) to eval format (source/target/type)
        predicted_rels: list[dict] = [
            {"source": r["source_name"], "target": r["target_name"], "type": r["edge_type"]}
            for r in raw_relations
        ]

        # Compute metrics
        exact_em = compute_entity_metrics(predicted, gt_entities, fuzzy=False)
        fuzzy_em = compute_entity_metrics(predicted, gt_entities, fuzzy=True)
        exact_rm = compute_relation_metrics(predicted_rels, gt_relations, fuzzy=False)
        fuzzy_rm = compute_relation_metrics(predicted_rels, gt_relations, fuzzy=True)

        result = {
            "id": entry["id"],
            "source": entry["source"],
            "difficulty": entry.get("difficulty", "unknown"),
            "text": text[:120] + "..." if len(text) > 120 else text,
            "gt_entity_count": len(gt_entities),
            "pred_entity_count": len(predicted),
            "gt_relation_count": len(gt_relations),
            "pred_relation_count": len(predicted_rels),
            "entities_exact": exact_em,
            "entities_fuzzy": fuzzy_em,
            "relations_exact": exact_rm,
            "relations_fuzzy": fuzzy_rm,
            "predicted_entities": predicted,
        }
        results_exact.append(result)
        results_fuzzy.append(result)  # same objects; we use the nested fields

    # 3. Aggregate metrics
    # Exact entity matching
    exact_all = [r["entities_exact"] for r in results_exact]
    exact_macro = macro_average(exact_all)
    exact_micro = micro_average(exact_all)

    # Fuzzy entity matching
    fuzzy_all = [r["entities_fuzzy"] for r in results_fuzzy]
    fuzzy_macro = macro_average(fuzzy_all)
    fuzzy_micro = micro_average(fuzzy_all)

    # Relation metrics (exact)
    rel_exact_all = [r["relations_exact"] for r in results_exact]
    rel_exact_macro = macro_average(rel_exact_all)
    rel_exact_micro = micro_average(rel_exact_all)

    # Relation metrics (fuzzy)
    rel_fuzzy_all = [r["relations_fuzzy"] for r in results_exact]
    rel_fuzzy_macro = macro_average(rel_fuzzy_all)
    rel_fuzzy_micro = micro_average(rel_fuzzy_all)

    # 4. Per-difficulty breakdown
    difficulty_metrics = defaultdict(list)
    for r in results_exact:
        difficulty_metrics[r["difficulty"]].append(r["entities_exact"])

    difficulty_summary = {}
    for diff, metrics_list in sorted(difficulty_metrics.items()):
        difficulty_summary[diff] = {
            "count": len(metrics_list),
            "macro": macro_average(metrics_list),
            "micro": micro_average(metrics_list),
        }

    # 5. Per-document-type breakdown
    def _doc_type(source: str) -> str:
        if "/docs/" in source:
            return "docs"
        if "/experiments/" in source:
            return "experiments"
        return "other"

    doctype_metrics = defaultdict(list)
    for r in results_exact:
        doctype_metrics[_doc_type(r["source"])].append(r["entities_exact"])

    doctype_summary = {}
    for dt, metrics_list in sorted(doctype_metrics.items()):
        doctype_summary[dt] = {
            "count": len(metrics_list),
            "macro": macro_average(metrics_list),
            "micro": micro_average(metrics_list),
        }

    # 5b. Per-domain breakdown (for noise, tables, metrics annotations)
    domain_metrics = defaultdict(list)
    for i, r in enumerate(results_exact):
        domain = labeled[i].get("domain", "general")
        domain_metrics[domain].append(r["entities_exact"])

    domain_summary = {}
    for dom, metrics_list in sorted(domain_metrics.items()):
        domain_summary[dom] = {
            "count": len(metrics_list),
            "macro": macro_average(metrics_list),
            "micro": micro_average(metrics_list),
        }

    # 6. Build a simulated corpus graph for corpus-level sanity metrics
    corpus_graph = InMemoryGraphStore()
    for entry in labeled:
        text = entry["text"]
        source = entry["source"]
        predicted = extract_from_markdown(text, source)
        raw_relations, _ = extract_relations(text, source, predicted)

        # Add entity nodes
        for entity in predicted:
            node_id = canonicalize(entity["name"], entity.get("type", "Entity"))
            if not corpus_graph.has_node(node_id):
                node = Node(
                    id=node_id,
                    type=entity.get("type", "Entity"),
                    properties={"name": entity["name"]},
                    sources=[source],
                )
                corpus_graph.add_node(node)

        # Add relation edges
        for rel in raw_relations:
            src_id = canonicalize(rel.get("source_name", ""))
            tgt_id = canonicalize(rel.get("target_name", ""))
            if not src_id or not tgt_id:
                continue
            # Ensure nodes exist (they might have been added in a different entry)
            if not corpus_graph.has_node(src_id):
                corpus_graph.add_node(Node(id=src_id, type="Entity", properties={"name": rel["source_name"]}, sources=[source]))
            if not corpus_graph.has_node(tgt_id):
                corpus_graph.add_node(Node(id=tgt_id, type="Entity", properties={"name": rel["target_name"]}, sources=[source]))
            try:
                edge = Edge(
                    type=rel.get("edge_type", "related_to"),
                    source=src_id,
                    target=tgt_id,
                    confidence=rel.get("confidence", 0.5),
                    evidence=source,
                )
                corpus_graph.add_edge(edge)
            except KeyError:
                pass

    # 7. Compute corpus-level health metrics
    corpus_report = corpus_health_report(corpus_graph)
    corpus_validity = corpus_report["node_validity_rate"]
    corpus_edge_quality = corpus_report["edge_quality_rate"]

    # 8. Build output
    output = {
        "total_examples": len(labeled),
        "entity_extraction": {
            "exact_match": {
                "macro": exact_macro,
                "micro": exact_micro,
            },
            "fuzzy_match": {
                "macro": fuzzy_macro,
                "micro": fuzzy_micro,
            },
        },
        "relation_extraction": {
            "exact_match": {
                "macro": rel_exact_macro,
                "micro": rel_exact_micro,
            },
            "fuzzy_match": {
                "macro": rel_fuzzy_macro,
                "micro": rel_fuzzy_micro,
            },
        },
        "corpus_health": {
            "node_count": corpus_report["node_count"],
            "edge_count": corpus_report["edge_count"],
            "node_validity_rate": corpus_report["node_validity_rate"],
            "edge_quality_rate": corpus_report["edge_quality_rate"],
            "invalid_node_count": corpus_report["invalid_node_count"],
            "invalid_node_examples": corpus_report["invalid_node_examples"],
            "node_type_distribution": corpus_report["node_type_distribution"],
        },
        "per_difficulty": difficulty_summary,
        "per_document_type": doctype_summary,
        "per_domain": domain_summary,
        "per_example": results_exact,
    }

    # 9. Double-gate check: snippet-level F1 AND corpus-level validity
    snippet_f1 = output["entity_extraction"]["fuzzy_match"]["micro"]["f1"]
    snippet_passed = snippet_f1 >= 0.80
    corpus_passed = corpus_validity >= 0.70
    gate_passed = snippet_passed and corpus_passed

    gate_failures = []
    if not snippet_passed:
        gate_failures.append(f"snippet F1={snippet_f1:.4f} < 0.80 (gap: {0.80 - snippet_f1:.4f})")
    if not corpus_passed:
        gate_failures.append(f"corpus validity={corpus_validity:.4f} < 0.70 (gap: {0.70 - corpus_validity:.4f})")

    output["gate_check"] = {
        "passed": gate_passed,
        "checks": {
            "snippet_f1": {
                "metric": "entity_extraction.fuzzy_match.micro.f1",
                "value": round(snippet_f1, 4),
                "threshold": 0.80,
                "passed": snippet_passed,
            },
            "corpus_validity": {
                "metric": "corpus_health.node_validity_rate",
                "value": round(corpus_validity, 4),
                "threshold": 0.70,
                "passed": corpus_passed,
            },
            "corpus_edge_quality": {
                "metric": "corpus_health.edge_quality_rate",
                "value": round(corpus_edge_quality, 4),
                "threshold": "informational_only",
                "passed": None,
            },
        },
        "failures": gate_failures,
    }

    # 8. Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to {RESULTS_FILE}")

    # 9. Print summary table
    print("\n" + "=" * 72)
    print("  ENTITY EXTRACTION EVALUATION -- Phase 1->2 Gate Check")
    print("=" * 72)

    print(f"\n{'Metric':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    print(f"{'Entity (exact, macro)':<30} {exact_macro['precision']:>10.4f} {exact_macro['recall']:>10.4f} {exact_macro['f1']:>10.4f}")
    print(f"{'Entity (exact, micro)':<30} {exact_micro['precision']:>10.4f} {exact_micro['recall']:>10.4f} {exact_micro['f1']:>10.4f}")
    print(f"{'Entity (fuzzy, macro)':<30} {fuzzy_macro['precision']:>10.4f} {fuzzy_macro['recall']:>10.4f} {fuzzy_macro['f1']:>10.4f}")
    print(f"{'Entity (fuzzy, micro)':<30} {fuzzy_micro['precision']:>10.4f} {fuzzy_micro['recall']:>10.4f} {fuzzy_micro['f1']:>10.4f}")
    print(f"{'Relation (exact, macro)':<30} {rel_exact_macro['precision']:>10.4f} {rel_exact_macro['recall']:>10.4f} {rel_exact_macro['f1']:>10.4f}")
    print(f"{'Relation (exact, micro)':<30} {rel_exact_micro['precision']:>10.4f} {rel_exact_micro['recall']:>10.4f} {rel_exact_micro['f1']:>10.4f}")
    print(f"{'Relation (fuzzy, macro)':<30} {rel_fuzzy_macro['precision']:>10.4f} {rel_fuzzy_macro['recall']:>10.4f} {rel_fuzzy_macro['f1']:>10.4f}")
    print(f"{'Relation (fuzzy, micro)':<30} {rel_fuzzy_micro['precision']:>10.4f} {rel_fuzzy_micro['recall']:>10.4f} {rel_fuzzy_micro['f1']:>10.4f}")

    print(f"\n{'Difficulty':<20} {'Count':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 58)
    for diff in sorted(difficulty_summary):
        d = difficulty_summary[diff]
        m = d["macro"]
        print(f"{diff:<20} {d['count']:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print(f"\n{'Doc Type':<20} {'Count':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 58)
    for dt in sorted(doctype_summary):
        d = doctype_summary[dt]
        m = d["macro"]
        print(f"{dt:<20} {d['count']:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    if any(k != "general" for k in domain_summary):
        print(f"\n{'Domain':<20} {'Count':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 58)
        for dom in sorted(domain_summary):
            d = domain_summary[dom]
            m = d["macro"]
            print(f"{dom:<20} {d['count']:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # Corpus health section
    print(f"\n{'---' * 24}")
    print("  CORPUS-LEVEL SANITY METRICS (simulated from labeled-set extraction)")
    print(f"{'---' * 24}")
    print(f"  Nodes in simulated corpus:    {corpus_report['node_count']:>6}")
    print(f"  Edges in simulated corpus:    {corpus_report['edge_count']:>6}")
    print(f"  Node validity rate:           {corpus_validity:>8.4f}  (threshold: 0.70)")
    print(f"  Edge quality rate (conf>=0.7): {corpus_edge_quality:>8.4f}")
    if corpus_report['invalid_node_count'] > 0:
        print(f"\n  {corpus_report['invalid_node_count']} invalid nodes (examples):")
        for ex in corpus_report["invalid_node_examples"][:5]:
            name = ex["node_id"]
            if len(name) > 50:
                name = name[:47] + "..."
            print(f"    [{ex['type']}] \"{name}\"  -- {', '.join(ex['reasons'])}")

    # Gate verdict
    print("\n" + "-" * 72)
    print(f"  GATE CHECK: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  -------------------------------------------------")
    status_f1 = "PASS" if snippet_passed else "FAIL"
    status_cv = "PASS" if corpus_passed else "FAIL"
    print(f"  Snippet entity F1 (fuzzy micro):   {snippet_f1:.4f}  (threshold: 0.80)  [{status_f1}]")
    print(f"  Corpus node validity rate:         {corpus_validity:.4f}  (threshold: 0.70)  [{status_cv}]")
    print(f"  Corpus edge quality (>=0.7 conf):  {corpus_edge_quality:.4f}  (informational)")
    if not gate_passed:
        print(f"\n  Gate failures:")
        for failure in gate_failures:
            print(f"    - {failure}")
        if not snippet_passed and not corpus_passed:
            print(f"\n  Both snippet F1 AND corpus validity are below threshold.")
            print(f"  This means the extraction quality degrades significantly")
            print(f"  on real-corpus noise (table rows, generic headers, single-token fragments).")
        elif not snippet_passed:
            print(f"\n  Snippet F1 is below threshold. Entity extraction accuracy on")
            print(f"  curated examples is insufficient for Phase 2.")
        elif not corpus_passed:
            print(f"\n  Corpus validity is below threshold. The entity extractor produces")
            print(f"  too many garbage nodes (stopwords, numbers, table fragments).")
            print(f"  Consider adding more header-filter rules or post-extraction cleanup.")
    print("=" * 72)

    # 10. Verify curated node facts surface in evidence
    _verify_curated_node_facts(output)


def _verify_curated_node_facts(results: dict) -> None:
    """
    Verify that curated node facts (key_finding/description) from the
    NEXUS graph properly surface in evidence packs.

    Loads the graph, builds evidence for a representative query, and
    checks that Experiment nodes contribute key_finding and Concept
    nodes contribute description to the node_facts section.
    """
    print("\n" + "=" * 72)
    print("  CURATED NODE FACTS VERIFICATION")
    print("=" * 72)

    from nexus.graph.store import InMemoryGraphStore
    from nexus.graph.traversal import traverse_with_intent
    from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
    from nexus.query.parser import parse_question
    from nexus.reasoning.evidence_builder import build_evidence

    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)

    # Test queries designed to pull in both Experiment and Concept nodes
    test_queries: list[dict[str, Any]] = [
        {
            "question": "What was the key finding of the oracle memory experiment?",
            "expect_node_type": "Experiment",
            "expect_property": "key_finding",
            "expected_fragment": "99.87%",
        },
        {
            "question": "What concept does the oracle memory experiment validate?",
            "expect_node_type": "Concept",
            "expect_property": "description",
            "expected_fragment": "external memory",
        },
        {
            "question": "How does SAM handle noise?",
            "expect_node_type": "Concept",
            "expect_property": "description",
            "expected_fragment": "91.6%",
        },
    ]

    all_passed = True
    for test in test_queries:
        question = test["question"]
        parsed = parse_question(question, graph, cutoff=0.6)
        if not parsed.entity_ids:
            print(f"\n  SKIP: '{question}' — no entities matched")
            continue

        query_entities = set(parsed.entity_ids)
        paths = traverse_with_intent(
            graph=graph,
            entry_nodes=parsed.entity_ids,
            query_entities=query_entities,
            intent=parsed.intent,
            max_depth=4,
            beam_width=5,
        )

        evidence_json = build_evidence(question, paths, graph, max_paths=3)
        evidence = json.loads(evidence_json)
        node_facts = evidence.get("node_facts", [])

        # Check: at least one node_fact contains the expected fragment
        matching = [nf for nf in node_facts if test["expected_fragment"] in nf.get("text", "")]
        passed = len(matching) > 0

        if passed:
            print(f"\n  PASS: '{question[:60]}...'")
            print(f"    Node facts found: {len(node_facts)} total, {len(matching)} matching")
            for mf in matching[:3]:
                conf_label = mf.get("confidence_label", "")
                print(f"    - [{conf_label}] {mf['text'][:120]}{'...' if len(mf['text']) > 120 else ''}")
        else:
            all_passed = False
            print(f"\n  FAIL: '{question[:60]}...'")
            print(f"    Expected fragment '{test['expected_fragment']}' not found in node_facts")
            print(f"    Node facts available: {[nf['text'][:80] for nf in node_facts]}")

    # Summary
    print("\n" + "-" * 72)
    if all_passed:
        print(f"  CURATED NODE FACTS: VERIFIED")
        print("  All test queries surface key_finding/description as node_facts.")
    else:
        print(f"  CURATED NODE FACTS: FAILURES DETECTED")
    print("=" * 72)


if __name__ == "__main__":
    evaluate()
