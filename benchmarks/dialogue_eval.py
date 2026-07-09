"""
Stage 3 dialogue evaluation harness.

Loads dialogues.jsonl, runs whole dialogues through answer_question
with DialogueState, and computes:
  - Per-turn: entity accuracy, intent accuracy
  - Per-dialogue: reference resolution accuracy (turns marked resolution_source="context")
  - Overall: reference resolution, single-turn no-regression check
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
from nexus.query.parser import parse_question
from nexus.reasoning.model_interface import SynthesizingModel
from nexus.utils.config import NEXUSConfig
from stack.dialogue.state import DialogueState


def load_dialogues(path: str | Path) -> list[dict[str, Any]]:
    """Load dialogue turns from a JSONL file."""
    turns: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def group_by_dialogue(turns: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group turns by dialogue_id."""
    dialogues: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        did = turn["dialogue_id"]
        dialogues.setdefault(did, []).append(turn)
    # Sort each dialogue's turns by turn_id
    for turns_list in dialogues.values():
        turns_list.sort(key=lambda t: t["turn_id"])
    return dialogues


def evaluate_dialogue_entities(
    graph: InMemoryGraphStore,
    config: NEXUSConfig,
) -> dict[str, Any]:
    """Run full dialogue evaluation.

    Returns:
        Dict with per-turn, per-dialogue, and overall metrics.
    """
    from nexus.query.parser import parse_question

    dataset_path = _PROJECT_ROOT / "benchmarks" / "qa-dataset" / "dialogues.jsonl"
    all_turns = load_dialogues(dataset_path)
    dialogues = group_by_dialogue(all_turns)

    # Results storage
    per_turn_results: list[dict[str, Any]] = []
    per_dialogue: dict[str, dict[str, Any]] = {}
    reference_correct = 0
    reference_total = 0
    global_correct = 0
    global_total = 0
    total_entity_accuracy = 0.0
    total_turns = 0

    total_state_time_ms = 0.0

    for did, turns in sorted(dialogues.items()):
        state = DialogueState(decay=config.dialogue_decay)
        dialogue_turns: list[dict[str, Any]] = []
        dialogue_ref_correct = 0
        dialogue_ref_total = 0

        for turn in turns:
            question = turn["question"]
            gt_entities = set(turn["gt_entities"])
            res_source = turn["resolution_source"]

            # Time ONLY the dialogue state operations within parse_question
            # (not the full answer_question pipeline which includes traversal etc.)
            t0 = time.perf_counter()
            parsed = parse_question(
                question, graph, cutoff=0.6, config=config,
                dialogue_state=state,
            )
            state_time = (time.perf_counter() - t0) * 1000  # ms
            total_state_time_ms += state_time

            predicted_entities = set(parsed.entity_ids)

            # Entity accuracy for this turn
            if gt_entities:
                intersection = gt_entities & predicted_entities
                turn_accuracy = len(intersection) / len(gt_entities)
            else:
                turn_accuracy = 1.0 if len(predicted_entities) == 0 else 0.0

            total_entity_accuracy += turn_accuracy
            total_turns += 1

            # Reference resolution tracking
            if res_source == "context":
                reference_total += 1
                dialogue_ref_total += 1
                if gt_entities & predicted_entities:
                    reference_correct += 1
                    dialogue_ref_correct += 1
            else:
                global_total += 1
                if gt_entities & predicted_entities:
                    global_correct += 1

            turn_result = {
                "dialogue_id": did,
                "turn_id": turn["turn_id"],
                "question": question,
                "gt_entities": list(gt_entities),
                "predicted_entities": list(predicted_entities),
                "entity_accuracy": round(turn_accuracy, 4),
                "resolution_source": res_source,
                "state_time_ms": round(state_time, 3),
                "intersection": list(gt_entities & predicted_entities),
            }
            dialogue_turns.append(turn_result)
            per_turn_results.append(turn_result)

            # Update dialogue state with resolved entities
            state.update(predicted_entities)

        # Per-dialogue metrics
        dialogue_ref_acc = (
            dialogue_ref_correct / dialogue_ref_total if dialogue_ref_total > 0 else 0.0
        )
        per_dialogue[did] = {
            "dialogue_id": did,
            "num_turns": len(turns),
            "reference_total": dialogue_ref_total,
            "reference_correct": dialogue_ref_correct,
            "reference_accuracy": round(dialogue_ref_acc, 4),
            "turns": dialogue_turns,
        }

    # Overall metrics
    overall_ref_acc = reference_correct / reference_total if reference_total > 0 else 0.0
    overall_global_acc = global_correct / global_total if global_total > 0 else 0.0
    overall_entity_acc = total_entity_accuracy / total_turns if total_turns > 0 else 0.0
    avg_state_time = total_state_time_ms / total_turns if total_turns > 0 else 0.0

    return {
        "num_dialogues": len(dialogues),
        "num_turns": total_turns,
        "reference_resolution_accuracy": round(overall_ref_acc, 4),
        "global_resolution_accuracy": round(overall_global_acc, 4),
        "overall_entity_accuracy": round(overall_entity_acc, 4),
        "reference_total": reference_total,
        "reference_correct": reference_correct,
        "global_total": global_total,
        "global_correct": global_correct,
        "avg_state_time_ms": round(avg_state_time, 3),
        "per_dialogue": per_dialogue,
        "per_turn": per_turn_results,
    }


def run_single_turn_baseline(
    graph: InMemoryGraphStore,
    config: NEXUSConfig,
    num_questions: int = 30,
) -> dict[str, Any]:
    """Run a single-turn baseline evaluation (no dialogue state).

    Spot-checks that dialogue state does not regress single-turn accuracy.
    Uses the first `num_questions` from questions.jsonl.
    """
    qa_path = _PROJECT_ROOT / "benchmarks" / "qa-dataset" / "questions.jsonl"
    questions: list[dict[str, Any]] = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
                if len(questions) >= num_questions:
                    break

    correct_without = 0
    correct_with = 0
    total = 0

    for q in questions:
        gt_entities = set(q.get("entities", []))
        if not gt_entities:
            continue

        # Without dialogue state
        parsed_without = parse_question(
            q["question"], graph, config=config,
        )
        predicted_without = set(parsed_without.entity_ids)
        if gt_entities & predicted_without:
            correct_without += 1

        # With fresh dialogue state (should behave identically)
        fresh_state = DialogueState(decay=config.dialogue_decay)
        parsed_with = parse_question(
            q["question"], graph, config=config, dialogue_state=fresh_state,
        )
        predicted_with = set(parsed_with.entity_ids)
        if gt_entities & predicted_with:
            correct_with += 1

        total += 1

    acc_without = correct_without / total if total > 0 else 0.0
    acc_with = correct_with / total if total > 0 else 0.0

    return {
        "num_questions": total,
        "accuracy_without_state": round(acc_without, 4),
        "accuracy_with_state": round(acc_with, 4),
        "regression": round(acc_without - acc_with, 4),
        "passes_check": acc_with >= acc_without - 0.02,  # within 2pp
    }


def report(results: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Print a formatted evaluation report."""
    print("=" * 70)
    print("Stage 3 -- Dialogue State Evaluation")
    print("=" * 70)
    print()

    print("-- Reference Resolution --")
    print(f"  Context turns:      {results['reference_correct']}/{results['reference_total']} "
          f"({results['reference_resolution_accuracy']:.1%})")
    print(f"  Global turns:       {results['global_correct']}/{results['global_total']} "
          f"({results['global_resolution_accuracy']:.1%})")
    print(f"  Overall entity acc: {results['overall_entity_accuracy']:.1%}")
    ref_pass = results['reference_resolution_accuracy'] >= 0.70
    print(f"  Gate (>=70%):       {'PASS' if ref_pass else 'FAIL'}")
    print()

    print("-- Single-turn No-Regression Check --")
    print(f"  Without state:      {baseline['accuracy_without_state']:.1%}")
    print(f"  With fresh state:   {baseline['accuracy_with_state']:.1%}")
    print(f"  Regression:         {baseline['regression']:+.2%}")
    print(f"  Gate (<=2pp):        {'PASS' if baseline['passes_check'] else 'FAIL'}")
    print()

    print("-- Latency --")
    print(f"  Avg state time:     {results['avg_state_time_ms']:.2f} ms")
    lat_pass = results['avg_state_time_ms'] <= 5.0
    print(f"  Gate (<=5ms p50):    {'PASS' if lat_pass else 'FAIL'}")
    print()

    print("-- Per-Dialogue Reference Accuracy --")
    for did in sorted(results["per_dialogue"]):
        dd = results["per_dialogue"][did]
        if dd["reference_total"] > 0:
            status = "OK" if dd["reference_accuracy"] >= 0.5 else "??"
            print(f"  {did}: {status} {dd['reference_correct']}/{dd['reference_total']} "
                  f"({dd['reference_accuracy']:.0%}) -- {dd['num_turns']} turns")

    print()
    all_pass = ref_pass and baseline['passes_check'] and lat_pass
    if all_pass:
        print("ALL GATES PASSED -- proceeding to Stage 4")
    else:
        print("GATE(S) FAILED -- write STAGE3_NEGATIVE.md and STOP")


def main():
    """Run Stage 3 gate evaluation."""
    config = NEXUSConfig()

    print("Loading graph...")
    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)
    print(f"Graph: {graph.node_count} nodes, {graph.edge_count} edges\n")

    # 1. Dialogue evaluation
    print("Running dialogue evaluation...")
    results = evaluate_dialogue_entities(graph, config)

    # 2. Single-turn no-regression check
    print("Running single-turn baseline check (30 questions)...")
    baseline = run_single_turn_baseline(graph, config)

    # 3. Report
    report(results, baseline)

    # 4. Write results
    output_path = _PROJECT_ROOT / "benchmarks" / "results" / "stage3_dialogue_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "dialogue_results": {
                "num_dialogues": results["num_dialogues"],
                "num_turns": results["num_turns"],
                "reference_resolution_accuracy": results["reference_resolution_accuracy"],
                "global_resolution_accuracy": results["global_resolution_accuracy"],
                "overall_entity_accuracy": results["overall_entity_accuracy"],
                "avg_state_time_ms": results["avg_state_time_ms"],
            },
            "single_turn_baseline": baseline,
            "gates": {
                "reference_resolution": results["reference_resolution_accuracy"] >= 0.70,
                "no_regression": baseline["passes_check"],
                "latency": results["avg_state_time_ms"] <= 5.0,
            },
            "all_pass": (
                results["reference_resolution_accuracy"] >= 0.70
                and baseline["passes_check"]
                and results["avg_state_time_ms"] <= 5.0
            ),
        }, f, indent=2)

    print(f"\nResults written to {output_path}")

    # Per-turn details
    detail_path = _PROJECT_ROOT / "benchmarks" / "results" / "stage3_per_turn.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results["per_turn"], f, indent=2)

    return results, baseline


if __name__ == "__main__":
    main()
