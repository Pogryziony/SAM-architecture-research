"""
Populate the NEXUS graph from existing SAM experiment results.

Extracts:
- Experiment nodes with all metrics as properties
- Metric nodes linked to experiments
- Key findings as Concept nodes
- Edges: derived_from, validates, depends_on, related_to
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add sam-lm to path to access its data
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexus.graph import Node, Edge
from nexus.graph.store import InMemoryGraphStore


EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "sam-lm" / "experiments"

# Experiment definitions (name -> description from reports)
EXPERIMENT_DEFS = {
    "exp_0_diagnosis": {
        "id": "Exp_0_Diagnosis",
        "title": "Experiment 0 — Pipeline Diagnosis",
        "question": "Is the experimental pipeline working correctly?",
        "key_finding": "Found and fixed 3 critical bugs. Retrieval is the bottleneck.",
        "phase": "Pipeline Setup",
        "depends_on": [],
    },
    "exp_0_2": {
        "id": "Exp_0_2_CompactPKM",
        "title": "Experiment 0.2 — Compact PKM Retrieval",
        "question": "Can compact PKM retrieval work?",
        "key_finding": "16K PKM: 25.8% Rec@8. Oracle text: 100% — core CAN use memory.",
        "phase": "Pipeline Setup",
        "depends_on": ["Exp_0_Diagnosis"],
    },
    "exp_0_3": {
        "id": "Exp_0_3_PKM_Candidates",
        "title": "Experiment 0.3 — PKM Candidate Generation",
        "question": "Is PKM candidate generation the bottleneck?",
        "key_finding": "Candidate gen: SOLVED (100%). Ranking generalizes poorly (29% val).",
        "phase": "Pipeline Setup",
        "depends_on": ["Exp_0_2_CompactPKM"],
    },
    "exp_0_5": {
        "id": "Exp_0_5_DenseDataset",
        "title": "Experiment 0.5 — Dense Dataset Fix",
        "question": "Can retrieval work with better data?",
        "key_finding": "Dense dataset with 21.8 ex/slot -> 99.0% Rec@8. Gate 1 PASSED.",
        "phase": "Pipeline Setup",
        "depends_on": ["Exp_0_3_PKM_Candidates"],
    },
    "exp_0_6": {
        "id": "Exp_0_6_Validation",
        "title": "Experiment 0.6 — Full Validation",
        "question": "Does SAM work end-to-end with retrieval?",
        "key_finding": "Oracle memory: 99.87%. Retrieved memory = core_only (68.74%). Query projection mismatch identified.",
        "phase": "Core Validation",
        "depends_on": ["Exp_0_5_DenseDataset"],
    },
    "exp_0_7": {
        "id": "Exp_0_7_ExternalText",
        "title": "Experiment 0.7 — External Text Query",
        "question": "Can external-text-query fix the projection mismatch?",
        "key_finding": "External text query bypasses hidden-state projection. Tested topK sweep.",
        "phase": "Core Validation",
        "depends_on": ["Exp_0_6_Validation"],
    },
    "exp_0_8": {
        "id": "Exp_0_8_Aggregation",
        "title": "Experiment 0.8 — Aggregation & Selection Variants",
        "question": "Which aggregation and selection variants work best?",
        "key_finding": "Tested weighted, threshold, softmax-mass, score-gap selection.",
        "phase": "Core Validation",
        "depends_on": ["Exp_0_7_ExternalText"],
    },
    "exp_0_9": {
        "id": "Exp_0_9_OracleFilter",
        "title": "Experiment 0.9 — Oracle-Filter & Multi-Query",
        "question": "Does oracle filtering or multi-query help?",
        "key_finding": "Oracle filter achieves 79.95%. Multi-query implemented but not yet effective.",
        "phase": "Core Validation",
        "depends_on": ["Exp_0_8_Aggregation"],
    },
    "exp_0_10": {
        "id": "Exp_0_10_RequiredSet",
        "title": "Experiment 0.10 — Required-Set Retrieval Diagnostics",
        "question": "Where exactly are the retrieval failures?",
        "key_finding": "all_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem.",
        "phase": "Retrieval Revolution",
        "depends_on": ["Exp_0_9_OracleFilter"],
    },
    "exp_0_11": {
        "id": "Exp_0_11_ChainRetrieval",
        "title": "Experiment 0.11 — Chain-Aware Retrieval",
        "question": "Can chain-aware retrieval solve the multi-hop bottleneck?",
        "key_finding": "Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.",
        "phase": "Retrieval Revolution",
        "depends_on": ["Exp_0_10_RequiredSet"],
    },
    "exp_0_12": {
        "id": "Exp_0_12_Selection",
        "title": "Experiment 0.12 — Candidate Selection & Memory-Use Training",
        "question": "Can we select the right slots from chain retrieval?",
        "key_finding": "Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck.",
        "phase": "Selection & Noise",
        "depends_on": ["Exp_0_11_ChainRetrieval"],
    },
    "exp_0_13A": {
        "id": "Exp_0_13A_NoisyMemory",
        "title": "Experiment 0.13A — Controlled Noisy Memory Tolerance",
        "question": "How much memory noise can SAM tolerate?",
        "key_finding": "SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.",
        "phase": "Selection & Noise",
        "depends_on": ["Exp_0_12_Selection"],
    },
    "exp_0_13B": {
        "id": "Exp_0_13B_RealisticDistractors",
        "title": "Experiment 0.13B — Realistic Retrieval Distractor Replay",
        "question": "Are realistic retrieval distractors harder than random?",
        "key_finding": "Testing in progress. Code implemented.",
        "phase": "Selection & Noise",
        "depends_on": ["Exp_0_13A_NoisyMemory"],
    },
}

# ── Alias maps for human-friendly entity resolution ──

_EXPERIMENT_ALIASES: dict[str, list[str]] = {
    "Exp_0_Diagnosis": ["pipeline diagnosis", "experiment 0", "diagnosis experiment", "initial experiment"],
    "Exp_0_2_CompactPKM": ["compact pkm", "experiment 0.2", "pkm retrieval", "product-key memory", "product key memory"],
    "Exp_0_3_PKM_Candidates": ["pkm candidates", "experiment 0.3", "candidate generation"],
    "Exp_0_5_DenseDataset": ["dense dataset", "experiment 0.5", "dataset fix"],
    "Exp_0_6_Validation": ["oracle memory", "full validation", "experiment 0.6", "validation experiment", "oracle memory experiment", "live memory", "memory slots", "sam experiments"],
    "Exp_0_7_ExternalText": ["external text", "experiment 0.7", "text query"],
    "Exp_0_8_Aggregation": ["aggregation", "experiment 0.8", "aggregation variants"],
    "Exp_0_9_OracleFilter": ["oracle filter", "experiment 0.9"],
    "Exp_0_10_RequiredSet": ["required set", "experiment 0.10", "required-set diagnostics"],
    "Exp_0_11_ChainRetrieval": ["chain retrieval", "experiment 0.11", "chain-set", "chain bce"],
    "Exp_0_12_Selection": ["selector", "slot selection", "experiment 0.12", "candidate selection", "learned selector"],
    "Exp_0_13A_NoisyMemory": ["noise tolerance", "experiment 0.13a", "noisy memory", "controlled noise"],
    "Exp_0_13B_RealisticDistractors": ["realistic distractors", "experiment 0.13b"],
}

_CONCEPT_ALIASES: dict[str, list[str]] = {
    "Concept_OracleMemory": ["oracle memory works", "oracle memory concept"],
    "Concept_SelectorBottleneck": ["selector bottleneck", "selection bottleneck"],
    "Concept_ChainRetrieval": ["chain retrieval solved", "retrieval solved"],
    "Concept_NoiseTolerance": ["noise tolerance concept", "noise handling"],
    "Concept_ArchitectureWorks": ["architecture validated", "architecture works", "core memory architecture"],
    "Concept_RetrievalMismatch": ["retrieval mismatch", "projection mismatch", "query projection"],
    "Concept_PivotToNEXUS": ["pivot to nexus", "architecture pivot", "nexus pivot"],
}

# Run-specific data mapping
RUN_MAP = {
    # Experiment 0.6 runs
    "exp_0_6/core_only": ("Exp_0_6_Validation", "core_only"),
    "exp_0_6/dense_baseline": ("Exp_0_6_Validation", "dense_baseline"),
    "exp_0_6/oracle_memory": ("Exp_0_6_Validation", "oracle_memory"),
    "exp_0_6/oracle_text_memory": ("Exp_0_6_Validation", "oracle_text_memory"),
    "exp_0_6/random_memory": ("Exp_0_6_Validation", "random_memory"),
    "exp_0_6/retrieval_dual_encoder": ("Exp_0_6_Validation", "retrieval_dual_encoder"),
    "exp_0_6/retrieved_memory": ("Exp_0_6_Validation", "retrieved_memory"),
    # Experiment 0.11 runs
    "exp_0_11/chain_set_bce": ("Exp_0_11_ChainRetrieval", "chain_set_bce"),
    "exp_0_11/sam_chain_aware": ("Exp_0_11_ChainRetrieval", "sam_chain_aware"),
    # Experiment 0.12 runs
    "exp_0_12/chain_equal_budget": ("Exp_0_12_Selection", "equal_budget"),
    "exp_0_12/chain_fixed_top_by_hop": ("Exp_0_12_Selection", "fixed_top_by_hop"),
    "exp_0_12/chain_learned_selector": ("Exp_0_12_Selection", "learned_selector"),
    "exp_0_12/chain_oracle_filter_top32": ("Exp_0_12_Selection", "oracle_filter_top32"),
    "exp_0_12/chain_oracle_filter_top64": ("Exp_0_12_Selection", "oracle_filter_top64"),
    # Experiment 0.13A runs
    "exp_0_13/noise_oracle_plus_0": ("Exp_0_13A_NoisyMemory", "noise_+0"),
    "exp_0_13/noise_oracle_plus_1": ("Exp_0_13A_NoisyMemory", "noise_+1"),
    "exp_0_13/noise_oracle_plus_2": ("Exp_0_13A_NoisyMemory", "noise_+2"),
    "exp_0_13/noise_oracle_plus_4": ("Exp_0_13A_NoisyMemory", "noise_+4"),
    "exp_0_13/noise_oracle_plus_8": ("Exp_0_13A_NoisyMemory", "noise_+8"),
    "exp_0_13/noise_oracle_plus_16": ("Exp_0_13A_NoisyMemory", "noise_+16"),
    # Experiment 0.7 runs
    "exp_0_7/retrieved_memory_external_text_query": ("Exp_0_7_ExternalText", "external_text_query"),
    "exp_0_7/retrieved_memory_hidden_adapter": ("Exp_0_7_ExternalText", "hidden_adapter"),
    # Experiment 0.8 runs
    "exp_0_8/retrieved_memory_external_text_query": ("Exp_0_8_Aggregation", "external_text_query"),
    "exp_0_8/retrieved_oracle_slots": ("Exp_0_8_Aggregation", "oracle_slots"),
    # Experiment 0.9 runs
    "exp_0_9/retrieved_memory_external_text_query": ("Exp_0_9_OracleFilter", "baseline"),
    "exp_0_9/oracle_filter": ("Exp_0_9_OracleFilter", "oracle_filter"),
    "exp_0_9/top1": ("Exp_0_9_OracleFilter", "top1"),
    "exp_0_9/weighted_t005_top8": ("Exp_0_9_OracleFilter", "weighted_t005_top8"),
    "exp_0_9/weighted_t005_top32": ("Exp_0_9_OracleFilter", "weighted_t005_top32"),
    # Other experiments
    "exp_0_2/retrieval_compact_16k": ("Exp_0_2_CompactPKM", "retrieval_compact_16k"),
    "exp_0_3/retrieval_compact_16k_improved": ("Exp_0_3_PKM_Candidates", "improved"),
    "exp_0_3/retrieval_compact_16k_subkey_loss": ("Exp_0_3_PKM_Candidates", "subkey_loss"),
    "exp_0_3/retrieval_compact_16k_top64": ("Exp_0_3_PKM_Candidates", "top64"),
    "exp_0_5/dual_encoder": ("Exp_0_5_DenseDataset", "dual_encoder"),
    "exp_000_dense_baseline": ("Exp_0_6_Validation", "dense_baseline_early"),
    "exp_000_dense_openbook": ("Exp_0_6_Validation", "dense_openbook"),
    "exp_001_pkm_retrieval": ("Exp_0_6_Validation", "pkm_retrieval_early"),
}


def load_metrics(metrics_path: Path) -> dict:
    """Load a metrics.json file."""
    with open(metrics_path) as f:
        return json.load(f)


def populate_graph(experiments_dir: Path, graph: InMemoryGraphStore) -> InMemoryGraphStore:
    """Populate the graph from all experiment results."""
    
    # Phase 1: Create experiment nodes
    for exp_key, exp_def in EXPERIMENT_DEFS.items():
        node = Node(
            id=exp_def["id"],
            type="Experiment",
            properties={
                "title": exp_def["title"],
                "question": exp_def["question"],
                "key_finding": exp_def["key_finding"],
                "phase": exp_def["phase"],
                "experiment_key": exp_key,
            },
            sources=[f"sam-lm/experiments/{exp_key}_report.md"],
            aliases=_EXPERIMENT_ALIASES.get(exp_def["id"], []),
        )
        graph.add_node(node)
    
    # Phase 2: Process individual runs and create metric nodes
    for pattern, (exp_id, run_name) in RUN_MAP.items():
        metrics_path = experiments_dir / pattern / "metrics.json"
        if not metrics_path.exists():
            # Check nested structure
            metrics_path = experiments_dir / pattern
            subdirs = list(metrics_path.glob("*/metrics.json"))
            if subdirs:
                metrics_path = subdirs[0]
            else:
                continue
        
        try:
            metrics = load_metrics(metrics_path)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        
        # Create a sub-experiment node (a specific run configuration)
        run_id = f"{exp_id}_{run_name}"
        run_node = Node(
            id=run_id,
            type="Experiment",
            properties={
                "title": f"{EXPERIMENT_DEFS.get(exp_key_from_id(exp_id), {}).get('title', exp_id)} — {run_name}",
                "mode": metrics.get("mode", ""),
                "run_name": metrics.get("run_name", ""),
                "phase": EXPERIMENT_DEFS.get(exp_key_from_id(exp_id), {}).get("phase", ""),
            },
            sources=[str(metrics_path.relative_to(experiments_dir.parent))],
        )
        graph.add_node(run_node)
        
        # Link run to parent experiment
        if graph.has_node(exp_id):
            edge = Edge(
                type="derived_from",
                source=run_id,
                target=exp_id,
                confidence=1.0,
                evidence=f"Metrics from {pattern}",
            )
            graph.add_edge(edge)
        
        # Create metric nodes for key metrics
        metric_keys = {
            "val_accuracy_overall": ("Overall Accuracy", "%"),
            "val_accuracy_single_hop": ("1-Hop Accuracy", "%"),
            "val_accuracy_two_hop": ("2-Hop Accuracy", "%"),
            "val_accuracy_three_hop": ("3-Hop Accuracy", "%"),
            "val_recall_at_1": ("Recall@1", "%"),
            "val_recall_at_8": ("Recall@8", "%"),
            "val_recall_at_32": ("Recall@32", "%"),
            "best_val_loss": ("Best Val Loss", ""),
            "param_count": ("Parameter Count", ""),
            "total_wall_s": ("Wall Time", "s"),
            "num_live_slots": ("Live Slots", ""),
        }
        
        for metric_key, (metric_name, unit) in metric_keys.items():
            if metric_key in metrics:
                metric_id = f"Metric_{run_id}_{metric_key}"
                metric_node = Node(
                    id=metric_id,
                    type="Metric",
                    properties={
                        "name": metric_name,
                        "value": metrics[metric_key],
                        "unit": unit,
                        "experiment": run_id,
                    },
                    sources=[str(metrics_path.relative_to(experiments_dir.parent))],
                )
                graph.add_node(metric_node)
                
                # Link metric to run
                edge = Edge(
                    type="derived_from",
                    source=metric_id,
                    target=run_id,
                    confidence=1.0,
                    evidence=f"Measured during {metrics.get('run_name', run_id)}",
                )
                graph.add_edge(edge)
    
    # Phase 3: Create dependency edges between experiments
    for exp_key, exp_def in EXPERIMENT_DEFS.items():
        for dep_id in exp_def.get("depends_on", []):
            if graph.has_node(exp_def["id"]) and graph.has_node(dep_id):
                edge = Edge(
                    type="depends_on",
                    source=exp_def["id"],
                    target=dep_id,
                    confidence=1.0,
                    evidence=f"Experiment {exp_def['id']} builds on {dep_id}",
                )
                graph.add_edge(edge)
    
    # Phase 4: Create key concepts and link to experiments
    concepts = {
        "Concept_OracleMemory": {
            "description": "Proves SAM core CAN use external memory — 100% accuracy",
            "validated_by": ["Exp_0_6_Validation"],
            "contradicted_by": [],
        },
        "Concept_SelectorBottleneck": {
            "description": "Learned selector precision is the critical bottleneck (50% precision, 96.6% recall)",
            "validated_by": ["Exp_0_12_Selection"],
            "contradicted_by": [],
        },
        "Concept_ChainRetrieval": {
            "description": "Chain-set BCE retriever achieves 100% all_required@32 — retrieval is solved",
            "validated_by": ["Exp_0_11_ChainRetrieval"],
            "contradicted_by": [],
        },
        "Concept_NoiseTolerance": {
            "description": "SAM tolerates controlled random noise (+8 distractors -> 91.6%), gate is NOT the bottleneck",
            "validated_by": ["Exp_0_13A_NoisyMemory"],
            "contradicted_by": [],
        },
        "Concept_RetrievalMismatch": {
            "description": "Dual encoder query projection mismatch prevents SAM from using retrieved memory",
            "validated_by": ["Exp_0_6_Validation"],
            "contradicted_by": ["Exp_0_12_Selection"],  # Oracle-filter = 100%, so path works
        },
        "Concept_ArchitectureWorks": {
            "description": "Oracle memory = 99.87-100%, oracle filter = 100% — the core+memory architecture IS valid",
            "validated_by": ["Exp_0_6_Validation", "Exp_0_12_Selection", "Exp_0_13A_NoisyMemory"],
            "contradicted_by": [],
        },
        "Concept_PivotToNEXUS": {
            "description": "Flat latent-vector memory can't solve selection quality -> pivot to graph-first architecture",
            "validated_by": ["Exp_0_12_Selection", "Exp_0_13A_NoisyMemory"],
            "contradicted_by": [],
        },
    }
    
    for concept_id, concept_def in concepts.items():
        node = Node(
            id=concept_id,
            type="Concept",
            properties={
                "description": concept_def["description"],
            },
            sources=["ANALYSIS_AND_ROADMAP.md"],
            aliases=_CONCEPT_ALIASES.get(concept_id, []),
        )
        graph.add_node(node)
        
        for exp_id in concept_def["validated_by"]:
            if graph.has_node(exp_id):
                edge = Edge(
                    type="validates",
                    source=exp_id,
                    target=concept_id,
                    confidence=0.95,
                    evidence=f"Experiment {exp_id} validates {concept_id}",
                )
                graph.add_edge(edge)
        
        for exp_id in concept_def["contradicted_by"]:
            if graph.has_node(exp_id):
                edge = Edge(
                    type="contradicts",
                    source=exp_id,
                    target=concept_id,
                    confidence=0.85,
                    evidence=f"Experiment {exp_id} partially contradicts {concept_id}",
                )
                graph.add_edge(edge)
    
    # Phase 5: Add decision nodes for the pivot
    decision = Node(
        id="Decision_PivotToNEXUS",
        type="Decision",
        properties={
            "description": "Pivot from SAM (latent-vector associative memory) to NEXUS (graph-first reasoning)",
            "rationale": "Selector precision bottleneck (50%) is structural — flat MLPs can't solve graph-structured selection. "
                         "Knowledge should be explicit entities + relations, not latent vectors.",
            "date": "2026-07-08",
        },
        sources=["ANALYSIS_AND_ROADMAP.md"],
    )
    graph.add_node(decision)
    
    # Link decision to supporting concepts
    for concept_id in ["Concept_SelectorBottleneck", "Concept_PivotToNEXUS", "Concept_ArchitectureWorks"]:
        if graph.has_node(concept_id):
            edge = Edge(
                type="derived_from",
                source=decision.id,
                target=concept_id,
                confidence=0.9,
                evidence="Pivot decision derived from experimental evidence",
            )
            graph.add_edge(edge)
    
    return graph


def exp_key_from_id(exp_id: str) -> str:
    """Reverse-map experiment ID to key."""
    for key, defn in EXPERIMENT_DEFS.items():
        if defn["id"] == exp_id:
            return key
    return ""


def main():
    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)
    
    print(f"Graph populated: {graph.node_count} nodes, {graph.edge_count} edges")
    print(f"\nNode types: {graph.stats()['node_types']}")
    
    # Demo: find key concepts
    print("\n=== Key Concepts ===")
    for concept_id in [n for n in graph._nodes if n.startswith("Concept_")]:
        node = graph.get_node(concept_id)
        print(f"  {concept_id}: {node.properties.get('description', '')}")
    
    print("\n=== Experiment Dependency Chain ===")
    for exp_key, exp_def in EXPERIMENT_DEFS.items():
        deps = exp_def.get("depends_on", [])
        if deps:
            print(f"  {exp_def['id']} -> depends_on -> {', '.join(deps)}")
    
    # Save graph stats
    print(f"\n=== Graph Stats ===")
    print(f"  Nodes: {graph.node_count}")
    print(f"  Edges: {graph.edge_count}")
    print(f"  Experiments: {len([n for n in graph._nodes if graph._nodes[n].type == 'Experiment'])}")
    print(f"  Metrics: {len([n for n in graph._nodes if graph._nodes[n].type == 'Metric'])}")
    print(f"  Concepts: {len([n for n in graph._nodes if graph._nodes[n].type == 'Concept'])}")
    print(f"  Decisions: {len([n for n in graph._nodes if graph._nodes[n].type == 'Decision'])}")


if __name__ == "__main__":
    main()
