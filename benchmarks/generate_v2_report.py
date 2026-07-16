"""Generate the final v2 pilot report."""
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

manifest = json.loads(Path("data/distillation/realizer_v1/manifest.json").read_text(encoding="utf-8"))
eval_data = json.loads(
    Path("benchmarks/results/realizer/v2_20260716T131357Z/eval_epoch_001.json").read_text(encoding="utf-8"),
)
config_path = Path("training/nexus_realizer_v2.json")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()

report = {
    "schema_version": "nexus-realizer-pilot-v2",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "REALIZER_NEURAL_CHECKPOINT_REJECTED",
    "status_detail": (
        "Epoch 1 neural model achieves 0% grounding rate. "
        "All neural outputs fail evidence support check. "
        "Training stopped per Section 6 (grounded_rate < 0.50)."
    ),
    "source_commit": commit,
    "source_tree_sha": tree,
    "run_id": "v2_20260716T131357Z",
    "epochs_completed": 1,
    "epochs_max": 3,
    "stop_reason": "grounded_rate_below_threshold",
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "dataset_sha256": manifest["dataset_sha256"],
    "model_architecture": "stable_transformer_v2",
    "source_format": "grounded_compact_v2",
    "training_summary": {
        "epoch": 1,
        "train_loss": 1.824445,
        "validation_loss": 1.457496,
        "initial_loss": 5.991638,
        "parameter_count": 951555,
        "elapsed_seconds": 129,
        "peak_rss_mb": 816.6,
    },
    "comparison": {
        "untrained_v2": {
            "exact_match": 0.0, "token_f1": 0.0, "similarity": 0.07,
            "grounded_rate": 0.0, "uniqueness": 0.64,
            "eos": 0.0, "empty": 0.46, "rep_3gram": 0.25,
        },
        "epoch_1_raw_neural": eval_data["raw_neural"],
        "epoch_1_grounded_hybrid": eval_data["grounded_hybrid"],
        "grounded_only_baseline": {
            "exact_match": 1.0, "token_f1": 1.0, "hallucination": 0.0,
            "fallback_rate": 1.0, "uniqueness": 1.0,
        },
    },
    "gates": {
        "exact_match": {"status": "FAIL", "value": 0.0, "threshold": ">= 0.70"},
        "token_f1": {"status": "FAIL", "value": 0.257, "threshold": ">= 0.85"},
        "similarity": {"status": "FAIL", "value": 0.506, "threshold": ">= 0.85"},
        "grounded_rate": {"status": "FAIL", "value": 0.0, "threshold": ">= 0.90"},
        "hallucination": {"status": "FAIL", "value": 1.0, "threshold": "<= 0.05"},
        "uniqueness": {"status": "PASS", "value": 0.922, "threshold": ">= 0.80"},
        "eos": {"status": "PASS", "value": 1.0, "threshold": ">= 0.95"},
        "empty_output": {"status": "PASS", "value": 0.0, "threshold": "= 0"},
        "mode_collapse": {"status": "PASS", "value": "None (1322 unique/1434)"},
    },
    "continuation_conditions": {
        "neural_uniqueness": {"status": "PASS", "value": 0.922, "threshold": ">= 0.50"},
        "max_single_output": {"status": "PASS", "value": 0.011, "threshold": "<= 0.10"},
        "neural_token_f1": {"status": "PASS", "value": 0.257, "threshold": ">= 0.10"},
        "grounded_rate_stop": {"status": "STOP", "value": 0.0, "threshold": ">= 0.50"},
        "hallucination_stop": {"status": "STOP", "value": 1.0, "threshold": "<= 0.25"},
        "empty_output": {"status": "PASS", "value": 0.0, "threshold": "<= 0.05"},
    },
    "failure_analysis": {
        "primary_issue": "neural_grounding_zero",
        "description": (
            "Despite measurable behavioral improvements (token-F1 0%->25.7%, "
            "similarity 7%->50.6%, EOS 0%->100%, empty 46%->0%), the neural model "
            "produces zero grounded answers. All 1434 neural outputs fail the "
            "grounding check (threshold 0.72). The Grounded Realizer falls back to "
            "evidence_copy for 100% of answers."
        ),
        "root_causes": [
            "Model capacity (951K params) may be insufficient",
            "Byte-level tokenizer limits semantic understanding",
            "Single-epoch training insufficient for evidence-to-answer mapping",
            "grounding_score requires exact numeric + token overlap",
        ],
    },
    "recommended_next_steps": [
        "Audit whether targets are complete evidence candidates before changing the model",
        "Use deterministic Pointer/Copy for full-candidate extractive targets",
        "Do not relax grounding thresholds to make an unsupported checkpoint pass",
        "Collect genuinely unique train-only abstractive targets before neural retraining",
        "Limit any future neural pilot to 1, then 3, then at most 5 epochs with raw-output gates",
    ],
}

out_path = Path("benchmarks/results/realizer/v2_20260716T131357Z/pilot_report_v2.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
)
digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
(out_path.with_suffix(out_path.suffix + ".sha256")).write_text(
    digest + "  " + out_path.name + "\n", encoding="ascii",
)
print("REALIZER_NEURAL_CHECKPOINT_REJECTED")
print("Report:", out_path)
