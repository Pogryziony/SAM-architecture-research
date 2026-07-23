import json
from pathlib import Path

from nexus.evaluation.aggregate import aggregate_question_records

nexus = json.loads(
    Path("benchmarks/results/eval_oracle_v1_grounded_phase3.json").read_text(
        encoding="utf-8"
    )
)
cb = json.loads(
    Path("benchmarks/results/phase4_qwen_closed_book_oracle_v1.json").read_text(
        encoding="utf-8"
    )
)
out = dict(nexus)
out["dataset_id"] = "oracle_v1"
out["dataset_sha256"] = cb["dataset_sha256"]
for row in out["per_question"]:
    row["dataset_id"] = "oracle_v1"
    row["dataset_sha256"] = cb["dataset_sha256"]
out["aggregates"] = aggregate_question_records(out["per_question"])
out["rebind_note"] = (
    "dataset_id/sha rebound to oracle_v1 for Phase4 paired compare; "
    "answers/config unchanged from eval_oracle_v1_grounded_phase3.json"
)
path = Path("benchmarks/results/eval_oracle_v1_grounded_phase3_dataset_rebind.json")
path.write_text(
    json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("wrote", path)
