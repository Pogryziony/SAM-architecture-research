"""Build and freeze the registered NEXUS oracle dataset (oracle_v1).

Writes:
  - benchmarks/qa-dataset/oracle_v1.jsonl
  - benchmarks/qa-dataset/oracle_v1.manifest.json

Does not read the consumed frozen test split. Re-running with identical
sources is deterministic; overwriting requires --force.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import sha256_file, sha256_json
from benchmarks.run_nexus_oracle import (
    ORACLE_SCHEMA_VERSION,
    build_oracle_records,
    validate_oracle_records,
)

DEFAULT_QUESTIONS = _project_root / "stack" / "encoder" / "data" / "val.jsonl"
DEFAULT_RELATIONS = _project_root / "benchmarks" / "qa-dataset" / "relation_gold.jsonl"
DEFAULT_FAMILIES = _project_root / "benchmarks" / "qa-dataset" / "oracle_families_v1.jsonl"
DEFAULT_OUTPUT = _project_root / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"
DEFAULT_MANIFEST = _project_root / "benchmarks" / "qa-dataset" / "oracle_v1.manifest.json"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def merge_oracle_records(
    base_records: list[dict],
    family_records: list[dict],
) -> list[dict]:
    """Merge curated family records into the base oracle contract."""
    by_id = {str(row["id"]): row for row in base_records}
    for row in family_records:
        record_id = str(row["id"])
        if record_id in by_id:
            raise ValueError(f"duplicate oracle record id: {record_id}")
        by_id[record_id] = row
    records = sorted(by_id.values(), key=lambda item: str(item["id"]))
    errors = validate_oracle_records(records)
    if errors:
        raise ValueError("invalid oracle records after family merge: " + "; ".join(errors))
    return records


def build_frozen_dataset(
    questions_path: Path,
    relations_path: Path,
    families_path: Path | None = None,
) -> tuple[list[dict], dict]:
    """Return (records, manifest) for the frozen oracle contract."""
    if questions_path.name.casefold() == "test.jsonl":
        raise ValueError("the consumed frozen test split is forbidden")
    question_rows = _read_jsonl(questions_path)
    relation_rows = _read_jsonl(relations_path)
    records = build_oracle_records(question_rows, relation_rows)
    sources = {
        str(questions_path.as_posix()): sha256_file(questions_path),
        str(relations_path.as_posix()): sha256_file(relations_path),
    }
    if families_path is not None and families_path.exists():
        family_rows = _read_jsonl(families_path)
        records = merge_oracle_records(records, family_rows)
        sources[str(families_path.as_posix())] = sha256_file(families_path)
    errors = validate_oracle_records(records)
    if errors:
        raise ValueError("invalid oracle records: " + "; ".join(errors))
    manifest = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "dataset_id": "oracle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "sha256": sha256_json(records),
        "sources": sources,
        "category_counts": {},
    }
    counts: dict[str, int] = {}
    for row in records:
        key = str(row["category"])
        counts[key] = counts.get(key, 0) + 1
    manifest["category_counts"] = dict(sorted(counts.items()))
    return records, manifest


def write_frozen_dataset(
    records: list[dict],
    manifest: dict,
    output_path: Path,
    manifest_path: Path,
    *,
    force: bool = False,
) -> None:
    if not force and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"refusing to overwrite {output_path} / {manifest_path}; pass --force"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Bind file hash after write for integrity checks.
    manifest = dict(manifest)
    manifest["file_sha256"] = sha256_file(output_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    records, manifest = build_frozen_dataset(
        args.questions, args.relations, families_path=args.families
    )
    write_frozen_dataset(
        records, manifest, args.output, args.manifest, force=args.force
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "records": len(records),
                "sha256": manifest["sha256"],
                "output": str(args.output),
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
