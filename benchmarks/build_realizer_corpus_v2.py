"""Build the large, non-expanded PL/EN NEXUS Realizer corpus v2.

Raw third-party datasets stay outside git. The committed source registry pins
their revisions, licenses and file hashes. One native source record can produce
at most one normalized record.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_corpus_v2_contracts import (
    ABSTENTION_TARGETS,
    MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SPLITS,
    canonical_json,
    normalized_text,
    record_id,
    record_text_key,
    sha256_file,
    sha256_json,
    text_fingerprint,
    validate_manifest,
    validate_record,
)


def _stable_eval_split(dataset: str, source_id: str, source_split: str) -> str:
    if source_split == "test":
        return "test"
    if source_split == "train":
        return "train"
    value = int(sha256_json({"dataset": dataset, "id": source_id})[:8], 16)
    return "validation" if value % 2 == 0 else "test"


def _evidence(
    dataset: str, source_id: str, index: int, title: str, text: str,
    locator: str,
) -> dict[str, Any]:
    return {
        "id": f"{dataset}:{source_id}:e{index}",
        "title": str(title).strip(),
        "text": str(text).strip(),
        "text_sha256": text_fingerprint(text),
        "source_locator": str(locator),
        "supporting": True,
    }


def _make_record(
    *, dataset: str, source: dict[str, Any], artifact_sha256: str,
    source_split: str, source_id: str, question: str, answer: str,
    aliases: list[str], answerable: bool, operator: str, hops: int,
    evidence: list[dict[str, Any]], groups: list[str], metadata: dict[str, Any],
) -> dict[str, Any]:
    split = _stable_eval_split(dataset, source_id, source_split)
    if not answerable:
        answer = ABSTENTION_TARGETS[source["language"]]
        aliases = []
        operator = "abstain"
        hops = 0
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": record_id(dataset, source["revision"], source_split, source_id),
        "dataset_split": split,
        "language": source["language"],
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "answer_aliases": sorted({str(item).strip() for item in aliases if str(item).strip()}),
        "answerable": answerable,
        "evidence": evidence,
        "document_group_ids": sorted(set(map(str, groups))),
        "semantic_plan": {
            "operator": operator,
            "hops": hops,
            "verified": True,
            "reasoning_owner": "nexus_upstream",
            "realizer_may_change_facts": False,
            "supporting_evidence_ids": [item["id"] for item in evidence],
        },
        "source": {
            "dataset": dataset,
            "revision": source["revision"],
            "source_split": source_split,
            "record_id": source_id,
            "license": source["license"],
            "url": source["url"],
            "artifact_sha256": artifact_sha256,
        },
        "metadata": metadata,
    }
    errors = validate_record(record)
    if errors:
        raise ValueError(f"invalid {dataset}/{source_id}: {','.join(errors)}")
    return record


def _iter_poquad(
    path: Path, source: dict[str, Any], file_meta: dict[str, Any], dataset: str,
) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_split = file_meta["source_split"]
    for document in payload["data"]:
        document_id = str(document["id"])
        for paragraph_index, paragraph in enumerate(document["paragraphs"]):
            context = str(paragraph["context"]).strip()
            for question_index, qa in enumerate(paragraph["qas"]):
                source_id = f"{document_id}:{paragraph_index}:{question_index}"
                answerable = not bool(qa.get("is_impossible"))
                answers = qa.get("answers", [])
                generative_targets = [
                    str(item.get("generative_answer") or "").strip()
                    for item in answers
                    if str(item.get("generative_answer") or "").strip()
                ]
                extractive_targets = [
                    str(item.get("text") or "").strip()
                    for item in answers
                    if str(item.get("text") or "").strip()
                ]
                targets = generative_targets or extractive_targets
                if answerable and not targets:
                    continue
                evidence = [_evidence(
                    dataset, source_id, 0, str(document.get("title", "")), context,
                    f"{document.get('url', '')}#paragraph-{paragraph_index}",
                )]
                yield _make_record(
                    dataset=dataset, source=source, artifact_sha256=file_meta["sha256"],
                    source_split=source_split, source_id=source_id,
                    question=qa["question"], answer=targets[0] if targets else "",
                    aliases=[
                        item for item in extractive_targets + generative_targets[1:]
                        if item.casefold() != (targets[0].casefold() if targets else "")
                    ],
                    answerable=answerable, operator="extract", hops=1,
                    evidence=evidence, groups=[f"{dataset}:document:{document_id}"],
                    metadata={"domain": "wikipedia", "task": "closed_domain_qa"},
                )


def _iter_musique(
    path: Path, source: dict[str, Any], file_meta: dict[str, Any], dataset: str,
) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = str(row["id"])
            supporting = [item for item in row["paragraphs"] if item.get("is_supporting")]
            evidence = [
                _evidence(
                    dataset, source_id, index, item.get("title", ""),
                    item["paragraph_text"], f"paragraph:{item['idx']}",
                )
                for index, item in enumerate(supporting)
            ]
            groups = [
                f"{dataset}:title:{text_fingerprint(item.get('title', ''))}"
                for item in supporting
            ]
            if not evidence or not groups:
                continue
            yield _make_record(
                dataset=dataset, source=source, artifact_sha256=file_meta["sha256"],
                source_split=file_meta["source_split"], source_id=source_id,
                question=row["question"], answer=row["answer"],
                aliases=list(row.get("answer_aliases", [])),
                answerable=bool(row.get("answerable", True)), operator="compose_path",
                hops=len(row.get("question_decomposition", [])), evidence=evidence,
                groups=groups,
                metadata={"domain": "wikipedia", "task": "multi_hop_qa"},
            )


def _iter_hotpot(
    path: Path, source: dict[str, Any], file_meta: dict[str, Any], dataset: str,
) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = str(row["id"])
            evidence = [
                _evidence(
                    dataset, source_id, index, item.get("title", ""), item["text"],
                    f"{item.get('title', '')}#sentence-{item.get('sentence_id', index)}",
                )
                for index, item in enumerate(row.get("evidence", []))
            ]
            groups = [
                f"{dataset}:title:{text_fingerprint(item.get('title', ''))}"
                for item in row.get("evidence", [])
            ]
            if not evidence:
                continue
            operator = "compare" if row.get("type") == "comparison" else "compose_path"
            yield _make_record(
                dataset=dataset, source=source, artifact_sha256=file_meta["sha256"],
                source_split=file_meta["source_split"], source_id=source_id,
                question=row["question"], answer=row["answer"], aliases=[],
                answerable=True, operator=operator, hops=max(2, len(groups)),
                evidence=evidence, groups=groups,
                metadata={
                    "domain": "wikipedia", "task": "multi_hop_qa",
                    "difficulty": row.get("level"), "question_type": row.get("type"),
                },
            )


def _iter_polqa(
    path: Path, source: dict[str, Any], file_meta: dict[str, Any], dataset: str,
) -> Iterator[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[str(row["question_id"])].append(row)
    for source_id, rows in grouped.items():
        first = rows[0]
        try:
            aliases = [str(item).strip() for item in ast.literal_eval(first["answers"])]
        except (SyntaxError, ValueError):
            aliases = []
        relevant: list[dict[str, str]] = []
        seen_passages: set[str] = set()
        for row in rows:
            if row["relevant"].casefold() != "true":
                continue
            passage_key = str(row.get("passage_id") or text_fingerprint(row["passage_text"]))
            if passage_key in seen_passages:
                continue
            seen_passages.add(passage_key)
            relevant.append(row)
        if not aliases or not relevant:
            continue
        evidence = [
            _evidence(
                dataset, source_id, index, row.get("passage_title", ""),
                row["passage_text"], f"passage:{row.get('passage_id', index)}",
            )
            for index, row in enumerate(relevant)
        ]
        groups = [
            f"{dataset}:passage:{row.get('passage_id') or text_fingerprint(row['passage_text'])}"
            for row in relevant
        ]
        yield _make_record(
            dataset=dataset, source=source, artifact_sha256=file_meta["sha256"],
            source_split=file_meta["source_split"], source_id=source_id,
            question=first["question"], answer=aliases[0], aliases=aliases[1:],
            answerable=True, operator="extract", hops=1, evidence=evidence, groups=groups,
            metadata={
                "domain": "open_domain_trivia", "task": "open_domain_qa",
                "question_formulation": first.get("question_formulation"),
                "question_type": first.get("question_type"),
            },
        )


def _iter_multihop_rag(
    path: Path, source: dict[str, Any], file_meta: dict[str, Any], dataset: str,
) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        source_id = f"query-{index:05d}"
        evidence = [
            _evidence(
                dataset, source_id, evidence_index, item.get("title", ""), item["fact"],
                item.get("url", f"evidence:{evidence_index}"),
            )
            for evidence_index, item in enumerate(row.get("evidence_list", []))
        ]
        groups = [
            f"{dataset}:url:{text_fingerprint(item.get('url') or item.get('title', ''))}"
            for item in row.get("evidence_list", [])
        ]
        if not evidence:
            continue
        question_type = str(row.get("question_type", ""))
        operator = "compare" if "comparison" in question_type else "compose_path"
        yield _make_record(
            dataset=dataset, source=source, artifact_sha256=file_meta["sha256"],
            source_split=file_meta["source_split"], source_id=source_id,
            question=row["query"], answer=row["answer"], aliases=[], answerable=True,
            operator=operator, hops=len(evidence), evidence=evidence, groups=groups,
            metadata={
                "domain": "news", "task": "cross_document_qa",
                "question_type": question_type,
            },
        )


ADAPTERS = {
    "hotpot_jsonl": _iter_hotpot,
    "multihop_rag_json": _iter_multihop_rag,
    "musique_jsonl": _iter_musique,
    "polqa_csv": _iter_polqa,
    "poquad_json": _iter_poquad,
}

PREVIOUS_REALIZER_MANIFESTS = (
    Path("data/distillation/realizer_v1/manifest.json"),
    Path("data/distillation/realizer_abstractive_v1/manifest.json"),
)


def _load_previous_realizer_text() -> tuple[set[str], set[str], list[dict[str, Any]]]:
    questions: set[str] = set()
    text_keys: set[str] = set()
    manifests: list[dict[str, Any]] = []
    for relative in PREVIOUS_REALIZER_MANIFESTS:
        manifest_path = _project_root / relative
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_count = 0
        for split_meta in manifest.get("splits", {}).values():
            path = manifest_path.parent / split_meta["path"]
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    questions.add(normalized_text(record.get("question", "")))
                    text_keys.add(sha256_json({
                        "question": normalized_text(record.get("question", "")),
                        "answer": normalized_text(record.get("answer", "")),
                    }))
                    record_count += 1
        manifests.append({
            "path": str(relative), "sha256": sha256_file(manifest_path),
            "records_scanned": record_count,
        })
    return questions, text_keys, manifests


def _iter_source_records(
    registry: dict[str, Any], source_root: Path, wanted: set[str],
) -> Iterable[dict[str, Any]]:
    for dataset in sorted(registry["sources"]):
        source = registry["sources"][dataset]
        adapter = ADAPTERS[source["adapter"]]
        for file_meta in source["files"]:
            if file_meta["source_split"] not in wanted:
                continue
            path = source_root / file_meta["path"]
            yield from adapter(path, source, file_meta, dataset)


def _verify_sources(registry: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for dataset, source in sorted(registry["sources"].items()):
        if source.get("adapter") not in ADAPTERS:
            raise ValueError(f"unsupported adapter for {dataset}")
        for file_meta in source["files"]:
            path = source_root / file_meta["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != file_meta["sha256"]:
                raise ValueError(f"source hash mismatch: {path}: {actual}")
            artifacts.append({
                "dataset": dataset, "path": file_meta["path"],
                "source_split": file_meta["source_split"], "sha256": actual,
                "derived_from_sha256": file_meta.get("derived_from_sha256"),
            })
    return artifacts


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count, sha256_file(path)


def build_corpus(registry_path: Path, source_root: Path, output_dir: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("policy", {}).get("maximum_records_per_source_record") != 1:
        raise ValueError("registry must prohibit record expansion")
    artifacts = _verify_sources(registry, source_root)
    previous_questions, previous_text_keys, previous_manifests = _load_previous_realizer_text()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rejected: Counter[str] = Counter()
    accepted: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    eval_candidates = list(_iter_source_records(
        registry, source_root, {"validation", "valid", "dev", "test"},
    ))
    by_split = {
        split: sorted(
            (row for row in eval_candidates if row["dataset_split"] == split),
            key=lambda row: row["id"],
        )
        for split in ("test", "validation")
    }
    seen_questions: set[str] = set(previous_questions)
    seen_text_keys: set[str] = set(previous_text_keys)
    protected_groups: set[str] = set()
    protected_evidence: set[str] = set()
    for split in ("test", "validation"):
        for row in by_split[split]:
            question_key = normalized_text(row["question"])
            text_key = record_text_key(row)
            groups = set(row["document_group_ids"])
            evidence_hashes = {item["text_sha256"] for item in row["evidence"]}
            if question_key in seen_questions or text_key in seen_text_keys:
                rejected[f"{split}_duplicate_text"] += 1
                continue
            if split == "validation" and (
                groups & protected_groups or evidence_hashes & protected_evidence
            ):
                rejected["validation_test_leakage"] += 1
                continue
            accepted[split].append(row)
            seen_questions.add(question_key)
            seen_text_keys.add(text_key)
            protected_groups.update(groups)
            protected_evidence.update(evidence_hashes)

    train_path = output_dir / "train.jsonl"
    train_count = 0
    train_language: Counter[str] = Counter()
    train_operator: Counter[str] = Counter()
    train_source: Counter[str] = Counter()
    train_answerable: Counter[str] = Counter()
    with train_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in _iter_source_records(registry, source_root, {"train"}):
            question_key = normalized_text(row["question"])
            text_key = record_text_key(row)
            groups = set(row["document_group_ids"])
            evidence_hashes = {item["text_sha256"] for item in row["evidence"]}
            if question_key in seen_questions or text_key in seen_text_keys:
                rejected["train_duplicate_text"] += 1
                continue
            if groups & protected_groups:
                rejected["train_eval_document_leakage"] += 1
                continue
            if evidence_hashes & protected_evidence:
                rejected["train_eval_evidence_leakage"] += 1
                continue
            handle.write(canonical_json(row) + "\n")
            train_count += 1
            seen_questions.add(question_key)
            seen_text_keys.add(text_key)
            train_language[row["language"]] += 1
            train_operator[row["semantic_plan"]["operator"]] += 1
            train_source[row["source"]["dataset"]] += 1
            train_answerable[str(row["answerable"]).lower()] += 1

    split_meta: dict[str, dict[str, Any]] = {
        "train": {
            "path": train_path.name,
            "count": train_count,
            "sha256": sha256_file(train_path),
        }
    }
    all_language = Counter(train_language)
    all_operator = Counter(train_operator)
    all_source = Counter(train_source)
    all_answerable = Counter(train_answerable)
    for split in ("validation", "test"):
        path = output_dir / f"{split}.jsonl"
        count, digest = _write_rows(path, accepted[split])
        split_meta[split] = {"path": path.name, "count": count, "sha256": digest}
        for row in accepted[split]:
            all_language[row["language"]] += 1
            all_operator[row["semantic_plan"]["operator"]] += 1
            all_source[row["source"]["dataset"]] += 1
            all_answerable[str(row["answerable"]).lower()] += 1

    statistics = {
        "records_by_split": {split: split_meta[split]["count"] for split in SPLITS},
        "records_by_language": dict(sorted(all_language.items())),
        "records_by_operator": dict(sorted(all_operator.items())),
        "records_by_source": dict(sorted(all_source.items())),
        "records_by_answerability": dict(sorted(all_answerable.items())),
        "train_by_language": dict(sorted(train_language.items())),
        "train_by_operator": dict(sorted(train_operator.items())),
        "train_by_source": dict(sorted(train_source.items())),
        "rejected": dict(sorted(rejected.items())),
        "source_records_per_output_record_max": 1,
        "synthetic_records": 0,
        "translated_records": 0,
        "paraphrase_expansions": 0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "schema_version": registry["schema_version"],
        },
        "source_artifacts": artifacts,
        "excluded_repository_manifests": previous_manifests,
        "splits": split_meta,
        "statistics": statistics,
        "quality_gates": {
            "minimum_train_records": 100_000,
            "minimum_validation_records": 3_000,
            "minimum_test_records": 10_000,
            "minimum_language_fraction": 0.30,
            "minimum_train_sources": 4,
            "minimum_abstention_fraction": 0.05,
            "minimum_comparison_fraction": 0.05,
            "minimum_composition_fraction": 0.45,
            "zero_synthetic_records": True,
            "zero_duplicate_questions": True,
            "zero_train_eval_document_leakage": True,
            "zero_train_eval_evidence_leakage": True,
        },
        "architectural_contract": {
            "knowledge_owner": "graph_or_source_evidence",
            "reasoning_owner": "nexus_upstream",
            "realizer_role": "language_interface",
            "realizer_may_invent_facts": False,
            "realizer_receives_verified_plan": True,
        },
    }
    manifest["dataset_sha256"] = sha256_json({
        split: split_meta[split]["sha256"] for split in SPLITS
    })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    errors = validate_manifest(manifest, output_dir)
    if errors:
        raise ValueError("generated corpus failed validation: " + "; ".join(errors[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("training/realizer_corpus_v2_sources.json"),
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_corpus(args.registry, args.source_root, args.output)
    print(json.dumps({
        "status": "REALIZER_CORPUS_V2_READY",
        "output": str(args.output),
        "dataset_sha256": manifest["dataset_sha256"],
        "statistics": manifest["statistics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
