"""Acquire genuinely distinct, train-only Realizer claims from repository sources.

The acquisition boundary is deliberately narrower than ``git ls-files``.  It
uses authored documentation, experiment reports, configuration files, and
public API docstrings, while excluding every evaluation split and generated
benchmark/result directory.  One record represents one source property, table
cell, prose claim, or API contract; it never creates paraphrases of a claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import lzma
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.realizer_contracts import canonical_json, normalize_question, sha256_file, sha256_json
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore


ACQUISITION_SCHEMA_VERSION = "nexus-realizer-acquisition-v1"
_DISALLOWED_PARTS = frozenset({"results", "data", ".git", ".pytest_cache", "__pycache__"})
_EVALUATION_NAMES = frozenset({"test.jsonl", "val.jsonl", "validation.jsonl", "holdout.jsonl"})
_EXCLUDED_SOURCE_PATHS = frozenset({
    "EXPERIMENT_NEXUS_REALIZER_V1.md",
    "docs/nexus-realizer-pretraining-status.md",
})
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "what", "when", "which", "with",
})


@dataclass(frozen=True)
class AtomicClaim:
    kind: str
    source_path: str
    locator: str
    subject: str
    predicate: str
    answer: str
    question: str

    @property
    def semantic_target_id(self) -> str:
        identity = {
            "kind": self.kind,
            "source_path": self.source_path,
            "locator": self.locator,
            "predicate": self.predicate,
        }
        return "target_" + sha256_json(identity)[:24]

    @property
    def claim_node_id(self) -> str:
        return "TrainClaim_" + self.semantic_target_id.removeprefix("target_")

    @property
    def source_node_id(self) -> str:
        return "TrainSource_" + hashlib.sha256(self.source_path.encode("utf-8")).hexdigest()[:20]

    def to_record(self, source_sha256: str) -> dict[str, Any]:
        return {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "id": self.semantic_target_id,
            "semantic_target_id": self.semantic_target_id,
            "kind": self.kind,
            "source_split": "train",
            "source_path": self.source_path,
            "source_sha256": source_sha256,
            "source_locator": self.locator,
            "source_family": self.source_node_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "question": self.question,
            "answer": self.answer,
            "entities": [self.claim_node_id, self.source_node_id],
            "question_type": "factual",
            "intent": "factual_lookup",
        }


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def discover_sources(root: Path = _PROJECT_ROOT) -> list[Path]:
    """Return the explicit train-only source corpus."""
    candidates: set[Path] = set()
    candidates.update(root.glob("*.md"))
    candidates.update((root / "docs").glob("*.md"))
    candidates.update((root / "sam-lm" / "docs").glob("*.md"))
    candidates.update((root / "sam-lm" / "experiments").glob("*report*.md"))
    candidates.update((root / "sam-lm" / "configs").glob("*.yaml"))
    candidates.update((root / "sam-lm" / "configs").glob("*.yml"))
    candidates.update((root / "models").glob("**/config.json"))
    # Evaluation and gate implementations are intentionally not training data.
    for package in ("nexus", "stack"):
        candidates.update((root / package).glob("**/*.py"))

    allowed: list[Path] = []
    for path in candidates:
        rel = _relative(path, root)
        parts = set(Path(rel).parts)
        if not path.is_file() or parts & _DISALLOWED_PARTS:
            continue
        if rel in _EXCLUDED_SOURCE_PATHS:
            continue
        if path.name.casefold() in _EVALUATION_NAMES:
            continue
        allowed.append(path)
    return sorted(allowed, key=lambda item: _relative(item, root))


def _clean_markdown(value: str) -> str:
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    value = re.sub(r"^\s*(?:[-*+] |\d+[.)] )", "", value)
    return re.sub(r"\s+", " ", value).strip(" |\t")


def _useful_text(value: str, *, minimum: int = 12, maximum: int = 900) -> bool:
    if not minimum <= len(value) <= maximum:
        return False
    if value.startswith(("http://", "https://")):
        return False
    words = re.findall(r"[A-Za-zÀ-ž0-9_%-]+", value)
    return len(words) >= 3 and len(set(word.casefold() for word in words)) >= 3


def _useful_scalar(value: str) -> bool:
    return bool(value and value not in {"-", "—", "n/a", "N/A"} and len(value) <= 500)


def _anchor(value: str, limit: int = 9) -> str:
    words = [
        word for word in re.findall(r"[A-Za-zÀ-ž0-9_./+%-]+", value)
        if word.casefold() not in _STOPWORDS
    ]
    return " ".join(words[:limit]).strip(".,:;-") or "the recorded finding"


def _split_prose(line: str) -> Iterator[str]:
    cleaned = _clean_markdown(line)
    for part in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ž0-9])", cleaned):
        part = part.strip()
        if _useful_text(part):
            yield part


def _markdown_claims(path: Path, root: Path) -> Iterator[AtomicClaim]:
    rel = _relative(path, root)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    section = "Overview"
    in_fence = False
    index = 0
    while index < len(lines):
        raw = lines[index]
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", raw)
        if heading:
            section = _clean_markdown(heading.group(1))[:160] or "Overview"
            index += 1
            continue

        # Header + separator + rows: each non-identity cell is one atomic target.
        if "|" in raw and index + 1 < len(lines) and re.match(
            r"^\s*\|?\s*:?-{3,}", lines[index + 1]
        ):
            headers = [_clean_markdown(cell) for cell in raw.strip().strip("|").split("|")]
            row_index = index + 2
            while row_index < len(lines) and "|" in lines[row_index]:
                cells = [_clean_markdown(cell) for cell in lines[row_index].strip().strip("|").split("|")]
                if len(cells) < 2:
                    break
                subject = cells[0] or f"row {row_index + 1}"
                for column, value in zip(headers[1:], cells[1:]):
                    if not column or not _useful_scalar(value):
                        continue
                    locator = f"L{row_index + 1}:table:{column}"
                    answer = f"For {subject}, {column} is {value}."
                    question = f"In {rel}, under {section}, what is {column} for {subject}?"
                    yield AtomicClaim("table_cell", rel, locator, subject, column, answer, question)
                row_index += 1
            index = row_index
            continue

        # Join wrapped prose/bullet continuations before sentence extraction.
        # This prevents line-wrapped Markdown from becoming truncated answers.
        start_index = index
        block = [raw]
        while index + 1 < len(lines):
            following = lines[index + 1]
            if not following.strip() or following.lstrip().startswith(("#", "```")):
                break
            if "|" in following or re.match(r"^\s*(?:[-*+] |\d+[.)] )", following):
                break
            index += 1
            block.append(following)
        cleaned = _clean_markdown(" ".join(block))
        if not cleaned or cleaned in {"---", "***", "___"}:
            index += 1
            continue
        key_value = re.match(r"^([^:]{2,100}):\s+(.+)$", cleaned)
        if key_value and _useful_text(key_value.group(2), minimum=2):
            subject, value = key_value.group(1).strip(), key_value.group(2).strip()
            answer = f"{subject}: {value}"
            question = f"In {rel}, under {section}, what is recorded for {subject}?"
            yield AtomicClaim("markdown_field", rel, f"L{start_index + 1}:field:{subject}", subject, subject, answer, question)
        else:
            for sentence_index, sentence in enumerate(_split_prose(raw), start=1):
                anchor = _anchor(sentence)
                question = f"What does {rel}, under {section}, state about {anchor}?"
                yield AtomicClaim(
                    "markdown_claim", rel, f"L{start_index + 1}:sentence:{sentence_index}",
                    anchor, "states", sentence, question,
                )
        index += 1


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(value[key], child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield prefix, value


def _config_claims(path: Path, root: Path) -> Iterator[AtomicClaim]:
    rel = _relative(path, root)
    text = path.read_text(encoding="utf-8", errors="strict")
    payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    for key_path, value in _flatten(payload):
        if not key_path:
            continue
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        answer = f"In {rel}, {key_path} is set to {rendered}."
        question = f"What value is assigned to {key_path} in {rel}?"
        yield AtomicClaim("config_value", rel, f"key:{key_path}", key_path, "configured_as", answer, question)


def _first_doc_sentence(docstring: str) -> str:
    paragraph = re.split(r"\n\s*\n", docstring.strip(), maxsplit=1)[0]
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    match = re.match(r"(.+?[.!?])(?:\s|$)", paragraph)
    return (match.group(1) if match else paragraph)[:900]


def _api_claims(path: Path, root: Path) -> Iterator[AtomicClaim]:
    rel = _relative(path, root)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    module = rel.removesuffix(".py").replace("/", ".")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node, clean=True)
        if not doc:
            continue
        sentence = _first_doc_sentence(doc)
        if not _useful_text(sentence):
            continue
        symbol = f"{module}.{node.name}"
        answer = f"{symbol}: {sentence}"
        question = f"What behavior or responsibility is documented for {symbol}?"
        yield AtomicClaim("api_contract", rel, f"symbol:{node.name}", symbol, "documents", answer, question)


def extract_claims(path: Path, root: Path = _PROJECT_ROOT) -> Iterable[AtomicClaim]:
    if path.suffix.casefold() == ".md":
        return _markdown_claims(path, root)
    if path.suffix.casefold() in {".yaml", ".yml", ".json"}:
        return _config_claims(path, root)
    if path.suffix.casefold() == ".py":
        return _api_claims(path, root)
    return ()


def acquire_claim_records(root: Path = _PROJECT_ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract and strictly deduplicate atomic source targets."""
    sources = discover_sources(root)
    records: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    rejected: Counter[str] = Counter()
    source_hashes = {_relative(path, root): sha256_file(path) for path in sources}

    for path in sources:
        rel = _relative(path, root)
        for claim in extract_claims(path, root):
            target = claim.semantic_target_id
            question = normalize_question(claim.question)
            answer = re.sub(r"\s+", " ", claim.answer).strip().casefold()
            if target in seen_targets:
                rejected["duplicate_semantic_target"] += 1
                continue
            if question in seen_questions:
                rejected["duplicate_normalized_question"] += 1
                continue
            if answer in seen_answers:
                rejected["duplicate_normalized_answer"] += 1
                continue
            seen_targets.add(target)
            seen_questions.add(question)
            seen_answers.add(answer)
            records.append(claim.to_record(source_hashes[rel]))

    records.sort(key=lambda item: item["semantic_target_id"])
    kind_counts = Counter(str(item["kind"]) for item in records)
    manifest = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_policy": "explicit_train_only_repository_sources_v1",
        "records": len(records),
        "semantic_targets_unique": len({item["semantic_target_id"] for item in records}),
        "normalized_questions_unique": len({normalize_question(item["question"]) for item in records}),
        "normalized_answers_unique": len({re.sub(r'\s+', ' ', item["answer"]).strip().casefold() for item in records}),
        "source_families": len({item["source_family"] for item in records}),
        "counts_by_kind": dict(sorted(kind_counts.items())),
        "rejected_duplicates": dict(sorted(rejected.items())),
        "sources": [{"path": path, "sha256": source_hashes[path]} for path in sorted(source_hashes)],
    }
    manifest["source_set_sha256"] = sha256_json(manifest["sources"])
    manifest["records_sha256"] = hashlib.sha256(
        b"".join((canonical_json(item) + "\n").encode("utf-8") for item in records)
    ).hexdigest()
    return records, manifest


def augment_graph_with_claims(
    graph: InMemoryGraphStore,
    records: Iterable[dict[str, Any]],
) -> InMemoryGraphStore:
    """Add source-family and atomic-claim nodes required by oracle building."""
    for record in records:
        claim_id, source_id = record["entities"]
        source_path = str(record["source_path"])
        if not graph.has_node(source_id):
            graph.add_node(Node(
                id=source_id,
                type="Document",
                properties={"name": source_path, "description": f"Train-only source: {source_path}."},
                sources=[source_path],
                aliases=[source_path],
            ))
        graph.add_node(Node(
            id=claim_id,
            type="Concept",
            properties={
                "name": str(record["subject"]),
                "key_finding": str(record["answer"]),
                "description": str(record["answer"]),
                "source_snippet": str(record["answer"]),
                "semantic_target_id": str(record["semantic_target_id"]),
            },
            sources=[f"{source_path}#{record['source_locator']}"],
            aliases=[str(record["subject"])],
        ))
        graph.add_edge(Edge(
            type="derived_from",
            source=claim_id,
            target=source_id,
            confidence=1.0,
            evidence=f"{source_path}#{record['source_locator']}",
        ))
    return graph


def write_acquisition(output: Path, records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "source_claims.jsonl.xz"
    canonical_records = b"".join(
        (canonical_json(record) + "\n").encode("utf-8") for record in records
    )
    records_path.write_bytes(lzma.compress(canonical_records, preset=9))
    final_manifest = dict(manifest)
    final_manifest["records_file"] = records_path.name
    final_manifest["records_file_sha256"] = sha256_file(records_path)
    (output / "manifest.json").write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_verified_acquisition(
    manifest_path: Path,
    root: Path = _PROJECT_ROOT,
    *,
    verify_current_sources: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and authenticate an immutable acquisition snapshot.

    ``verify_current_sources`` additionally requires today's worktree files to
    match the archived source hashes. It may be disabled when reproducing a
    dataset from the already hash-identified compressed snapshot after normal
    documentation/code evolution. Record provenance, source-set identity,
    uniqueness and the compressed archive hash are always verified.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ACQUISITION_SCHEMA_VERSION:
        raise ValueError("unsupported acquisition manifest schema")
    records_name = str(manifest.get("records_file", ""))
    if not records_name or Path(records_name).name != records_name:
        raise ValueError("acquisition records_file must be a local filename")
    records_path = manifest_path.parent / records_name
    if not records_path.is_file() or sha256_file(records_path) != manifest.get("records_file_sha256"):
        raise ValueError("acquisition records file hash mismatch")
    raw_records = (
        lzma.decompress(records_path.read_bytes()).decode("utf-8")
        if records_path.suffix == ".xz"
        else records_path.read_text(encoding="utf-8")
    )
    records = [json.loads(line) for line in raw_records.splitlines() if line]
    if len(records) != manifest.get("records"):
        raise ValueError("acquisition record count mismatch")
    if hashlib.sha256(b"".join(
        (canonical_json(item) + "\n").encode("utf-8") for item in records
    )).hexdigest() != manifest.get("records_sha256"):
        raise ValueError("acquisition canonical records hash mismatch")

    declared_sources = {str(item["path"]): str(item["sha256"]) for item in manifest.get("sources", [])}
    if sha256_json([
        {"path": path, "sha256": declared_sources[path]} for path in sorted(declared_sources)
    ]) != manifest.get("source_set_sha256"):
        raise ValueError("acquisition source-set hash mismatch")
    for rel, expected_hash in declared_sources.items():
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"source escapes repository root: {rel}") from exc
        parts = set(Path(rel).parts)
        if Path(rel).name.casefold() in _EVALUATION_NAMES or parts & _DISALLOWED_PARTS:
            raise ValueError(f"evaluation/generated source is forbidden: {rel}")
        if verify_current_sources and (
            not candidate.is_file() or sha256_file(candidate) != expected_hash
        ):
            raise ValueError(f"source hash mismatch: {rel}")

    target_ids: set[str] = set()
    questions: set[str] = set()
    answers: set[str] = set()
    for index, record in enumerate(records):
        if record.get("schema_version") != ACQUISITION_SCHEMA_VERSION or record.get("source_split") != "train":
            raise ValueError(f"record {index} is not an acquisition train record")
        source_path = str(record.get("source_path", ""))
        if source_path not in declared_sources or record.get("source_sha256") != declared_sources[source_path]:
            raise ValueError(f"record {index} has invalid source provenance")
        target_id = str(record.get("semantic_target_id", ""))
        normalized_question = normalize_question(str(record.get("question", "")))
        normalized_answer = re.sub(r"\s+", " ", str(record.get("answer", ""))).strip().casefold()
        if not target_id or target_id in target_ids:
            raise ValueError(f"record {index} duplicates a semantic target")
        if not normalized_question or normalized_question in questions:
            raise ValueError(f"record {index} duplicates a normalized question")
        if not normalized_answer or normalized_answer in answers:
            raise ValueError(f"record {index} duplicates a normalized answer")
        target_ids.add(target_id)
        questions.add(normalized_question)
        answers.add(normalized_answer)
    return records, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("data/realizer_train/source_claims_v1"))
    args = parser.parse_args()
    records, manifest = acquire_claim_records(args.root.resolve())
    write_acquisition(args.output, records, manifest)
    print(canonical_json({
        "records": len(records),
        "source_families": manifest["source_families"],
        "counts_by_kind": manifest["counts_by_kind"],
        "output": str(args.output),
    }))
    return 0 if len(records) >= 5000 else 2


if __name__ == "__main__":
    raise SystemExit(main())
