"""Convert pinned HotpotQA Parquet shards to the compact corpus-v2 input.

Only gold supporting sentences are retained. Install the ``data`` extra before
running this one-time acquisition step. The converter does not generate,
translate or paraphrase any text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    supporting = {
        (str(title), int(sentence_id))
        for title, sentence_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
            strict=True,
        )
    }
    evidence = []
    for title, sentences in zip(
        row["context"]["title"], row["context"]["sentences"], strict=True,
    ):
        for sentence_id, text in enumerate(sentences):
            if (str(title), sentence_id) in supporting:
                evidence.append({
                    "title": str(title), "sentence_id": sentence_id, "text": str(text),
                })
    return {
        "id": str(row["id"]),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "type": str(row["type"]),
        "level": str(row["level"]),
        "evidence": evidence,
    }


def convert(input_path: Path, output_path: Path) -> int:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("install the data extra: pip install -e '.[data]'") from exc
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    count = 0
    source = parquet.ParquetFile(input_path)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for batch in source.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                handle.write(json.dumps(
                    compact_row(row), ensure_ascii=False, separators=(",", ":"),
                ) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps({"records": convert(args.input, args.output), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
