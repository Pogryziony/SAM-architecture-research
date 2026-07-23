"""Hash a corpus directory for sealed freeze records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.corpus_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    entries = []
    root_hasher = hashlib.sha256()
    for path in files:
        digest = hash_file(path)
        rel = path.relative_to(root).as_posix()
        entries.append({"path": rel, "sha256": digest, "bytes": path.stat().st_size})
        root_hasher.update(f"{rel}:{digest}\n".encode("utf-8"))
    report = {
        "schema_version": "nexus-corpus-hash-v1",
        "corpus_dir": str(root),
        "file_count": len(entries),
        "corpus_sha256": root_hasher.hexdigest(),
        "files": entries,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
