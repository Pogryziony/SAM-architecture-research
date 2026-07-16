"""Acquire every pinned external artifact for Realizer corpus v2."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.convert_hotpot_parquet import convert
from benchmarks.realizer_corpus_v2_contracts import sha256_file


def _download(url: str, target: Path, expected_sha256: str) -> str:
    if target.is_file():
        actual = sha256_file(target)
        if actual == expected_sha256:
            return "verified_existing"
        raise ValueError(f"existing artifact hash mismatch: {target}: {actual}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "NEXUS-corpus-v2/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise ValueError(f"download hash mismatch: {target}: {actual}")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return "downloaded"


def acquire(registry_path: Path, output_root: Path) -> list[dict[str, str]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    results: list[dict[str, str]] = []
    for dataset, source in sorted(registry["sources"].items()):
        for file_meta in source["files"]:
            target = output_root / file_meta["path"]
            if file_meta.get("derived_from_sha256"):
                origin = output_root / file_meta["origin_path"]
                origin_status = _download(
                    file_meta["download_url"], origin, file_meta["derived_from_sha256"],
                )
                if target.is_file():
                    actual = sha256_file(target)
                    if actual != file_meta["sha256"]:
                        raise ValueError(f"derived artifact hash mismatch: {target}: {actual}")
                    status = "verified_existing"
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    convert(origin, target)
                    actual = sha256_file(target)
                    if actual != file_meta["sha256"]:
                        target.unlink(missing_ok=True)
                        raise ValueError(f"derived artifact hash mismatch: {target}: {actual}")
                    status = "converted"
                results.append({
                    "dataset": dataset, "path": str(target), "status": status,
                    "origin_status": origin_status,
                })
            else:
                status = _download(file_meta["download_url"], target, file_meta["sha256"])
                results.append({"dataset": dataset, "path": str(target), "status": status})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("training/realizer_corpus_v2_sources.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        {"status": "SOURCES_READY", "artifacts": acquire(args.registry, args.output_root)},
        ensure_ascii=False, indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
