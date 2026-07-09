"""
Distillation logging for Stage 2+ — appends verifier-passed
(evidence_pack → answer) pairs to data/distillation/pairs.jsonl.

Free training data for Stage 4 Realization L2.

Usage:
    from benchmarks.distillation_logger import log_distillation_pairs
    log_distillation_pairs(results)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _get_pairs_path() -> Path:
    """Return the path to the distillation pairs file."""
    project_root = Path(__file__).parent.parent
    pairs_dir = project_root / "data" / "distillation"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    return pairs_dir / "pairs.jsonl"


def _compute_dedup_key(evidence_text: str, answer: str) -> str:
    """Compute a stable deduplication key for an (evidence, answer) pair."""
    content = f"{evidence_text.strip()}\n{answer.strip()}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def _load_existing_keys(pairs_path: Path) -> set[str]:
    """Load deduplication keys from existing pairs file."""
    keys: set[str] = set()
    if not pairs_path.exists():
        return keys
    try:
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        key = entry.get("dedup_key", "")
                        if key:
                            keys.add(key)
                    except json.JSONDecodeError:
                        continue
    except (OSError, IOError):
        pass
    return keys


def log_distillation_pairs(
    results: list[dict[str, Any]],
    pairs_path: Path | None = None,
) -> int:
    """Append verifier-passed evidence→answer pairs to pairs.jsonl.

    Only logs NEXUS arm results that:
    - Passed verification (hallucination <= threshold)
    - Have non-empty evidence_raw
    - Have non-empty answer
    - Are not "Insufficient evidence"

    Returns the number of new pairs written.
    """
    if pairs_path is None:
        pairs_path = _get_pairs_path()

    existing_keys = _load_existing_keys(pairs_path)
    new_count = 0

    with open(pairs_path, "a", encoding="utf-8") as f:
        for result in results:
            # Only log NEXUS arm results
            if result.get("arm_mode") != "nexus":
                continue

            nexus = result.get("nexus", {})
            if not nexus.get("passed", False):
                continue

            evidence_raw = nexus.get("evidence_raw", "")
            answer = nexus.get("answer", "")

            if not evidence_raw or not answer:
                continue

            if "insufficient evidence" in answer.lower():
                continue

            # Dedup
            dedup_key = _compute_dedup_key(evidence_raw, answer)
            if dedup_key in existing_keys:
                continue

            pair = {
                "dedup_key": dedup_key,
                "question_id": result.get("question_id", ""),
                "question": result.get("question", ""),
                "evidence": evidence_raw,
                "answer": answer,
                "hallucination_rate": nexus.get("hallucination_rate", 0.0),
            }

            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            existing_keys.add(dedup_key)
            new_count += 1

    return new_count


def get_pair_count(pairs_path: Path | None = None) -> int:
    """Return the number of pairs currently in the distillation file."""
    if pairs_path is None:
        pairs_path = _get_pairs_path()
    if not pairs_path.exists():
        return 0
    count = 0
    try:
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except (OSError, IOError):
        pass
    return count
